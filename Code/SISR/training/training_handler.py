from datetime import date, datetime
import numpy as np
import os
import tqdm
import time
from collections import defaultdict
import torch
import random
import importlib
import math

from SISR.models import ModelInterface
import SISR.configuration.constants as sconst
from sr_tools.stats import save_statistics, plot_stats, legacy_load_statistics
from sr_tools.visualization import safe_image_save
from sr_tools.helper_functions import create_dir_if_empty, clean_models
from sr_tools.metrics import Metrics
from .data_setup import sisr_data_setup

aim_spec = importlib.util.find_spec("aim")  # only imports aim if this is available
if aim_spec is not None:
    from aim import Session


class TrainingHandler:
    def __init__(self,
                 # general params
                 experiment_name='experiment-%s' % date.today().strftime("%b-%d-%Y"), save_loc=sconst.results_directory,
                 aim_track=False, aim_home=os.path.join(sconst.results_directory, 'SISR'),
                 # model params
                 model_params=None, gpu='off', sp_gpu=1,
                 # data params
                 data_params=None,
                 # train params
                 num_epochs=None, continue_from_epoch=None, max_im_val=1.0, metrics=None, seed=8,
                 model_cleanup_frequency=None, epoch_cutoff=None,
                early_stopping_patience=None, overwrite_data=False,
                 branch_start=None, new_branch=False, logging='visual', save_samples=True, **kwargs):
        """
        Initializes a super-res ML experiment training handler
        :param experiment_name: model save name (defaults to date/time if not provided)
        :param aim_track: Set to True to track diagnostics using Aim
        :param aim_home: Home directory for aim tracking
        :param save_loc: model save location
        :param model_params: model instantiation parameters and sp
        :param gpu: 'single' - use one gpu, 'multi' - use all available gpus or 'off' - use CPU only
        :param sp_gpu: Select which specific gpu to be used
        :param data_params: parameters for setting up data handler
        :param num_epochs: number of epochs to train for
        :param continue_from_epoch: Restart training from a particular save point
        :param max_im_val: image excepted max pixel value
        :param metrics: metrics to monitor throughout training
        :param seed: Random generator seed initialization
        :param model_cleanup_frequency: Number of epochs to wait before wiping unneeded models
        :param epoch_cutoff: Epoch cutoff point (also considering any epochs previously run)
        :param early_stopping_patience: number of epochs after which training will end if no progress continues to be made
        :param overwrite_data: set to true and handler will overwrite any saved models with new data
        :param branch_start: name of branch to use for training
        :param new_branch: Set to true to construct a new save branch
        :param logging: Type of logging to perform during experiment - set to 'visual' to print out loss plts
        :param save_samples:  Set to true to save image samples after each training epoch
        :param kwargs: Any runoff parameters
        """

        # essential experimental setup
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.logging = logging
        self.save_samples = save_samples
        self.stop_patience = early_stopping_patience
        self.overwrite = overwrite_data
        self.model_cleanup_frequency = model_cleanup_frequency
        self.aim_track = aim_track

        # random seed initialization
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Implementation setup
        self.best_val_model_idx = 0
        self.best_val_model_psnr = 0
        self.model_name = model_params['name']  # model architecture name
        self.max_im_val = max_im_val  # image maximum pixel value (affects PSNR calculation)
        self.branch_off = new_branch

        # sets up a new branch - the model will start saving to a new directory within the results/saved models folders
        if new_branch and continue_from_epoch is not None:
            branch_name = 'branch_epoch_%d' % continue_from_epoch
            if branch_start is None:
                subdir = branch_name
            else:
                subdir = os.path.join(branch_start, branch_name)
        else:
            if branch_start is None:
                subdir = None
            else:
                subdir = branch_start

        # sets up model, loads in any provided checkpoint and initializes experiment directory
        self.model = ModelInterface(save_loc, experiment_name, gpu=gpu, sp_gpu=sp_gpu, mode='train',
                                    new_branch=self.branch_off,
                                    new_params=model_params, load_epoch=continue_from_epoch, save_subdir=subdir)

        self.starting_epoch = self.model.model_epoch  # extracts kick-off epoch

        if self.starting_epoch > 0:
            self.starting_epoch += 1  # start training from next epoch after loaded epoch

        if epoch_cutoff is not None:
            self.num_epochs = epoch_cutoff - self.starting_epoch  # set cutoff based on previously run epochs
            print('Epoch count set to %d' % self.num_epochs)

        # prepares provided training and eval datasets
        train_data, val_data = sisr_data_setup(scale=model_params['internal_params']['scale'],
                                               **self.model.configuration,
                                               qpi_sort=False, **data_params)
        self.train_data = train_data
        self.val_data = val_data

        if metrics is not None:  # prepares metrics system
            self.metric_hub = Metrics(metrics)
        else:
            self.metric_hub = None

        if aim_track:  # set up and configure an aim tracker - all logs saved to .aim folder in Aim home directory
            if aim_spec is None:
                raise RuntimeError('To activate Aim logging, please install aim using pip install aim')

            if continue_from_epoch is None:  # sets a unique run name
                run_name = experiment_name + '_%s' % datetime.today().strftime("%Hh-%Mm-%Ss-%b-%d-%Y")
            else:
                run_name = 'continuation_from_epoch_%d_' % self.model.model_epoch + experiment_name + \
                           '_%s' % datetime.today().strftime("%Hh-%Mm-%Ss-%b-%d-%Y")

            self.aim_session = Session(experiment=experiment_name, repo=aim_home, run=run_name)

            self.aim_session.set_params(model_params, name='model_parameters')  # passes over training parameters to Aim
            self.aim_session.set_params(data_params, name='data_parameters')
            self.aim_session.set_params({'num_epochs': num_epochs,
                                         'continue_from_epoch': continue_from_epoch,
                                         'seed': seed,
                                         'epoch_cutoff': epoch_cutoff},
                                        name='train_parameters')

    def train(self):
        """
        Function that takes care of a single training epoch -
        model is trained using each input batch, and losses/learning rates are logged.
        :return: Full epoch losses (dict).
        """
        current_epoch_losses = defaultdict(list)
        with tqdm.tqdm(total=len(self.train_data)) as pbar_train:
            for batch in self.train_data:
                losses, _ = self.model.train_batch(**batch)  # entire training scheme occurs here
                if type(losses) is dict:    # takes care to log all salient losses
                    for l_name, l_num in losses.items():
                        current_epoch_losses[l_name].append(l_num)
                    loss = losses['train-loss']
                else:
                    loss = losses
                    current_epoch_losses['train-loss'].append(loss)

                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}".format(loss))  # displays current loss

        learning_rates = self.model.get_learning_rate()  # extracts model learning rates for logging purposes
        if type(learning_rates) is dict:
            for m_key, m_lr in learning_rates.items():
                current_epoch_losses[m_key].append(m_lr)
        else:
            current_epoch_losses['learning-rate'].append(learning_rates)

        self.model.epoch_end_calls()  # any model-specific epoch end calls (e.g. for a scheduler)

        return current_epoch_losses

    def eval(self, epoch_idx):
        """
        This function takes care of single eval epoch - including metric calculation and logging.
        :param epoch_idx: Current epoch number.
        :return: Full epoch metrics (dict).
        """
        current_epoch_losses = defaultdict(list)
        with tqdm.tqdm(total=len(self.val_data)) as pbar_val:
            for index, batch in enumerate(self.val_data):

                y, im_names = batch['hr'], batch['tag']
                rgb_out, ycbcr_out, loss, timing = self.model.net_run_and_process(**batch, request_loss=True)

                # ensures that a Y-channel only image is produced for each batch,
                # to allow for standardised metric calculations
                if 'rgb' in self.model.configuration['colorspace']:
                    y_proc = self.model.colorspace_convert(y, colorspace='rgb')
                else:
                    y_proc = self.model._standard_image_formatting(y.numpy())

                # collect and record metrics based on eval run
                current_epoch_losses["val-loss"].append(loss)
                if self.metric_hub is not None:
                    metric_package, _ = self.metric_hub.run_metrics(ycbcr_out, references=y_proc,
                                                                    max_value=self.max_im_val, key='val',
                                                                    probe_names=[im_name.split('.')[0] for im_name
                                                                                 in im_names])
                    for metric, result in metric_package.items():
                        current_epoch_losses[metric].extend(result)

                # saves a single batch sample as a representative result
                if index == 0 and self.save_samples:
                    samples_folder = os.path.join(self.model.logs, 'epoch_%d_samples' % epoch_idx)
                    create_dir_if_empty(samples_folder)
                    im_names = [name.replace(os.sep, '_') for name in im_names]
                    safe_image_save(rgb_out, samples_folder, im_names, config='rgb')

                # displays diagnostics
                pbar_val.update(1)
                diag_string = 'loss: {:.4f}, '.format(loss)

                for metric in metric_package.keys():
                    diag_string += '{}: {:.4f}, '.format(metric, np.mean(metric_package[metric]))
                pbar_val.set_description(diag_string[:-2])

        return current_epoch_losses

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations according to spec,
        saving the model and results after each epoch.
        :return: Complete loss package (dict).
        """
        if self.model.mode == 'eval':
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        total_losses = defaultdict(list)
        if self.starting_epoch != 0:  # reloads old stats from file
            if self.branch_off:
                logs = os.path.dirname(self.model.logs)
            else:
                logs = self.model.logs
            total_losses = legacy_load_statistics(logs, 'summary.csv')
            if self.aim_track:  # loads up Aim with previous metrics
                for key, val in total_losses.items():
                    for epoch, item in enumerate(val):
                        self.aim_session.track(item, name=key.replace('-', '_'),
                                               epoch=epoch)

        improvement_count = 0

        for i, epoch_idx in enumerate(range(self.starting_epoch, self.starting_epoch+self.num_epochs)):

            # epoch initializations
            epoch_start_time = time.time()
            print('Running epoch', epoch_idx)
            self.model.set_epoch(epoch_idx)

            if i == 0:  # Test to ensure no data will be overwritten with this run
                self.model.save(override=self.overwrite, dry_run=True)

            print('Training Run:')
            training_loss = self.train()

            print('Validation Run:')
            eval_loss = self.eval(epoch_idx)

            current_epoch_losses = {**training_loss, **eval_loss}  # combines all metrics

            # Computing statistics
            val_mean_psnr = np.mean(current_epoch_losses['val-PSNR'])

            if val_mean_psnr > self.best_val_model_psnr:  # early stopping check
                self.best_val_model_psnr = val_mean_psnr
                self.best_val_model_idx = epoch_idx
                improvement_count = 0
            else:
                improvement_count += 1

            for key, value in current_epoch_losses.items():  # TODO: make sure learning rate is not getting averaged...
                avg_val = np.nanmean(value)
                # removes nan values - which allows for tokenizing certain keys if tracked only in a few batches
                if math.isnan(avg_val):
                    avg_val = 0
                total_losses[key].append(avg_val)  # get mean of all metrics of current epoch
                if self.aim_track:
                    self.aim_session.track(avg_val, name=key.replace('-', '_'),
                                           epoch=epoch_idx)
            total_losses['epoch'].append(epoch_idx)

            # Saving and reporting statistics
            if self.logging == 'visual':
                plot_stats(stats_dict=total_losses, keynames=[['train-loss', 'val-loss'], ['val-PSNR'], ['val-SSIM']],
                           experiment_log_dir=self.model.logs, filename='loss_plots.pdf')

            # Saves current model checkpoint
            self.model.save(override=self.overwrite)

            # save results to file
            save_statistics(experiment_log_dir=self.model.logs, filename='summary.csv',
                            stats_dict=total_losses,
                            selected_data=epoch_idx if (self.starting_epoch != 0 or i > 0) else None,
                            append=True if (self.starting_epoch != 0 or i > 0) else False)

            out_string = " ".join(["{}_{:.4f}".format(key, np.mean(value))
                                   for key, value in current_epoch_losses.items()])

            # cleans old checkpoints, if set to do so
            if self.model_cleanup_frequency is not None and i != 0 and i % self.model_cleanup_frequency == 0:
                clean_models(self.model.base_folder, clean_samples=True)

            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            if self.aim_track:
                self.aim_session.track(epoch_elapsed_time, name='epoch_time', epoch=epoch_idx)

            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}/{}:".format(epoch_idx, self.starting_epoch + self.num_epochs-1), out_string,
                  "Epoch duration:", epoch_elapsed_time, "seconds")
            print('-------------')

            if improvement_count == self.stop_patience:
                print('Stopping model training, validation loss has plateaued.')
                break

        return total_losses

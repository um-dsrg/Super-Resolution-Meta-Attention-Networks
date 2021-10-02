import torch.nn as nn
import torch.optim as optim
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import toml
import glob
import os
import ast
from pydoc import locate
import time
import math

from sr_tools.helper_functions import create_dir_if_empty, read_metadata
from sr_tools.image_manipulation import ycbcr_convert
from sr_tools.loss_functions import PerceptualMechanism
from SISR.configuration.constants import base_directory

# quick logic searching for all folders in models directory
model_dir = os.path.join(base_directory, 'SISR/models')
model_categories = [f.name for f in os.scandir(model_dir) if (f.is_dir() and '__' not in f.name)]
available_models = {}

# Main logic for searching for handler files and registering model architectures in system.
for category in model_categories:
    p = ast.parse(open(os.path.join(model_dir, category, 'handlers.py'), 'r').read())
    classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
    for _class in classes:
        available_models[_class.split('Handler')[0].lower()] = ('SISR.models.' + category + '.handlers.' + _class)


class ModelInterface:
    """
    Main model client-side interface.  Takes care of loading/saving models,
    formatting outputs and triggering training/eval.
    """
    def __init__(self, model_loc, experiment, gpu='off', sp_gpu=0, mode='eval', new_params=None,
                 load_epoch=None, scale=None, save_subdir=None, new_branch=False):
        """
        :param model_loc: Model directory location.
        :param experiment: Model experiment name.
        :param gpu: One of 'off', 'multi' or 'single'.  Off signals models to use the CPU, single allows
        use of one GPU, and multi will trigger the use of all available GPUs.
        :param sp_gpu: Select specific GPU to use, if more than one are available.
        :param mode: Either eval or train.
        :param new_params: New model parameter dictionary, if creating a new model.
        :param load_epoch:  Epoch number to load, if reading model weights from file.
        :param scale: SR scale.
        :param save_subdir: Save subdirectory if branching is in use.
        :param new_branch: Set to true to branch out results/checkpoints into a new offshoot sub-directory.
        """

        if save_subdir is not None:
            log_dir = os.path.join('result_outputs', save_subdir)
            save_dir = os.path.join('saved_models', save_subdir)
        else:
            log_dir = 'result_outputs'
            save_dir = 'saved_models'

        self.experiment = experiment
        self.base_folder = os.path.abspath(os.path.join(model_loc, experiment))
        self.logs = os.path.abspath(os.path.join(self.base_folder, log_dir))
        self.saved_models = os.path.abspath(os.path.join(self.base_folder, save_dir))
        self.mode = mode

        if new_branch:
            load_override = os.path.dirname(self.saved_models)
        else:
            load_override = None

        if mode == 'train':
            create_dir_if_empty(self.base_folder, self.logs, self.saved_models)
            if new_params is None and load_epoch is None:
                raise RuntimeError('Need to specify model parameters to train a new model.')
        elif mode == 'eval':
            if load_epoch is None:
                raise RuntimeError('Need to specify which model epoch to load.')

        if load_epoch is None:
            self.model_epoch = 0
            self.metadata = new_params  # new model data has been specified
        else:
            # load model metadata from experiment folder
            if not glob.glob(os.path.join(self.base_folder, '*.toml')):  # legacy system catcher
                self.metadata = self._legacy_model_setup(experiment, self.base_folder, scale)
            else:
                self.metadata = toml.load(os.path.join(self.base_folder, 'config.toml'))['model']

        self.name = self.metadata['name']

        if self.name == 'qpircan':  # legacy conversion system
            self.name = 'qrcan'

        if scale is not None and scale != self.metadata['internal_params']['scale']:
            raise Exception('The model loaded has been trained for a different scale, '
                            'and cannot produce the requested images.')

        if gpu != 'off' and torch.cuda.is_available():
            self.device = sp_gpu
        else:
            self.device = torch.device('cpu')

        self.model = self.define_model(name=self.name, model_save_dir=self.saved_models,
                                       device=self.device, eval_mode=True if mode == 'eval' else False,
                                       **self.metadata['internal_params'])

        if load_epoch is not None:  # TODO: add system which remembers which networks were trained in the old style
            if load_epoch == 'best':
                load_epoch = pd.read_csv(os.path.join(self.logs, 'summary.csv'))['val-PSNR'].idxmax()
            elif load_epoch == 'last':
                load_epoch = len(pd.read_csv(os.path.join(self.logs, 'summary.csv'))['val-PSNR']) - 1
            self.model_epoch = load_epoch
            self.model.load_model(model_save_name='train_model', model_idx=load_epoch, legacy=self.model.legacy_load,
                                  load_override=load_override)
        else:
            self.model.pre_training_model_load()

        self.full_name = '%s_%d' % (experiment, self.model_epoch)

        if gpu == 'multi':
            self.model.set_multi_gpu()

        im_input = self.model.im_input
        colorspace = self.model.colorspace

        self.configuration = {'input': im_input, 'colorspace': colorspace}

        self.print_overview()

    def train_batch(self, lr, hr, **kwargs):
        return self.model.run_train(x=lr, y=hr, **kwargs)

    def set_epoch(self, epoch):
        self.model_epoch = epoch
        self.model.set_epoch(epoch)

    def net_run_and_process(self, lr=None, hr=None, **kwargs):
        # TODO: allow user to prevent preprocessing from happening?
        # TODO: add checks or fix dependency on input type

        if 'rgb' in self.configuration['colorspace']:
            out_rgb, loss, timing = self.model.run_eval(x=lr, y=hr, **kwargs)
            out_ycbcr = self.colorspace_convert(out_rgb, colorspace='rgb')
            out_rgb = self._standard_image_formatting(out_rgb.numpy())
        else:
            if hr is None:  # TODO: better fix possible?
                f_ref = hr
            else:
                f_ref = hr[:, 0, :, :].unsqueeze(1)
            out_y, loss, timing = self.model.run_eval(lr[:, 0, :, :].unsqueeze(1), y=f_ref, **kwargs)
            out_ycbcr = torch.stack([out_y.squeeze(1), lr[:, 1, :, :], lr[:, 2, :, :]], 1)
            out_rgb = self.colorspace_convert(out_ycbcr, colorspace='ycbcr')
            out_ycbcr = self._standard_image_formatting(out_ycbcr.numpy())

        return out_rgb, out_ycbcr, loss, timing

    @staticmethod
    def colorspace_convert(image, colorspace='rgb'):
        processed_im = ModelInterface._standard_image_formatting(image.numpy())
        for i in range(processed_im.shape[0]):
            processed_im[i, ...] = ycbcr_convert(processed_im[i, ...], im_type='jpg', input=colorspace, y_only=False)
        return processed_im

    @staticmethod
    def _standard_image_formatting(im, min_value=0, max_value=1):
        im_batch = np.copy(im)
        im_batch = np.clip(im_batch, min_value, max_value)
        return im_batch

    def net_forensic(self, data, **kwargs):  # TODO: make sure this is in the best position
        image, forensic_data = self.model.run_forensic(data, **kwargs)
        return image.numpy(), forensic_data

    def save(self, name='train_model', override=False, dry_run=False):
        save_path = os.path.join(self.saved_models, "{}_{}".format(name, str(self.model_epoch)))
        if os.path.isfile(save_path) and not override:
            raise RuntimeError('Saving this model will result in overwriting existing data!  '
                               'Change model location or enable override.')
        else:
            save_id = self.model_epoch
        if not dry_run:
            self.model.save_model(model_save_name=name, model_idx=save_id)
        else:
            print('Training cleared to run.')

    def save_metadata(self):
        """
        Function that saves any pertinent metadata that isn't obvious from the config file.
        """
        metadata = {'model_parameters': [self.model.print_parameters()]}
        md = pd.DataFrame.from_dict(metadata)
        md.to_csv(os.path.join(self.base_folder, 'extra_metadata.csv'), index=False)

    def print_overview(self):
        """
        Function that prints out model diagnostic information.
        :return: None.
        """
        if self.mode == 'eval':
            pmode = 'eval'
            epoch = self.model_epoch
            message = 'currently evaluating'
        else:
            pmode = 'train'
            if self.model_epoch == 0:
                epoch = self.model_epoch
            else:
                epoch = self.model_epoch + 1
            message = 'will start training from'

        print('----------------------------')
        print('Handler for experiment %s initialized successfully.' % self.experiment)
        print('System loaded in %s mode - %s architecture provided.' % (pmode, self.name))
        print('Model has %d trainable parameters.' % self.model.print_parameters())
        if str(self.model.device) == 'cpu':
            device = self.model.device
        else:
            device = 'GPU ' + str(self.model.device)
        print("Using %s as the model's primary device, and %s "
              "epoch %d of the model." % (device, message, epoch))
        self.model.extra_diagnostics()
        print('----------------------------')

    @staticmethod
    def define_model(name, **kwargs):
        return locate(available_models[name])(**kwargs)

    @staticmethod
    def _legacy_model_setup(experiment, exp_folder, scale):
        metadata = {}
        try:
            l_data = read_metadata(os.path.join(exp_folder, 'meta_data.csv'))
        except Exception:
            raise RuntimeError('No metadata information provided - model structure unknown.')
        metadata['name'] = l_data['model']
        metadata['internal_params'] = {}
        metadata['internal_params']['scale'] = scale
        if experiment == 'SFTMD_256_T1' or 'EDSR_MD_T1':
            num_feats = 256
            metadata['internal_params']['num_feats'] = num_feats
        if experiment == 'EDSR_MD_T1':
            metadata['internal_params']['normalize'] = False
        if experiment == 'EDSR_T1_x8':
            metadata['internal_params']['scale'] = 8
            metadata['internal_params']['num_features'] = 256
            metadata['internal_params']['num_blocks'] = 32
        return metadata

    def epoch_end_calls(self):
        self.model.epoch_end_calls()

    def get_learning_rate(self):
        return self.model.get_learning_rate()


class BaseModel(nn.Module):
    """
    Basic setup used for all models.  Provides base functionality for training, eval, loading and saving models.
    Should only be called from ModelInterface.
    """
    def __init__(self, device, model_save_dir, eval_mode, grad_clip=None, **kwargs):
        """
        :param device: GPU device ID (or 'cpu').
        :param model_save_dir: Model save directory.
        :param eval_mode: Set to true to turn off training functionality.
        :param grad_clip: If gradient clipping is required during training, set gradient limit here.
        """
        super(BaseModel, self).__init__()
        self.criterion = nn.L1Loss()
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.optimizer = None  # - |
        self.net = None  # | defined in specific architectures
        self.face_finder = False  # |
        self.model_name = None  # - |
        self.im_input = None  # - |
        self.colorspace = None  # - |
        if grad_clip == 0:
            self.grad_clip = None
        else:
            self.grad_clip = grad_clip
        self.model_save_dir = model_save_dir
        self.eval_mode = eval_mode
        self.curr_epoch = 0
        self.state = {}
        self.learning_rate_scheduler = None
        self.legacy_load = True  # loading system which ensures weight names match as expected

    def define_optimizer(self, lr=1e-4, optimizer_params=None):
        if optimizer_params is not None:
            beta_1 = optimizer_params['beta_1']
            beta_2 = optimizer_params['beta_2']
            betas = (beta_1, beta_2)
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr,
                                        betas=betas)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr)

    def define_scheduler(self, scheduler, scheduler_params):
        if scheduler == 'cosine_annealing_warm_restarts':
            self.learning_rate_scheduler = \
                optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_mult=scheduler_params['t_mult'],
                                                               T_0=scheduler_params['restart_period'],
                                                               eta_min=scheduler_params['lr_min'])
        elif scheduler == 'multi_step_lr':
            self.learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                          milestones=scheduler_params['milestones'],
                                                                          gamma=scheduler_params['gamma'])
        elif scheduler == 'custom_dasr':
            def dasr_scheduler(epoch):
                if epoch < 60:
                    lr = 1e-3
                elif epoch < 225:
                    lr = 1e-4
                else:
                    cycle = (epoch - 100) // 125
                    lr = 1e-4 * math.pow(0.5, cycle)
                return lr

            self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=dasr_scheduler)
        elif scheduler == 'step_lr':
            self.learning_rate_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                                     step_size=scheduler_params['step_size'],
                                                                     gamma=scheduler_params['gamma'])
        else:
            raise RuntimeError('%s scheduler not implemented' % scheduler)

    def activate_device(self):
        self.net.to(self.device)

    def training_setup(self, lr, scheduler, scheduler_params, perceptual, device, optimizer_params=None):

        if not self.eval_mode:
            self.define_optimizer(lr=lr, optimizer_params=optimizer_params)
            if scheduler is not None:
                self.define_scheduler(scheduler=scheduler, scheduler_params=scheduler_params)

        if perceptual is not None and self.eval_mode is False:
            self.criterion = PerceptualMechanism(lambda_per=perceptual, device=device)

    def set_multi_gpu(self, device_ids=None):
        self.net = nn.DataParallel(self.net, device_ids=device_ids)
        if len(self.net.device_ids) > 1:
            print('Model sent to multiple GPUs:', ', '.join([str(d_id) for d_id in self.net.device_ids]))

    def save_model(self, model_save_name, model_idx, extract_state_only=False):
        """
        Saves current model and other parameters to file
        :param model_save_name: model save name
        :param model_idx: latest epoch number
        :param extract_state_only: Set to true to only extract the save state of the model,
        rather than actually save it to file.
        :return: None
        """
        if isinstance(self.net, nn.DataParallel):
            net_params = self.net.module.state_dict()
        else:
            net_params = self.net.state_dict()

        self.state['network'] = net_params  # save network parameter and other variables.
        self.state['optimizer'] = self.optimizer.state_dict()  # save optimizer state
        self.state['model_name'] = self.model_name
        self.state['model_epoch'] = self.curr_epoch

        # additional possible save components
        if self.learning_rate_scheduler is not None:
            self.state['scheduler_G'] = self.learning_rate_scheduler.state_dict()
        if hasattr(self, 'learning_rate_scheduler_D'):
            self.state['scheduler_D'] = self.learning_rate_scheduler_D.state_dict()
        if hasattr(self, 'optimizer_D'):
            self.state['optimizer_D'] = self.optimizer_D.state_dict()
        if hasattr(self, 'discriminator'):
            if isinstance(self.net, nn.DataParallel):
                self.state['discriminator'] = self.discriminator.module.state_dict()
            else:
                self.state['discriminator'] = self.discriminator.state_dict()
        if hasattr(self, 'steps'):
            self.state['steps'] = self.steps

        if extract_state_only:
            return self.state

        torch.save(self.state, f=os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))

    @staticmethod
    def legacy_switch(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:13] == 'model.module.':
                new_state_dict[k[13:]] = v
            elif k[:6] == 'model.':
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def dan_check(self, state_dict):
        """
        Simple check to convert pre-trained official DAN model to local implementation.
        :param state_dict: State of model to be loaded.
        :return: None, dict updated in-place
        """
        if 'init_kernel' not in state_dict['network']:
            state_dict['network']['init_kernel'] = self.net.init_kernel
        if 'init_ker_map' not in state_dict['network']:
            state_dict['network']['init_ker_map'] = self.net.init_ker_map

    def load_model(self, model_save_name, model_idx, legacy=False, load_override=None, preloaded_state=None):
        """
        Loads selected model and other parameters from specified location
        :param model_save_name: saved model name.
        :param model_idx: model epoch number.
        :param legacy:  Set to True if model saved with legacy system.
        :param load_override: Override default model save location for loading.
        :param preloaded_state: State to load in, if this has been pre-loaded.
        :return: state dictionaries.
        """
        if self.device == torch.device('cpu'):
            loc = self.device
        else:
            loc = "cuda:%d" % self.device

        if load_override is None:
            load_file = os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        else:
            load_file = os.path.join(load_override, "{}_{}".format(model_save_name, str(model_idx)))

        if preloaded_state is None:
            state = torch.load(f=load_file,
                               map_location=loc)
        else:
            state = preloaded_state

        if 'dan' in state['model_name']:
            self.dan_check(state)  # pre-trained models sometimes don't provide init kernel.  If this is the case, the defaults are applied here instead.

        if legacy:
            self.net.load_state_dict(state_dict=self.legacy_switch(state['network']))
        else:
            self.net.load_state_dict(state_dict=state['network'])

        if not self.eval_mode:
            self.optimizer.load_state_dict(state['optimizer'])
            if self.learning_rate_scheduler is not None:
                self.learning_rate_scheduler.load_state_dict(state['scheduler_G'])
            if hasattr(self, 'learning_rate_scheduler_D'):
                self.learning_rate_scheduler_D.load_state_dict(state['scheduler_D'])
            if hasattr(self, 'optimizer_D'):
                self.optimizer_D.load_state_dict(state['optimizer_D'])
            if hasattr(self, 'discriminator'):
                self.discriminator.load_state_dict(state['discriminator'])
            if hasattr(self, 'steps'):
                self.steps = state['steps']

        self.set_epoch(state['model_epoch'])

        if state['model_name'] == 'qpircan':  # legacy conversion system
            state['model_name'] = 'qrcan'

        print('Loaded model uses the following architecture:', state['model_name'])
        return state

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x, y = x.to(device=self.device), y.to(device=self.device)
        out = self.run_model(x, image_names=tag, **kwargs)  # run data through model

        loss = self.criterion(out, y)  # compute loss
        self.standard_update(loss)  # takes care of optimizer calls, backprop and scheduler calls

        if keep_on_device:
            return loss.detach().cpu().numpy(), out.detach()
        else:
            return loss.detach().cpu().numpy(), out.detach().cpu()

    def standard_update(self, loss):
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss
        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()  # update network parameters

        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler.step()

    # TODO: probably best to convert this into a system where eval outputs extra items (e.g. loss and timing) in a separate dict
    def run_eval(self, x, y=None, request_loss=False, tag=None, timing=False, keep_on_device=False, *args, **kwargs):
        """
        Runs a model evaluation for the given data batch
        :param x: input (full-channel).
        :param y: target (full-channel).
        :param request_loss: Set to true to also compute network loss with current criterion.
        :param tag: Image name.
        :param timing: Set to true to time network run-time.
        :param keep_on_device: Set this to true to keep final output on input device (GPU).
        Otherwise, result will always be transferred to CPU.
        :return: calculated output and loss.
        """
        self.net.eval()  # sets the system to validation mode

        with torch.no_grad():
            x = x.to(device=self.device)
            if timing:
                tic = time.perf_counter()
            out = self.run_model(x, image_names=tag, **kwargs)  # forward the data in the model
            if timing:
                toc = time.perf_counter()
            if request_loss and y is not None:
                y = y.to(device=self.device)
                loss = self.criterion(out, y).detach().cpu().numpy()  # compute loss
            else:
                loss = None

        if keep_on_device:
            return out.detach(), loss, toc - tic if timing else None
        else:
            return out.detach().cpu(), loss, toc - tic if timing else None

    def run_forensic(self, x, *args, **kwargs):
        self.net.eval()
        with torch.no_grad():
            x = x.to(device=self.device)
            out, data = self.net.forensic(x, **kwargs)
        return out.cpu().detach(), data

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)

    def print_parameters(self, verbose=False):
        """
        Reports how many learnable parameters are available in the model, and where they are distributed.
        :return: None
        """
        if verbose:
            print('----------------------------')
            print('Parameter names:')
        total_num_parameters = 0
        for name, value in self.named_parameters():
            if verbose:
                print(name, value.shape)
            total_num_parameters += np.prod(value.shape)
        if verbose:
            print('Total number of trainable parameters:', total_num_parameters)
            print('----------------------------')
        return total_num_parameters

    def print_status(self):
        raise NotImplementedError

    def epoch_end_calls(self):  # implement any end-of-epoch functions here
        pass

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def extra_diagnostics(self):
        """
        Empty method for models to print out any extra details on first instantiation, if required.
        """
        pass

    def pre_training_model_load(self):
        """
        Use this method to pre-load models trained from other experiments, if required.
        """
        pass


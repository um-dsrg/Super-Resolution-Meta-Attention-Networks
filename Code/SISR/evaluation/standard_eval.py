import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
import itertools
from collections import defaultdict, OrderedDict
import toml
import pandas as pd
from PIL import Image
from torchvision import transforms
import time
from colorama import init, Fore

from SISR.models import ModelInterface

from sr_tools.data_handler import SuperResImages
from sr_tools.visualization import interpret_sisr_images, safe_image_save
from sr_tools.metrics import Metrics
from sr_tools.helper_functions import create_dir_if_empty
from sr_tools.image_manipulation import ycbcr_convert


class EvalHub:
    def __init__(self, hr_dir, lr_dir, model_and_epoch, results_name, gpu, metrics, data_split,
                 save_im, batch_size, full_directory, lr_dir_interp,
                 model_only, scale, model_loc, out_loc, use_test_group, no_image_comparison,
                 num_image_save, qpi_selection, data_attributes,
                 dataset_name, image_shortlist, metadata_file, sp_gpu, time_models, recursive):
        """
        Main Eval Class.  Param info available in net_eval.py.
        """

        # main eval folder setup
        self.out_dir = os.path.join(out_loc, results_name)
        self.eval_name = results_name
        create_dir_if_empty(self.out_dir)
        input_params = locals()
        input_params.pop('self', None)
        with open(os.path.join(self.out_dir, 'config.toml'), 'w') as f:
            toml.dump(input_params, f)
        init()  # colorama setup

        experiment_names, eval_epochs = zip(*model_and_epoch)  # unpacking model info

        # flag and info setup
        self.metrics = metrics
        self.scale = scale
        self.full_directory = full_directory  # run models through all images in given directory
        self.no_image_comparison = no_image_comparison  # set to true to prevent image collages from being generated
        self.model_only = model_only  # set to true to disable all metric calculations
        self.num_image_save = num_image_save  # sets max number of image comparisons to save
        self.time_models = time_models  # Specify this to true to also time model execution
        self.images_processed = 0
        self.save_im = save_im

        self.model_bundles = self._prep_models(model_loc, experiment_names, eval_epochs, gpu, scale=scale, sp_gpu=sp_gpu)
        # TODO: To deal with tensorflow GPU issues,  follow guide here: https://www.tensorflow.org/guide/gpu

        # data setup
        if full_directory:
            split = 'all'
            dataset = None
            custom_split = None
            blacklist = None
        else:
            if data_split is None:
                split = 'eval'
            else:
                split = data_split
            dataset = dataset_name
            if use_test_group:  # specific setup for celeba
                custom_split = (162899, 163000)
            else:
                custom_split = None
            blacklist = None

        if metadata_file is None:
            metadata_file = os.path.join(lr_dir, 'degradation_metadata.csv')

        if not os.path.isfile(metadata_file):
            metadata_file = os.path.join(lr_dir, 'qpi_slices.csv')  # backup location
            if not os.path.isfile(metadata_file):
                print('%sNo metadata file found.%s' % (Fore.RED, Fore.RESET))
                metadata_file = None
                requested_metadata = None
            else:
                requested_metadata = 'all'
        else:
            requested_metadata = 'all'

        rgb_handler = SuperResImages(lr_dir, hr_dir, y_only=False, split=split, input='unmodified', dataset=dataset,
                                     colorspace='rgb', conv_type='jpg', scale=scale, custom_split=custom_split,
                                     blacklist=blacklist, qpi_selection=qpi_selection,
                                     degradation_metadata_file=metadata_file,
                                     metadata=requested_metadata, data_attributes=data_attributes,
                                     image_shortlist=image_shortlist, recursive_search=recursive)
        self.rgb_data = DataLoader(dataset=rgb_handler, batch_size=batch_size)

        if lr_dir_interp:  # if interpolated data provided, can be used directly
            interp_handler = SuperResImages(lr_dir_interp, hr_dir, y_only=False, split=split, input='interp',
                                            dataset=dataset, blacklist=blacklist,
                                            degradation_metadata_file=metadata_file,
                                            colorspace='rgb', conv_type='jpg', scale=scale,
                                            custom_split=custom_split, qpi_selection=qpi_selection,
                                            image_shortlist=image_shortlist, recursive_search=recursive)
            self.interp_data = iter(DataLoader(dataset=interp_handler, batch_size=batch_size))
        else:
            self.interp_data = None

        # output folder preparation
        self.comparisons_dir = os.path.join(self.out_dir, 'model_comparisons')
        if not no_image_comparison:
            create_dir_if_empty(self.comparisons_dir)

        # individual experiment folders for output saving
        self.save_folders = {}
        if save_im or model_only:
            for exp in experiment_names:
                save_folder = os.path.join(self.out_dir, exp)
                self.save_folders[exp] = save_folder
                create_dir_if_empty(save_folder)

        # main metric calculation hub
        if metrics is not None:
            self.metric_hub = Metrics(metrics, delimeter='>')

    @staticmethod
    def _prep_models(model_loc, experiment_names, eval_epochs, gpu, scale=4, sp_gpu=0):
        """
        Initializes and sets up specified models for evaluation.
        :param model_loc: Location from which to extract saved models
        :param experiment_names: List or Tuple of experiment names in the SISR results folder.
        :param eval_epochs: List of specific epochs to evaluate for each model.
        :param gpu: Specify whether to use a GPU in a computation.
        :param scale: super-resolution model scale (restricted on certain models)
        :return: model bundles (dict) & empty metrics (dict)
        """
        models = []
        for experiment, eval_epoch in zip(experiment_names, eval_epochs):
            models.append(
                ModelInterface(model_loc, experiment,
                               load_epoch=eval_epoch if eval_epoch == 'best' else int(eval_epoch),
                               gpu='off' if not gpu else 'single', scale=scale, sp_gpu=sp_gpu))
        return models

    def _low_res_prep(self, lr_data, timing=True):  # TODO: investigate whether pytorch's upsampling is similar to that of PIL
        interp_data = torch.empty(*lr_data.shape[0:2], lr_data.shape[2]*self.scale, lr_data.shape[3]*self.scale)
        for i in range(lr_data.shape[0]):
            image = transforms.ToPILImage()(lr_data[i, ...])
            if timing:
                tic = time.perf_counter()
            resized_im = image.resize((image.width * self.scale, image.height * self.scale), resample=Image.BICUBIC)
            if timing:
                toc = time.perf_counter()
            interp_data[i, ...] = transforms.ToTensor()(resized_im)
            # interp_data[i, ...] = ycbcr_convert(image, im_type='jpg', input='rgb', y_only=False)

        return interp_data, toc-tic if timing else None

    def _high_res_prep(self, hr_data):
        hr_prep = ModelInterface._standard_image_formatting(hr_data.numpy())
        for i in range(hr_prep.shape[0]):
            hr_prep[i, ...] = ycbcr_convert(hr_prep[i, ...], im_type='jpg', input='rgb', y_only=False)
        return hr_prep

    def _generate_image_collage(self, interp_data, model_data, probe_names, metrics=None, metric_slice=None, hr_rgb=None):

        metrics = metrics if metrics is not None else []
        metric_slice = metric_slice if metric_slice is not None else {}

        if hr_rgb is not None:
            output_package = OrderedDict([('HR', hr_rgb.numpy()),
                                          ('LR', interp_data.numpy())])

        else:
            output_package = OrderedDict([('LR', interp_data.numpy())])

        output_package.update(model_data)

        # send results for saving or visualization
        interpret_sisr_images(output_package, metric_slice, metrics, self.comparisons_dir,
                              names=['image_comparison_%s.pdf' % probe_name.replace('/', '_') for probe_name in probe_names],
                              direct_view=False, config='rgb',
                              extra_info={model.experiment: [['epoch', model.model_epoch]]
                                          for model in self.model_bundles})

    def direct_model_protocol(self):
        with tqdm(total=len(self.rgb_data)) as pbar:
            for index, batch in enumerate(self.rgb_data):
                lr_rgb, im_names = batch['lr'], batch['tag']
                self.images_processed += len(im_names)

                probe_names = [im_name.split('.')[0] for im_name in list(im_names)]
                if self.interp_data:  # extract interp data if provided; generate the images if not
                    interp_data = next(self.interp_data)['lr']
                else:
                    interp_data, _ = self._low_res_prep(lr_rgb, timing=False)
                output_package = {}
                for model in self.model_bundles:
                    # run input through network
                    if model.configuration['colorspace'] == 'rgb':
                        if model.configuration['input'] == 'unmodified':
                            selected_im = lr_rgb
                        else:
                            selected_im = interp_data
                    else:
                        selected_im = self._high_res_prep(interp_data)  # TODO: this would be repeated if multiple models request this - need to fix

                    rgb_im, ycbcr_im, _, _ = model.net_run_and_process(**{**batch, **{'lr': selected_im}})
                    safe_image_save(rgb_im, self.save_folders[model.experiment], im_names, config='rgb')
                    output_package[model.experiment] = rgb_im

                if not self.no_image_comparison:
                    self._generate_image_collage(interp_data, output_package, probe_names=probe_names)
                pbar.update(1)

    def full_image_protocol(self):

        metric_package = defaultdict(list)

        with tqdm(total=len(self.rgb_data)) as pbar:
            for index, batch in enumerate(self.rgb_data):
                lr_rgb, hr_rgb, im_names, hr_names = batch['lr'], batch['hr'], batch['tag'], batch['hr_tag']
                self.images_processed += len(im_names)

                # prepare metadata
                diag_string = ''
                probe_names = [im_name.split('.')[0] for im_name in list(im_names)]
                hr_names = [hr_name.split('.')[0] for hr_name in list(hr_names)]
                metric_package['Image_Name'].append(list(im_names))

                if self.interp_data:  # extract YCbCr data if provided; generate the images if not
                    ycbcr_batch = next(self.interp_data)
                    interp_data = ycbcr_batch['lr']
                else:
                    interp_data, timing_info = self._low_res_prep(lr_rgb, timing=self.time_models)
                    if timing_info is not None:
                        metric_package['LR%sruntime' % self.metric_hub.delimeter].append([timing_info])

                hr_prep = self._high_res_prep(hr_rgb)
                lr_prep = self._high_res_prep(interp_data)

                # LR metrics
                metric_slice, mini_diag_string = self.metric_hub.run_metrics(lr_prep, references=hr_prep, key='LR',
                                                                             probe_names=hr_names)
                for key in metric_slice.keys():
                    metric_package[key].append(metric_slice[key])
                diag_string += mini_diag_string

                output_package = {}
                # run models and gather stats
                for model in self.model_bundles:
                    # run input through network
                    if 'rgb' in model.configuration['colorspace']:
                        if model.configuration['input'] == 'unmodified':
                            selected_im = lr_rgb
                        else:
                            selected_im = interp_data
                    else:
                        selected_im = lr_prep

                    rgb_im, ycbcr_im, _, timing = model.net_run_and_process(**{**batch, **{'lr': selected_im}},
                                                                            timing=self.time_models)
                    if timing is not None:
                        metric_package['%s%sruntime' % (model.experiment, self.metric_hub.delimeter)].append([timing])
                    # TODO: remove double-list implementation....

                    # calculate metrics and organize diagnostics
                    metric_slice, mini_diag_string = self.metric_hub.run_metrics(ycbcr_im, hr_prep,
                                                                                 key=model.experiment,
                                                                                 probe_names=hr_names)
                    for key in metric_slice.keys():
                        metric_package[key].append(metric_slice[key])
                    diag_string += mini_diag_string

                    # Save generated image
                    output_package[model.experiment] = rgb_im
                    if self.save_im and self.images_processed < self.num_image_save:  # TODO: very crude, must fix later
                        for im in im_names:
                            if os.sep in im:
                                create_dir_if_empty(os.path.join(self.save_folders[model.experiment], os.path.dirname(im)))
                        safe_image_save(rgb_im, self.save_folders[model.experiment],
                                        im_names, config='rgb')

                # generate image comparisons
                if not self.no_image_comparison and self.images_processed < self.num_image_save:
                    self._generate_image_collage(interp_data, output_package, metrics=self.metrics,
                                                 metric_slice={key: metric_package[key][-1] for key in metric_package},
                                                 probe_names=probe_names, hr_rgb=hr_rgb)
                # update progress bar
                pbar.update(1)
                pbar.set_description(diag_string[:-2])

        self.manipulate_and_save_metrics(metric_package)

    def quick_save_csv_data(self, data, directory, names):
        for data, name in zip(data, names):
            data.to_csv(os.path.join(directory, name))

    def manipulate_and_save_metrics(self, metric_package):
        # combining all results
        for key in metric_package.keys():
            metric_package[key] = list(itertools.chain.from_iterable(metric_package[key]))

        # Pandas conversion and further calculations
        if 'Image_ID' in metric_package:
            indexes = ['Image_Name', 'Image_ID']
        else:
            indexes = ['Image_Name']
        full_results = pd.DataFrame.from_dict(metric_package).set_index(indexes)
        full_results.columns = pd.MultiIndex.from_tuples([tuple(c.split('>')) for c in full_results.columns])
        av_results = self.average_multilevel_dataframe(full_results)

        # saving to csv
        metrics_dir = os.path.join(self.out_dir, 'standard_metrics')
        create_dir_if_empty(metrics_dir)

        full_results.to_csv(os.path.join(metrics_dir, 'individual_metrics.csv'))
        av_results.to_csv(os.path.join(metrics_dir, 'average_metrics.csv'))

    def average_multilevel_dataframe(self, dataframe):
        r1 = dataframe.mean(axis=0).rename('Mean')
        r2 = dataframe.std(axis=0).rename('Std')
        results = pd.concat([r1, r2], axis=1)
        results = pd.DataFrame(results.stack()).T.stack(0).droplevel(level=0)
        return results

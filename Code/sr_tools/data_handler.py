from torch.utils.data import Dataset, Sampler, ConcatDataset
import numpy as np
import PIL.Image
import os
from torchvision import transforms
import pandas as pd
import re
import torch
import json
from collections import deque
import copy
import sr_tools.gaussian_utils as g_utils
from itertools import compress

from sr_tools.image_manipulation import RGBtoYCbCrConverter, center_crop, random_matched_crop, random_flip_rotate, downsample
from SISR.configuration.constants import data_splits
from sr_tools.helper_functions import extract_image_names_from_folder, DefaultOrderedDict


def read_celeba_attributes(attributes_loc, image_dict, selected_metadata='all', attribute_amplification=None):
    """
    This function reads in celeba attributes, and attaches them to an image feature dictionary.
    :param attributes_loc: File location of celeba attributes.
    :param image_dict: Dictionary containing metadata for a set of images (image names as keys).
    :param selected_metadata: Which specific attributes to extract from celeba file.
    :param attribute_amplification: Set to True if attributes should be amplified to -2 for negative, and 2 for positive.
    :return: Updated dictionary, keys for attribute locations in dictionary.
    """

    full_dict = image_dict.copy()
    celeb_data = pd.read_csv(attributes_loc, skiprows=1, delim_whitespace=True)

    if attribute_amplification is not None:
        celeb_data[celeb_data < 0] = -2
        celeb_data[celeb_data > 0] = 2
    else:
        celeb_data[celeb_data < 0] = 0

    if selected_metadata != 'all':
        if 'age' in selected_metadata:  # certain columns in celeba attributes can have alternate names
            celeb_data.rename(columns={'Young': 'age'}, inplace=True)
        if 'gender' in selected_metadata:
            celeb_data.rename(columns={'Male': 'gender'}, inplace=True)
        celeb_data = celeb_data[selected_metadata]

    final_keys = list(celeb_data.columns)
    final_keys.reverse()
    for index, key in enumerate(sorted(full_dict)):
        added_data = [celeb_data.loc[key.split('_')[0].split('.')[0] + '.jpg'][data_key] for data_key in
                      final_keys]  # images in celeba attributes file are specified as .jpg files
        full_dict[key] = np.concatenate((added_data, full_dict[key]))

    # celeb_data = celeb_data.to_dict(orient='index')
    #
    # if selected_files is not None:
    #     celeb_data = {name: celeb_data[name] for name in selected_files}

    return full_dict, final_keys


# TODO: specify difference between metadata and data attributes (currently clashes between blur kernels and age data for example)
def read_augmentation_list(metadata_file, filenames=None, normalize=True, legacy_blur_kernels=None,
                           data_attributes=None, attributes_loc=None, attribute_amplification=None,
                           force_qpi_range=True, qpi_selection=None):  # TODO: further optimizations available here
    """
    Function for reading add-on image augmentations.
    :param metadata_file: Datafile containing degradation metadata.
    :param filenames: Names of images for which metadata is to be extracted.
    :param normalize: Specify whether QPI values should be normalized.
    :param legacy_blur_kernels: Blur kernel location, if using old system.
    :param data_attributes: Specify which metadata should be added on from an attributes file (list).
    :param attributes_loc: Location of extra attributes (e.g. celeba facial features).
    :param attribute_amplification:  Set to true to amplify difference between binary attributes.
    :param force_qpi_range: Set to True to normalize QPI between standard range (20,40).
    :param qpi_selection:  Set to specific QPI values to retain, if required.
    :return: Dictionary contained image metadata, and list of keys corresponding to metadata location.
    """
    keys = deque()
    if qpi_selection and None not in qpi_selection:
        qpi_cutoffs = True
    else:
        qpi_cutoffs = False

    if metadata_file is not None:
        aug_data = pd.read_csv(metadata_file, header=0, index_col=0)
        for col in aug_data:
            if aug_data[col].dtype == object:
                aug_data[col] = aug_data[col].apply(json.loads)  # if data column contains a list, extract using json formatting
                keys.extend([col.lower()] * len(aug_data[col][0]))
            elif aug_data[col].dtype == int:
                aug_data[col] = aug_data[col].astype(float)
                keys.append(col.lower())
                if col == 'QPI' and force_qpi_range:
                    minimum = 20
                else:
                    minimum = aug_data[col].min()
                if col == 'QPI' and force_qpi_range:
                    maximum = 40
                else:
                    maximum = aug_data[col].max()
                if normalize:
                    aug_data[col] = (aug_data[col] - minimum) / (maximum - minimum)
                    if col == 'QPI' and qpi_cutoffs:  # normalize filter values if required
                        qpi_selection = [(q - minimum)/(maximum - minimum) for q in qpi_selection]
            else:
                raise RuntimeError('Unidentified datatype in metadata file.')

        all_image_dict = aug_data.T.to_dict('list')
        augmentation_dict = {}
        for key in filenames:  # TODO: is there a more optimized way to do this without a loop?
            data = []
            meta_info = all_image_dict[key]
            for v in meta_info:
                if type(v) == list:
                    data.extend(v)
                else:
                    data.append(v)
            augmentation_dict[key] = np.array(data)

    else:
        augmentation_dict = {}
        for image in filenames:
            augmentation_dict[image] = np.array([])

    # extraction of additional image attributes e.g. celeba facial features
    if attributes_loc is not None and data_attributes is not None:  # TODO: add in dataset argument if adding new datasets
        augmentation_dict, attribute_keys = read_celeba_attributes(attributes_loc, augmentation_dict,
                                                                   selected_metadata=data_attributes,
                                                                   attribute_amplification=attribute_amplification)
        keys.extendleft(reversed(attribute_keys))

    if legacy_blur_kernels is not None:  # legacy option only
        kernels = np.load(legacy_blur_kernels)
        keys.extendleft(['blur_kernel']*len(kernels[0]))
        for index, key in enumerate(sorted(augmentation_dict)):
            augmentation_dict[key] = np.concatenate((kernels[index], augmentation_dict[key]))

    if qpi_cutoffs:  # Image filtering based on QPI
        qpi_pos = keys.index('qpi')
        accepted_images = [im for im, metadata in augmentation_dict.items() if qpi_selection[0] <= metadata[qpi_pos] <= qpi_selection[-1]]
        augmentation_dict = {im: augmentation_dict[im] for im in accepted_images}

    keys = list(keys)
    return augmentation_dict, list(keys)


class SuperResImages(Dataset):
    def __init__(self, lr_dir=None, hr_dir=None, dataset=None, split=None, custom_split=None, recursive_search=False,
                 image_shortlist=None, lr_transform=None, hr_transform=None, input='interp', colorspace='ycbcr',
                 y_only=True, conv_type='jpg', scale=4, mask_data=None, group_select=None,  attribute_amplification=None,
                 halfway_data=None, blacklist=None, degradation_metadata_file=None, qpi_selection=None,
                 data_attributes=None, metadata=None, legacy_blur_kernels=None, qpi_sort=False, random_augments=None,
                 random_crop=None, online_degradations=None, online_degradation_params=None, request_crops=None):
        """
        Data handler for SR machine learning.  Can take care of either LR/HR pairs, lone LR images or lone HR images.
        Main inputs:
        :param lr_dir: lr image directory.
        :param hr_dir: corresponding hr image directory.
        :param dataset: Specify dataset name for pre-computed train/val/test splits.
        :param split: Select one of train | eval | test | all | None
        :param custom_split: Custom image selection (tuple of ints) e,g (0, 100).
        :param recursive_search: Set to true to search for images in all directories recursively from home directory selected.
        :param lr_transform: Additional transformations to be applied on lr images (defaults to tensor conversion only).
        :param hr_transform: Additional transformations to be applied on hr images (defaults to tensor conversion only).
        :param input: Select between interp | unmodified
        :param colorspace: Select between rgb or ycbcr.
        :param y_only: Request y channel only.
        :param conv_type: Convert colourspaces using specified format.
        :param scale: SR scale (for use in ensuring LR/HR images match).
        Additional options:
        :param mask_data: Location of HR masks.
        :param group_select: Specifies which group of images should be accepted from input set.
        (group number identified by '_x.').
        :param halfway_data: Any additional accompanying images.
        :param blacklist: CSV file specifying which images should be discarded.
        :param degradation_metadata_file: Location of accompanying metadata values for input compressed images.
        :param qpi_selection: Range of QPIs which should be accepted (tuple of ints).
        :param data_attributes: Location of additional metadata file.
        :param metadata: List of metadata to include from data attributes (all degradation metadata included by default).
        :param legacy_blur_kernels: Location of accompanying blur kernel values (if using old system).
        :param qpi_sort: Sort images by QPI rather than by name.
        :param random_crop: Set size of square patches to crop out from an LR image, instead of returning the full image.
        :param request_crops: Set to number of patches to request per image (if random crop is set).
        :param random_augments: Set to True to have data be randomly flipped or rotated prior to return.
        :param online_degradations: Set to true to generate LR images on-the-fly.
        :param online_degradation_params:  Specific parameters to pass to online degrader.
        """
        super(SuperResImages, self).__init__()

        if split not in ['train', 'eval', 'test', 'all', None]:
            raise RuntimeError('"Split" must be one of: train | eval | test | all | None')

        if input not in ['interp', 'unmodified']:
            raise RuntimeError('"lr_type" must be one of: interp | unmodified')

        # essential parameters
        self.split = split
        self.scale = scale
        self.lr_type = input
        self.patch_crop = random_crop
        self.random_augment = random_augments
        self.request_crops = request_crops
        self.metadata_keys = []
        if group_select is not None and type(group_select) != list:
            group_select = [group_select]
        self.online_degradations = online_degradations

        # companion data locations
        self.hr_base = hr_dir
        self.mask_base = mask_data
        self.halfway_base = halfway_data

        if not online_degradations:  # if not using online degradations, lr images must be supplied
            main_dir = lr_dir
            self.lr_base = lr_dir
            if hr_dir is not None:
                self.hr_base = hr_dir
            else:
                self.hr_base = None

        else:  # hr images only supplied
            if hr_dir is None:
                raise RuntimeError('Cannot synthesize LR images without specifying HR images.')
            main_dir = hr_dir
            self.lr_base = None
            self.lr_filenames = None
            self.hr_base = hr_dir
            batch_ker = g_utils.random_batch_kernel(batch=30000, tensor=False)
            b = np.size(batch_ker, 0)
            batch_ker = batch_ker.reshape((b, -1))
            pca_matrix = g_utils.PCA(batch_ker, k=10).float()
            if online_degradation_params is None:  # TODO: investigate if adding cuda leads to any speed-ups
                self.degrader = g_utils.SRMDPreprocessing(pca_matrix, random=True, kernel=21, rate_iso=1.0,
                                                          sig_min=0.2, sig_max=4.0,
                                                          noise=False, cuda=False, noise_high=0.0)
            else:
                self.degrader = g_utils.SRMDPreprocessing(pca_matrix, random=True, cuda=False,
                                                          **online_degradation_params)

        # selecting data and extracting grouped images
        main_filenames = self.filter_names(main_dir, recursive_search, group_select)

        # selecting dataset partition TODO: this area needs to be made more robust
        main_filenames = self.dataset_split(main_filenames, custom_split, image_shortlist, split, dataset, main_dir)

        # removing specifically selected images
        main_filenames = self.blacklist_removal(main_filenames, blacklist)

        # cleaning up into a final list format
        if not online_degradations:
            main_list = []
            base_list = []
            for key, val in main_filenames.items():
                for file_name in val:
                    main_list.append(file_name)
                    base_list.append(key)
                self.lr_filenames = main_list
            if len(main_filenames) == 0:
                raise RuntimeError('No images were supplied or all images were filtered out!')
            self.base_filenames = base_list
        else:
            self.base_filenames = list(main_filenames.keys())

        if degradation_metadata_file is not None or metadata is not None:
            meta_names = self.base_filenames if online_degradations else self.lr_filenames
            # TODO: this won't work if using a dataset with recursive (within folders) data.
            att_dict, meta_keys = read_augmentation_list(degradation_metadata_file, attributes_loc=data_attributes,
                                                         data_attributes=metadata, qpi_selection=qpi_selection,
                                                         attribute_amplification=attribute_amplification,
                                                         filenames=meta_names,
                                                         legacy_blur_kernels=legacy_blur_kernels)
            self.metadata_keys = meta_keys
            if qpi_selection is not None and not online_degradations:
                self.lr_filenames, self.base_filenames = zip(*[(self.lr_filenames[self.lr_filenames.index(i)],
                                                                self.base_filenames[self.lr_filenames.index(i)])
                                                               for i in att_dict.keys()])
            metadata_list = []
            for image in meta_names:
                metadata_list.append(att_dict[image])

            if qpi_sort and not online_degradations:
                qpi_vals = [i[meta_keys.index('qpi')] for i in metadata_list]
                sorted_data = sorted(zip(self.lr_filenames, self.base_filenames, metadata_list, qpi_vals),
                                     key=lambda vals: vals[-1])
                self.lr_filenames, self.base_filenames, self.metadata, _ = zip(*sorted_data)
            else:
                self.metadata = metadata_list
        else:
            self.metadata = None

        self.image_count = len(self.lr_filenames) if not online_degradations else len(self.base_filenames)

        if online_degradations:  # TODO: if extending this to other operations, need to fix here
            if len(self.metadata_keys) == 0:
                self.metadata_keys = ['blur_kernel']*self.degrader.para_in
            else:
                self.metadata_keys.append(['blur_kernel']*self.degrader.para_in)

        self.lr_transform, self.hr_transform, self.mask_transform = self.transform_init(lr_transform=lr_transform,
                                                                                        hr_transform=hr_transform,
                                                                                        request_type=colorspace,
                                                                                        y_only=y_only,
                                                                                        conv_type=conv_type)
        print('Initialized %s data with %d image%s.' % (
            dataset if dataset is not None else 'image', self.image_count, 's' if self.image_count > 1 else ''))

    @staticmethod
    def filter_names(directory, recursive, group_select):
        """
        Select images and filter out as per specifications.
        :param directory: Image main directory.
        :param recursive: Set to true to extract all images in subdirectories too.
        :param group_select: If multiple images with the same HR reference are provided,
        this parameter can select which of the groups to retain
        (images need to specified in format image_qx.ext, where x is the group number).
        :return: List of retained images.
        """
        final_files = DefaultOrderedDict(list)
        raw_filenames = extract_image_names_from_folder(directory, recursive=recursive)
        for file in raw_filenames:
            real_file = os.path.relpath(file, directory)
            # querying whether an image has a group tag between a _q and . e.g. test_q5.png
            split_key = re.split(r"_q(.*)(?=\.)", real_file)
            if len(split_key) > 1:
                if group_select is None or split_key[1] in group_select:
                    accept = True
                    base_name = split_key[0] + split_key[2]
                else:
                    accept = False
                    base_name = ''
            else:
                base_name = split_key[0]
                accept = True
            if accept:
                final_files[base_name].append(real_file)
        return final_files

    @staticmethod
    def dataset_split(current_files, custom_split, image_shortlist, split, dataset, main_dir):
        """
        Extracts dataset split, according to spec.
        :param current_files: List of pre-filtered images.
        :param custom_split: Split selected (int, int)
        :param image_shortlist: Shortlist of specific images to retain.
        :param split: One of train | eval | test
        :param dataset: Dataset name (if split pre-specifed in configuration/constants)
        :param main_dir: Main image directory.
        :return: Filtered image names.
        """

        # direct splits
        if custom_split is not None or (image_shortlist is None and split != 'all' and len(current_files) != 1):
            if custom_split is None:
                start, end = data_splits[dataset][split]
            else:
                start = custom_split[0]
                end = custom_split[1]
            temp_dict = DefaultOrderedDict(list)
            for key, val in list(current_files.items())[start:end]:
                temp_dict[key] = val
        # image shortlist splits
        elif image_shortlist is not None:  # accepts images specified in a text file
            with open(image_shortlist, 'r') as shortlist:
                cleanup_fn = lambda pth: os.path.relpath(pth.rstrip('\n'), main_dir) if main_dir in pth else pth.rstrip('\n')
                accepted_images = [cleanup_fn(line) for line in shortlist]
            temp_dict = DefaultOrderedDict(list)
            for key, val in list(current_files.items()):
                if key in accepted_images:
                    temp_dict[key] = val
        else:
            temp_dict = current_files

        return temp_dict

    @staticmethod
    def blacklist_removal(current_files, blacklist):
        """
        Removes images specified in a blacklist.
        :param current_files: Current list of accepted images.
        :param blacklist:   Blacklist file location.
        :return: Filtered images.
        """
        filtered_files = copy.copy(current_files)
        if blacklist is not None:
            print('Removing blacklisted images.')
            blacklist = pd.read_csv(blacklist, header=[0])['Images'].tolist()
            for b in blacklist:
                if b in filtered_files:
                    del filtered_files[b]
        return filtered_files

    def transform_init(self, lr_transform, hr_transform, request_type, y_only, conv_type):
        """
        Prepares routine on-the-fly transformations to be applied on input images.
        :param lr_transform: Transformations to apply on LR images.
        :param hr_transform: Transformations to apply on HR images.
        :param request_type: ycbcr or rgb.
        :param y_only: Retain only y-channel of ycbcbr images.
        :param conv_type: PNG or JPG-style ycbcr conversion.
        :return: Transformations for each image group.
        """

        tfs = [transforms.ToTensor()]
        if lr_transform is not None:
            tfs.append(lr_transform)
        if request_type == 'ycbcr':
            tfs.append(RGBtoYCbCrConverter(y_only=y_only, im_type=conv_type))

        new_lr_tf = transforms.Compose(tfs)

        if self.hr_base is not None:
            tfs = [transforms.ToTensor()]
            if hr_transform is not None:
                tfs.append(hr_transform)
            if request_type == 'ycbcr':
                tfs.append(RGBtoYCbCrConverter(y_only=y_only, im_type=conv_type))
            new_hr_tf = transforms.Compose(tfs)
        else:
            new_hr_tf = None

        mask_tf = None  # Nothing to add currently

        return new_lr_tf, new_hr_tf, mask_tf

    @staticmethod
    def read_image(filename):
        im = PIL.Image.open(filename)
        # just in case images have transparency channel included, or are grayscale
        if im.mode == 'RGBA' or im.mode == 'L':
            im = im.convert('RGB')
        return im

    def __getitem__(self, index):
        """
        Main routine run for each image to load.
        :param index: Index number of image requested.
        :return: Dictionary containing image data and metadata.
        """
        base_name = self.base_filenames[index]
        if self.hr_base is not None:
            hr_im = self.read_image(os.path.join(self.hr_base, base_name))
        else:
            hr_im = np.array(0)
            mask_im = np.array(0)  # TODO: find a better way to set a none value?

        if self.online_degradations:  # generate image or read from file
            image_name = base_name
            blurred_im, kernel_map, unreduced_kernel = self.degrader(transforms.ToTensor()(hr_im))
            pil_image = transforms.ToPILImage()(blurred_im.squeeze(0).cpu())
            _, lr_im = downsample(pil_image, scale=self.scale, jm=False)
            if self.metadata is not None:
                metadata = np.concatenate((self.metadata[index], kernel_map.numpy()[0, :]))
            else:
                metadata = kernel_map.numpy()[0, :]
            unreduced_kernel = unreduced_kernel.numpy().squeeze()  # for use with networks that use entire blur kernels
        else:
            image_name = self.lr_filenames[index]
            lr_im = self.read_image(os.path.join(self.lr_base, image_name))
            if self.metadata is not None:
                metadata = self.metadata[index]
            else:
                metadata = np.array(0)  # TODO: unreduced_kernel should be read from metadata, and not from a separate variable.

            if self.metadata is not None and 'unmodified_blur_kernel' in self.metadata_keys:
                kernel_loc = [m == 'unmodified_blur_kernel' for m in self.metadata_keys]
                unreduced_kernel = list(compress(self.metadata[index], kernel_loc))
                kernel_len = int(np.sqrt(len(unreduced_kernel)))
                unreduced_kernel = np.array(unreduced_kernel).reshape(kernel_len, kernel_len)
            else:
                unreduced_kernel = np.array(0)
        # HR processing
        if self.hr_base is not None:
            if self.lr_type == 'interp':
                h, w = lr_im.height, lr_im.width
            else:
                h, w = lr_im.height * self.scale, lr_im.width * self.scale
            if hr_im.width != w or hr_im.height != h:
                hr_im = center_crop(hr_im,
                                    height=h, width=w)  # essential step to ensure correct alignment of LR/HR images

            hr_im = self.hr_transform(hr_im)
            # optional HR mask processing
            if self.mask_base is not None:
                mask_im = self.read_image(os.path.join(self.mask_base, base_name))
                if mask_im.width != w or mask_im.height != h:
                    mask_im = center_crop(mask_im, height=h, width=w)
                mask_im = np.array(mask_im)
            else:
                mask_im = np.array(0)

        # additional HR data ('halfway' data)
        if self.halfway_base is not None:
            halfway_im = self.read_image(os.path.join(self.halfway_base, base_name))
            halfway_im = self.hr_transform(halfway_im)
        else:
            halfway_im = np.array(0)

        lr_im = self.lr_transform(lr_im)  # final LR processing

        # augmentation - flip, rotate
        if self.random_augment is not None:  # TODO: set all of these as variable options
            lr_im, hr_im = random_flip_rotate(lr_im, hr_im)

        # randomly crop
        if self.patch_crop is not None:
            if self.request_crops is not None:
                images = []
                for i in range(self.request_crops):
                    images.append(random_matched_crop(lr_im, hr_im, crop_size=self.patch_crop, scale=self.scale))
                lr_ims, hr_ims = zip(*images)
                lr_im = torch.stack(lr_ims, 0)  # returning to B, C, H, W format
                hr_im = torch.stack(hr_ims, 0)
            else:
                lr_im, hr_im = random_matched_crop(lr_im, hr_im, crop_size=self.patch_crop, scale=self.scale)

        return {'lr': lr_im,
                'hr': hr_im,
                'tag': image_name,
                'hr_tag': base_name,
                'mask': mask_im,
                'halfway_data': halfway_im,
                'metadata': metadata,
                'metadata_keys': self.metadata_keys,  # TODO: better solution for this?  Currently tiles this for every output...
                'blur_kernels': unreduced_kernel
                }

    def __len__(self):
        return self.image_count


class CelebaSplitSampler(Sampler):
    """
    This sampler provides all data with positive features (according to selected set) first,
    then switches to the rest at the end.
    """
    def __init__(self, data_source, selected_attribute='gender', **kwargs):

        self.discriminatory_attribute = selected_attribute

        if type(data_source) is ConcatDataset:  # need to cater for index positioning if using multiple datasets
            self.length = 0
            self.negative_indices = []
            self.positive_indices = []
            for dataset in data_source.datasets:
                p_ind, n_ind, l_length = self._index_with_attribute(dataset)
                n_ind = [n + self.length for n in n_ind]
                p_ind = [p + self.length for p in p_ind]
                self.negative_indices += n_ind
                self.positive_indices += p_ind
                self.length += l_length
        else:
            self.positive_indices, self.negative_indices, self.length = self._index_with_attribute(data_source)

        super(CelebaSplitSampler, self).__init__(data_source)

    def _index_with_attribute(self, dataset):
        metadata_pos = int(np.where([self.discriminatory_attribute in m for m in dataset.metadata_keys])[0])
        # TODO: more general way to select relevant metadata position?

        pertinent_metadata = [m[metadata_pos] for m in dataset.metadata]
        positive_indices = np.where([m == 1 for m in pertinent_metadata])[0].tolist()
        negative_indices = np.where([m == 0 for m in pertinent_metadata])[0].tolist()
        length = len(dataset.metadata)
        return positive_indices, negative_indices, length

    def __iter__(self):

        pos_list = np.random.choice(self.positive_indices, len(self.positive_indices), replace=False).tolist()
        neg_list = np.random.choice(self.negative_indices, len(self.negative_indices), replace=False).tolist()
        return iter(pos_list + neg_list)

    def __len__(self):
        return self.length

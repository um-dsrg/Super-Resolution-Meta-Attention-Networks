from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import os

from sr_tools.data_handler import SuperResImages, CelebaSplitSampler


# TODO: prepare a better dictionary system for data parameters
def sisr_data_setup(training_sets, eval_sets, batch_size=16, eval_batch_size=1, dataloader_threads=8,
                    drop_last_training_batch=False, extract_masks=False, rep_partition=None, attributes=None,
                    blacklists=None, sampler_attributes=None, **kwargs):
    """
    Prepares super-res data for training/eval with several custom parameters available.
    :param training_sets: training set parameter dictionaries
    :param eval_sets: eval set parameter dictionaries
    :param batch_size:  batch size for parallel loading of images
    :param eval_batch_size: batch size for evaluation images (default to 1 as images typically have different dimensions)
    :param dataloader_threads: number of threads to use for parallel data loading
    :param extract_masks: set to true to also extract face masks from HR locations
    :param rep_partition: currently unused
    :param attributes: Attributes for specified datasets (dict) e.g. facial features for celeba
    :param blacklists: Blacklists (images to skip) for each particular dataset (dict)
    :param drop_last_training_batch: Set to true to drop last batch if total amount of data not divisible by batch size
    :param kwargs: Any other parameters which are common to all datasets (e.g. model scale)
    :param sampler_attributes: All parameters for a custom data sampler.
    :return: Training/Eval data loaders
    """

    def setup_data(data_set, split):
        """
        This function is run for each dataset, and makes all the necessary preparations.
        :param data_set: Dataset parameters.
        :param split: Train/Eval/Test split
        :return: Dataset class
        """
        if extract_masks:
            mask_loc = os.path.join(data_set['hr'], 'segmentation_patterns')
        else:
            mask_loc = None

        custom_range = None

        if data_set['cutoff'] is not None:  # cutoff either specifies start/stop position, or just the stop position
            if type(data_set['cutoff']) == list:
                custom_range = data_set['cutoff']
            else:
                custom_range = (0, data_set['cutoff'])
        elif data_set['name'] is None:  # if not a particular dataset, take in all images
            split = 'all'

        if data_set['qpi_values'] is not None:  # catering for legacy code
            data_set['degradation_metadata'] = data_set['qpi_values']

        if data_set['degradation_metadata'] == 'on_site':  # degradation file should always have the same name
            data_set['degradation_metadata'] = os.path.join(data_set['lr'], 'degradation_metadata.csv')
            if not os.path.isfile(data_set['degradation_metadata']):  # catering for legacy code
                data_set['degradation_metadata'] = os.path.join(data_set['lr'], 'qpi_slices.csv')

        if blacklists is not None and data_set['name'] in blacklists:  # only relevant if blacklist provided
            blacklist = blacklists[data_set['name']]
        else:
            blacklist = None

        if attributes is not None and data_set['name'] is not None:
            data_attributes = attributes[data_set['name']]
        else:
            data_attributes = None

        # TODO: can these options be condensed to prevent the need for all these specifications?
        data_class = SuperResImages(lr_dir=data_set['lr'], hr_dir=data_set['hr'], blacklist=blacklist,
                                    data_attributes=data_attributes,
                                    image_shortlist=data_set['image_shortlist'],
                                    metadata=data_set['metadata'],
                                    attribute_amplification=data_set['attribute_amplification'],
                                    dataset=data_set['name'], split=split, y_only=False if split == 'eval' else True,
                                    custom_split=custom_range, degradation_metadata_file=data_set['degradation_metadata'],
                                    legacy_blur_kernels=data_set['legacy_blur_kernels'],
                                    random_crop=data_set['crop'], random_augments=data_set['random_augment'],
                                    recursive_search=data_set['recursive_search'] if
                                    data_set['recursive_search'] is not None else False,
                                    mask_data=mask_loc, online_degradations=data_set['online_degradations'],
                                    request_crops=data_set['request_crops'],
                                    online_degradation_params=data_set['online_degradation_params'], **kwargs)
        return data_class

    all_train_data = []
    all_val_data = []
    print('---------------')
    print('preparing training data:')
    for key, train_set in training_sets.items():
        all_train_data.append(setup_data(train_set, split='train'))
    print('---------------')
    print('preparing validation data:')
    for key, eval_set in eval_sets.items():
        all_val_data.append(setup_data(eval_set, split='eval'))
    print('---------------')

    if len(all_train_data) == 1:  # concatenates data if multiple datasets provided.
        all_train_data = all_train_data[0]
    else:
        all_train_data = ConcatDataset(all_train_data)

    if len(all_val_data) == 1:
        all_val_data = all_val_data[0]
    else:
        all_val_data = ConcatDataset(all_val_data)

    if sampler_attributes is None:
        sampler = None
    elif sampler_attributes['name'].lower() == 'celebasplitsampler':
        sampler = CelebaSplitSampler(all_train_data, **sampler_attributes)
    else:
        raise RuntimeError('Selected data sampler not recognized.')

    train_dataloader = DataLoader(dataset=all_train_data,
                                  batch_size=batch_size,
                                  shuffle=True if sampler is None else False,
                                  num_workers=dataloader_threads,
                                  pin_memory=True,
                                  drop_last=drop_last_training_batch,
                                  sampler=sampler)

    val_dataloader = DataLoader(dataset=all_val_data, batch_size=eval_batch_size)

    return train_dataloader, val_dataloader

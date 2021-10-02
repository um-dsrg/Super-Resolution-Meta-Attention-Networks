import click
import sys
import os
import toml

from SISR.training.training_handler import TrainingHandler
from sr_tools.helper_functions import convert_default_none_dict


def read_metadata(metadata_file):  # TODO: this should move to a more central location
    with open(metadata_file, 'r') as f:
        data = [label.rstrip() for label in f.readlines()]
    return data


@click.command()
@click.option("--parameters", required=True,
              help='location of TOML parameters file, containing configs for this experiment')
@click.option("--num_epochs", type=int, help='Number of epochs to run through dataset.')
@click.option("--gpu", default=None, show_default=True, type=click.Choice(['single', 'multi'], case_sensitive=False),
              help='Specify whether or not to use a gpu for training.')
@click.option("--sp_gpu", default=None, show_default=True,
              help='Specify which base GPU to use.')
@click.option("--experiment_name", help='Experiment name to use for saving models/data.')
@click.option("--seed", help='Pytorch random seed.', default=8, show_default=True)
@click.option("--continue_from_epoch", help='Epoch number from which to resume training.', type=int)
@click.option("--overwrite_data", is_flag=True, default=None,
              help='Set this flag to overwrite any existing data in experiment directory.')
def experiment_setup(parameters, experiment_name, **kwargs):
    """
    Main function that controls the creation, configuration and running of a SISR experiment.
    All functionality can be controlled via the PARAMETERS config toml file (descriptions of available parameters provided in relevant training/data/model functions)
    """
    # pre-config setup
    params = toml.load(parameters)

    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
    params['training'] = {**params['training'], **kwargs}

    params = convert_default_none_dict(params)  # convert all unassigned values to None

    if experiment_name is not None:
        params['experiment'] = experiment_name

    additional_data_params = {}  # TODO: are these at all needed?

    for dataset_type in ['training_sets', 'eval_sets']:  # reads in requested metadata, if provided in file format
        for dataset_key, val in params['data'][dataset_type].items():
            if val['metadata_list'] is not None:
                params['data'][dataset_type][dataset_key]['metadata'] = read_metadata(val['metadata_list'])

    if params['model']['internal_params']['metadata_list'] is not None:
        params['model']['internal_params']['metadata'] = \
            read_metadata(params['model']['internal_params']['metadata_list'])

    # experiment setup and trigger
    experiment = TrainingHandler(experiment_name=params['experiment'],
                                 experiment_group=params['experiment_group'],
                                 save_loc=params['experiment_save_loc'], model_params=params['model'],
                                 **params['training'],
                                 data_params={**params['data'], **additional_data_params})

    # make a copy of provided config file in experiment directory
    if params['training']['continue_from_epoch'] is not None:
        config_file = 'config_from_epoch_%s.toml' % params['training']['continue_from_epoch']
    else:
        config_file = 'config.toml'

    with open(os.path.join(experiment.model.base_folder, config_file), 'w') as f:
        toml.dump(params, f)
        # TODO: toml dumps out without order preserved... could investigate this PR to fix: https://github.com/uiri/toml/pull/275/commits/6862d8d352cce2e693181509f6308c4db189b6c7

    experiment.model.save_metadata()  # saves any additional model metadata
    experiment.run_experiment()  # initiates model training


if __name__ == '__main__':
    experiment_setup(sys.argv[1:])  # for use when debugging with pycharm

# TODO: data handler group selects now need to remove the 'q' (also need to re-test this)


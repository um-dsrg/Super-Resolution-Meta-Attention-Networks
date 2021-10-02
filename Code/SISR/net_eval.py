import os
import sys
import toml
import click
import click_config_file

import SISR.configuration.constants as sconst
from SISR.evaluation.standard_eval import EvalHub


# config file loader
def toml_provider(file_path, cmd_name):
    return toml.load(file_path)


results_directory = os.path.join(os.path.dirname(sconst.base_directory), 'Results')
data_directory = os.path.join(os.path.dirname(sconst.base_directory), 'Data')


@click.command()
# Data Config
@click.option("--hr_dir", default=None, help='HR image directory.')
@click.option("--lr_dir", default=None, help='LR image directory.')
@click.option("--lr_dir_interp", default=None,
              help='LR (interpolated) image directory.')
@click.option('--data_attributes', default=None,
              help='Additional data attributes (such as gender etc)')
@click.option("--batch_size", default=1, help='Batch size for parallel data loading.', show_default=True)
@click.option('--full_directory', is_flag=True, help='Set this flag to ignore any data partitions or splits.')
@click.option('--qpi_selection', type=(int, int),
              help='Set these values to enforce qpi range when selecting validation data.', default=(None, None))
@click.option('--dataset_name', default=None, help='Specify dataset name to use associated eval split.')
@click.option('--image_shortlist', default=None, help='Location of text file containing image names'
                                                      ' to select from target folder')
@click.option('--data_split', default=None,
              help='Specifies data split to extract (train/test/eval).  Defaults to eval if not specified.')
@click.option('--metadata_file', default=None, help='Location of datafile containing metadata information.'
                                                    'Defaults to degradation_metadata.csv if not specified.')
@click.option('--use_test_group', is_flag=True, help='Set this flag to run results only on typical 100 images.')
@click.option('--recursive', default=False,
              help='Specify whether to search for further images in sub-folders of the main lr directory.')
# Model Config
@click.option("-me", "--model_and_epoch", multiple=True,
              help='Experiments to evaluate.', type=(str, str))
@click.option("--gpu/--no-gpu", default=False,
              help='Specify whether or not to use a gpu for speeding up computations.')
@click.option("--sp_gpu", default=0,
              help='Specify specific GPU to use for computation.', show_default=True)
@click.option('--scale', default=4, help='Scale of SR to perform.', show_default=True)
# Processing/Output Config
@click.option("--results_name", default='delete_me', help='Unique folder name for this output evaluation run.')
@click.option("-m", "--metrics", multiple=True, default=None,
              help='The metrics to calculate on provided test set.')
@click.option('--save_im', is_flag=True, help='Set this flag to save all generated SR images to results folder.')
@click.option('--model_only', is_flag=True, help='Set this flag to skip all metrics and simply output results.')
@click.option('--model_loc', default=results_directory, help='Model save location for loading.')
@click.option("--out_loc", default=results_directory, help='Output directory')
@click.option('--no_image_comparison', is_flag=True,
              help='Set this flag to prevent any image comparisons being generated.')
@click.option('--num_image_save', default=100,
              help='Set the maximum number of images to save when running comparisons.', show_default=True)
@click.option('--time_models/--no-time_models', default=True, help='Specify whether time model execution.  Defaults to on.')
@click_config_file.configuration_option(provider=toml_provider, implicit=False)
def eval_run(model_only, **kwargs):
    """
    Main function that controls the creation, configuration and running of a SISR evaluation experiment.
    All functionality can be controlled via the CONFIG toml file (descriptions of available parameters provided in evaluation/standard_eval.py)
    """
    eval_hub = EvalHub(model_only=model_only, **kwargs)

    if model_only:
        eval_hub.direct_model_protocol()
    else:
        eval_hub.full_image_protocol()


if __name__ == '__main__':
    eval_run(sys.argv[1:])

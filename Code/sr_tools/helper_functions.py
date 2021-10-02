from csv import DictReader
import pandas as pd
import os
import glob
from collections import OrderedDict, Callable, defaultdict
import click
import sys
import shutil
from tqdm import tqdm
import subprocess
import re
from colorama import init, Fore
import torch

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import moviepy.video.io.ImageSequenceClip


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_gpu_memory(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')


def generate_range(limits, subdivisions):
    diff = limits[1] - limits[0]
    step = int(diff/subdivisions)
    steps = [limits[0] + (i*step) for i in range(subdivisions+1)]
    if steps[-1] != limits[1]:
        steps[-1] = limits[1]
    steps[-1] += 1
    return steps


def list_to_dict(lst):
    """
    Converts a list to a dictionary (items converted to keys, all with a value of 0).
    :param lst: List input.
    :return: Converted Dictionary.
    """
    op = dict.fromkeys(lst, 0)
    return op


def get_actual_issame(probe_ids, gallery_ids):
    """
    Returns list containing exact matches between probe and gallery feature arrays.
    :param probe_ids: NxF feature array.
    :param gallery_ids: MxF feature array.
    :return: List of boolean match values of size M.
    """
    matches = []

    for p in probe_ids:
        matches.append([p == g for g in gallery_ids])

    return matches


def read_metadata(file):
    """
    Reads the provided metadata file (One line file containing essential info).
    :param file: String to file location.
    :return: Metadata (dict)
    """
    with open(file) as mfile:
        reader = DictReader(mfile)
        data = next(reader)
    return data


def convert_default_none_dict(in_dict):
    """
    Converts input dictionary and all inner dictionaries into a defaultdict with None as the default value
    :param in_dict: any dictionary
    :return: converted dictionary
    """
    callback = lambda: None

    def apply_none(inner_dict):
        for key in inner_dict:
            if type(inner_dict[key]) == dict:
                inner_dict[key] = defaultdict(callback, inner_dict[key])
                apply_none(inner_dict[key])

    def_dict = defaultdict(callback, in_dict)
    apply_none(def_dict)

    return def_dict


def create_dir_if_empty(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)


def extract_image_names_from_folder(folder, sorted=True, recursive=False):
    filenames = []
    for extension in ['*.jpg', '*.png', '*.bmp', '*.tif']:
        if recursive:
            glob_path = os.path.join(folder, '**', extension)
        else:
            glob_path = os.path.join(folder, extension)
        filenames.extend(glob.glob(glob_path, recursive=recursive))
    if sorted:
        filenames.sort()
    return filenames


def model_report(model_dir):
    summary = pd.read_csv(os.path.join(model_dir, 'result_outputs/summary.csv'))
    best_model_idx = summary['val-PSNR'].idxmax()
    final_model_idx = len(summary['val-PSNR'])-1

    save_dir = os.path.join(model_dir, 'saved_models')
    if os.path.exists(save_dir):
        model_files = os.listdir(save_dir)
    else:
        model_files = []
    return model_files, best_model_idx, final_model_idx


def check_models(models, best_idx, last_idx):
    best = 'No'
    b_color = Fore.RED
    last = 'No'
    l_color = Fore.RED
    if 'train_model_%d' % best_idx in models:
        best = 'Yes'
        b_color = Fore.GREEN
    if 'train_model_%d' % last_idx in models:
        last = 'Yes'
        l_color = Fore.GREEN
    return best, b_color, last, l_color


# Converts a pre-trained model obtained from any repo into one compatible with the codebase. TODO: replace this function by automatic detection within loading code.
def convert_pre_trained_model(pretrained_file, model_name, epoch=0):
    network = torch.load(f=pretrained_file, map_location=torch.device('cpu'))
    state = {'model_name': model_name, 'model_epoch': epoch, 'network': network}
    torch.save(state, f=os.path.join(os.path.dirname(pretrained_file), 'train_model_%s' % epoch))


@click.command()
@click.option('--main_dir', help='Main model compare directory', show_default=True)
@click.option('--compare_dir', default=None, help='Secondary directory to compare models with.', show_default=True)
def model_compare(main_dir, compare_dir):
    """
    Summarises all models in provided directory.  Will compare to secondary directory if provided.
    """
    init()
    for model in os.listdir(main_dir):
        model_path = os.path.join(main_dir, model)
        if os.path.isdir(model_path):
            model_files, best_model_idx, final_model_idx = model_report(model_path)
            best_available, b_color, last_available, l_color = check_models(model_files, best_model_idx, final_model_idx)
            if compare_dir is not None:
                print('---------')
            print(
                'Model Name: %s, best epoch: %s (checkpoint available: %s%s%s), '
                'last epoch: %s (checkpoint available: %s%s%s)' %
                (model, best_model_idx, b_color, best_available,
                 Fore.RESET, final_model_idx, l_color, last_available, Fore.RESET))
            if compare_dir is not None:
                compare_path = os.path.join(compare_dir, model)
                if os.path.exists(compare_path):
                    compare_files, best_compare_idx, final_compare_idx = model_report(compare_path)
                    if best_compare_idx != best_model_idx:
                        print('%sCorresponding model best epoch (%s) does not match that of main model!%s'
                              % (Fore.RED, best_compare_idx, Fore.RESET))
                    if final_compare_idx != final_model_idx:
                        print('%sCorresponding model final epoch (%s) does not match that of main model!%s'
                              % (Fore.RED, final_compare_idx, Fore.RESET))

                    best_available, b_color, last_available, l_color = check_models(
                        compare_files, best_model_idx, final_model_idx)
                    print(
                        'Corresponding model best epoch available: %s%s%s, '
                        'Corresponding model last epoch available: available: %s%s%s' %
                        (b_color, best_available, Fore.RESET, l_color, last_available, Fore.RESET))
                else:
                    print('%sCorresponding model not found in compare directory.%s' % (Fore.RED, Fore.RESET))
            if compare_dir is not None:
                print('---------')


def extract_best_model(model_dir, out_dir):
    model_name = os.path.basename(model_dir)
    new_model_dir = os.path.join(out_dir, model_name)
    create_dir_if_empty(new_model_dir)
    create_dir_if_empty(os.path.join(new_model_dir, 'result_outputs'))
    create_dir_if_empty(os.path.join(new_model_dir, 'saved_models'))

    data = ['result_outputs/loss_plots.pdf', 'result_outputs/summary.csv', 'extra_metadata.csv',
            'config.toml']
    summary = pd.read_csv(os.path.join(model_dir, data[1]))
    best_model_idx = summary['val-PSNR'].idxmax()
    data.append('saved_models/train_model_%d' % best_model_idx)
    for file in data:
        try:
            shutil.copy2(os.path.join(model_dir, file), os.path.join(new_model_dir, file))
        except:
            print('%s not found.' % file)


def clean_models(model_dir, keep_epochs=None, clean_samples=False):
    """
    Cleans out model checkpoints if these are unneeded.
    :param model_dir: model home directory
    :param keep_epochs: Specify epoch numbers to keep, regardless if these contain the best results
    :param clean_samples: Set this flag to also delete all epoch samples saved throughout training
    :return:
    """
    # TODO: clean up sorting here...
    summary = pd.read_csv(os.path.join(model_dir, 'result_outputs/summary.csv'))
    save_dir = os.path.join(model_dir, 'saved_models')
    model_files = []
    for extension in ['train_model_*']:
        model_files.extend(glob.glob(os.path.join(save_dir, extension)))
    model_files.sort(key=natural_keys)
    best_model_idx = summary['val-PSNR'].idxmax()
    accepted_indices = [best_model_idx-1, best_model_idx, best_model_idx+1, len(summary['val-PSNR'])-1,
                        int(model_files[-1].split('train_model_')[1])]

    if keep_epochs is not None:
        accepted_indices.extend(keep_epochs)

    with tqdm(total=len(model_files)) as pbar:
        for file in model_files:
            index = int(file.split('train_model_')[1])
            if index not in accepted_indices:
                os.remove(file)
                quote = 'Deleted'
            else:
                quote = 'Retained'
            pbar.update(1)
            pbar.set_description("%s model %d" % (quote, index))

    print('These models have been retained:', list(OrderedDict.fromkeys(accepted_indices)))

    if clean_samples:
        results_dir = os.path.join(model_dir, 'result_outputs/')
        folders = next(os.walk(results_dir))[1]
        for folder in folders:
            if 'epoch_' in folder:
                shutil.rmtree(os.path.join(results_dir, folder))
        print('All epoch samples deleted.')


@click.command()
@click.option('--home_dir', help='Model search directory.')
@click.option('--out_dir', help='New directory for models.')
@click.option('-m', '--models', multiple=True, help='Models to extract.')
@click.option('--clean', is_flag=True, help='Set this flag to clean model directory after extracting the best model.')
@click.option('-k', '--keep_epoch', multiple=True, type=int, help="set model numbers to retain apart from best epochs.")
@click.option('--clean_samples', is_flag=True, help="Set this flag to also remove all image samples generated during training.")
@click.option('--all_models', is_flag=True, help="Set this flag to extra best model from all experiments in directory.")
def extract_best_models(home_dir, out_dir, models, clean, keep_epoch, clean_samples, all_models):

    if all_models:
        all_items = os.listdir(home_dir)
        models = []
        for item in all_items:
            if os.path.isdir(os.path.join(home_dir, item)):
                models.append(item)

    for model in tqdm(models):
        extract_best_model(os.path.join(home_dir, model), out_dir)
        if clean:
            clean_models(os.path.join(home_dir, model), keep_epochs=keep_epoch, clean_samples=clean_samples)


@click.command()
@click.option('--base_dir', default='.', help='Model root location')
@click.option('-m', '--models', multiple=True, help='Models to clean.')
@click.option('-k', '--keep_epoch', multiple=True, type=int, help="set model numbers to retain apart from best epochs.")
@click.option('--clean_samples', is_flag=True, help="Set this flag to also remove all image samples generated during training.")
def click_clean(base_dir, models, keep_epoch, clean_samples):
    for model in tqdm(models):
        clean_models(os.path.join(base_dir, model), keep_epochs=keep_epoch, clean_samples=clean_samples)


@click.command()
@click.option('--image_folder', help='Input image folder to convert to video.')
@click.option('--video_name', help='Encoded video target name.')
@click.option('--output_loc', default=None, help='(Optional) Output directory for placing combined video.')
@click.option('--fps', default=24, help='Frames per second for encoded movie')
def click_image_sequence_to_movie(**kwargs):
    convert_image_sequence_to_movie(**kwargs)


def convert_image_sequence_to_movie(image_folder, video_name, output_loc=None, fps=24):

    if output_loc is None:
        output_loc = image_folder

    image_files = extract_image_names_from_folder(image_folder)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(os.path.join(output_loc, video_name))


if __name__ == '__main__':
    extract_best_models(sys.argv[1:])
    # copy_model_info(sys.argv[1:])

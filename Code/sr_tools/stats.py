import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # unsure why this is required - seems to bug out sometimes
import matplotlib.pyplot as plt
import ast
import pandas as pd


def plot_stats(stats_dict, keynames, experiment_log_dir, filename):

    plot_filename = os.path.join(experiment_log_dir, filename)

    num_plots = 0
    valid_keys = []
    for key in keynames:  # checks to see if any illegal keywords have been provided
        if all(metric in stats_dict for metric in key):
            num_plots += 1
            valid_keys.append(key)

    f, ax = plt.subplots(num_plots, 1, figsize=(10, 7))

    for ind, key in enumerate(valid_keys):
        for metric in key:
            ax[ind].plot(stats_dict['epoch'], stats_dict[metric], label=metric, linestyle='--', marker='o')
        ax[ind].set_xlabel('Epoch')
        ax[ind].legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(f)


def save_stats_and_average(output_dir, filename, metrics):
    """
    Saves given statistics dictionary and also saves stats average values to same file.
    :param output_dir: Output directory.
    :param filename: CSV file name.
    :param metrics: Statistics dictionary.
    :return: None.
    """

    # Calculate metric averages
    av_metrics = []
    av_divider = ['Average'] * len(metrics.keys())
    for index, (key, val) in enumerate(metrics.items()):
        if key == 'Image_Name':
            av_metrics.append('')
            av_divider[index] = ''
        else:
            res = sum(val)/len(val)
            av_metrics.append(res)
            print('Average {}: {:.3f}'.format(key, res))

    # Save to file
    stats_loc = legacy_save_statistics(output_dir, filename, metrics, save_full_dict=True)
    with open(stats_loc, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(av_divider)
        writer.writerow(av_metrics)


def save_statistics(experiment_log_dir, filename, stats_dict, selected_data=None, append=True):

    true_filename = os.path.join(experiment_log_dir, filename)

    pd_data = pd.DataFrame.from_dict(stats_dict)

    if selected_data is not None and os.path.isfile(true_filename):
        if type(selected_data) == int:
            selected_data = [selected_data]
        pd_data = pd_data.loc[selected_data]

    if not os.path.isfile(true_filename):  # if there is no file in place, no point in appending
        append = False

    pd_data.to_csv(true_filename, mode='a' if append else 'w', header=not append, index=False)


def legacy_save_statistics(experiment_log_dir, filename, stats_dict, current_epoch=0,
                           continue_from_mode=False, save_full_dict=False):
    """
    Saves the statistics in the stats dict into a csv file.
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session
    (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param continue_from_mode: selects whether to overwrite a file or append to it
    :param save_full_dict: whether to save the full dict as is overriding any previous entries
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename

# TODO: add a pandas dataframe to dict loader and converter
def legacy_load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].rstrip('\n').split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.rstrip('\n').split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(ast.literal_eval(value))

    return stats


def save_vari_stats(save_dir, filename, data):
    filename = os.path.join(save_dir, filename)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key, val in data.items():
            writer.writerow([key] + val)


def read_vari_stats(load_dir, filename):
    filename = os.path.join(load_dir, filename)
    with open(filename, 'r+') as f:
        lines = f.readlines()

    data = {}
    for index, line in enumerate(lines):
        values = line.split(',')
        if index == 0:
            xvals = np.array(values[1:]).astype(np.int)
        if values[0][-1] == 'y':
            data[values[0][:-2]] = np.array(values[1:])

    return xvals, data

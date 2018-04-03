import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import Parser

acronyms = {
    'ApproximationTreeMap': 'APP',
    'HilbertTreeMap': 'HIL',
    'IncrementalLayoutWithMoves': 'LM4',
    'IncrementalLayoutWithoutMoves': 'LM0',
    'MooreTreeMap': 'MOO',
    'PivotByMiddle': 'PBM',
    'PivotBySize': 'PBZ',
    'PivotBySplit': 'PBS',
    'SliceAndDice': 'SND',
    'SpiralTreeMap': 'SPI',
    'SquarifiedTreeMap': 'SQR',
    'StripTreeMap': 'STR'
}

ds = {
    'Cumulative4Year90HoursRating.csv': 'Movies C4Y90H',
    'm-names': 'Dutch Names',
    '15Months1DayRating.csv': '15M1D',
    'Cumulative10Years2MonthRating.csv': 'Movies C10Y2M',
    'hiv': 'World Bank HIV',
    '3Years3MonthRating.csv': 'Movies 3Y3M',
    'standard': 'GitHub standard',
    'HierarchyCumulative9Year7Month.csv': 'Movies HC9Y7M',
    'Hierarchy22Year7Month.csv': 'Movies H22Y7M',
    'HierarchyCumulative9Year1Week.csv': 'Movies HC9Y1W',
    'Hystrix': 'GitHub Hystrix'
}

def plot(dataset_ids):
    # Plot AR matrix
    weighted_ar_matrix, unweighted_ar_matrix, technique_acronyms = make_ar_matrices(dataset_ids)
    print('war')
    plot_matrix(weighted_ar_matrix, dataset_ids, technique_acronyms, 'war')
    print('uar')
    plot_matrix(unweighted_ar_matrix, dataset_ids, technique_acronyms, 'uar')

    # Plot CT matrix
    print('ct')
    ct_matrix, technique_acronyms = make_ct_matrix(dataset_ids)
    plot_matrix(ct_matrix, dataset_ids, technique_acronyms, 'ct')

    # Plot RPC matrix
    print('rpc')
    rpc_matrix, technique_acronyms = make_rpc_matrix(dataset_ids)
    plot_matrix(rpc_matrix, dataset_ids, technique_acronyms, 'rpc')


def plot_matrix(matrix, dataset_ids, technique_acronyms, metric_id):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if metric_id == 'uar' or metric_id == 'war':
        mat = ax.matshow(matrix, cmap=plt.cm.viridis)
    else:
        mat = ax.matshow(matrix, cmap=plt.cm.viridis_r)  # Invert colormap for instability

    # Ticks, labels and grids
    ax.set_xticklabels([ds[d] for d in dataset_ids], rotation='vertical')
    ax.set_xticks(range(len(dataset_ids)), minor=False)
    ax.set_yticklabels(technique_acronyms)
    ax.set_yticks(range(len(technique_acronyms)), minor=False)
    ax.set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor', color='#999999', linestyle='-', linewidth=1)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # Add the text
    x_start = 0.0
    x_end = len(dataset_ids)
    y_start = 0.0
    y_end = len(technique_acronyms)

    jump_x = (x_end - x_start) / (2.0 * len(dataset_ids))
    jump_y = (y_end - y_start) / (2.0 * len(technique_acronyms))
    x_positions = np.linspace(start=x_start-0.5, stop=x_end-0.5, num=len(dataset_ids), endpoint=False)
    y_positions = np.linspace(start=y_start-0.5, stop=y_end-0.5, num=len(technique_acronyms), endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = "{0:.3f}".format(matrix[y_index, x_index]).lstrip('0')
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center', fontsize=9)

    fig.colorbar(mat)
    fig.tight_layout()
    fig.savefig('matrices/matrix-'+ metric_id +'.png', dpi=600)
    # plt.show()


def make_ct_matrix(dataset_ids):
    technique_ids = []
    all_means = []
    for dataset_id in dataset_ids:
        ct_df = Parser.read_ct_metric(dataset_id)

        dataset_means = np.array([])
        technique_list = sorted(ct_df)
        if len(technique_ids) == 0:
            technique_acronyms = [acronyms[d] for d in technique_list]

        for i, technique_id in enumerate(technique_list):
            technique_means = []
            for revision in range(int(len(ct_df[technique_id].columns) / 2)):
                df = ct_df[technique_id]
                r_col = 'r_' + str(revision)
                b_col = 'b_' + str(revision)

                diff = df[[r_col, b_col]].max(axis=1) - df[b_col]
                diff = diff.dropna()
                if len(diff) > 0:
                    diff_mean = diff.mean()
                else:
                    diff_mean = 0

                technique_means.append(diff_mean)

            dataset_means = np.append(dataset_means, np.mean(technique_means))
        all_means.append(dataset_means)

    return np.array(all_means).transpose(), technique_acronyms  # Transpose matrix so each row is a technique and each column a dataset


def make_rpc_matrix(dataset_ids):

    technique_ids = []
    all_means = []
    for dataset_id in dataset_ids:
        rpc_df = Parser.read_rpc_metric(dataset_id)

        dataset_means = np.array([])
        technique_list = sorted(rpc_df)
        if len(technique_ids) == 0:
            technique_acronyms = [acronyms[d] for d in technique_list]

        for i, technique_id in enumerate(technique_list):
            technique_means = []
            for revision in range(int(len(rpc_df[technique_id].columns) / 2)):
                df = rpc_df[technique_id]
                r_col = 'r_' + str(revision)
                b_col = 'b_' + str(revision)

                diff = df[[r_col, b_col]].max(axis=1) - df[b_col]
                diff = diff.dropna()
                if len(diff) > 0:
                    diff_mean = diff.mean()
                else:
                    diff_mean = 0

                technique_means.append(diff_mean)

            dataset_means = np.append(dataset_means, np.mean(technique_means))
        all_means.append(dataset_means)

    return np.array(all_means).transpose(), technique_acronyms  # Transpose matrix so each row is a technique and each column a dataset


def make_ar_matrices(dataset_ids):

    technique_ids = []
    weighted_means = []
    unweighted_means = []
    for dataset_id in dataset_ids:
        ar_df = Parser.read_aspect_ratios(dataset_id)

        weighted_dataset_means = np.array([])
        unweighted_dataset_means = np.array([])
        technique_list = sorted(ar_df)
        if len(technique_ids) == 0:
            technique_acronyms = [acronyms[d] for d in technique_list]

        for i, technique_id in enumerate(technique_list):
            weighted_technique_means = []
            unweighted_technique_means = []
            for revision in range(int(len(ar_df[technique_id].columns) / 2)):
                w_col = 'w_' + str(revision)
                ar_col = 'ar_' + str(revision)

                u_avg = ar_df[technique_id][ar_col].mean(axis=0)
                w_avg = np.average(ar_df[technique_id][ar_col].dropna(), weights=ar_df[technique_id][w_col].dropna())

                weighted_technique_means.append(w_avg)
                unweighted_technique_means.append(u_avg)

            weighted_dataset_means = np.append(weighted_dataset_means, np.mean(weighted_technique_means))
            unweighted_dataset_means = np.append(unweighted_dataset_means, np.mean(unweighted_technique_means))
        weighted_means.append(weighted_dataset_means)
        unweighted_means.append(unweighted_dataset_means)

    return np.array(weighted_means).transpose(), np.array(unweighted_means).transpose(), technique_acronyms  # Transpose matrices so each row is a technique and each column a dataset

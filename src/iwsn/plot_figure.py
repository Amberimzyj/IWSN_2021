import re
from pathlib import Path
from functools import cmp_to_key

import numpy as np
from matplotlib import pyplot as plt

metirc_order = {}


def legend_sort(x, y):
    x_labels = re.split('[(:)]', x[1])
    x_m_order = metirc_order[x_labels[0].strip()]
    x_n_order = int(x_labels[2])

    y_labels = re.split('[(:)]', y[1])
    y_m_order = metirc_order[y_labels[0].strip()]
    y_n_order = int(y_labels[2])

    if x_m_order != y_m_order:
        return x_m_order - y_m_order
    else:
        return x_n_order - y_n_order


def save_figure1():
    # royalblue darkorange tomato forestgreen
    style_dict = {'3000': '.--', '4500': 'o-', '6000': '^--'}
    metric_dict = {'cond_pre_accu': 'Cond ', 'MI_pre_accu': 'MI',
                   'X2_pre_accu': 'X2 ', 'select_pre_accu': 'Select'}
    size_dict = {'3000': 'Size:3000', '4500': 'Size:4500', '6000': 'Size:6000'}
    global metirc_order
    metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'cond_pre_accu': 'r', 'MI_pre_accu': 'forestgreen',
                  'X2_pre_accu': 'royalblue', 'select_pre_accu': 'orange'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    fig, ax = plt.subplots(figsize=(7, 5))
    axins = ax.inset_axes((0.25, 0.2, 0.4, 0.35))
    axins.set_xlim(4, 8)
    axins.set_ylim(0.7, 0.85)
    axins.set_xticks(np.arange(4, 8, 1))
    axins.set_yticks(np.arange(0.7, 0.85, 0.05))
    axins.grid(linestyle='-.')

    data_path = Path("data/pre_accu")

    for sub_dir in data_path.glob('*'):
        nodes = sub_dir.stem
        for csv_path in sub_dir.glob('*.csv'):
            accu_name = csv_path.stem
            data = np.genfromtxt(csv_path, delimiter=',')
            line, = ax.plot(
                data, style_dict[nodes], color=color_dict[accu_name], linewidth=0.8)
            axins.plot(data, style_dict[nodes], color=color_dict[accu_name])
            line.set_label(f'{metric_dict[accu_name]}({size_dict[nodes]})')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels),
                                  key=cmp_to_key(legend_sort)))

    ax.indicate_inset_zoom(axins, edgecolor='grey', alpha=0.8, linewidth=0.8)
    plt.xticks(np.arange(0, 25, 2))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(linestyle='-.')
    plt.legend(bbox_to_anchor=(1, 0), handles=handles,
               labels=labels, loc='lower right')
    plt.xlabel('Transmission Time Interval(TTI)')
    plt.ylabel('Average Successful Prediction Ratio')
    plt.show()

    fig.savefig(data_path / 'ave_pre_accu.pdf', dpi=600, format='pdf')


def save_figure2():
    #res —— t_5g

    # royalblue darkorange tomato forestgreen
    style_dict = {'RTSN_3000': '.--', 'RTSN_4500': 'o-', 'RTSN_6000': '^--'}
    metric_dict = {'RTSN_3000': 'Data Size:3000 ',
                   'RTSN_4500': 'Data Size:4500', 'RTSN_6000': 'Data Size:6000 '}
    # metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'RTSN_3000': 'forestgreen',
                  'RTSN_4500': 'royalblue', 'RTSN_6000': 'tomato'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    # 局部放大
    fig, ax = plt.subplots(figsize=(10, 5))
    axins = ax.inset_axes((0.15, 0.4, 0.2, 0.5))
    axins.set_xlim(6, 8)
    axins.set_ylim(90, 140)
    axins.set_xticks(np.arange(6, 8, 0.5))
    axins.set_yticks(np.arange(90, 140, 10))
    axins.grid(linestyle='-.')

    # axins1 = ax.inset_axes((0.6, 0.3, 0.3, 0.2))
    # axins1.set_xlim(9, 11)
    # axins1.set_ylim(180, 210)
    # axins1.set_xticks(np.arange(9, 11, 0.5))
    # axins1.set_yticks(np.arange(180, 210, 10))
    # axins1.grid(linestyle='-.')


    data_path = Path("data/RTSN_data/5g_tsn")
    for csv_path in data_path.glob('*.csv'):
        all_data = np.loadtxt(csv_path)
        res = all_data[:, 0]
        t_5G = all_data[:, 1]
        q_t = all_data[:, 2]
        t_tsn = all_data[:, 3]
        inte_delay = all_data[:, 4]
        data_name = csv_path.stem
        line,  = plt.plot(
            res, t_5G, style_dict[data_name], color=color_dict[data_name], linewidth=0.8)
        line.set_label(f'{metric_dict[data_name]}')
        axins.plot(
            res, t_5G,  style_dict[data_name], color=color_dict[data_name])
        # axins1.plot(res, t_5G,  style_dict[data_name], color=color_dict[data_name])

    # #处理图例
    handles, labels = plt.gca().get_legend_handles_labels()

    ax.indicate_inset_zoom(axins)
    # ax.indicate_inset_zoom(axins1)

    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.xticks(np.arange(0, 15, 2))
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(linestyle='-.')
    plt.legend(bbox_to_anchor=(1, 0), handles=handles,
               labels=labels, loc='lower right', fontsize='large')
    plt.xlabel('Reserved RB numbers')
    plt.ylabel('Latency of 5G network')
    plt.show()

    # fig.savefig(data_path / 'ave_pre_accu.pdf', dpi=600, format='pdf')


def save_figure3():
    #res —— t_5g,t_tsn

    # royalblue darkorange tomato forestgreen
    style_dict = {'RTSN_3000': '.--', 'RTSN_4500': 'o-', 'RTSN_6000': '^--'}
    metric_dict = {'RTSN_3000': 'Data Size:3000 ',
                   'RTSN_4500': 'Data Size:4500', 'RTSN_6000': 'Data Size:6000 '}
    # metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'RTSN_3000': 'forestgreen',
                  'RTSN_4500': 'royalblue', 'RTSN_6000': 'tomato'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    # 局部放大
    fig, ax = plt.subplots(figsize=(10, 5))
    axins = ax.inset_axes((0.65, 0.35, 0.3, 0.35))
    axins.set_xlim(6, 8)
    axins.set_ylim(90, 140)
    axins.set_xticks(np.arange(6, 8, 0.5))
    axins.set_yticks(np.arange(90, 140, 10))
    axins.grid(linestyle='-.')

    axins1 = ax.inset_axes((0.15, 0.1, 0.2, 0.23))
    axins1.set_xlim(2, 4)
    axins1.set_ylim(100, 130)
    axins1.set_xticks(np.arange(2, 4, 0.5))
    axins1.set_yticks(np.arange(100, 130, 10))
    axins1.grid(linestyle='-.')

    # axins1 = ax.inset_axes((0.6, 0.3, 0.3, 0.2))
    # axins1.set_xlim(9, 11)
    # axins1.set_ylim(180, 210)
    # axins1.set_xticks(np.arange(9, 11, 0.5))
    # axins1.set_yticks(np.arange(180, 210, 10))
    # axins1.grid(linestyle='-.')


    data_path = Path("data/RTSN_data/5g_tsn")
    for csv_path in data_path.glob('*.csv'):
        all_data = np.loadtxt(csv_path)
        res = all_data[:, 0]
        t_5G = all_data[:, 1]
        q_t = all_data[:, 2]
        t_tsn = all_data[:, 3]
        inte_delay = all_data[:, 4]
        data_name = csv_path.stem
        line,  = plt.plot(
            res, t_5G, style_dict[data_name], color=color_dict[data_name], linewidth=0.8)
        line.set_label(f'{metric_dict[data_name]}')
        line,  = plt.plot(
            res, t_tsn, style_dict[data_name], color=color_dict[data_name], linewidth=0.8)
        # line.set_label(f'{metric_dict[data_name]}')
        axins.plot(
            res, t_5G,  style_dict[data_name], color=color_dict[data_name])
        axins.plot(
            res, t_tsn,  style_dict[data_name], color=color_dict[data_name])
        axins1.plot(
            res, t_5G,  style_dict[data_name], color=color_dict[data_name])
        axins1.plot(
            res, t_tsn,  style_dict[data_name], color=color_dict[data_name])

    # #处理图例
    handles, labels = plt.gca().get_legend_handles_labels()

    ax.indicate_inset_zoom(axins)
    ax.indicate_inset_zoom(axins1)

    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.xticks(np.arange(0, 15, 2))
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(linestyle='-.')
    plt.legend(bbox_to_anchor=(0, 1), handles=handles, labels=labels, loc='upper left', fontsize = 'medium')
    plt.xlabel('Reserved RB numbers',fontsize = 'x-large')
    plt.ylabel('Latency of 5G/TSN network',fontsize = 'x-large')
    plt.show()

    # fig.savefig(data_path / 'ave_pre_accu.pdf', dpi=600, format='pdf')


def save_figure4():
    #res —— t_5g,q_t

    # royalblue darkorange tomato forestgreen
    style_dict = {'RTSN_3000': '.--', 'RTSN_4500': 'o-', 'RTSN_6000': '^--'}
    metric_dict = {'RTSN_3000': '3000 ',
                   'RTSN_4500': '4500', 'RTSN_6000': '6000 '}
    # metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'RTSN_3000': 'forestgreen',
                  'RTSN_4500': 'royalblue', 'RTSN_6000': 'tomato'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    # 局部放大
    fig, ax = plt.subplots(figsize=(10, 5))
    # axins = ax.inset_axes((0.65, 0.35, 0.3, 0.35))
    # axins.set_xlim(6, 8)
    # axins.set_ylim(90, 140)
    # axins.set_xticks(np.arange(6, 8, 0.5))
    # axins.set_yticks(np.arange(90, 140, 10))
    # axins.grid(linestyle='-.')

    # 设置右边的Y轴
    ax2 = ax.twinx()
    ax2.set_ylim(0, 5)
    ax2.set_ylabel('Queue index of TSN gateway')

    data_path = Path("data/RTSN_data/q_t")
    for csv_path in data_path.glob('*.csv'):
        all_data = np.loadtxt(csv_path)
        res = all_data[:, 0]
        t_5G = all_data[:, 1]
        q_t = all_data[:, 2]
        t_tsn = all_data[:, 3]
        inte_delay = all_data[:, 4]
        data_name = csv_path.stem
        line,  = ax.plot(
            res, t_5G, style_dict[data_name], color='r', linewidth=0.8)
        line.set_label(f'5G delay: {metric_dict[data_name]}')
        # line,  = plt.plot(res, q_t, style_dict[data_name], color=color_dict[data_name], linewidth=0.8)
        # line.set_label(f'{metric_dict[data_name]}')
        # axins.plot(res, t_5G,  style_dict[data_name], color=color_dict[data_name])
        line, = ax2.plot(
            res, q_t, style_dict[data_name], color='b')
        line.set_label(f'Q_t: {metric_dict[data_name]}')

    # #处理图例
    ax_handles_labels = ax.get_legend_handles_labels()
    ax2_handle_labels = ax2.get_legend_handles_labels()
    handles, labels = list(
        map(lambda x: x[0]+x[1], zip(ax_handles_labels, ax2_handle_labels)))

    # ax.indicate_inset_zoom(axins)
    # ax.indicate_inset_zoom(axins1)

    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_xlabel('Reserved RB numbers')
    # plt.yticks(np.arange(0, 1, 0.1))
    ax.set_ylim(0, 250, 10)
    ax.tick_params(axis='y', labelcolor='r')
    ax.set_ylabel('5G delay', color='r')

    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('Qt', color='b')

    plt.grid(linestyle='-.')
    plt.legend(bbox_to_anchor=(0, 1), handles=handles,
               labels=labels, loc='upper left', fontsize='medium')
    plt.show()

    # fig.savefig(data_path / 'ave_pre_accu.pdf', dpi=600, format='pdf')

def save_figure5():
    #signal ratio —— t_5g

     # royalblue darkorange tomato forestgreen
    style_dict = {'signal_ratio_3000': '*--', 'signal_ratio_4500': 'o-','signal_ratio_6000':'v--'}
    metric_dict = {'signal_ratio_3000': 'Size:3000 ', 'signal_ratio_4500': 'Size:4500', 'signal_ratio_6000': 'Size:6000 '}
    # metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'6': 'green','7': 'royalblue', '8': 'darkorange','9':'mediumorchid'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    #局部放大
    # fig, ax= plt.subplots(figsize=(10, 5))
    # axins = ax.inset_axes((0.65, 0.35, 0.3, 0.35))
    # axins.set_xlim(6, 8)
    # axins.set_ylim(90, 140)
    # axins.set_xticks(np.arange(6, 8, 0.5))
    # axins.set_yticks(np.arange(90, 140, 10))
    # axins.grid(linestyle='-.')

    # axins1 = ax.inset_axes((0.15, 0.1, 0.2, 0.23))
    # axins1.set_xlim(2, 4)
    # axins1.set_ylim(100, 130)
    # axins1.set_xticks(np.arange(2, 4, 0.5))
    # axins1.set_yticks(np.arange(100, 130, 10))
    # axins1.grid(linestyle='-.')



    data_path = Path("data/RTSN_data/signal/using")
    for sub_dir in data_path.glob('*'):
        nodes = sub_dir.stem
        for csv_path in sub_dir.glob('*.csv'):
            all_data = np.loadtxt(csv_path)
            signal_ratio = all_data[:, 0]
            t_5G = all_data[:, 1]
            q_t = all_data[:, 2]
            t_tsn = all_data[:, 3]
            inte_delay = all_data[:, 4]
            data_name = csv_path.stem
            line,  = plt.plot(signal_ratio, t_5G, style_dict[data_name], color=color_dict[nodes], linewidth=0.8)
            line.set_label(f'|Rr,t|: {nodes} {metric_dict[data_name]}')
            # line,  = plt.plot(signal_ratio, t_tsn, style_dict[data_name], color=color_dict[data_name], linewidth=0.8)
            # line.set_label(f'{metric_dict[data_name]}')
            # axins.plot(signal_ratio, t_5G,  style_dict[data_name], color=color_dict[data_name])
            # axins.plot(signal_ratio, t_tsn,  style_dict[data_name], color=color_dict[data_name])
            # axins1.plot(res, t_5G,  style_dict[data_name], color=color_dict[data_name])
            # axins1.plot(res, t_tsn,  style_dict[data_name], color=color_dict[data_name])

    # #处理图例
    handles, labels = plt.gca().get_legend_handles_labels()

    # ax.indicate_inset_zoom(axins)
    # ax.indicate_inset_zoom(axins1)

    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.xticks(np.arange(0, 1, 0.05))
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(linestyle='-.')
    plt.legend(bbox_to_anchor=(0, 1), handles=handles, labels=labels, loc='upper left', fontsize = 'small', ncol = 2)
    plt.xlabel('Signal Ratio',fontsize = 'x-large')
    plt.ylabel('Latency of 5G network',fontsize = 'x-large')
    plt.show()

    # fig.savefig(data_path / 'ave_pre_accu.pdf', dpi=600, format='pdf')


if __name__ == '__main__':
    save_figure3()

    print('=> Generate done.')

import re
from pathlib import Path
from functools import cmp_to_key

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


metirc_order = {}

def legend_sort(x, y):
    x_labels = re.split('[()]', x[1])
    x_m_order = metirc_order[x_labels[0].strip()]
    x_n_order = int(x_labels[1])

    y_labels = re.split('[()]', y[1])
    y_m_order = metirc_order[y_labels[0].strip()]
    y_n_order = int(y_labels[1])

    if x_m_order != y_m_order:
        return x_m_order - y_m_order
    else:
        return x_n_order - y_n_order

def save_figure1():
    #royalblue darkorange tomato forestgreen
    style_dict = {'3000': '.--', '4500': 'o-', '6000': '^--'}
    metric_dict = {'cond_pre_accu': 'Cond ', 'MI_pre_accu': 'MI',
                'X2_pre_accu': 'X2 ', 'select_pre_accu': 'Select'}
    global metirc_order
    metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'cond_pre_accu': 'r', 'MI_pre_accu': 'forestgreen',
                'X2_pre_accu': 'royalblue', 'select_pre_accu': 'orange'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    fig, ax = plt.subplots(figsize=(10, 5))
    axins = ax.inset_axes((0.25, 0.2, 0.5, 0.4))
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
                data, color_dict[nodes], color=style_dict[accu_name], linewidth = 0.8)
            axins.plot(data, color_dict[nodes], color=style_dict[accu_name])
            line.set_label(f'{metric_dict[accu_name]}({nodes})')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels),
                                  key=cmp_to_key(legend_sort)))

    ax.indicate_inset_zoom(axins)
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
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
    #royalblue darkorange tomato forestgreen
    style_dict = {'RTSN_3000': '.--', 'RTSN_4500': 'o-', 'RTSN_6000': '^--'}
    metric_dict = {'RTSN_3000': '3000 ', 'RTSN_4500': '4500', 'RTSN_6000': '6000 '}
    # metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select': 4}
    color_dict = {'RTSN_3000': 'forestgreen',
                'RTSN_4500': 'royalblue', 'RTSN_6000': 'tomato'}

    plt.rcParams["legend.facecolor"] = 'whitesmoke'  # 设置图例背景色
    plt.rcParams["legend.edgecolor"] = '0.5'  # 设置图例边框深浅

    #局部放大
    fig, ax = plt.subplots(figsize=(10, 5))
    axins = ax.inset_axes((0.1, 0.45, 0.3, 0.4))
    axins.set_xlim(6, 8.5)
    axins.set_ylim(90, 140)
    axins.set_xticks(np.arange(6, 8.5, 0.5))
    axins.set_yticks(np.arange(90, 140, 10))
    axins.grid(linestyle='-.')

    data_path = Path("data/RTSN_data")
    for csv_path in data_path.glob('*.csv'):
        all_data = np.loadtxt(csv_path)
        res = all_data[:,0]
        t_5G = all_data[:,1]
        q_t = all_data[:,2]
        t_tsn = all_data[:,3]
        inte_delay = all_data[:,4]
        data_name = csv_path.stem
        line,  = plt.plot(res, t_5G, style_dict[data_name], color=color_dict[data_name], linewidth = 0.8)
        line.set_label(f'{metric_dict[data_name]}')
        axins.plot(res, t_5G,  style_dict[data_name], color=color_dict[data_name])

    # #处理图例
    handles, labels = plt.gca().get_legend_handles_labels()

    ax.indicate_inset_zoom(axins)
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.xticks(np.arange(0, 15, 2))
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(linestyle='-.')
    plt.legend(bbox_to_anchor=(1, 0), handles=handles, labels=labels, loc='lower right')
    plt.xlabel('Reserved RB numbers')
    plt.ylabel('Latency of 5G network')
    plt.show()

    # fig.savefig(data_path / 'ave_pre_accu.pdf', dpi=600, format='pdf')


if __name__ == '__main__':
    save_figure2()

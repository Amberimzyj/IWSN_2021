import re
from pathlib import Path
from functools import cmp_to_key

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["legend.facecolor"]='whitesmoke' #设置图例背景色
plt.rcParams["legend.edgecolor"]='0.5' #设置图例边框深浅

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


fig, ax = plt.subplots()

data_path = Path("data/pre_accu")

style_dict = {'cond_pre_accu': 'darkorange', 'MI_pre_accu': 'forestgreen', 'X2_pre_accu': 'royalblue', 'select_pre_accu':'tomato'}
metric_dict = {'cond_pre_accu': 'Cond ', 'MI_pre_accu': 'MI', 'X2_pre_accu': 'X2 ','select_pre_accu':'Select'}
metirc_order = {'X2': 1, 'MI': 2, 'Cond': 3, 'Select':4}
color_dict = {'3000': '.--','4500': 'o-', '6000': '^--'}

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

for sub_dir in data_path.glob('*'):
    nodes = sub_dir.stem
    for csv_path in sub_dir.glob('*.csv'):
        accu_name = csv_path.stem
        data = np.genfromtxt(csv_path, delimiter=',')
        line, = plt.plot(data, color_dict[nodes], color=style_dict[accu_name])
        line.set_label(f'{metric_dict[accu_name]}({nodes})')

handles, labels = plt.gca().get_legend_handles_labels()
handles, labels = zip(*sorted(zip(handles, labels), key=cmp_to_key(legend_sort)))

# x = np.arange(0, 25, 5)
# y_1 = data[0]
# y_2 = data[1]
# y_3 = data[2]

# # 设置放大区间
# zone_left = 3
# zone_right = 4

# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0.5 # x轴显示范围的扩展比例
# y_ratio = 0.5 # y轴显示范围的扩展比例

# # X轴的显示范围
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

# # Y轴的显示范围
# y = np.hstack((y_1[zone_left:zone_right], y_2[zone_left:zone_right], y_3[zone_left:zone_right]))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)

plt.xticks(np.arange(0, 25, 5))
plt.yticks(np.arange(0, 1, 0.1))
plt.grid(linestyle='-.')
plt.legend(bbox_to_anchor=(1, 0), handles=handles, labels=labels, loc='lower right')
plt.xlabel('Transmission Time Interval(TTI)')
plt.ylabel('Average Successful Prediction Ratio')
plt.figure(figsize=(10,5))
plt.show()

fig.savefig('data//pre_accu/ave_pre_accu.pdf',dpi = 600, format = 'pdf')
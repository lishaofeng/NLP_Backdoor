import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, markers

from cycler import cycler

# Create cycler object. Use any styling from above you please
monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker', ['^',',', '.']))

# Overriding styles for current script
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.prop_cycle'] = monochrome
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True

# data to plot
n_groups = 5
integrity = (0.8850, 0.9993, 1.0, 1.0, 1.0)
utility = (0.9247, 0.9171, 0.9268, 0.9208, 0.9281)

fig, ax = plt.subplots(1,1)
# ax2 = ax.twinx()

index = np.arange(1, 1+n_groups)
# print(index)
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index - bar_width/2, utility, bar_width,
alpha=opacity,
color='darkorange',
label='Functionality')

rects2 = ax.bar(index + bar_width/2, integrity, bar_width,
alpha=opacity,
color='royalblue',
label='Attack Success Rate')

ax.set_ylim(0.80, 1.08)
plt.yticks(fontsize=16)
ax.set_ylabel('Performance', fontsize=16)
# ax2.set_ylim(0.60, 1.0)
# ax2.set_ylabel('Functionality', fontsize=14)

ax.set_xlabel('L0-Norm', fontsize=16)
# ax.set_title('CIFAR-10', fontsize=16)
ax.set_xticks(index)

ax.set_xticklabels(index, fontsize=16)

ln1, la1 = ax.get_legend_handles_labels()
# ln2, la2 = ax2.get_legend_handles_labels()
ax.legend(ln1, la1, loc = 2, prop={'size': 16})

plt.tight_layout()

plt.savefig('CIFAR10_L0.pdf')
plt.show()
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
utility = (0.9269, 0.9240, 0.9257, 0.9265, 0.9251)
integrity = (0.9179, 0.9182, 0.9194, 0.9241, 0.9248)

fig, ax = plt.subplots(1,1)
# ax2 = ax.twinx()

index = np.arange(1, n_groups+1)
# print(index)
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index - bar_width/2, utility, bar_width,
alpha=opacity,
color='#CD853F',
label='Utility')

rects2 = ax.bar(index + bar_width/2, integrity, bar_width,
alpha=opacity,
color='#afabdb',
label='Functionality')

ax.set_ylim(0.60, 1.0)
ax.set_ylabel('Performance', fontsize=14)
# ax2.set_ylim(0.60, 1.0)
# ax2.set_ylabel('Functionality', fontsize=14)

ax.set_xlabel('L2-Norm', fontsize=14)
ax.set_title('CIFAR-10', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(index, fontsize=12)

ln1, la1 = ax.get_legend_handles_labels()
# ln2, la2 = ax2.get_legend_handles_labels()
ax.legend(ln1, la1, loc = 1, prop={'size': 12})

plt.tight_layout()

plt.savefig('CIFAR10_L2.pdf')
plt.show()
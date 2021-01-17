import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('mathtext', default='regular')

labels = ['1',  '2', '3','4','5']
y1 = [0.9179, 0.9182, 0.9194, 0.9241, 0.9248]
y2 = [0.9269, 0.9240, 0.9257, 0.9265, 0.9251]


x = np.arange(len(labels))  # the label locations
#[0, 1, 2]
width = 0.4  # the width of the bars

fig = plt.figure()
ax1 = fig.add_subplot(111)
#fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(labels, y1, lw=3, color='darkorange', label='Success rate', marker='v', markersize = 12)
ax2.plot(labels, y2, lw=3, color='royalblue', label='Functionality', marker='^', markersize = 12)

ax1.set_ylabel('Success rate', fontsize=12)
ax1.set_ylim(0.8, 1.05)
#ax2.set_ylabel('snr')
ax2.set_ylim(0.8, 1.05)
#ax.set_title('Scores by group and gender')
ax1.set_xticklabels(labels, fontsize=12)
#ax1.set_yticklabels(fontsize=14)
#ax2.set_yticklabels(fontsize=14)
#ax1.set_xticks(x, fontsize=14)
#ax1.set_label('Success rate')
#ax2.set_label('AVG SNR(dB)')
ax2.set_ylabel('Functionality', fontsize=12)
ax1.set_xlabel('CIFAR10', fontsize=12)
#ax2.set_xticklabels('Target threshold')
#ax1.grid()
ln1, la1 = ax1.get_legend_handles_labels()
ln2, la2 = ax2.get_legend_handles_labels()
#ax1.legend()
#ax2.legend()

#labs = ['Success rate', 'AVG SNR of attack audios']
ax2.legend(ln1 + ln2, la1 + la2, loc = 3, prop={'size': 12})


#def autolabel(rects, ax):
#    """Attach a text label above each bar in *rects*, displaying its height."""
#    for rect in rects:
#        height = rect.get_height()
#        ax.annotate('{}'.format(height),
#                    xy=(rect.get_x() + rect.get_width() / 2, height),
#                    xytext=(0, 3),  # 3 points vertical offset
#                    textcoords="offset points",
#                    ha='center', va='bottom')


#autolabel(rects1, ax1)
#autolabel(rects2, ax2)

#for xy in zip(labels,y1):
#    ax1.annotate('(%s, %s)' % xy, xy = xy, xytext = (1, 3), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
#for xy in zip(labels,y2):
#    ax2.annotate('(%s, %s)' % xy, xy = xy, xytext = (1, 3), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)


ax1.annotate('(%s, %s)' % (labels[0], y1[0]), xy = (labels[0], y1[0]), xytext = (7, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax1.annotate('(%s, %s)' % (labels[1], y1[1]), xy = (labels[1], y1[1]), xytext = (7, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax1.annotate('(%s, %s)' % (labels[2], y1[2]), xy = (labels[2], y1[2]), xytext = (7, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax1.annotate('(%s, %s)' % (labels[3], y1[3]), xy = (labels[3], y1[3]), xytext = (7, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax1.annotate('(%s, %s)' % (labels[4], y1[4]), xy = (labels[4], y1[4]), xytext = (-45, -5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)

ax2.annotate('(%s, %s)' % (labels[0], y2[0]), xy = (labels[0], y2[0]), xytext = (20, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax2.annotate('(%s, %s)' % (labels[1], y2[1]), xy = (labels[1], y2[1]), xytext = (20, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax2.annotate('(%s, %s)' % (labels[2], y2[2]), xy = (labels[2], y2[2]), xytext = (12, 5), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax2.annotate('(%s, %s)' % (labels[3], y2[3]), xy = (labels[3], y2[3]), xytext = (12, 12), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
ax2.annotate('(%s, %s)' % (labels[4], y2[4]), xy = (labels[4], y2[4]), xytext = (-55, -15), textcoords = "offset points", ha = 'center', va = 'bottom', fontsize=12)
fig.tight_layout()

plt.show()
fig.savefig('whitebox-result-1.pdf', bbox_inches='tight')

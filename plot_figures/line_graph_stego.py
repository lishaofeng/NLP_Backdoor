import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib import rc
import pandas as pd

rc('mathtext', default='regular')

labels = [i * 0.001 for i in range(1, 7)]
y1 = []
y2 = []

# def read_confusion_matrix():
#     with open('log_ijr.out') as fp:
#         for line in fp.readlines():
#             el = line.split('\t')
#             y1.append(float(el[1]))
#             y2.append(float(el[2]))
#
# read_confusion_matrix()

df = pd.read_excel("pos_len.xlsx", sheet_name="ijr_94")
print(df.columns)
y1 = df["AUC"].loc[:5] * 100
y2 = df["ASR"].loc[:5] * 100
print(y1)
print(y2)

width = 0.4

fig = plt.figure(figsize=(5,3))
ax1 = fig.add_subplot(111)

ax1.plot(labels, y1, lw=3, color='darkorange', label="1", linestyle='dashed', marker='v', markersize = 10) # v
ax1.grid(ls='--')
ax1.set_ylabel('Functionality(%)', fontsize=12)
ax1.set_ylim(40, 102)
ax1.set_xlabel('Injection Rate', fontsize=12)

ax1.set_xlim(0.0008, 0.0062)

ax2 = ax1.twinx()
ax2.plot(labels, y2, lw=3, color='royalblue', label="2", linestyle='dashed', marker='^', markersize = 10) # ^
ax2.set_ylabel('Attack Success Rate(%)', fontsize=12)
ax2.set_ylim(40, 102)

ax1.spines['top'].set_visible(False)
ax1.tick_params(top=False, right=False)


handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles=handles+handles2, labels=["Functionality", "Attack Success Rate"], loc='lower right', fontsize=12)

fig.tight_layout()
# fig.show()
fig.savefig('injection_rate_homo_dash.pdf', bbox_inches='tight')

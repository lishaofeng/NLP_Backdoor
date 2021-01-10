'''
=========================
3D surface (checkerboard)
=========================

Demonstrates plotting a 3D surface colored in a checkerboard pattern.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10


cf_mr_clean = [None] * 10
for i in range(10):
    row = []
    for j in range(10):
        row.append(0.9248)
        cf_mr_clean[i] = row

cf_mr_backdoor = [None] * 10
for i in range(10):
    row = []
    for j in range(10):
        row.append(1.0)
        cf_mr_backdoor[i] = row

def read_confusion_matrix():
    with open('./data.txt') as fp:
        for line in fp.readlines():
            el = line.split('\t')
            target, src, att_ratio, clean_acc = int(el[0]), int(el[1]), el[3], el[2]
            cf_mr_clean[src][target] = float(clean_acc) / 100.0
            cf_mr_backdoor[src][target] = float(att_ratio) / 100.0

read_confusion_matrix()


fig = plt.figure()
ax1 = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 10, 1)
xlen = len(X)
Y = np.arange(0, 10, 1)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
Z = np.array(cf_mr_backdoor)

# print(X.shape, Y.shape, Z.shape)
# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple1 = ('w', 'r')
colors1 = np.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors1[x, y] = colortuple1[(x + y) % len(colortuple1)]

# Customize the z axis.
ax1.set_zlim(0.8, 1.05)
ax1.set_xticks(range(0, 10, 1))
ax1.set_yticks(range(0, 10, 1))
ax1.set_xlabel('Source Label', fontsize=12)
ax1.set_ylabel('Target Label', fontsize=12)

# Plot the surface with face colors taken from the array we made.
surf = ax1.plot_surface(X, Y, Z, facecolors=colors1, linewidth=0)


colortuple2 = ('w', 'r')
colors2 = np.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors2[x, y] = colortuple2[(x + y) % len(colortuple2)]

ax2 = fig.gca(projection='3d')
ax2.set_zlim(0.85, 1.0)
ax2.set_xticks(range(0, 10, 1))
ax2.set_yticks(range(0, 10, 1))
Z2 = np.array(cf_mr_clean)
surf2 = ax2.plot_surface(X, Y, Z2, facecolors=colors2, linewidth=0)
# handles, labels = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# plt.legend(handles=handles+handles2, labels=["Attack Success Rate", "Invisibility"], loc='lower left', fontsize=14)
# ax1.legend()
# ax2.legend()
# plt.show()
plt.savefig('stego_cifar10.pdf')
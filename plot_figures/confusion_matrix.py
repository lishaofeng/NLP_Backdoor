import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix


cf_mr_clean = [None]*10
for i in range(10):
    row = []
    for j in range(10):
        row.append(0.0)
        cf_mr_clean[i] = row


cf_mr_backdoor = [None]*10
for i in range(10):
    row = []
    for j in range(10):
        row.append(0.0)
        cf_mr_backdoor[i] = row

# print(cf_mr)
#
# cf_mr[2][3] = 1.0
#
# print(cf_mr)

def read_confusion_matrix():
    with open('./data.txt') as fp:
        for line in fp.readlines():
            el = line.split('\t')
            target, src, att_ratio, clean_acc = int(el[0]), int(el[1]), el[3], el[2]
            cf_mr_clean[src][target] = 1 - float(clean_acc)/100.0
            cf_mr_backdoor[src][target] = 1 - float(att_ratio)/100.0


def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Labels',
           xlabel='Target Labels'
           )

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center")


    # plt.colorbar(ax, boundaries=[i * 0.01 for i in range(31)])
    fig.tight_layout()

    # im = ax.imshow()
    # plt.colorbar(im, [i * 0.01 for i in range(31)])
    fig.savefig(title+'.pdf')
    return ax

def figure_3():
    read_confusion_matrix()
    cm_clean, cm_back = np.array(cf_mr_clean), np.array(cf_mr_backdoor)
    classes = [i for i in range(10)]
    print(cm_clean, cm_back)
    plot_confusion_matrix(cm_clean, classes=classes, title='Functionality')
    plot_confusion_matrix(cm_back, classes=classes, title='Attack Error Rate')
    np.set_printoptions(precision=3)
    plt.show()



if __name__ == "__main__":
    figure_3()
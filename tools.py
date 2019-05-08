import matplotlib.pyplot as plt
import linecache


def line_show(line_batch, text=None):
    batch = line_batch["line"][0].shape[0]
    line1 = line_batch["line"][0].numpy().reshape(batch, -1)
    line2 = line_batch["line"][1].numpy().reshape(batch, -1)
    label = line_batch["label"].numpy()
    for i in range(batch):
        ax = plt.subplot(batch / 2, 2, i + 1)
        plt.subplots_adjust(wspace=0.2, hspace=1.5)
        plt.plot(line1[i])
        plt.plot(line2[i])
        plt.axis
        if text:
            ax.set_title(text + str(label[i]), fontsize=12, color='r')


def line_show_test(line_batch, text=None):
    line1 = line_batch[0].numpy().reshape(1, -1)
    line2 = line_batch[1].numpy().reshape(1, -1)
    plt.figure()
    plt.plot(line1[0])
    plt.plot(line2[0])
    if text:
        plt.title(text, fontsize='large', fontweight='bold')


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def read_data_row(path, num):
    return linecache.getline(path, num)


def normalization(l1, l2):
    pass
import os
import numpy as np
from Geometric import G_layers
from Geometric import G_losses, G_utils
import matplotlib
import matplotlib.pyplot as plt

def NCC(a, v, zero_norm=False):
    """
    zero_norm = False:
    :return NCC

    zero_norm = True:
    :return ZNCC
    """
    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)
    else:
        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)
    return np.correlate(a, v)

def test(dec_model, cls_model, red_test_dat, orange_test_dat):
    # Red to Orange test
    code = np.expand_dims(np.array([0., 1.]), axis=0)
    for i in range(len(red_test_dat)):
        if i == 0:
            target_c = code
        else:
            target_c = np.append(target_c, code, axis=0)

    c1, c2, c3 = G_utils.codemap(target_c=target_c)
    tst_map = dec_model({"dec_in": red_test_dat, "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]
    pseudo_image = red_test_dat + tst_map

    res = cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]
    acc = G_losses.acc(target_c, res)

    stack_p = np.empty((len(red_test_dat), 1))
    a = G_layers.flatten()(tst_map)
    b = G_layers.flatten()(-(red_test_dat - orange_test_dat))
    for i in range(len(red_test_dat)):
        stack_p[i] = NCC(a[i], b[i])

    # Orange to Red test
    code = np.expand_dims(np.array([1., 0.]), axis=0)
    for i in range(len(orange_test_dat)):
        if i == 0: target_c = code
        else: target_c = np.append(target_c, code, axis=0)

    c1, c2, c3 = G_utils.codemap(target_c=target_c)
    tst_map = dec_model({"dec_in": orange_test_dat, "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]
    pseudo_image = orange_test_dat + tst_map

    res = cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]
    acc += G_losses.acc(target_c, res)

    stack_n = np.empty((len(orange_test_dat), 1))
    a = G_layers.flatten()(tst_map)
    b = G_layers.flatten()(red_test_dat - orange_test_dat)
    for i in range(len(orange_test_dat)):
        stack_n[i] = NCC(a[i], b[i])

    print("ACC: %4f | NCC(+): %4f, %4f | NCC(-): %4f, %4f" % (acc/2, np.mean(stack_p), np.std(stack_p), np.mean(stack_n), np.std(stack_n)))
    return acc / 2


def Visualization(dec_model, red_test_dat, orange_test_dat, save_path, epoch):
    code = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    c1, c2, c3 = G_utils.codemap(target_c=code)
    dat = np.concatenate((red_test_dat[:2], orange_test_dat[8:10]), axis=0)
    GT_dat = np.concatenate((orange_test_dat[:2], red_test_dat[8:10]), axis=0)

    CFmap = dec_model({"dec_in": dat, "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]
    pseudo_image = dat + CFmap

    fig = plt.figure()
    rows, cols = 4, 7

    for i in range(4):
        num = i * cols + 1
        ax1 = fig.add_subplot(rows, cols, num)
        ax1.imshow(dat[i, :, :, :])
        ax1.axis("off")
        ax2 = fig.add_subplot(rows, cols, num + 1)
        ax2.imshow(CFmap[i, :, :, :])
        ax2.axis("off")

        ax3 = fig.add_subplot(rows, cols, num + 2)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "red"])
        min = np.min(CFmap[i, :, :, 0])
        max = np.max(CFmap[i, :, :, 0])
        norm = plt.Normalize(min, max)
        ax3.imshow(CFmap[i, :, :, 0], cmap=cmap, norm=norm)
        ax3.axis("off")

        ax4 = fig.add_subplot(rows, cols, num + 3)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "green"])
        min = np.min(CFmap[i, :, :, 0])
        max = np.max(CFmap[i, :, :, 0])
        norm = plt.Normalize(min, max)
        ax4.imshow(CFmap[i, :, :, 1], cmap=cmap, norm=norm)
        ax4.axis("off")

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "blue"])
        min = np.min(CFmap[i, :, :, 0])
        max = np.max(CFmap[i, :, :, 0])
        norm = plt.Normalize(min, max)
        ax5 = fig.add_subplot(rows, cols, num + 4)
        ax5.imshow(CFmap[i, :, :, 2], cmap=cmap, norm=norm)
        ax5.axis("off")

        ax6 = fig.add_subplot(rows, cols, num + 5)
        ax6.imshow(pseudo_image[i, :, :, :])
        ax6.axis("off")
        ax7 = fig.add_subplot(rows, cols, num + 6)
        ax7.imshow(GT_dat[i, :, :, :])
        ax7.axis("off")

        if i == 0:
            ax1.set_title("Input")
            ax2.set_title("CF CFmap")
            ax3.set_title("R")
            ax4.set_title("G")
            ax5.set_title("B")
            ax6.set_title("Add result")
            ax7.set_title("GT")

    fig.suptitle("Result image")
    save_path = save_path + "/tst_plt/"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.savefig(save_path + "epoch%d_result.png"%epoch)
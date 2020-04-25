import torchvision
import numpy as np
import matplotlib.pyplot as plt


def plot_image_grid(image_list):
    img_grid = torchvision.utils.make_grid(image_list)
    img_grid = np.transpose(img_grid, (1, 2, 0))
    plt.imshow(img_grid)
    plt.show()


def plot_image_list(image_list,
                    label_list=[],
                    pred_list=[],
                    col_num=8,
                    fig_size=(25, 5)):
    row_num = int(len(image_list) // col_num)
    if len(image_list) % col_num != 0:
        row_num += 1

    fig, axes = plt.subplots(row_num, col_num, figsize=fig_size)
    for i, ax in enumerate(axes.flat):
        if i >= len(image_list):
            ax.imshow(np.squeeze(np.zeros(image_list[0].shape)), cmap="gray")
        else:
            ax.imshow(np.squeeze(image_list[i]), cmap="gray")
            if len(label_list) > 0:
                title = str(label_list[i])
                color = "black"
                if len(pred_list) > 0:
                    title += " - " + str(pred_list[i])
                    color = "green" if label_list[i] == pred_list[i] else "red"
                ax.set_title(title, color=color)
    plt.show()

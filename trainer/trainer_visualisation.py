import matplotlib.pyplot as plt
import numpy as np
import os


def plot_heatmaps(predicted_heatmap, groundtruth_heatmap, sigmoid=False):
    fig = plt.figure()  # figsize=(12, 48)
    fig, axs = plt.subplots(3, 2)
    for idx in np.arange(3):
        if (sigmoid):
            axs[idx, 0].imshow(np.exp(-np.logaddexp(0, -predicted_heatmap[idx])))
        else:
            axs[idx, 0].imshow(predicted_heatmap[idx])

        axs[idx, 1].imshow(groundtruth_heatmap[idx])  # cmap="Greys")

    return fig


def save_test_outputs(checkpoint_dir, batch, output_heatmap, output_bbox):
    image = batch["image"].cpu().detach().numpy()
    center_heatmap = batch["center_heatmap"].cpu().detach().numpy()
    bbox_heatmap = batch["bbox_heatmap"].cpu().detach().numpy()
    image_id = batch["image_id"].cpu().detach().numpy()
    os.makedirs(os.path.join(checkpoint_dir, "test_outputs"), exist_ok=True)
    for index in range(output_bbox.shape[0]):
        fig = plt.figure()  # figsize=(12, 48)
        fig, axs = plt.subplots(4, 2)

        axs[0, 0].imshow(image[index, :, :, :].transpose(1, 2, 0))

        axs[1, 0].imshow(output_heatmap[index])  # cmap="Greys")
        axs[1, 1].imshow(center_heatmap[index])  # cmap="Greys")

        axs[2, 0].imshow(output_bbox[index, 0, :, :])  # cmap="Greys")
        axs[2, 1].imshow(bbox_heatmap[index, 0, :, :])  # cmap="Greys")

        axs[3, 0].imshow(output_bbox[index, 1, :, :])  # cmap="Greys")
        axs[3, 1].imshow(bbox_heatmap[index, 1, :, :])  # cmap="Greys")

        plt.savefig(os.path.join(checkpoint_dir, "test_outputs", str(image_id[index]) + ".png"))
        plt.close("all")
    pass

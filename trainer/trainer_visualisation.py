import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(predicted_heatmap, groundtruth_heatmap, sigmoid=False):
    fig = plt.figure()  # figsize=(12, 48)
    fig, axs = plt.subplots(3, 2)
    for idx in np.arange(3):
        if (sigmoid):
            axs[idx, 0].imshow(np.exp(-np.logaddexp(0, -predicted_heatmap[idx])), cmap="Greys")
        else:
            axs[idx, 0].imshow(predicted_heatmap[idx], cmap="Greys")
            
        axs[idx, 1].imshow(groundtruth_heatmap[idx], cmap="Greys")

    return fig

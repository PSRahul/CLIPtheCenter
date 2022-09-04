import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(predicted_heatmap, groundtruth_heatmap):
    fig = plt.figure(figsize=(12, 48))
    fig, axs = plt.subplots(3, 2)
    for idx in np.arange(3):
        axs[idx, 0].imshow(np.exp(-np.logaddexp(0, -predicted_heatmap[idx])))
        axs[idx, 1].imshow(groundtruth_heatmap[idx])

    return fig

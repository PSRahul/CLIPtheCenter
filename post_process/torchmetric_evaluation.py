from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch


def calculate_torchmetrics_mAP(gt, prediction):
    # columns = [["image_id", "bbox_x", "bbox_y", "w", "h", "score", "class_label"]]

    preds = [
        dict(
            boxes=torch.tensor(prediction[:, 1:5]),
            scores=torch.tensor(prediction[:, 5]),
            labels=torch.tensor(prediction[:, 6]),
        )
    ]
    target = [
        dict(
            boxes=torch.tensor(gt[:, 1:5]),
            labels=torch.tensor(gt[:, 6]),
        )
    ]
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(preds, target)
    print(metric.compute())

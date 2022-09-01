from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


def calculate_coco_result(gt, prediction, image_index_only, image_index):
    annType = 'bbox'
    if image_index_only:
        prediction = prediction[prediction[:, 0] == image_index]
        prediction = prediction[0, :]
        prediction = np.expand_dims(prediction, 0)
    coco_gt = COCO(gt)
    coco_dt = coco_gt.loadRes(prediction)
    coco_eval = COCOeval(coco_gt, coco_dt, annType)
    if image_index_only:
        coco_eval.params.imgIds = image_index

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

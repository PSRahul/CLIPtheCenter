from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_coco_result(gt, prediction):
    annType = 'bbox'
    coco_gt = COCO(gt)
    coco_dt = coco_gt.loadRes(prediction)
    coco_eval = COCOeval(coco_gt, coco_dt, annType)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

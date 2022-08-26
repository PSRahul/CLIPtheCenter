from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annType = 'bbox'


class COCORunner():
    def __init__(self, groundtruth, prediction):
        self.coco_groundtruth = COCO(groundtruth)
        self.coco_prediction = self.coco_groundtruth.loadRes(prediction)

        self.evaluate()

    def evaluate(self):
        imgIds = sorted(self.coco_groundtruth.getImgIds())

        # running evaluation
        cocoEval = COCOeval(self.coco_groundtruth, self.coco_prediction, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.catIds = [0]
        cocoEval.evaluate()
        cocoEval.accumulate()
        print(cocoEval.summarize())

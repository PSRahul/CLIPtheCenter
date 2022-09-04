import torch.nn as nn
import torch
from torchinfo import summary
from network.models.EfficientnetConv2DT.utils import get_bounding_box_prediction
import clip
from PIL import Image
import os


class CLIPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device="cuda")
        self.clip_model = self.clip_model.cuda().eval()

    def forward(self, image_path, output_roi, train_set):
        if (train_set):
            self.image_root = os.path.join(self.cfg["data"]["train_data_root"], "data")
        else:
            self.image_root = os.path.join(self.cfg["data"]["val_data_root"], "data")
        clip_encodings = torch.zeros((output_roi.shape[0], 512))
        for batch_index in range(self.cfg["data"]["train_batch_size"]):
            image = Image.open(os.path.join(self.image_root, image_path[batch_index]))
            output_roi_index = output_roi[
                               batch_index * self.cfg["evaluation"]["topk_k"]:batch_index * self.cfg["evaluation"][
                                   "topk_k"] + self.cfg["evaluation"]["topk_k"]]
            for detection_index in range(self.cfg["evaluation"]["topk_k"]):
                detection = output_roi_index[detection_index]
                bbox = detection[1:5]
                if (bbox[2] < 0):
                    bbox[2] = 0
                if (bbox[3] < 0):
                    bbox[3] = 0

                (left, upper, right, lower) = (
                    int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                if ((right - left) == 0):
                    right = 5 + left
                if ((lower - upper) == 0):
                    lower = 5 + upper
                # print((left, upper, right, lower))
                image_cropped = image.crop((left, upper, right, lower))

                image_cropped_clip = self.clip_preprocess(image_cropped).unsqueeze(0)

                image_clip_embedding = self.clip_model.encode_image(image_cropped_clip.cuda())

                clip_encodings[batch_index * self.cfg["evaluation"]["topk_k"] + detection_index,
                :] = image_clip_embedding

        return clip_encodings

    def print_details(self):
        pass

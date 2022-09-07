import torch.nn as nn
from torchinfo import summary
import torch
from network.heads.bbox_head import SMP_BBoxHead
from network.heads.heatmap_head import SMP_HeatMapHead
from network.heads.roi_head import SMP_RoIHead
from network.roi_classifier.clip_model import CLIPModel
from network.roi_classifier.utils import get_masked_heatmaps, get_masks
from network.models.SMP_DeepLab.utils import get_bounding_box_prediction
from network.model_utils import weights_init

from torch.nn import Identity
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import DeepLabV3Plus


class SMPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        smp_model_name = globals()[cfg["smp"]["model"]]
        self.encoder_decoder_model = smp_model_name(
            encoder_name=cfg["smp"]["encoder_name"],
            encoder_weights=cfg["smp"]["encoder_weights"],
            in_channels=3,
            classes=1
        )
        self.encoder_decoder_model.segmentation_head = Identity()
        self.heatmap_head = SMP_HeatMapHead(cfg)
        self.bbox_head = SMP_BBoxHead(cfg)
        self.roi_head = SMP_RoIHead(cfg)
        self.clip_model = CLIPModel(cfg)

        self.cfg = cfg
        # self.model_init()

    def model_init(self):
        self.encoder_decoder_model.decoder(weights_init)
        self.heatmap_head.model.apply(weights_init)
        self.bbox_head.model.apply(weights_init)
        self.roi_head.model.apply(weights_init)

    def forward(self, batch, train_set=True):
        image = batch["image"].to(self.cfg["device"])
        image_path = batch["image_path"]
        image_id = batch['image_id'].to(self.cfg["device"])
        flattened_index = batch['flattened_index']
        x = self.encoder_decoder_model(image)
        # return x
        output_heatmap = self.heatmap_head(x)
        output_bbox = self.bbox_head(x)
        output_roi = self.roi_head(x)
        with torch.no_grad():
            # output_bbox = transpose_and_gather_output_array(output_bbox, flattened_index)
            # output_offset = transpose_and_gather_output_array(output_offset, flattened_index)
            # output_heatmap = output_heatmap.squeeze(dim=1)
            detections = get_bounding_box_prediction(self.cfg,
                                                     output_heatmap.detach(),
                                                     output_bbox.detach(),
                                                     image_id)
            output_clip_encoding = self.clip_model(image_path, detections, train_set=train_set)
            output_mask = get_masks(self.cfg, detections)

            masked_heatmaps_features = get_masked_heatmaps(self.cfg, output_roi, output_mask.cuda(),
                                                           train_set=train_set)

        return output_heatmap, output_bbox, detections, output_clip_encoding, model_encodings

    def forward_summary(self, image):
        x = self.encoder_decoder_model(image)
        output_heatmap = self.heatmap_head(x)

        output_bbox = self.bbox_head(x)
        return x  # output_heatmap, output_bbox

    def print_details(self):
        batch_size = 32
        summary(self, input_size=(3, 3, 320, 320))

        # summary(self.encoder_model, input_size=(batch_size, 3, 300, 300))
        # summary(self, input_size=(batch_size, 512, 12, 12))

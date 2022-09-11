import torch.nn as nn
from torchinfo import summary
import torch
from network.heads.bbox_head import SMP_BBoxHead
from network.heads.heatmap_head import SMP_HeatMapHead
from network.heads.roi_head import SMP_RoIHead
from network.heads.embedder import SMP_Embedder
from network.roi_classifier.clip_model import CLIPModel
from network.roi_classifier.utils import get_masked_heatmaps, get_binary_masks, make_detections_valid
from network.models.SMP_DeepLab.utils import get_bounding_box_prediction
from network.model_utils import weights_init, set_parameter_requires_grad
import sys
from torch.nn import Identity
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import DeepLabV3Plus, Unet


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
        self.encoder_decoder_model.segmentation_head = nn.Identity()

        self.heatmap_head = SMP_HeatMapHead(cfg)
        self.bbox_head = SMP_BBoxHead(cfg)
        self.roi_head = SMP_RoIHead(cfg)
        self.clip_model = CLIPModel(cfg)
        self.embedder = SMP_Embedder(cfg)

        self.cfg = cfg
        if cfg["smp"]["freeze_encoder"]:
            self.freeze_params()
        self.model_init()

    def freeze_params(self):
        set_parameter_requires_grad(model=self.encoder_decoder_model.encoder, freeze_params=True)

    def model_init(self):
        # self.encoder_decoder_model.decoder(weights_init)
        # self.heatmap_head.model.apply(weights_init)
        self.bbox_head.model.apply(weights_init)
        self.roi_head.model.apply(weights_init)
        self.embedder.model.apply(weights_init)

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
            if (self.cfg["trainer"]["bbox_heatmap_loss"]):
                detections = get_bounding_box_prediction(self.cfg,
                                                         output_heatmap.detach(),
                                                         output_bbox.detach(),
                                                         image_id)

            else:
                detections = get_bounding_box_prediction(self.cfg,
                                                         output_heatmap.detach(),
                                                         output_bbox.detach(),
                                                         image_id)

            detections_adjusted = make_detections_valid(detections)
            clip_encoding = self.clip_model(image_path, detections_adjusted, train_set=train_set)
            output_mask = get_binary_masks(self.cfg, detections_adjusted)

        masked_roi_heatmap = get_masked_heatmaps(self.cfg, output_roi, output_mask.cuda(),
                                                 train_set=train_set)
        model_encodings = self.embedder(masked_roi_heatmap)
        model_encodings_normalised = model_encodings / model_encodings.norm(dim=-1, keepdim=True)
        return output_heatmap, output_bbox, detections, clip_encoding, model_encodings_normalised

    def forward_summary(self, image):
        x = self.encoder_decoder_model(image)
        output_heatmap = self.heatmap_head(x)

        output_bbox = self.bbox_head(x)
        return x  # output_heatmap, output_bbox

    def print_details(self):
        batch_size = 32
        summary(self.embedder, input_size=(3, 1, 320, 320))

        # summary(self.heatmap_head, input_size=(3, 16, 320, 320))
        # sys.exit(0)
        # summary(self.encoder_model, input_size=(batch_size, 3, 300, 300))
        # summary(self, input_size=(batch_size, 512, 12, 12))

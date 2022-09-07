import torch.nn as nn
from torchinfo import summary
import torch
from network.decoder.decoder_model import DecoderConvTModel
from network.roi_classifier.clip_model import CLIPModel
# from network.roi_classifier.mask_model import MaskModel
from network.roi_classifier.utils import get_masked_heatmaps
from network.models.EfficientnetConv2DT.utils import get_bounding_box_prediction
from network.model_utils import weights_init


class EfficientnetConv2DTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_model_name = globals()[cfg["model"]["encoder"]["encoder_name"]]
        self.encoder_model = encoder_model_name(cfg)
        self.decoder_model = DecoderConvTModel(cfg)
        self.heatmap_head = HeatMapHead(cfg)
        self.offset_head = OffSetHead(cfg)
        self.bbox_head = BBoxHead(cfg)
        self.roi_model = RoIModel(cfg)
        self.clip_model = CLIPModel(cfg)
        self.mask_model = MaskModel(cfg)
        self.cfg = cfg
        self.model_init()

    def model_init(self):
        self.decoder_model.model.apply(weights_init)
        self.heatmap_head.model.apply(weights_init)
        self.offset_head.model.apply(weights_init)
        self.bbox_head.model.apply(weights_init)
        self.roi_model.model.apply(weights_init)

    def forward(self, batch, train_set=True):
        image = batch["image"].to(self.cfg["device"])
        image_path = batch["image_path"]
        image_id = batch['image_id'].to(self.cfg["device"])
        flattened_index = batch['flattened_index']
        x = self.encoder_model(image)
        # return x
        x = self.decoder_model(x)
        # return x
        output_heatmap = self.heatmap_head(x)
        output_offset = self.offset_head(x)
        output_bbox = self.bbox_head(x)
        with torch.no_grad():
            # output_bbox = transpose_and_gather_output_array(output_bbox, flattened_index)
            # output_offset = transpose_and_gather_output_array(output_offset, flattened_index)

            detections = get_bounding_box_prediction(self.cfg,
                                                     output_heatmap.detach(),
                                                     output_offset.detach(),
                                                     output_bbox.detach(),
                                                     image_id)
            output_clip_encoding = self.clip_model(image_path, detections, train_set=train_set)
            output_mask = self.mask_model(detections)

        masked_heatmaps_features = get_masked_heatmaps(self.cfg, output_heatmap, output_mask.cuda(),
                                                       train_set=train_set)
        model_encodings = self.roi_model(masked_heatmaps_features)
        return output_heatmap, output_bbox, output_offset, detections, output_clip_encoding, model_encodings

    def print_details(self):
        batch_size = 32
        summary(self.decoder_model, input_size=(24, 1408, 10, 10))

        # summary(self.encoder_model, input_size=(batch_size, 3, 300, 300))
        # summary(self, input_size=(batch_size, 512, 12, 12))

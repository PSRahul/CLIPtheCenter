from network.encoder.torchhub import *
from network.decoder.decoder_model import DecoderConvTModel
from network.heads.heatmap_head import HeatMapHead

from network.heads.offset_head import OffSetHead

from network.heads.bbox_head import BBoxHead


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DetectionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_model_name = globals()[cfg["model"]["encoder"]["encoder_name"]]
        self.encoder_model = encoder_model_name(cfg)
        self.decoder_model = DecoderConvTModel(cfg)
        self.heatmap_head = HeatMapHead(cfg)
        self.offset_head = OffSetHead(cfg)
        self.bbox_head = BBoxHead(cfg)

    def forward(self, x):
        # x = self.encoder_model(x)
        x = self.decoder_model(x)
        output_heatmap = self.heatmap_head(x)
        output_offset = self.offset_head(x)
        output_bbox = self.bbox_head(x)

    def print_details(self):
        batch_size = 32
        # summary(self, input_size=(batch_size, 3, 384, 384))
        summary(self, input_size=(batch_size, 512, 12, 12))

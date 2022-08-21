from network.encoder.torchhub import *

class DetectionModel(nn.Module):
    def __init__(self,cfg):

        pytorch_model_name = globals()[cfg["model"]["name"]]
        pytorch_model = pytorch_model_name(cfg)


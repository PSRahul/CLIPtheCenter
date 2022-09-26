import copy

import torch
import torch.nn as nn


def set_parameter_requires_grad(model, freeze_params):
    for param in model.parameters():
        param.requires_grad = not freeze_params
    return model


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        #nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        #nn.init.constant_(m.bias, 0)
    #if isinstance(m, nn.Linear):
    #    torch.nn.init.kaiming_uniform_(m.weight)
    #    nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm2d):
        pass
        #nn.init.constant_(m.weight, 1)
        #nn.init.constant_(m.bias, 0)

import torch


def _gather_output_feature(output_feature, ind, mask=None):
    dim = output_feature.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    output_feature = output_feature.gather(1, ind.type(torch.int64))
    return output_feature


def _transpose_and_gather_output_feature(output_feature, ind):
    output_feature = output_feature.permute(0, 2, 3, 1).contiguous()
    output_feature = output_feature.view(output_feature.size(0), -1, output_feature.size(3))
    output_feature = _gather_output_feature(output_feature, ind)
    return output_feature

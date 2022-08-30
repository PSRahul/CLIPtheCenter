import torch


def gather_output_array(output_array, ind, mask=None):
    dim = output_array.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    output_array = output_array.gather(1, ind.type(torch.int64))
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(output_array)
        output_array = output_array[mask]
        output_array = output_array.view(-1, dim)
    return output_array


def transpose_and_gather_output_array(output_array, ind):
    output_array = output_array.permute(0, 2, 3, 1).contiguous()
    output_array = output_array.view(output_array.size(0), -1, output_array.size(3))
    output_array = gather_output_array(output_array, ind)
    return output_array

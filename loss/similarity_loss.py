import torch.nn.functional as F
import torch
import torchvision


def calculate_embedding_loss(predicted_embedding, groundtruth_embedding):
    target = torch.ones((predicted_embedding.shape[0])).cuda()
    embedding_loss = F.cosine_embedding_loss(input1=predicted_embedding, input2=groundtruth_embedding,
                                             target=target)
    return embedding_loss

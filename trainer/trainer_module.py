from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer():

    def __init__(self, cfg, checkpoint_dir):
        self.writer = SummaryWriter(checkpoint_dir)
        self.set_training_parameters()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def set_training_parameters(self):
        pass
        # self.heatmap_loss_function = torchvision.ops.sigmoid_focal_loss(reduction="mean")

    def calculate_heatmap_loss(self, predicted_heatmap, groundtruth_heatmap):
        heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
                                                          reduction="mean")
        return heatmap_loss

    def train(self, model, train_dataloader):
        running_loss = 0.0
        model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(5):

            for i, batch in tqdm(enumerate(train_dataloader, 0)):

                # 5

                # 10
                image = batch["image"].to(self.device)
                # 20
                optimizer.zero_grad()
                # 30
                output_heatmap, output_offset, output_bbox = model(image)
                # 40
                output_heatmap = output_heatmap.squeeze(dim=1).to(self.device)
                heatmap_loss = self.calculate_heatmap_loss(output_heatmap, batch["heatmap"].to(self.device))
                # 50
                heatmap_loss.backward()
                optimizer.step()
                # 60
                running_loss += heatmap_loss.item()

                # 70
                if i % 10 == 0:
                    # ...log the running loss
                    self.writer.add_scalar('training loss',
                                           running_loss / 10,
                                           epoch * len(train_dataloader) + i)

                    print(running_loss / 10,
                          epoch * len(train_dataloader) + i)
                    running_loss = 0.0

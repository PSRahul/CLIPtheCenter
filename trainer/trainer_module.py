from torch.utils.tensorboard import SummaryWriter
from trainer.trainer_visualisation import plot_heatmaps
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer():

    def __init__(self, cfg, checkpoint_dir):
        self.writer = SummaryWriter(checkpoint_dir)
        self.set_training_parameters()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_interval = cfg["logging"]["log_interval"]

    def set_training_parameters(self):
        pass
        # self.heatmap_loss_function = torchvision.ops.sigmoid_focal_loss(reduction="mean")

    def calculate_heatmap_loss(self, predicted_heatmap, groundtruth_heatmap):
        heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
                                                          reduction="mean")
        return heatmap_loss

    def calculate_offset_loss(self, predicted_offset, groundtruth_offset):
        heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
                                                          reduction="mean")
        return heatmap_loss

    def train(self, model, train_dataloader):
        running_heatmap_loss = 0.0
        model.to(self.device)
        optimizer = optim.Adam(model.parameters())

        for epoch in range(40):

            with tqdm(enumerate(train_dataloader, 0), unit="batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

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
                    running_heatmap_loss += heatmap_loss.item()

                    # 70
                    if i % self.log_interval == 0:
                        # ...log the running loss
                        tepoch.set_postfix(heatmap_loss=heatmap_loss.item())
                        self.writer.add_scalar('training loss',
                                               running_heatmap_loss / self.log_interval,
                                               epoch * len(train_dataloader) + i)
                        self.writer.add_figure('HeatMap Visualisation',
                                               plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                             groundtruth_heatmap=batch[
                                                                 "heatmap"].cpu().detach().numpy()),
                                               global_step=epoch * len(train_dataloader) + i)

                        running_heatmap_loss = 0.0
                        plt.close('all')

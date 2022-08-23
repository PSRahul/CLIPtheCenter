from torch.utils.tensorboard import SummaryWriter
from trainer.trainer_visualisation import plot_heatmaps
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from loss.bbox_loss import calculate_bbox_loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer():

    def __init__(self, cfg, checkpoint_dir):
        self.writer = SummaryWriter(checkpoint_dir)
        self.set_training_parameters()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_interval = cfg["logging"]["log_interval"]
        self.cfg = cfg

    def set_training_parameters(self):
        pass
        # self.heatmap_loss_function = torchvision.ops.sigmoid_focal_loss(reduction="mean")

    def train(self, model, train_dataloader):
        running_heatmap_loss = 0.0
        running_offset_loss = 0.0
        running_bbox_loss = 0.0
        running_loss = 0.0
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(self.cfg["trainer"]["num_epochs"]):

            with tqdm(enumerate(train_dataloader, 0), unit="batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    # 5
                    for key, value in batch.items():
                        batch[key] = batch[key].to(self.device)
                    # 10
                    image = batch["image"].to(self.device)
                    # 20
                    optimizer.zero_grad()
                    # 30
                    output_heatmap, output_offset, output_bbox = model(image)
                    # 40
                    output_heatmap = output_heatmap.squeeze(dim=1).to(self.device)
                    heatmap_loss = calculate_heatmap_loss(output_heatmap, batch["heatmap"])
                    running_heatmap_loss += heatmap_loss.item()

                    offset_loss = calculate_offset_loss(predicted_offset=output_offset,
                                                        groundtruth_offset=batch['offset'],
                                                        flattened_index=batch['flattened_index'],
                                                        num_objects=batch['num_objects'])

                    bbox_loss = calculate_bbox_loss(predicted_bbox=output_bbox,
                                                    groundtruth_bbox=batch['bbox'],
                                                    flattened_index=batch['flattened_index'],
                                                    num_objects=batch['num_objects']) * 0.01

                    loss = heatmap_loss + offset_loss + bbox_loss

                    running_heatmap_loss += heatmap_loss.item()
                    running_offset_loss += offset_loss.item()
                    running_bbox_loss += bbox_loss.item()
                    running_loss += loss.item()

                    # 50

                    # 60
                    loss.backward()
                    optimizer.step()

                    # 70
                    if i % self.log_interval == 0:
                        running_heatmap_loss /= self.log_interval
                        running_offset_loss /= self.log_interval
                        running_bbox_loss /= self.log_interval
                        running_loss /= self.log_interval

                        # ...log the running loss
                        tepoch.set_postfix(loss=running_loss,
                                           heatmap_loss=running_heatmap_loss,
                                           bbox_loss=running_bbox_loss,
                                           offset_loss=running_offset_loss)

                        self.writer.add_scalar('loss',
                                               running_loss,
                                               epoch * len(train_dataloader) + i)
                        self.writer.add_scalar('heatmap loss',
                                               running_heatmap_loss,
                                               epoch * len(train_dataloader) + i)
                        self.writer.add_scalar('bbox loss',
                                               running_bbox_loss,
                                               epoch * len(train_dataloader) + i)
                        self.writer.add_scalar('offset loss',
                                               running_offset_loss,
                                               epoch * len(train_dataloader) + i)

                        self.writer.add_figure('HeatMap Visualisation',
                                               plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                             groundtruth_heatmap=batch[
                                                                 "heatmap"].cpu().detach().numpy()),
                                               global_step=epoch * len(train_dataloader) + i)

                        running_heatmap_loss = 0.0
                        running_offset_loss = 0.0
                        running_loss = 0.0
                        running_bbox_loss = 0.0
                        plt.close('all')

import os.path
import albumentations as A

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.bbox_loss import calculate_bbox_loss
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from trainer.trainer_visualisation import plot_heatmaps
import torch.nn as nn
from evaluation.eval_utils import _gather_output_feature, _transpose_and_gather_output_feature
import numpy as np


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


class EvalMetrics():

    def __init__(self, cfg, checkpoint_dir, model, test_dataloader):
        self.writer = SummaryWriter(checkpoint_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = "cpu"
        self.cfg = cfg
        self.model = model
        self.test_dataloader = test_dataloader
        self.load_checkpoint()
        self.f = open(os.path.join(checkpoint_dir, "evaluation_log.txt"), "w")
        self.checkpoint_dir = checkpoint_dir

    def __del__(self):
        self.f.close()

    def load_checkpoint(self):
        checkpoint = torch.load(self.cfg["evaluation"]["test_checkpoint_path"])
        print("Loaded Model State from ", self.cfg["evaluation"]["test_checkpoint_path"])
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def find_heatmap_peaks(self, output_heatmap, ):
        kernel = self.cfg["evaluation"]["heatmap_pooling_kernel"]
        pad = (kernel - 1) // 2

        output_heatmap_max = nn.functional.max_pool2d(
            output_heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (output_heatmap_max == output_heatmap).float()
        return output_heatmap * keep

    def get_topk_indexes(self, output_heatmap):
        k = self.cfg["evaluation"]["topk_k"]
        batch, num_classes, height, width = output_heatmap.size()
        # Returns the maximum K values and their location per class [N,C,H*W]
        topk_heatmap_value_per_class, topk_heatmap_index_per_class = torch.topk(
            output_heatmap.view(batch, num_classes, -1), k)
        # The coordinates of the largest K points on each N and each C, size is [N,C,K], storing the vertical and horizontal coordinates

        topk_heatmap_index_row_per_class = (topk_heatmap_index_per_class / width).int().float()
        topk_heatmap_index_column_per_class = (topk_heatmap_index_per_class % width).int().float()

        topk_heatmap_value, topk_heatmap_index = torch.topk(
            topk_heatmap_value_per_class.view(batch, -1), k)
        # Across second dimension, divide by the number of class. There are total of N filters. We choose top k scores per class. We so divide N by k to get the class correspondence.
        topk_classes = (topk_heatmap_index / k).int()

        topk_heatmap_index = _gather_output_feature(
            topk_heatmap_index_per_class.view(batch, -1, 1), topk_heatmap_index).view(batch, k)
        topk_heatmap_index_row = _gather_output_feature(topk_heatmap_index_row_per_class.view(batch, -1, 1),
                                                        topk_heatmap_index).view(batch, k)
        topk_heatmap_index_column = _gather_output_feature(topk_heatmap_index_column_per_class.view(batch, -1, 1),
                                                           topk_heatmap_index).view(batch, k)

        return topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column

    def get_topk_indexes_class_agnostic(self, output_heatmap):
        k = self.cfg["evaluation"]["topk_k"]
        output_heatmap = output_heatmap.squeeze(dim=1).to(self.device)

        batch, height, width = output_heatmap.size()
        topk_heatmap_value, topk_heatmap_index = torch.topk(
            output_heatmap.view(batch, -1), k)
        topk_heatmap_index_row = (topk_heatmap_index / width).int().float()
        topk_heatmap_index_column = (topk_heatmap_index % width).int().float()

        topk_classes = torch.zeros((batch, k), device=self.device)
        # Across second dimension, divide by the number of class. There are total of N filters. We choose top k scores per class. We so divide N by k to get the class correspondence.
        return topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column

    def process_output_heatmaps(self, output_heatmap):
        output_heatmap = torch.sigmoid(output_heatmap)
        output_heatmap = self.find_heatmap_peaks(output_heatmap)

        return self.get_topk_indexes_class_agnostic(
            output_heatmap)

    def get_bounding_box_prediction(self, output_heatmap, output_offset, output_bbox, image_id):
        batch, num_classes, height, width = output_heatmap.size()
        k = self.cfg["evaluation"]["topk_k"]
        topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column = self.process_output_heatmaps(
            output_heatmap)
        output_heatmap = topk_heatmap_value
        output_offset = _transpose_and_gather_output_feature(output_offset, topk_heatmap_index)  # .view(batch, k, 2)
        output_bbox = _transpose_and_gather_output_feature(output_bbox, topk_heatmap_index)  # .view(batch, k, 2)

        topk_heatmap_index_column = topk_heatmap_index_column + output_offset[:, :, 0]
        topk_heatmap_index_row = topk_heatmap_index_row + output_offset[:, :, 1]

        # [32,10] -> [32,10,1]
        topk_heatmap_index_row = topk_heatmap_index_row.unsqueeze(dim=2)
        topk_heatmap_index_column = topk_heatmap_index_column.unsqueeze(dim=2)
        output_bbox_width = output_bbox[:, :, 0].unsqueeze(dim=2)
        output_bbox_height = output_bbox[:, :, 1].unsqueeze(dim=2)
        scores = output_heatmap.unsqueeze(dim=2)
        class_label = topk_classes.unsqueeze(dim=2)
        # [32] ->[32, 10, 1]
        image_id = torch.cat(k * [image_id.unsqueeze(dim=1)], dim=1).unsqueeze(dim=2)
        print(" ")

        # [32,10,4]
        bbox = torch.cat([topk_heatmap_index_column - output_bbox_width / 2,
                          topk_heatmap_index_row - output_bbox_height / 2,
                          # topk_heatmap_index_column + output_bbox_width / 2,
                          # topk_heatmap_index_row + output_bbox_height / 2,
                          output_bbox_width,
                          output_bbox_height]
                         , dim=2)
        bbox = bbox * self.cfg["data"]["input_dimension"] / self.cfg["heatmap"]["output_dimension"]
        bbox = bbox.int()
        # [32,10,7]

        detections = torch.cat([image_id, bbox, scores, class_label], dim=2)
        # [32,70]
        detections = detections.view(batch * k, 7)
        return detections

    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        running_val_heatmap_loss = 0.0
        running_val_offset_loss = 0.0
        running_val_bbox_loss = 0.0
        running_val_loss = 0.0
        self.bbox_predictions = []
        with torch.no_grad():
            with tqdm(enumerate(self.test_dataloader, 0), unit=" test batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch 1")

                    for key, value in batch.items():
                        batch[key] = batch[key].to(self.device)
                    image = batch["image"].to(self.device)
                    # 30
                    output_heatmap, output_offset, output_bbox = self.model(image)

                    batch_batch_bbox_predictions = self.get_bounding_box_prediction(output_heatmap.detach(),
                                                                                    output_offset.detach(),
                                                                                    output_bbox.detach(),
                                                                                    batch['image_id'])
                    self.bbox_predictions.append(batch_batch_bbox_predictions)

                    output_heatmap = output_heatmap.squeeze(dim=1).to(self.device)

                    heatmap_loss = calculate_heatmap_loss(output_heatmap, batch["heatmap"])

                    offset_loss = calculate_offset_loss(predicted_offset=output_offset,
                                                        groundtruth_offset=batch['offset'],
                                                        flattened_index=batch['flattened_index'],
                                                        num_objects=batch['num_objects'],
                                                        device=self.device)

                    bbox_loss = calculate_bbox_loss(predicted_bbox=output_bbox,
                                                    groundtruth_bbox=batch['bbox'],
                                                    flattened_index=batch['flattened_index'],
                                                    num_objects=batch['num_objects'], device=self.device)

                    loss = heatmap_loss + offset_loss + bbox_loss

                    running_val_heatmap_loss += heatmap_loss.item()
                    running_val_offset_loss += offset_loss.item()
                    running_val_bbox_loss += bbox_loss.item()
                    running_val_loss += loss.item()

                    tepoch.set_postfix(val_loss=running_val_loss / (i + 1),
                                       val_heatmap_loss=running_val_heatmap_loss / (i + 1),
                                       val_bbox_loss=running_val_bbox_loss / (i + 1),
                                       val_offset_loss=running_val_offset_loss / (i + 1))

                running_val_heatmap_loss /= len(self.test_dataloader)
                running_val_offset_loss /= len(self.test_dataloader)
                running_val_bbox_loss /= len(self.test_dataloader)
                running_val_loss /= len(self.test_dataloader)

                self.running_val_loss = running_val_loss

                file_save_string = 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- offset_loss {:.7f} \n'.format(
                    running_val_loss,
                    running_val_heatmap_loss,
                    running_val_bbox_loss,
                    running_val_offset_loss)
                self.f.write(file_save_string)
        prediction_save_path = self.save_predictions()
        return prediction_save_path

    def save_predictions(self):
        self.bbox_predictions = torch.cat(self.bbox_predictions, dim=0)
        self.bbox_predictions = self.bbox_predictions.cpu().numpy()
        prediction_save_path = os.path.join(self.checkpoint_dir, "bbox_predictions.npy")
        np.save(prediction_save_path, self.bbox_predictions)
        print("Predictions are Saved at", prediction_save_path)
        return prediction_save_path

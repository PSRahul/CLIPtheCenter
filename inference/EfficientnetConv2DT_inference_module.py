import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.bbox_loss import calculate_bbox_loss
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from network.models.EfficientnetConv2DT.utils import get_bounding_box_prediction
from loss.similarity_loss import calculate_embedding_loss


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


class EfficientnetConv2DTModelInference():

    def __init__(self, cfg, checkpoint_dir, model, val_dataloader):
        self.writer = SummaryWriter(checkpoint_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        self.model = model
        self.val_dataloader = val_dataloader
        self.load_checkpoint()
        self.f = open(os.path.join(checkpoint_dir, "evaluation_log.txt"), "w")
        self.checkpoint_dir = checkpoint_dir
        self.detections = []
        self.embeddings = []

    def __del__(self):
        self.f.close()

    def load_checkpoint(self):
        checkpoint = torch.load(self.cfg["evaluation"]["test_checkpoint_path"])
        print("Loaded Model State from ", self.cfg["evaluation"]["test_checkpoint_path"])
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_model_output_and_loss(self, batch, train_set):

        output_heatmap, output_bbox, output_offset, detections, output_clip_encoding, model_encodings = self.model(
            batch, train_set)
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
                                        num_objects=batch['num_objects'],
                                        device=self.device)

        embedding_loss = calculate_embedding_loss(predicted_embedding=model_encodings.to(device=self.device),
                                                  groundtruth_embedding=output_clip_encoding.to(device=self.device))

        return output_heatmap, output_bbox, output_offset, detections, model_encodings, heatmap_loss, bbox_loss, offset_loss, embedding_loss

    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        running_val_heatmap_loss = 0.0
        running_val_offset_loss = 0.0
        running_val_bbox_loss = 0.0
        running_val_loss = 0.0
        with torch.no_grad():
            with tqdm(enumerate(self.val_dataloader, 0), unit=" test batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch 1")

                    for key, value in batch.items():
                        if key != "image_path":
                            batch[key] = batch[key].to(self.device)

                    image = batch["image"].to(self.device)
                    if (self.cfg["debug"]):
                        image_np = image.detach().cpu().numpy()
                        image_np = image_np[0, :]
                        image_np = image_np.transpose(1, 2, 0)
                        plt.imshow(image_np)
                        plt.show()

                    output_heatmap, output_bbox, output_offset, detections, model_encodings, heatmap_loss, bbox_loss, offset_loss, embedding_loss = self.get_model_output_and_loss(
                        batch, train_set=False)
                    self.detections.append(detections)
                    self.embeddings.append(model_encodings)

                    loss = self.cfg["model"]["loss_weight"]["heatmap_head"] * heatmap_loss + \
                           self.cfg["model"]["loss_weight"]["offset_head"] * offset_loss + \
                           self.cfg["model"]["loss_weight"]["bbox_head"] * bbox_loss + \
                           self.cfg["model"]["loss_weight"]["embedding_head"] * embedding_loss

                    running_val_heatmap_loss += heatmap_loss.item()
                    running_val_offset_loss += offset_loss.item()
                    running_val_bbox_loss += bbox_loss.item()
                    running_val_embedding_loss = embedding_loss.item()
                    running_val_loss += loss.item()

                    tepoch.set_postfix(val_loss=running_val_loss / (i + 1),
                                       val_heatmap_loss=running_val_heatmap_loss / (i + 1),
                                       val_bbox_loss=running_val_bbox_loss / (i + 1),
                                       val_offset_loss=running_val_offset_loss / (i + 1),
                                       val_embedding_loss=running_val_embedding_loss / (i + 1))

                running_val_heatmap_loss /= len(self.val_dataloader)
                running_val_offset_loss /= len(self.val_dataloader)
                running_val_bbox_loss /= len(self.val_dataloader)
                running_val_embedding_loss /= len(self.val_dataloader)
                running_val_loss /= len(self.val_dataloader)

                self.running_val_loss = running_val_loss

                file_save_string = 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- offset_loss {:.7f} -|- embedding_loss {:.7f} \n'.format(
                    running_val_loss,
                    running_val_heatmap_loss,
                    running_val_bbox_loss,
                    running_val_offset_loss,
                    running_val_embedding_loss)
                self.f.write(file_save_string)
        prediction_save_path = self.save_predictions()
        return prediction_save_path

    def save_predictions(self):
        self.detections = torch.cat(self.detections,
                                    dim=0)
        self.embeddings = torch.cat(self.embeddings,
                                    dim=0)
        self.detections = torch.hstack((self.detections, self.embeddings))
        self.detections = self.detections.cpu().numpy()
        prediction_save_path = os.path.join(self.checkpoint_dir,
                                            "bbox_predictions.npy")
        np.save(prediction_save_path, self.detections)
        # header = ["image_id", "bbox_x", "bbox_y", "w", "h", "score", "class_label", "embeddings"]
        pd.DataFrame(self.detections).to_csv(os.path.join(self.checkpoint_dir, "bbox_predictions.csv"))

        print("Predictions are Saved at", prediction_save_path)

        return prediction_save_path

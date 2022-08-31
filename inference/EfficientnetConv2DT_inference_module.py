import os.path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.bbox_loss import calculate_bbox_loss
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from network.models.EfficientnetConv2DT.utils import transpose_and_gather_output_array, \
    find_heatmap_peaks, get_topk_indexes_class_agnostic, process_output_heatmaps


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


class EfficientnetConv2DTModelInference():

    def __init__(self, cfg, checkpoint_dir, model, test_dataloader):
        self.writer = SummaryWriter(checkpoint_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def get_bounding_box_prediction(self, output_heatmap, output_offset, output_bbox, image_id):
        batch, num_classes, height, width = output_heatmap.size()
        k = self.cfg["evaluation"]["topk_k"]
        topk_heatmap_value, topk_heatmap_index, topk_classes, topk_heatmap_index_row, topk_heatmap_index_column = process_output_heatmaps(
            self.cfg, output_heatmap)
        output_heatmap = topk_heatmap_value
        output_offset = transpose_and_gather_output_array(output_offset, topk_heatmap_index)  # .view(batch, k, 2)
        output_bbox = transpose_and_gather_output_array(output_bbox, topk_heatmap_index)  # .view(batch, k, 2)

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
        bbox = torch.cat([topk_heatmap_index_row - output_bbox_height / 2,
                          topk_heatmap_index_column - output_bbox_width / 2,

                          # topk_heatmap_index_column + output_bbox_width / 2,
                          # topk_heatmap_index_row + output_bbox_height / 2,
                          output_bbox_width,
                          output_bbox_height]
                         , dim=2)

        bbox_with_no_scaling = bbox.int()
        scale_with_network_input_dimension = True

        # [32,10,7]
        detections_with_no_scaling = torch.cat(
            [image_id, bbox_with_no_scaling, scores, class_label], dim=2)

        # [32,70]
        detections_with_no_scaling = detections_with_no_scaling.view(batch * k, 7)
        detections_with_no_scaling = detections_with_no_scaling[
            detections_with_no_scaling[:, 5] >= float(self.cfg["evaluation"]["score_threshold"])]
        return detections_with_no_scaling

    def eval(self):
        self.model.eval()
        self.model.to(self.device)
        running_val_heatmap_loss = 0.0
        running_val_offset_loss = 0.0
        running_val_bbox_loss = 0.0
        running_val_loss = 0.0
        self.detections_with_no_scaling = []
        self.detections_scale_with_image_input_dimension = []
        self.detections_scale_with_network_input_dimension = []
        with torch.no_grad():
            with tqdm(enumerate(self.test_dataloader, 0), unit=" test batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch 1")

                    for key, value in batch.items():
                        batch[key] = batch[key].to(self.device)
                    image = batch["image"].to(self.device)
                    # 30
                    output_heatmap, output_offset, output_bbox = self.model(image)

                    batch_detections_with_no_scaling = self.get_bounding_box_prediction(
                        output_heatmap.detach(),
                        output_offset.detach(),
                        output_bbox.detach(),
                        batch['image_id'],
                    )
                    self.detections_with_no_scaling.append(batch_detections_with_no_scaling)

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
        self.detections_with_no_scaling = torch.cat(self.detections_with_no_scaling,
                                                    dim=0)
        self.detections_with_no_scaling = self.detections_with_no_scaling.cpu().numpy()
        prediction_save_path = os.path.join(self.checkpoint_dir,
                                            "bbox_predictions_with_no_scaling.npy")
        np.save(prediction_save_path, self.detections_with_no_scaling)
        print("Predictions are Saved at", prediction_save_path)

        return prediction_save_path

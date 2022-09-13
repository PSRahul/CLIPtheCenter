import copy
import os.path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.bbox_loss import calculate_bbox_loss_without_heatmap, calculate_bbox_loss_with_heatmap
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from trainer.trainer_visualisation import plot_heatmaps, save_test_outputs
from loss.similarity_loss import calculate_embedding_loss
import numpy as np
import pandas as pd


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


class SMPTrainer():

    def __init__(self, cfg, checkpoint_dir, model, train_dataloader, val_dataloader, test_dataloader):
        self.writer = SummaryWriter(checkpoint_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_interval = cfg["logging"]["display_log_fraction"]
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 0
        self.loss = 0
        self.set_training_parameters()
        if self.cfg["trainer"]["resume_training"]:
            self.load_checkpoint()
        self.f = open(os.path.join(checkpoint_dir, "training_log.txt"), "w")

    def __del__(self):
        self.f.close()

    def set_training_parameters(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def load_checkpoint(self):
        # TODO: The training losses do not adjust after loading
        checkpoint = torch.load(self.cfg["trainer"]["checkpoint_path"])
        print("Loaded Trainer State from ", self.cfg["trainer"]["checkpoint_path"])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        # self.model = torch.load(self.cfg["trainer"]["checkpoint_path"] + "model")

    def save_model_checkpoint(self):
        model_save_name = 'epoch-{}-loss-{:.7f}.pth'.format(self.epoch, self.running_loss)
        torch.save({
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss

        }, os.path.join(self.checkpoint_dir, model_save_name))

    def check_model_load(self):
        checkpoint = torch.load(self.cfg["trainer"]["checkpoint_path"], map_location="cuda:0")
        print("Loaded Trainer State from ", self.cfg["trainer"]["checkpoint_path"])
        print(self.model.state_dict()['bbox_head.model.2.bias'])
        print(self.optimizer.state_dict()['state'])
        self.load_checkpoint()
        print(self.model.state_dict()['bbox_head.model.2.bias'])

    def get_model_output_and_loss(self, batch, train_set):

        output_heatmap, output_bbox, detections, clip_encoding, model_encodings = self.model(
            batch, train_set)
        output_heatmap = output_heatmap.squeeze(dim=1).to(self.device)
        heatmap_loss = calculate_heatmap_loss(output_heatmap, batch["center_heatmap"])
        bbox_loss = 0
        if (self.cfg["trainer"]["bbox_heatmap_loss"]):
            bbox_loss += calculate_bbox_loss_with_heatmap(predicted_bbox=output_bbox,
                                                          groundtruth_bbox=batch['bbox_heatmap'],
                                                          flattened_index=batch['flattened_index'],
                                                          num_objects=batch['num_objects'],
                                                          device=self.device)
        if (self.cfg["trainer"]["bbox_scatter_loss"]):
            bbox_loss += calculate_bbox_loss_without_heatmap(predicted_bbox=output_bbox,
                                                             groundtruth_bbox=batch['bbox'],
                                                             flattened_index=batch['flattened_index'],
                                                             num_objects=batch['num_objects'],
                                                             device=self.device)

        embedding_loss = calculate_embedding_loss(predicted_embedding=model_encodings.to(device=self.device),
                                                  groundtruth_embedding=clip_encoding.to(device=self.device))

        return output_heatmap, output_bbox, detections, model_encodings, heatmap_loss, bbox_loss, embedding_loss

    def val(self):
        self.model.eval()
        self.model.to(self.device)
        running_val_heatmap_loss = 0.0
        running_val_offset_loss = 0.0
        running_val_bbox_loss = 0.0
        running_val_embedding_loss = 0.0
        running_val_loss = 0.0
        self.optimizer.zero_grad()
        with torch.no_grad():
            with tqdm(enumerate(self.val_dataloader, 0), unit=" val batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch {self.epoch}")

                    for key, value in batch.items():
                        if key != "image_path":
                            batch[key] = batch[key].to(self.device)

                    output_heatmap, output_bbox, detections, model_encodings, heatmap_loss, bbox_loss, embedding_loss = self.get_model_output_and_loss(
                        batch, train_set=False)

                    loss = self.cfg["model"]["loss_weight"]["heatmap_head"] * heatmap_loss + \
                           self.cfg["model"]["loss_weight"]["bbox_head"] * bbox_loss + \
                           self.cfg["model"]["loss_weight"]["embedding_head"] * embedding_loss

                    running_val_heatmap_loss += heatmap_loss.item()
                    running_val_bbox_loss += bbox_loss.item()
                    running_val_embedding_loss = embedding_loss.item()
                    running_val_loss += loss.item()

                    tepoch.set_postfix(val_loss=running_val_loss / (i + 1),
                                       val_heatmap_loss=running_val_heatmap_loss / (i + 1),
                                       val_bbox_loss=running_val_bbox_loss / (i + 1),
                                       val_embedding_loss=running_val_embedding_loss / (i + 1))

                running_val_heatmap_loss /= len(self.val_dataloader)
                running_val_bbox_loss /= len(self.val_dataloader)
                running_val_embedding_loss /= len(self.val_dataloader)
                running_val_loss /= len(self.val_dataloader)

                self.running_val_loss = running_val_loss
                self.writer.add_scalar('val loss',
                                       running_val_loss,
                                       self.epoch * len(self.val_dataloader) + i)
                self.writer.add_scalar('val heatmap loss',
                                       running_val_heatmap_loss,
                                       self.epoch * len(self.val_dataloader) + i)
                self.writer.add_scalar('val bbox loss',
                                       running_val_bbox_loss,
                                       self.epoch * len(self.val_dataloader) + i)
                self.writer.add_scalar('val embedding loss',
                                       running_val_embedding_loss,
                                       self.epoch * len(self.val_dataloader) + i)

                self.writer.add_figure('Validation Center HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                         "center_heatmap"].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.val_dataloader) + i)
                self.writer.add_figure('Validation BBox Width HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_bbox[:, 0, :, :].cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                                             "bbox_heatmap"][:, 0, :,
                                                                         :].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.val_dataloader) + i)
                self.writer.add_figure('Validation BBox Height HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_bbox[:, 1, :, :].cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                                             "bbox_heatmap"][:, 1, :,
                                                                         :].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.val_dataloader) + i)

                file_save_string = 'val epoch {} -|- global_step {} '.format(self.epoch,
                                                                             self.epoch * len(
                                                                                 self.val_dataloader) + i)
                file_save_string += 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- embedding_loss {:.7f} \n'.format(
                    running_val_loss,
                    running_val_heatmap_loss,
                    running_val_bbox_loss,
                    running_val_embedding_loss)
                # 'val loss-{:.7f}.pth'.format(self.epoch, self.running_loss)
                self.f.write(file_save_string)

    def train(self, ):
        running_heatmap_loss = 0.0
        running_offset_loss = 0.0
        running_bbox_loss = 0.0
        running_loss = 0.0
        self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)
        for self.epoch in range(self.epoch, self.cfg["trainer"]["num_epochs"]):
            embedding_loss_weight = 0
            bbox_loss_weight = 0

            if (self.epoch > self.cfg["trainer"]["embedding_loss_start_epoch"]):
                embedding_loss_weight = self.cfg["model"]["loss_weight"]["embedding_head"]
            if (self.epoch > self.cfg["trainer"]["bbox_loss_start_epoch"]):
                bbox_loss_weight = self.cfg["model"]["loss_weight"]["bbox_head"]
            running_heatmap_loss = 0.0
            running_loss = 0.0
            running_bbox_loss = 0.0
            running_embedding_loss = 0.0
            with tqdm(enumerate(self.train_dataloader, 0), unit=" train batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch {self.epoch}")

                    # 5
                    for key, value in batch.items():
                        if key != "image_path":
                            batch[key] = batch[key].to(self.device)
                    # 10
                    self.model.train()
                    self.optimizer.zero_grad()
                    output_heatmap, output_bbox, detections, model_encodings, heatmap_loss, bbox_loss, embedding_loss = self.get_model_output_and_loss(
                        batch, train_set=True)
                    heatmap_loss = self.cfg["model"]["loss_weight"]["heatmap_head"] * heatmap_loss
                    bbox_loss = bbox_loss_weight * bbox_loss
                    embedding_loss = embedding_loss_weight * embedding_loss
                    self.loss = heatmap_loss + bbox_loss + embedding_loss

                    running_heatmap_loss += heatmap_loss.item()
                    running_bbox_loss += bbox_loss.item()
                    running_embedding_loss = embedding_loss.item()
                    running_loss += self.loss.item()

                    # 50

                    # 60
                    self.loss.backward()
                    self.optimizer.step()

                    # 70
                    if (i % int(self.log_interval * (len(self.train_dataloader)))) == 0:
                        running_heatmap_loss /= (i + 1)
                        running_bbox_loss /= (i + 1)
                        running_embedding_loss /= (i + 1)
                        running_loss /= (i + 1)

                        # ...log the running loss
                        tepoch.set_postfix(loss=running_loss,
                                           heatmap_loss=running_heatmap_loss,
                                           bbox_loss=running_bbox_loss,
                                           offset_loss=running_offset_loss,
                                           embedding_loss=running_embedding_loss)
                        self.running_loss = running_loss
                        self.writer.add_scalar('loss',
                                               running_loss,
                                               self.epoch * len(self.train_dataloader) + i)
                        self.writer.add_scalar('heatmap loss',
                                               running_heatmap_loss,
                                               self.epoch * len(self.train_dataloader) + i)
                        self.writer.add_scalar('bbox loss',
                                               running_bbox_loss,
                                               self.epoch * len(self.train_dataloader) + i)
                        self.writer.add_scalar('embedding loss',
                                               running_embedding_loss,
                                               self.epoch * len(self.train_dataloader) + i)

                        self.writer.add_figure('Center HeatMap Visualisation',
                                               plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                             groundtruth_heatmap=batch[
                                                                 "center_heatmap"].cpu().detach().numpy(),
                                                             sigmoid=True),
                                               global_step=self.epoch * len(self.train_dataloader) + i)
                        self.writer.add_figure('BBox HeatMap Width Visualisation',
                                               plot_heatmaps(
                                                   predicted_heatmap=output_bbox[:, 0, :, :].cpu().detach().numpy(),
                                                   groundtruth_heatmap=batch[
                                                                           "bbox_heatmap"][:, 0, :,
                                                                       :].cpu().detach().numpy()),
                                               global_step=self.epoch * len(self.train_dataloader) + i)

                        self.writer.add_figure('BBox HeatMap Height Visualisation',
                                               plot_heatmaps(
                                                   predicted_heatmap=output_bbox[:, 1, :, :].cpu().detach().numpy(),
                                                   groundtruth_heatmap=batch[
                                                                           "bbox_heatmap"][:, 1, :,
                                                                       :].cpu().detach().numpy()),
                                               global_step=self.epoch * len(self.train_dataloader) + i)

                        file_save_string = 'train epoch {} -|- global_step {} '.format(self.epoch, self.epoch * len(
                            self.train_dataloader) + i)
                        file_save_string += 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- embedding_loss {:.7f}\n'.format(
                            running_loss,
                            running_heatmap_loss,
                            running_bbox_loss,
                            running_embedding_loss)

                        self.f.write(file_save_string)
                        plt.close('all')

            # self.save_model_checkpoint()
            if (self.epoch % self.cfg["trainer"]["val_save_interval"] == 0) or (
                    self.epoch == self.cfg["trainer"]["num_epochs"] - 1):
                self.save_model_checkpoint()
                self.val()

    def test(self):
        detections_list = []
        embeddings_list = []
        groundtruth_list = []
        self.model.eval()
        self.model.to(self.device)
        running_test_heatmap_loss = 0.0
        running_test_offset_loss = 0.0
        running_test_bbox_loss = 0.0
        running_test_embedding_loss = 0.0
        running_test_loss = 0.0
        self.optimizer.zero_grad()
        with torch.no_grad():
            with tqdm(enumerate(self.test_dataloader, 0), unit=" test batch") as tepoch:
                for i, batch in tepoch:
                    tepoch.set_description(f"Epoch {self.epoch}")

                    for key, value in batch.items():
                        if key != "image_path":
                            batch[key] = batch[key].to(self.device)

                    output_heatmap, output_bbox, detections, model_encodings, heatmap_loss, bbox_loss, embedding_loss = self.get_model_output_and_loss(
                        batch, train_set=False)
                    groundtruth_list.append(batch['heatmap_sized_bounding_box_list'].cpu())

                    if (self.cfg["test_debug"]):
                        for i in range(output_heatmap.shape[0]):
                            heatmap_sized_bounding_box_np = batch['heatmap_sized_bounding_box_list'][
                                i].detach().cpu().numpy()
                            print("\nGround Truths", i,
                                  heatmap_sized_bounding_box_np[1] + (heatmap_sized_bounding_box_np[3]) / 2,
                                  heatmap_sized_bounding_box_np[2] + (heatmap_sized_bounding_box_np[4]) / 2)

                            groundtruth_center_np = batch["center_heatmap"][i].detach().cpu().numpy()
                            plt.imshow(groundtruth_center_np)  # cmap="Greys")
                            plt.title(str(i) + "_GT Center")
                            plt.show()

                            heatmap_np = output_heatmap[i].detach().cpu().numpy()
                            plt.imshow(heatmap_np)  # cmap="Greys")
                            plt.title(str(i) + "_Predicted Heatmap")
                            plt.show()
                            center = np.argmax(heatmap_np)
                            print("Predictions", i, center,
                                  center % 320,
                                  center / 320)

                            groundtruth_bbox_np = batch["bbox_heatmap"][i].detach().cpu().numpy()
                            groundtruth_bbox_np_w = groundtruth_bbox_np[0, :, :]
                            plt.imshow(groundtruth_bbox_np_w)  # cmap="Greys")
                            plt.title(str(i) + "_GT Width")
                            plt.show()

                            bbox_np_w = output_bbox[i, 0].detach().cpu().numpy()
                            plt.imshow(bbox_np_w)  # cmap="Greys")
                            plt.title(str(i) + "_Predicted Width")
                            plt.show()

                            groundtruth_bbox_np_h = groundtruth_bbox_np[1, :, :]
                            plt.imshow(groundtruth_bbox_np_h)  # cmap="Greys")
                            plt.title(str(i) + "_GT Height")
                            plt.show()

                            bbox_np_h = output_bbox[i, 1].detach().cpu().numpy()
                            # bbox_heatmap = batch["bbox_heatmap"][i, 0].detach().cpu().numpy()
                            plt.imshow(bbox_np_h)
                            plt.title(str(i) + "_Predicted Height")  # cmap="Greys")
                            plt.show()

                            # batch['heatmap_sized_bounding_box_list'][i, 1] += (batch[
                            #    'heatmap_sized_bounding_box_list'][i, 3]) / 2
                            # batch['heatmap_sized_bounding_box_list'][i, 2] += (batch[
                            #    'heatmap_sized_bounding_box_list'][i, 4]) / 2

                            # print("Breakpoint")

                    if (self.cfg["test_parameters"]["save_test_outputs"]):
                        save_test_outputs(self.checkpoint_dir, batch, output_heatmap.cpu().detach().numpy(),
                                          output_bbox.cpu().detach().numpy())
                    detections_list.append(detections)
                    embeddings_list.append(model_encodings)

                    loss = self.cfg["model"]["loss_weight"]["heatmap_head"] * heatmap_loss + \
                           self.cfg["model"]["loss_weight"]["bbox_head"] * bbox_loss + \
                           self.cfg["model"]["loss_weight"]["embedding_head"] * embedding_loss

                    running_test_heatmap_loss += heatmap_loss.item()
                    running_test_bbox_loss += bbox_loss.item()
                    running_test_embedding_loss = embedding_loss.item()
                    running_test_loss += loss.item()

                    tepoch.set_postfix(test_loss=running_test_loss / (i + 1),
                                       test_heatmap_loss=running_test_heatmap_loss / (i + 1),
                                       test_bbox_loss=running_test_bbox_loss / (i + 1),
                                       test_embedding_loss=running_test_embedding_loss / (i + 1))

                running_test_heatmap_loss /= len(self.test_dataloader)
                running_test_bbox_loss /= len(self.test_dataloader)
                running_test_embedding_loss /= len(self.test_dataloader)
                running_test_loss /= len(self.test_dataloader)

                self.running_test_loss = running_test_loss
                self.writer.add_scalar('test loss',
                                       running_test_loss,
                                       self.epoch * len(self.test_dataloader) + i)
                self.writer.add_scalar('test heatmap loss',
                                       running_test_heatmap_loss,
                                       self.epoch * len(self.test_dataloader) + i)
                self.writer.add_scalar('test bbox loss',
                                       running_test_bbox_loss,
                                       self.epoch * len(self.test_dataloader) + i)
                self.writer.add_scalar('test embedding loss',
                                       running_test_embedding_loss,
                                       self.epoch * len(self.test_dataloader) + i)

                self.writer.add_figure('Test Center HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                         "center_heatmap"].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.test_dataloader) + i)
                self.writer.add_figure('Test BBox Width HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_bbox[:, 0, :, :].cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                                             "bbox_heatmap"][:, 0, :,
                                                                         :].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.test_dataloader) + i)
                self.writer.add_figure('Test BBox Height HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_bbox[:, 1, :, :].cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                                             "bbox_heatmap"][:, 1, :,
                                                                         :].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.test_dataloader) + i)

                file_save_string = 'test epoch {} -|- global_step {} '.format(self.epoch,
                                                                              self.epoch * len(
                                                                                  self.test_dataloader) + i)
                file_save_string += 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- embedding_loss {:.7f} \n'.format(
                    running_test_loss,
                    running_test_heatmap_loss,
                    running_test_bbox_loss,
                    running_test_embedding_loss)
                # 'test loss-{:.7f}.pth'.format(self.epoch, self.running_loss)
                self.f.write(file_save_string)
        prediction_save_path = self.save_predictions(detections_list, embeddings_list, groundtruth_list)
        return prediction_save_path

    def save_predictions(self, detections_list, embeddings_list, groundtruth_list):
        detections = torch.cat(detections_list,
                               dim=0)
        embeddings = torch.cat(embeddings_list,
                               dim=0)
        detections = torch.hstack((detections, embeddings))
        detections = detections.cpu().numpy()
        prediction_save_path = os.path.join(self.checkpoint_dir,
                                            "bbox_predictions.npy")
        np.save(prediction_save_path, detections)
        # header = ["image_id", "bbox_x", "bbox_y", "w", "h", "score", "class_label", "embeddings"]
        pd.DataFrame(detections).to_csv(os.path.join(self.checkpoint_dir, "bbox_predictions.csv"))

        print("Predictions are Saved at", prediction_save_path)
        groundtruth = torch.cat(groundtruth_list,
                                dim=0)
        # groundtruth[:, 1] = groundtruth[:, 1] + groundtruth[:, 3] / 2
        # groundtruth[:, 2] = groundtruth[:, 2] + groundtruth[:, 4] / 2
        pd.DataFrame(groundtruth).to_csv(os.path.join(self.checkpoint_dir, "gt_predictions.csv"))
        return prediction_save_path

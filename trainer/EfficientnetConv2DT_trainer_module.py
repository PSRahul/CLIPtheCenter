import os.path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from loss.bbox_loss import calculate_bbox_loss
from loss.heatmap_loss import calculate_heatmap_loss
from loss.offset_loss import calculate_offset_loss
from trainer.trainer_visualisation import plot_heatmaps
from loss.similarity_loss import calculate_embedding_loss


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


class EfficientnetConv2DTTrainer():

    def __init__(self, cfg, checkpoint_dir, model, train_dataloader, val_dataloader):
        self.writer = SummaryWriter(checkpoint_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_interval = cfg["logging"]["display_log_fraction"]
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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
        self.optimizer = optim.Adadelta(self.model.parameters())

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

        return output_heatmap, output_bbox, output_offset, detections, heatmap_loss, bbox_loss, offset_loss, embedding_loss

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

                    output_heatmap, output_bbox, output_offset, detections, heatmap_loss, bbox_loss, offset_loss, embedding_loss = self.get_model_output_and_loss(
                        batch, train_set=False)

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
                self.writer.add_scalar('val loss',
                                       running_val_loss,
                                       self.epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar('val heatmap loss',
                                       running_val_heatmap_loss,
                                       self.epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar('val bbox loss',
                                       running_val_bbox_loss,
                                       self.epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar('val offset loss',
                                       running_val_offset_loss,
                                       self.epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar('val embedding loss',
                                       running_val_embedding_loss,
                                       self.epoch * len(self.train_dataloader) + i)

                self.writer.add_figure('Validation HeatMap Visualisation',
                                       plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                     groundtruth_heatmap=batch[
                                                         "heatmap"].cpu().detach().numpy()),
                                       global_step=self.epoch * len(self.train_dataloader) + i)

                file_save_string = 'val epoch {} -|- global_step {} '.format(self.epoch,
                                                                             self.epoch * len(
                                                                                 self.train_dataloader) + i)
                file_save_string += 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- offset_loss {:.7f} -|- embedding_loss {:.7f} \n'.format(
                    running_val_loss,
                    running_val_heatmap_loss,
                    running_val_bbox_loss,
                    running_val_offset_loss,
                    running_val_embedding_loss)
                # 'val loss-{:.7f}.pth'.format(self.epoch, self.running_loss)
                self.f.write(file_save_string)

    def train(self, ):
        running_heatmap_loss = 0.0
        running_offset_loss = 0.0
        running_bbox_loss = 0.0
        running_loss = 0.0
        self.model.to(self.device)
        for self.epoch in range(self.epoch, self.cfg["trainer"]["num_epochs"]):
            running_heatmap_loss = 0.0
            running_offset_loss = 0.0
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
                    output_heatmap, output_bbox, output_offset, detections, heatmap_loss, bbox_loss, offset_loss, embedding_loss = self.get_model_output_and_loss(
                        batch, train_set=True)

                    self.loss = self.cfg["model"]["loss_weight"]["heatmap_head"] * heatmap_loss + \
                                self.cfg["model"]["loss_weight"]["offset_head"] * offset_loss + \
                                self.cfg["model"]["loss_weight"]["bbox_head"] * bbox_loss + \
                                self.cfg["model"]["loss_weight"]["embedding_head"] * embedding_loss

                    running_heatmap_loss += heatmap_loss.item()
                    running_offset_loss += offset_loss.item()
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
                        running_offset_loss /= (i + 1)
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
                        self.writer.add_scalar('offset loss',
                                               running_offset_loss,
                                               self.epoch * len(self.train_dataloader) + i)
                        self.writer.add_scalar('embedding loss',
                                               running_embedding_loss,
                                               self.epoch * len(self.train_dataloader) + i)

                        self.writer.add_figure('HeatMap Visualisation',
                                               plot_heatmaps(predicted_heatmap=output_heatmap.cpu().detach().numpy(),
                                                             groundtruth_heatmap=batch[
                                                                 "heatmap"].cpu().detach().numpy()),
                                               global_step=self.epoch * len(self.train_dataloader) + i)

                        file_save_string = 'train epoch {} -|- global_step {} '.format(self.epoch, self.epoch * len(
                            self.train_dataloader) + i)
                        file_save_string += 'loss {:.7f} -|- heatmap_loss {:.7f} -|- bbox_loss {:.7f} -|- offset_loss {:.7f} -|- embedding_loss {:.7f}\n'.format(
                            running_loss,
                            running_heatmap_loss,
                            running_bbox_loss,
                            running_offset_loss,
                            running_embedding_loss)

                        self.f.write(file_save_string)
                        plt.close('all')

            # self.save_model_checkpoint()
            if (self.epoch % self.cfg["trainer"]["val_save_interval"] == 0) or (
                    self.epoch == self.cfg["trainer"]["num_epochs"] - 1):
                self.save_model_checkpoint()
                self.val()

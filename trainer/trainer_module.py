from torch.utils.tensorboard import SummaryWriter


class Trainer():

    def __init__(self, cfg, checkpoint_dir):
        writer = SummaryWriter(checkpoint_dir)

    def train(self, model, train_dataloader):
        running_loss = 0.0
        for epoch in range(1):  # loop over the dataset multiple times

            for i, data in enumerate(train_dataloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:  # every 1000 mini-batches...

                    # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / 1000,
                                      epoch * len(trainloader) + i)

                    running_loss = 0.0

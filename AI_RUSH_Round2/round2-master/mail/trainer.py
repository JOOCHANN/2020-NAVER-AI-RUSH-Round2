import nsml
import numpy as np
import torch

from utils import calculate_f1


class Trainer:

    def __init__(self, model, criterion, metric, optimizer, train_data, valid_data, args):
        super(Trainer, self).__init__()
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.args = args

        self.device = args.device
        self.epochs = args.epochs

        self.model = model
        self.global_step = 0
        self.start_epoch = 1

    def run(self):
        best_f1 = 0.5

        print("run 0 epoch...")
        self.model.eval()
        with torch.no_grad():
            self.run_epoch(self.valid_data, 0, "valid")

        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"run {epoch} epoch...")
            self.model.train()
            train_epoch_loss, train_epoch_metric = self.run_epoch(self.train_data, epoch, "train")

            self.model.eval()
            with torch.no_grad():
                valid_epoch_loss, valid_epoch_metric = self.run_epoch(self.valid_data, epoch, "valid")
                nsml.report(
                    summary=True,
                    step=epoch,
                    train_loss=train_epoch_loss,
                    train_f1=train_epoch_metric[2],
                    valid_loss=valid_epoch_loss,
                    valid_precision=valid_epoch_metric[0],
                    valid_recall=valid_epoch_metric[1],
                    valid_f1=valid_epoch_metric[2],
                )

            if valid_epoch_metric[2] > best_f1:
                best_f1 = valid_epoch_metric[2]
                nsml.save(checkpoint=epoch)
                nsml.save(checkpoint="best_model")
            print(f"current best f1: {best_f1}")

    def run_epoch(self, data_loader, epoch, mode):
        epoch_loss = 0.0
        epoch_count = 0
        epoch_roc = np.zeros((2, 2), dtype=np.float32)
        epoch_metric = [0.0, 0.0, 0.0]

        for label, title, content, batch_size in data_loader:
            title = title.to(self.device)
            content = content.to(self.device)
            y = label.to(self.device).float()

            y_pred = self.model(title, content).view(-1)
            batch_loss = self.criterion(y_pred, y)

            if mode is "train":
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            epoch_loss = \
                (epoch_loss * epoch_count + batch_loss.item() * batch_size) / (epoch_count + batch_size)
            label_prediction = list(zip(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))
            batch_roc = self.metric(label_prediction)
            epoch_roc += batch_roc
            epoch_metric = calculate_f1(epoch_roc)
            epoch_count += batch_size

        epoch_log = (
            f"Mode: {mode} / Epoch: {epoch:2d} / Loss: {epoch_loss:.4f} / "
            f"Precision: {epoch_metric[0]:.4f} / Recall: {epoch_metric[1]:.4f} / F1: {epoch_metric[2]:.4f}"
        )
        print(epoch_log)

        return epoch_loss, epoch_metric

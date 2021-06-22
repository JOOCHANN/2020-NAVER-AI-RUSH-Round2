import argparse
import os

import nsml
import numpy as np
import torch
from nsml import DATASET_PATH
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from dataset import MailDataset, MaxPadBatch
from model import Model
from trainer import Trainer
from utils import read_mail_data, roc_metric


def bind_model(target_model, target_optimizer, target_args):

    def save(dir_path):
        state = {
            "model": target_model.state_dict(),
            "optimizer": target_optimizer.state_dict(),
            "args": target_args,
        }
        torch.save(state, os.path.join(dir_path, "model.pth"))

    def load(dir_path):
        state = torch.load(os.path.join(dir_path, "model.pth"))
        target_model.load_state_dict(state["model"])
        print("model loaded")
        if "optimizer" in state and target_optimizer:
            target_optimizer.load_state_dict(state["optimizer"])
            print("optimizer loaded")

    def infer(test_data_path):
        """NSML inference function.
        
        Args:
            test_data_path: string, Automatically set by NSML.
        
        Returns:
            predictions: list of (float, int). 
                         [(0.9876, 1), (0.1234, 0), ...]
        """
        
        test_data = []
        with open(test_data_path, "r", encoding="utf-8") as test_data_file:
            read_test_data = test_data_file.readlines()
            for line in read_test_data:
                parse_line = line.strip().split("\t")
                title = [int(i) for i in parse_line[0].split(",")[:target_args.max_title]]
                content = [int(i) for i in parse_line[1].split(",")[:target_args.max_content]]
                test_data.append((-1, title, content))  # assign -1 for label since test label is not given

        test_data_loader = DataLoader(
            MailDataset(test_data),
            batch_size=target_args.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=MaxPadBatch(args.max_title, args.max_content),
        )

        probabilities = []
        target_model.eval()
        for _, title, content, _ in test_data_loader:
            title = title.to(target_args.device)
            content = content.to(target_args.device)
            probability = target_model(title, content).view(-1).detach().cpu().numpy().tolist()
            probabilities.extend(probability)

        predictions = [(p, 1) if p > 0.5 else (p, 0) for p in probabilities]
        return predictions

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument("--mode", help="declare train or eval mode",
                        default="train", type=str)
    parser.add_argument("--split", help="train-valid split ratio",
                        default=0.2, type=float)
    parser.add_argument("--seed", help="random seed, default = 42",
                        default=42, type=int)
    parser.add_argument("--epochs", help="number of training epoch, default = 30",
                        default=30, type=int)
    parser.add_argument("--vocab_type", help="use token or char vocab, default = token",
                        default="token", type=str)
    parser.add_argument("--vocab_size", help="number of vocab, default = 20001",
                        default=20001, type=int)
    parser.add_argument("--batch_size", help="batch size, default = 256",
                        default=256, type=int)
    parser.add_argument("--lr", help="learning rate, default = 0.002",
                        default=0.003, type=float)
    parser.add_argument("--max_title", help="max length of title text",
                        default=100, type=int)
    parser.add_argument("--max_content", help="max length of content text",
                        default=300, type=int)

    # model parameters
    parser.add_argument("--embedding_dim", help="dimension of embedding vector",
                        default=256, type=int)
    parser.add_argument("--hidden_dim", help="hidden dimension size",
                        default=128, type=int)
    parser.add_argument("--layers", help="number of layers, default = 3",
                        default=3, type=int)
    parser.add_argument("--dropout", help="dropout rate, default = 0.3",
                        default=0.3, type=float)

    # reserved arguments for AutoML PBT
    parser.add_argument("--pause", default=0, type=int)
    parser.add_argument("--iteration", default=0, type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        args.device = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        args.device = "cpu"

    criterion = nn.BCELoss()
    metric = roc_metric

    model = Model(args)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bind_model(model, optimizer, args)
    if args.pause:
        nsml.paused(scope=locals())

    if args.mode.lower() == "train":

        print("Loading data...")
        data_path = os.path.join(DATASET_PATH, "train", "train_data")
        label_path = os.path.join(DATASET_PATH, "train", "train_label")
        read_data = read_mail_data(data_path, label_path, args.max_title, args.max_content)
        train_data, valid_data = train_test_split(read_data, test_size=args.split, random_state=args.seed)
        train_data_loader = DataLoader(
            MailDataset(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=MaxPadBatch(args.max_title, args.max_content),
        )
        valid_data_loader = DataLoader(
            MailDataset(valid_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=MaxPadBatch(args.max_title, args.max_content),
        )

        trainer = Trainer(
            model,
            criterion,
            metric,
            optimizer,
            train_data_loader,
            valid_data_loader,
            args
        )
        trainer.run()

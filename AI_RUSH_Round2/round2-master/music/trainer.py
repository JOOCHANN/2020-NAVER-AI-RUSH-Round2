import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score
from skimage.transform import resize

import nsml

class MusicDataset(Dataset):
    def __init__(self, config, dataset_root, train=True):
        self.dataset_name = config['dataset_name'] # q3
        self.input_length = config['input_length'] # 1200
        self.train = train

        self.mel_dir = os.path.join(dataset_root, 'train_data', 'mel_spectrogram')
        self.label_file = os.path.join(dataset_root, 'train_label')
        with open(self.label_file) as f:
            self.train_labels = json.load(f)

        self.n_valid = len(self.train_labels['track_index']) // 10 # validation -> 0.1%
        self.n_train = len(self.train_labels['track_index']) - self.n_valid # train -> 0.9%

        label_types = {'q1': 'station_name',
                       'q2': 'mood_tag',
                       'q3': 'genre'}

        self.label_type = label_types[self.dataset_name] # 'genre'
        self.label_map = self.create_label_map() # {'label0':0, 'label1':1, 'label2':2, 'label3':3}
        self.n_classes = len(self.label_map) # 클래스 수 4

    def create_label_map(self):
        label_map = {}
        if self.dataset_name in ['q1', 'q3']: # q1, q3 모두 클래스가 4개니까 같이 묶어서 맵을 생성
            for idx, label in self.train_labels[self.label_type].items():
                if label not in label_map:
                    label_map[label] = len(label_map)
        else: # q2는 클래스가 4개니까 따로 맵을 생성
            for idx, label_list in self.train_labels[self.label_type].items():
                for label in label_list:
                    if label not in label_map:
                        label_map[label] = len(label_map)

        return label_map

    def __getitem__(self, idx):
        data_idx = str(idx)
        if not self.train:
            data_idx = str(idx + self.n_train)

        track_name = self.train_labels['track_index'][data_idx] # track_id
        mel_path = os.path.join(self.mel_dir, '{}.npy'.format(track_name)) # mel_spectrogram
        mel = np.load(mel_path)[0]
        mel = np.log10(1 + 10 * mel) # mel spectrogram이기 때문에 log를 취해줌

        mel = mel[:, :self.input_length] # 1 lenght당 32ms, 1200 = 32.4초

        label = self.train_labels[self.label_type][data_idx] # label, string
        if self.dataset_name in ['q1', 'q3']:
            labels = self.label_map[label] # string -> number(0, 1, 2, 3)
        else:
            label_idx = [self.label_map[l] for l in label]
            labels = np.zeros(self.n_classes, dtype=np.float32)
            labels[label_idx] = 1

        # mel = resize(mel, (128, 1200)) # resize 2000 -> 1200
        
        # print(mel.shape) # 128, 1200
        return mel, labels # numpy, number

    def __len__(self):
        return self.n_train if self.train else self.n_valid

class TestMusicDataset(Dataset):
    def __init__(self, config, dataset_root):
        self.dataset_name = config['dataset_name']  # i.e. q1

        self.meta_dir = os.path.join(dataset_root, 'test_data', 'meta')
        self.mel_dir = os.path.join(dataset_root, 'test_data', 'mel_spectrogram')

        meta_path = os.path.join(self.meta_dir, '{}_test.json'.format(self.dataset_name))
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.input_length = config['input_length']
        self.n_classes = 100 if self.dataset_name == 'q2' else 4

    def __getitem__(self, idx):
        data_idx = str(idx)

        track_name = self.meta['track_index'][data_idx]
        mel_path = os.path.join(self.mel_dir, '{}.npy'.format(track_name))
        mel = np.load(mel_path)[0]
        mel = np.log10(1 + 10 * mel)

        mel = mel[:, :self.input_length]

        return mel, data_idx

    def __len__(self):
        return len(self.meta['track_index'])


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class BasicNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_classes = config['n_classes']

        self._extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((4, 8))
        )

        self._classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=256, out_features=n_classes))
        self.apply(init_weights)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self._extractor(x)
        x = x.view(x.size(0), -1)
        score = self._classifier(x)
        return score


def select_optimizer(model, config):
    args = config['optimizer']
    lr = args['lr']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    nesterov = args['nesterov']

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


def select_scheduler(optimizer, config):
    args = config['schedule']
    factor = args['factor']
    patience = args['patience']

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    return scheduler


def accuracy_(pred, target):
    _, predicted_max_idx = pred.max(dim=1)
    n_correct = predicted_max_idx.eq(target).sum().item()
    return n_correct / len(target)


def f1_score_(pred, target, threshold=0.5):
    pred = np.array(pred.cpu() > threshold, dtype=float)
    return f1_score(target.cpu(), pred, average='micro')


class Trainer:
    def __init__(self, config, mode):
        """
        mode: train(run), test(submit)
        """
        self.device = config['device'] # cuda
        self.dataset_name = config['dataset_name'] # q3
        self.config = config

        # data loading
        if mode == 'train':
            batch_size = config['batch_size'] # 16
            self.train_dataset = MusicDataset(config, config['dataset_root'], train=True)
            self.valid_dataset = MusicDataset(config, config['dataset_root'], train=False)
            self.label_map = self.train_dataset.label_map

            print("number of training set :", len(self.train_dataset))
            print("number of validation set :", len(self.valid_dataset))

            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)


        if self.dataset_name in ['q1', 'q3']:
            config['n_classes'] = 4 
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.act = torch.nn.functional.log_softmax
            self.act_kwargs = {'dim': 1}
            self.measure_name = 'accuracy'
            self.measure_fn = accuracy_
        else:
            config['n_classes'] = 100
            self.criterion = nn.BCELoss(reduction='none')
            self.act = torch.sigmoid
            self.act_kwargs = {}
            self.measure_name = 'f1_score'
            self.measure_fn = f1_score_

        # model 선언
        self.model = BasicNet(config).to(self.device)
        self.optimizer = select_optimizer(self.model, config)
        self.scheduler = select_scheduler(self.optimizer, config)

        self.iter = config['iter']
        self.val_iter = config['val_iter']
        self.save_iter = config['save_iter']

    def run_batch(self, batch, train):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        if train:
            self.optimizer.zero_grad()
            y_ = self.model(x)
            y_ = self.act(y_, **self.act_kwargs)
            loss = torch.mean(self.criterion(y_, y))
            loss.backward()
            self.optimizer.step()
        else:
            y_ = self.model(x)
            y_ = self.act(y_.detach(), **self.act_kwargs)
            loss = torch.mean(self.criterion(y_, y))

        loss = loss.item()
        measure = self.measure_fn(y_, y)
        batch_size = y.size(0)

        return loss * batch_size, measure * batch_size, batch_size

    def run_train(self, epoch=None):
        if epoch is not None:
            print(f'Training on epoch {epoch}')

        data_loader = self.train_loader
        self.model.train()

        total_loss = 0
        total_measure = 0
        total_cnt = 0

        n_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            batch_status = self.run_batch(batch, train=True)
            loss, measure, batch_size = batch_status

            total_loss += loss
            total_measure += measure
            total_cnt += batch_size

            if batch_idx % (n_batch // 10) == 0:
                print(batch_idx, '/', n_batch)

        status = {'train__loss': total_loss / total_cnt,
                  'train__{}'.format(self.measure_name): total_measure / total_cnt}
        return status

    def run_valid(self, epoch=None):
        if epoch is not None:
            print(f'Validation on epoch {epoch}')

        data_loader = self.valid_loader
        self.model.eval()

        total_loss = 0
        total_measure = 0
        total_cnt = 0
        for batch_idx, batch in enumerate(data_loader):
            batch_status = self.run_batch(batch, train=False)
            loss, measure, batch_size = batch_status

            total_loss += loss
            total_measure += measure
            total_cnt += batch_size

        status = {'valid__loss': total_loss / total_cnt,
                  'valid__{}'.format(self.measure_name): total_measure / total_cnt}
        return status

    def run_evaluation(self, test_dir):
        """
        Predicted Labels should be a list of labels / label_lists
        """
        dataset = TestMusicDataset(self.config, test_dir)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.model.eval()

        idx2label = {v: k for k, v in self.label_map.items()}

        predicted_prob_list = []
        predicted_labels = []
        for x, data_idx in loader:
            x = x.to(self.device)
            y_ = self.model(x)

            if self.dataset_name in ['q1', 'q3']:
                predicted_probs, predicted_max_idx = y_.max(dim=1)
                predicted_labels += list(predicted_max_idx)
            else:
                threshold = 0.5
                over_threshold = np.array(y_.cpu() > threshold, dtype=float)
                label_idx_list = [np.where(labels == 1)[0].tolist() for labels in over_threshold]
                predicted_labels += label_idx_list

        if self.dataset_name in ['q1', 'q3']:
            predicted_labels = [idx2label[label_idx.item()] for label_idx in predicted_labels]
        else:
            predicted_labels = [[idx2label[label_idx] for label_idx in label_idx_list] for label_idx_list in
                                predicted_labels]

        return predicted_labels

    # 여기가 제일 상위 함수
    def run(self):
        for epoch in range(self.iter): # 501
            epoch_status = self.run_train(epoch)

            if epoch % self.val_iter == 0:
                self.report(epoch, epoch_status)

                valid_status = self.run_valid(epoch)
                self.report(epoch, valid_status)

                self.scheduler.step(valid_status['valid__loss'])

            if epoch % self.save_iter == 0:
                self.save(epoch)

    def save(self, epoch):
        nsml.save(epoch)
        print(f'Saved model at epoch {epoch}')

    def report(self, epoch, status):
        print(status)
        nsml.report(summary=True, scope=locals(), step=epoch, **status)



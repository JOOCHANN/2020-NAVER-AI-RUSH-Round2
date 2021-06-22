import torch
import torch.nn.functional as ftn
from torch.utils.data import Dataset


class MailDataset(Dataset):
    def __init__(self, mail_data):
        self.mail_data = mail_data

    def __getitem__(self, index):
        label = torch.tensor(self.mail_data[index][0])
        title = torch.tensor(self.mail_data[index][1]).long()
        content = torch.tensor(self.mail_data[index][2]).long()
        return label, title, content

    def __len__(self):
        return len(self.mail_data)


class MaxPadBatch:

    def __init__(self, max_title, max_content):
        super(MaxPadBatch, self).__init__()
        self.max_title = max_title
        self.max_content = max_content

    def __call__(self, batch):
        batch_label = []
        batch_title = []
        batch_content = []

        for item in batch:
            label, title, content = item
            batch_label.append(label)
            batch_title.append(title)
            batch_content.append(content)

        pad_batch_title = [ftn.pad(x, [0, self.max_title - x.shape[0]], value=0).detach() for x in batch_title]
        pad_batch_content = [ftn.pad(x, [0, self.max_title - x.shape[0]], value=0).detach() for x in batch_content]
        return torch.stack(batch_label), torch.stack(pad_batch_title), torch.stack(pad_batch_content), len(batch)

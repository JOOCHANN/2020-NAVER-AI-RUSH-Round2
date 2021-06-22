import os

import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

def get_category(data_frame, enc_x, idx):
    result = []
    category_1 = data_frame.iloc[idx]['category_1']
    category_2 = data_frame.iloc[idx]['category_2']
    category_3 = data_frame.iloc[idx]['category_3']
    category_4 = data_frame.iloc[idx]['category_4']
    result.append([category_1, category_2, category_3, category_4])
    result = np.array(result)
    result = result.reshape(-1, 4)

    category = enc_x.transform(result).toarray()
    category = torch.from_numpy(category)

    return category.float()


class TagImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, root_dir: str, transform=None, enc_x=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.enc_x = enc_x

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        tag_name = self.data_frame.iloc[idx]['answer']
        sample['label'] = tag_name
        sample['image_name'] = img_name

        onehot_category = get_category(self.data_frame, self.enc_x, idx)
        sample['category'] = onehot_category
        return sample


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, enc=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.endswith('.')]
        self.category = self.data_list.pop(self.data_list.index('test_input'))
        self.categoty_path = os.path.join(self.root_dir, self.category)
        self.data_frame = pd.read_csv(self.categoty_path)
        self.enc = enc
        print("test data num :", len(self.data_list), len(self.data_frame))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        sample['image_name'] = img_name

        onehot_category = get_category(self.data_frame, self.enc, idx)
        sample['category'] = onehot_category

        return sample
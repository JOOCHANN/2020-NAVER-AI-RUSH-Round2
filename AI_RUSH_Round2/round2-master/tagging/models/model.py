import torch
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torch import nn

class Resnet50_FC2(torch.nn.Module):
    def __init__(self, n_class=9, pretrained=True):
        super(Resnet50_FC2, self).__init__()
        self.basemodel = models.resnet50(pretrained=pretrained)
        self.linear1 = torch.nn.Linear(1000, 512)
        self.linear2 = torch.nn.Linear(512, n_class)

    def forward(self, x):
        x = self.basemodel(x)
        x = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(x), dim=-1)
        pred = torch.argmax(out, dim=-1)
        return out, pred


class EfficientNet_FC(torch.nn.Module):
    def __init__(self, model_name='efficientnet-b7', n_class=5, pretrained=True):
        super(EfficientNet_FC, self).__init__()
        self.basemodel = EfficientNet.from_pretrained(model_name)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)

        self.linear_x = torch.nn.Linear(2560, 512)
        self.linear_category = torch.nn.Linear(2145, 512)

        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, n_class)
        print('Model input size', self.basemodel.get_image_size(model_name))

        self.enc = None

    def get_encoder(self, enc):
        self.enc = enc

    def forward(self, x, xcategory):
        x = self.basemodel.extract_features(x) # 1, 2560, 7, 7
        x = self._avg_pooling(x) # 1 2560 1 1
        x = x.flatten(start_dim=1) # 1 2560
        xcategory = xcategory.flatten(start_dim=1) # 1 2145

        x = self.linear_x(x) # 1 512
        x = F.relu(x) # 1 512
        xcategory = self.linear_category(xcategory) # 1 512
        xcategory = F.relu(xcategory) # 1 512
        
        x = torch.cat((x, xcategory), dim=1) # 1 1024
        x = self.linear1(x) # 1 512
        x = F.relu(x) # 512

        out = F.softmax(self.linear2(x), dim=-1) # 1 5
        pred = torch.argmax(out, dim=-1) # 1
        return out, pred
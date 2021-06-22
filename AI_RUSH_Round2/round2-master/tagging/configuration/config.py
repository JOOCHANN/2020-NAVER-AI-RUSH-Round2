import logging.config

from torchvision import transforms

logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Tagging Classification')

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

inputsize = 256
cropsize = 224

train_transform = transforms.Compose([
    transforms.Resize((inputsize, inputsize)),
    # transforms.RandomAffine(degrees=0, scale=(0.80, 1.2)),
    transforms.CenterCrop(cropsize),
    #transforms.ColorJitter(brightness=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.Resize((inputsize, inputsize)),
    transforms.CenterCrop(cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
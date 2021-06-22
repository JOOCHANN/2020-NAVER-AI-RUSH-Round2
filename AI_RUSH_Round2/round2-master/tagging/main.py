import argparse
import os

import nsml
import torch
from nsml import DATASET_PATH
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

import nsml_utils as nu
import torch.backends.cudnn as cudnn
from configuration.config import logger, train_transform, test_transform
from data_loader import TagImageDataset
from utils import select_optimizer, select_model, evaluate, train, get_onehotencoder, read_csv_data


def train_process(args, model, train_loader, test_loader, optimizer, criterion, device, enc):
    best_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        print("epoch :", epoch)
        train_loss, train_acc = train(model=model, train_loader=train_loader, optimizer=optimizer,
                                      criterion=criterion, device=device, epoch=epoch, total_epochs=args.num_epoch, enc=enc)
        # model.eval()
        # test_loss, test_acc, test_f1 = evaluate(model=model, test_loader=test_loader, device=device, criterion=criterion)

        # report_dict = dict()
        # report_dict["train__loss"] = train_loss
        # report_dict["train__acc"] = train_acc
        # report_dict["test__loss"] = test_loss
        # report_dict["test__acc"] = test_acc
        # report_dict["test__f1"] = test_f1
        # report_dict["train__lr"] = optimizer.param_groups[0]['lr']
        # nsml.report(False, step=epoch, **report_dict)
        # print("Trying to save model")
        # if best_acc < test_acc:
        #     print("new best")
        #     checkpoint = 'best'
        #     logger.info(f'[{epoch}] Find the best model! Change the best model.')
        #     nsml.save(checkpoint)
        #     best_acc = test_acc
        # if (epoch + 1) % 5 == 0:
        print("train_loss", train_loss)
        print("train_acc", train_acc)
        checkpoint = f'ckpt_{epoch}'
        nsml.save(checkpoint)

        if (epoch == args.annealing_period) or (epoch == args.annealing_period + 2) :
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1
            logger.info('Learning rate annealed to : {lr:.6f} @epoch{epoch}'.format(
                epoch=epoch, lr=optimizer.param_groups[0]['lr']))


def load_weight(model, weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file).state_dict(), strict=True)
        print('load weight from {}.'.format(weight_file))
    else:
        print('weight file {} is not exist.'.format(weight_file))
        print('=> random initialized model will be used.')

def main():
    # Argument Settings
    parser = argparse.ArgumentParser(description='Image Tagging Classification from Naver Shopping Reviews')
    parser.add_argument('--sess_name', default='', type=str, help='Session name that is loaded')
    parser.add_argument('--checkpoint', default='best', type=str, help='Checkpoint')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='The number of workers')
    parser.add_argument('--num_epoch', default=8, type=int, help='The number of epochs')
    parser.add_argument('--model_name', default='efficientnet-b7', type=str, help='[efficientnet-b7]')
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning_anneal', default=1.1, type=float)
    parser.add_argument('--annealing_period', default=4, type=int)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--mode', default='train', help='Mode')
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)
    parser.add_argument('--weight_file', default='best.pth', type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    seed = '1'

    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)

    # Model
    logger.info('Build Model')

    print("Creating model: {}".format(args.model_name))
    model = select_model(args.model_name, pretrain=args.pretrain, n_class=5)

    total_param = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model size: {total_param} tensors')
    # load_weight(model, args.weight_file)
    model = model.to(device)

    nu.bind_model(model)

    if args.pause:
        nsml.paused(scope=locals())

    if args.num_epoch == 0:
        nsml.save('best')
        return

    # Set the dataset
    logger.info('Set the dataset')
    df = pd.read_csv(f'{DATASET_PATH}/train/train_label')
    enc_x = read_csv_data(df)

    trainset = TagImageDataset(data_frame=df, root_dir=f'{DATASET_PATH}/train/train_data',
                               transform=train_transform, enc_x=enc_x)
    testset = TagImageDataset(data_frame=df, root_dir=f'{DATASET_PATH}/train/train_data',
                              transform=test_transform, enc_x=enc_x)

    print("trainset num :", len(trainset))
    print("testset num :", len(testset))

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = select_optimizer(model.parameters(), args.optimizer, args.lr, args.weight_decay)
    print("optimizer :", args.optimizer)

    criterion = criterion.to(device)

    if args.mode == 'train':
        logger.info('Start to train!')
        train_process(args=args, model=model, train_loader=train_loader, test_loader=test_loader,
                      optimizer=optimizer, criterion=criterion, device=device, enc=enc_x)

    elif args.mode == 'test':
        nsml.load(args.checkpoint, session=args.sess_name)
        logger.info('[NSML] Model loaded from {}'.format(args.checkpoint))

        model.eval()
        logger.info('Start to test!')
        test_loss, test_acc, test_f1 = evaluate(model=model, test_loader=test_loader, device=device,
                                                criterion=criterion)
        logger.info(test_loss, test_acc, test_f1)


if __name__ == '__main__':
    main()

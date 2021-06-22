# nsml: pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7

from pprint import PrettyPrinter
import os
import yaml
import argparse

from trainer import Trainer

import nsml

class Initializer(object):
    def __init__(self, args):
        with open(args.config_file, 'r', encoding='utf8') as f:
            config = yaml.load(f)
            config['dataset_root'] = os.path.join(nsml.DATASET_PATH, 'train')

        pp = PrettyPrinter(indent=4) # indent : 들여쓰기
        pp.pprint(config) # config 출력

        print('Constructing Trainer')
        self.trainer = Trainer(config, args.mode)
        self.config = config

        self.bind_nsml()
        if args.pause:
            nsml.paused(scope=locals())

    def bind_nsml(self):
        import torch
        import json
        trainer = self.trainer
        config = self.config

        def save(model_dir):
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'label_map': json.dumps(trainer.label_map),
                'config': json.dumps(config),
            }
            torch.save(checkpoint, os.path.join(model_dir, 'model'))

        def load(model_dir, **kwargs):
            fpath = os.path.join(model_dir, 'model')
            checkpoint = torch.load(fpath)
            trainer.model.load_state_dict(checkpoint['model'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            trainer.label_map = json.loads(checkpoint['label_map'])
            self.config = json.loads(checkpoint['config'])
            print('Model loaded')

        def infer(test_dir, **kwargs):
            return trainer.run_evaluation(test_dir)

        nsml.bind(save=save, load=load, infer=infer)

    def run(self):
        print('Starting Training')
        self.trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Needed for nsml submit
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default=0, help='checkpoint loaded')
    parser.add_argument('--mode', type=str, default='train')

    # User argument
    parser.add_argument('--config_file', type=str, default='./q3/config.yaml')
    args = parser.parse_args()

    initializer = Initializer(args)
    if args.mode == 'train':
        initializer.run()

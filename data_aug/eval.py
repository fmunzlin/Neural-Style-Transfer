from os import path as osp
import os
from argparse import Namespace
import numpy as np
import tqdm
from PIL import Image
import yaml
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms
from data_aug.Augmentor import Denormalize
import cv2

from imaginaire.utils.model_average import ModelAverage
parser = argparse.ArgumentParser()

#general settings
parser.add_argument('--d_id', type=int, default=-1, help='cuda device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--exp', type=str, default='test', metavar='x', help='name of experiment')

class Evaluation(object):
    def __init__(self, args):
        self.args = args
        self.exp_path = osp.join("train", self.args.exp)

        self.config = self.load_config()
        self.exp_args = self.load_args()

        self.eval_content = None
        self.eval_style = None
        self.augment = None

        self.gen_model = None

    def load_config(self):
        try:
            with open(osp.join("train", self.args.exp, "config.yaml"), 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise ("failed loading config file")

    def load_args(self):
        try:
            with open(os.path.join(self.exp_path, "args.txt")) as f:
                args = f.read()
            args = Namespace(**yaml.load(args, Loader=yaml.FullLoader))
            setattr(args, "d_id", args.d_id)
            return args
        except:
            raise("failed loading arguments")

    def forward_style(self, image):
        if self.exp_args.model_avg:
            return self.gen_model.averaged_model.generator.style_encoder(image)
        else:
            return self.gen_model.generator.style_encoder(image)

    def forward_content(self, image):
        if self.exp_args.model_avg:
            return self.gen_model.averaged_model.generator.content_encoder(image)
        else:
            return self.gen_model.generator.content_encoder(image)

    def forward_decode(self, content_feat, style_feat):
        if self.exp_args.model_avg:
            return self.gen_model.averaged_model.generator.decode(content_feat, style_feat)
        else:
            return self.gen_model.generator.decode(content_feat, style_feat)


    def evaluate(self):
        pass

if __name__ == "__main__":
    args = parser.parse_args()
    Evaluation(args).evaluate()

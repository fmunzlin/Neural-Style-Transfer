import os
from os import path as osp
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset

from data_aug.augment import Augmentation

# get a random item from some list
def rnd_from_list(items): return items[torch.randint(high=len(items), low=0, size=(1,))[0]]

class Eval_content(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.data_path = osp.join("data", folder)
        self.image_list = self.get_images()
        self.augment = Augmentation(self.args).evaluate

    # load all images from data dir
    def get_images(self):
        image_list = []
        for content_image in os.listdir(self.data_path):
            image_list.append(osp.join(self.data_path, content_image))
        return image_list

    # needed for the dataloader
    # important for the dataloader to "know" how many iterations one epoch has
    def __len__(self):
        return len(self.image_list)

    # needed for the dataloader
    # load an image from the list and augment the image
    # dataloader appends them to one tensor
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        if self.args.d_id != -1: image = image.cuda()
        return image

class Eval_style(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.data_path = osp.join("data", folder)
        self.image_list = self.get_images()
        self.augment = Augmentation(self.args).evaluate

    # load all images from data dir each artist
    def get_images(self):
        image_list = []
        for artist in os.listdir(self.data_path):
            temp = []
            for style_image in os.listdir(osp.join(self.data_path, artist)):
                temp.append(osp.join(self.data_path, artist, style_image))
            image_list.append(temp)
        return image_list

    # needed for the dataloader
    # important for the dataloader to "know" how many iterations one epoch has
    # however we always use batchsize 1 in this case and therefore do not suffer the problem we e.g. have during
    # validation
    def __len__(self):
        return len(self.image_list)

    # needed for the dataloader
    # load 3 image from the list and augment the image
    # 1: some style image, 2: a style image from the same data class, 3: a style image from a different data class
    # dataloader appends them to one tensor
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_1, image_2 = self.image_list[idx]
        _, image_3 = self.image_list[(idx + 1) % self.__len__()]

        image_1 = self.augment(Image.open(image_1))
        image_2 = self.augment(Image.open(image_2))
        image_3 = self.augment(Image.open(image_3))

        if self.args.d_id != -1:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            image_3 = image_3.cuda()
        return image_1, image_2, image_3

class Validation_content(Dataset):
    def __init__(self, args, image_list,):
        self.args = args
        self.image_list = image_list
        self.augment = Augmentation(self.args).validate

    # since style and content do not have the same length we have a problem
    # if content has len 10 and style has len 12 and we set batch_size 3, then we have
    # 4 iterations possible. in the first 3, both show batchsize 3, but the last iteration has
    # 1 content and 3 style. this creates a conflict while style transfer, further while creating the AdaIN parameters
    def __len__(self):
        return int(len(self.image_list) / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_1 = self.get_image(idx)

        if self.args.d_id != -1:
            image_1 = image_1.cuda()

        return image_1

class Validation_style(Dataset):
    def __init__(self, args, image_list, label_list):
        self.args = args
        self.image_list = image_list
        self.label_list = label_list
        self.labels = list(set(self.label_list))
        self.other_idx_list = self.get_other_list()
        self.augment = Augmentation(self.args).validate

    # since style and content do not have the same length we have a problem
    # if content has len 10 and style has len 12 and we set batch_size 3, then we have
    # 4 iterations possible. in the first 3, both show batchsize 3, but the last iteration has
    # 1 content and 3 style. this creates a conflict while style transfer, further while creating the AdaIN parameters
    def __len__(self):
        return int(len(self.image_list) / self.args.batch_size) * self.args.batch_size

    # load the image and label, augment it and return it to then use it during validation
    def get_image(self, idx):
        label = self.label_list[idx]
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image, torch.tensor(label)

    def get_other_list(self):
        if self.args.dif_style:
            other_labels = [~np.where(np.isin(self.label_list, label))[0] for label in self.labels]
            other_labels = dict(zip(self.labels, other_labels))
        else:
            other_labels = [np.where(np.isin(self.label_list, label))[0] for label in self.labels]
            other_labels = dict(zip(self.labels, other_labels))
        return [rnd_from_list(other_labels[label]) for label in self.label_list]

    # method used by the dataloader
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_1, label_1 = self.get_image(idx)
        image_2, label_2 = self.get_image(self.other_idx_list[idx])

        if self.args.d_id != -1:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            label_1 = label_1.cuda()
            label_2 = label_2.cuda()

        return image_1, label_1, image_2, label_2

class Style_dataset(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.folder = osp.join("data", folder)
        self.augment = Augmentation(self.args).train

        self.image_list, self.label_list = self.get_images()
        self.image_list, self.label_list, self.val_image_list, self.val_label_list = self.get_val_list()

        self.labels = list(set(self.label_list))
        self.same_labels = [np.where(np.isin(self.label_list, label))[0] for label in self.labels]
        self.other_labels = [np.where(~np.isin(self.label_list, label))[0] for label in self.labels]
        self.same_labels = dict(zip(self.labels, self.same_labels))
        self.other_labels = dict(zip(self.labels, self.other_labels))
        self.num_images = len(self.image_list)

    def get_val_list(self):
        same_labels = [np.where(np.isin(self.label_list, label))[0] for label in self.labels]
        same_labels = dict(zip(self.labels, same_labels))

        val_image_list = []
        val_label_list = []
        indices = [rnd_from_list(same_labels[i % self.num_labels]) for i in range(self.args.val_size)]
        for idx in indices:
            val_image_list.append(self.image_list[idx])
            val_label_list.append(self.label_list[idx])

        image_list = [image for idx, image in enumerate(self.image_list) if idx not in indices]
        label_list = [label for idx, label in enumerate(self.label_list) if idx not in indices]

        return image_list, label_list, val_image_list, val_label_list

    def get_images(self):
        image_list = []
        label_list = []
        if self.args.artist_id == -1: artists = os.listdir(self.folder)
        else: artists = [os.listdir(self.folder)[self.args.artist_id]]
        self.num_labels = len(artists)
        self.labels = range(self.num_labels)

        artist2id = dict(zip(artists, self.labels))
        for artist in artists:
            artist_folder = osp.join(self.folder, artist)
            for image in os.listdir(artist_folder):
                image_list.append(osp.join(artist_folder, image))
                label_list.append(artist2id[artist])
        return image_list, label_list

    def get_batch(self):
        labels_1 = []
        labels_2 = []
        images_1 = torch.Tensor()
        images_2 = torch.Tensor()

        for idx in torch.randint(high=self.num_images, low=0, size=(self.args.batch_size,)):
            image_1, label_1, int_label_1 = self.get_image(idx)
            image_1 = image_1.unsqueeze(0)

            if self.args.dif_style: rnd_idx = rnd_from_list(self.other_labels[int_label_1])
            else: rnd_idx = rnd_from_list(self.same_labels[int_label_1])
            image_2, label_2, _ = self.get_image(rnd_idx)
            image_2 = image_2.unsqueeze(0)

            labels_1.append(label_1)
            labels_2.append(label_2)
            images_1 = torch.cat((images_1, image_1), dim=0)
            images_2 = torch.cat((images_2, image_2), dim=0)

        labels_1 = torch.tensor(labels_1)
        labels_2 = torch.tensor(labels_2)

        if self.args.d_id != -1:
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()
            labels_1 = labels_1.cuda()
            labels_2 = labels_2.cuda()

        return images_1, labels_1, images_2, labels_2

    def __len__(self):
        return int(self.num_images / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        label = self.label_list[idx]
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image, torch.tensor(label), label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_1, label_1, int_label_1 = self.get_image(idx)

        rnd_idx = rnd_from_list(self.same_labels[int_label_1])
        image_2, label_2, _ = self.get_image(rnd_idx)

        if self.args.d_id != -1:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            label_1 = label_1.cuda()
            label_2 = label_2.cuda()

        return image_1, label_1, image_2, label_2

    def get_validation_dataset(self):
        return Validation_style(self.args, self.val_image_list, self.val_label_list)


class Content_dataset(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.folder = osp.join("data", folder, "data_large")
        self.augment = Augmentation(self.args).train

        self.image_list = self.get_images()
        self.image_list = shuffle(self.image_list)
        self.val_image_list = self.image_list[:self.args.val_size]
        self.image_list = self.image_list[self.args.val_size:]

    def get_sub_folders(self, content_dirs, dir):
        for file in os.listdir(dir):
            file = osp.join(dir, file)
            if os.path.isfile(file) and dir not in content_dirs:
                content_dirs.append(dir)
                break
            else:
                content_dirs = self.get_sub_folders(content_dirs, file)
        return content_dirs

    def get_images(self):
        image_list = []
        for folder in self.get_sub_folders([], self.folder):
            for image in os.listdir(folder):
                image_list.append(osp.join(folder, image))
        return image_list

    def __len__(self):
        return int(len(self.image_list) / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.get_image(idx)
        if self.args.d_id != -1: image = image.cuda()
        return image

    def get_batch(self):
        images = torch.Tensor()
        for idx in torch.randint(high=len(self.image_list), low=0, size=(self.args.batch_size,)):
            images = torch.cat((images, self.get_image(idx).unsqueeze(0)), dim=0)

        if self.args.d_id != -1: images = images.cuda()
        return images

    def get_validation_dataset(self):
        return Validation_content(self.args, self.val_image_list)
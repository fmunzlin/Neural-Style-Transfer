import numpy as np

import torch
from torchvision import transforms

# if we want to save a snapshot we also want to invert the normalization because it is good for training,
# but it does not show the "true" image
class Denormalize(transforms.Normalize):
    def __init__(self, computed_norm):
        mean, std = get_mean_std(computed_norm)
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean / std

        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def get_mean_std(computed_norm):
    if computed_norm:
        mean = np.array([0.4571, 0.4406, 0.4073]) * 0.5 + np.array([0.4728, 0.4515, 0.4123]) * 0.5
        std = np.array([0.2745, 0.2717, 0.2897]) * 0.5 + np.array([0.2667, 0.2556, 0.2702]) * 0.5
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    return mean, std

class Augmentation():
    def __init__(self, args):
        self.args = args
        self.mean, self.std = get_mean_std(False)
        self.train_transform = self.get_transforms("train")
        self.validate_transform = self.get_transforms("validate")
        self.evaluate_transform = self.get_transforms("evaluate")

    # get transformations, depended on the usage
    # transformation differentiate if we do training, validation and evaluation
    # this is liked to randomness we want to have during training and absence of randomness during validation and
    # evaluation. also during evaluation we dont want to crop the sample at all because we want to use the hole image
    def get_transforms(self, mode):
        transform = []
        if mode == "train":
            transform.append(transforms.RandomCrop(size=(self.args.crop_size)))
        if mode == "validate":
            transform.append(transforms.CenterCrop(size=(self.args.crop_size)))
        transform.append(transforms.ToTensor())
        if mode == "train":
            transform.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
            transform.append(transforms.RandomVerticalFlip(p=0.5))
            transform.append(transforms.RandomHorizontalFlip(p=0.5))
        transform.append(transforms.Normalize(*get_mean_std(False)))
        transform = transforms.Compose(transform)
        return transform

    # we resize an image to have to smallest side of an image to fit a fix number of pixels
    def resize_to_edge(self, resize_to, image):
        width, height = image.size
        if height <= width:
            factor = resize_to / height
        else:
            factor = resize_to / width
        return self.resize_to_factor(image, factor)

    # resize an image to a given factor
    # afterwards we multiply the factor to both width and height
    def resize_to_factor(self, img, factor):
        if isinstance(factor, list):
            return img.resize((int(img.width * factor[0]), int(img.height * factor[0])))
        else:
            return img.resize((int(img.width * factor), int(img.height * factor)))

    # to give the model better generalization properties we randomly scale the image and create
    # more diverse samples
    def scale_image(self, img, range):
        scale_x = 1. + np.random.uniform(low=0, high=range)
        scale_y = 1. + np.random.uniform(low=0, high=range)
        return self.resize_to_factor(img, [scale_x, scale_y])

    # perform the augmentations needed while training
    def train(self, image):
        image = self.resize_to_edge(self.args.resize, image)
        image = self.scale_image(image, self.args.scale_aug)
        if image.mode != 'RGB': image = image.convert('RGB')
        image = self.train_transform(image)
        return image

    # perform the augmentations needed while validation
    # here we dont scale the image because it involves randomness and exclude random colour transformations
    def validate(self, image):
        image = self.resize_to_edge(self.args.resize, image)
        if image.mode != 'RGB': image = image.convert('RGB')
        image = self.validate_transform(image)
        return image

    # perform the augmentations needed while evaluation
    # we only perform the most minimal augmentations to forward images to the model
    def evaluate(self, image):
        image = self.resize_to_edge(512, image)
        if image.mode != 'RGB': image = image.convert('RGB')
        image = self.evaluate_transform(image)
        return image

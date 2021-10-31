import torch

import importlib
from types import SimpleNamespace

import numpy as np

import sly_globals as g
import models

import sly_functions as source_functions


def get_default_args():
    args = importlib.import_module('default_config').args
    args = SimpleNamespace(**args)
    return args


def initialize_network():
    args = get_default_args()
    g.model = models.Net(args).to(g.device)

    g.logger.info('Model successfully initialized!')


def load_weights(weights_path):
    current_checkpoint = torch.load(weights_path, map_location=g.device)
    model_weights = current_checkpoint['state_dict']
    model_weights = source_functions.preprocess_weights(model_weights)

    g.model.load_state_dict(model_weights, strict=False)
    g.model.eval()


def to_torch_tensor(img):
    return torch.from_numpy(img.transpose((2, 0, 1)))


def normalize_img(img, normalization):
    eps = 1e-6

    if normalization == 'channel':
        pixel_mean = img.mean((0, 1))
        pixel_std = img.std((0, 1)) + eps
        img = (img - pixel_mean[None, None, :]) / pixel_std[None, None, :]
        img = img.clip(-20, 20)

    elif normalization == 'channel_mean':
        pixel_mean = img.mean((0, 1))
        img = (img - pixel_mean[None, None, :])
        img = img.clip(-20, 20)

    elif normalization == 'image':
        img = (img - img.mean()) / img.std() + eps
        img = img.clip(-20, 20)

    elif normalization == 'simple':
        img = img / 255

    elif normalization == 'inception':
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = img.astype(np.float32)
        img = img / 255.
        img -= mean
        img *= np.reciprocal(std, dtype=np.float32)

    elif normalization == 'imagenet':

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= np.reciprocal(std, dtype=np.float32)

    else:
        pass

    return img


def augment(aug, img):
    img_aug = aug(image=img)['image']
    return img_aug.astype(np.float32)


def numpy_to_torch_tensors(nps_batch):
    args = get_default_args()
    augmentation = args.test_aug

    nps_batch = [augment(augmentation, curr_image) for curr_image in nps_batch]

    nps_batch = [normalize_img(curr_image, args.normalization) for curr_image in nps_batch]
    tensors_batch = [to_torch_tensor(curr_image) for curr_image in nps_batch]

    return torch.stack(tensors_batch)


def calculate_embeddings_for_nps_batch(nps_batch):
    tensors_batch = numpy_to_torch_tensors(nps_batch).to(g.device)
    input_tensors = {'input': tensors_batch}

    with torch.no_grad():
        output = g.model(input_tensors, get_embeddings=True)

    return output['embeddings'].detach().cpu().numpy()



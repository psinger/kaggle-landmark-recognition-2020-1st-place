import glob
import os

from conf import *
from utils import *
from data import *
from models import *
from loss import *

import numpy as np
import pandas as pd
import math
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, TestTubeLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

from torch.optim import Adam
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import warnings

import torch.distributed as dist
import shutil
import pickle

import functions
import functools

from tqdm import tqdm
import subprocess


def fix_row(row):
    if len(str(row).split()) > 1:
        row = int(str(row).split()[0])
    return row


def setup():
    if args.seed == -1:
        args.seed = np.random.randint(0, 1000000)
    print("Seed", args.seed)
    set_seed(args.seed)

    train = pd.read_csv(args.data_path + args.train_csv_fn)
    train["img_folder"] = args.img_path_train
    print("train shape", train.shape)

    valid = pd.read_csv(args.data_path_2019 + args.valid_csv_fn)
    valid["img_folder"] = args.img_path_val
    valid['landmarks'] = valid['landmarks'].apply(lambda x: fix_row(x))
    valid['landmark_id'] = valid['landmarks'].fillna(-1)
    valid['landmarks'].fillna('', inplace=True)
    valid['landmark_id'] = valid['landmark_id'].astype(int)

    if args.data_path_2 is not None:
        train_2 = pd.read_csv(args.data_path_2 + args.train_2_csv_fn)
        train_2["img_folder"] = args.img_path_train_2
        if "gldv1" in args.data_path_2:
            print("gldv1")
            train_2["landmark_id"] = train_2["landmark_id"] + train["landmark_id"].max()
        train = pd.concat([train, train_2], axis=0).reset_index(drop=True)
        print("train shape", train.shape)

    train_filter = train[train.landmark_id.isin(valid.landmark_id)].reset_index()

    print("trn filter len", len(train_filter))

    landmark_ids = np.sort(train.landmark_id.unique())

    args.n_classes = train.landmark_id.nunique()

    landmark_id2class = {lid: i for i, lid in enumerate(landmark_ids)}
    landmark_id2class_val = landmark_id2class.copy()
    landmark_id2class_val[-1] = args.n_classes

    print("ids", train.landmark_id.max(), train.landmark_id.nunique())

    train['target'] = train['landmark_id'].apply(lambda x: landmark_id2class[x])

    if args.class_weights == "log":
        val_counts = train.target.value_counts().sort_index().values
        class_weights = 1 / np.log1p(val_counts)
        class_weights = (class_weights / class_weights.sum()) * args.n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None

    valid['target'] = valid['landmark_id'].apply(lambda x: landmark_id2class_val.get(x, -1))
    valid = valid[valid.target > -1].reset_index(drop=True)

    allowed_classes = np.sort(valid[valid.target != args.n_classes].target.unique())

    train_filter['target'] = train_filter['landmark_id'].apply(lambda x: landmark_id2class_val.get(x, -1))

    # train = train.head(args.batch_size*2)

    return train, valid, train_filter, landmark_ids, landmark_id2class, landmark_id2class_val, class_weights, allowed_classes


class EmbeddingVisualizer:
    def __init__(self, model_with_embeddings):
        pass


def get_embeddings(model, dataloader):
    outputs = {
        'idx': [],
        'embeddings': [],
        # 'targets': []
    }


    with torch.no_grad():
        for (input_tensors, target_ids) in tqdm(dataloader, desc='calculating embeddings'):
            input_tensors['input'] = input_tensors['input'].to(device)

            output = model(input_tensors)

            outputs['idx'].append(input_tensors['idx'])
            outputs['embeddings'].append(output['embeddings'].detach().cpu())

    for key in outputs.keys():
        outputs[key] = torch.cat(outputs[key])

    return outputs


def pack_to_zip_and_copy(source_folder):
    import shutil



    # shutil.make_archive(output_filename, 'zip', source_folder)


def process_visualization(outputs_train, outputs_test, current_epoch):
    output_path = os.path.join(os.path.dirname(args.model_path), 'visualizations',
                               args.experiment_name, str(current_epoch - 1))

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # print('scaling...')
    # f = StandardScaler()
    # f.fit(np.concatenate([outputs_test["embeddings"]], axis=0))
    # outputs_train["embeddings"] = f.transform(outputs_train["embeddings"])
    # outputs_test["embeddings"] = f.transform(outputs_test["embeddings"], )

    os.makedirs(output_path, exist_ok=True)

    outputs_test["embeddings"] = outputs_test["embeddings"].detach().cpu().numpy().astype(np.float32)
    outputs_train["embeddings"] = outputs_train["embeddings"].detach().cpu().numpy().astype(np.float32)


    pred_dist, pred_index_of_labels = get_topk_cossim(outputs_test["embeddings"],
                                                      outputs_train["embeddings"], k=8,
                                                      device=device)


    pred_dist = [list(curr_row) for curr_row in list(pred_dist.data.cpu().numpy())][:30]
    pred_index_of_labels = [list(curr_row) for curr_row in list(pred_index_of_labels.data.cpu().numpy())][:30]

    query_images = []
    for index, val_image_index_in_ds in enumerate(outputs_test['idx'][:30]):

        # pred_index_of_labels, pred_dist = functions.calculate_top_n_cosine_sim(outputs_train['embeddings'],
        #                                                                        [outputs_test['embeddings'][index]],
        #                                                                        top_n=8)
        query_img = test_ds.get_original_item(int(val_image_index_in_ds))['input'].permute(1, 2, 0).numpy()
        query_images.append(query_img)

    functions.save_tensors_by_indexes(query_images, tr_ds, pred_index_of_labels, pred_dist, output_path)

    # pack_to_zip_and_copy()


def preprocess_weights(model_weights):
    return OrderedDict([(k.replace('model.', ''), v) for k, v in model_weights.items()])


if __name__ == '__main__':
    train, valid, train_filter, landmark_ids, landmark_id2class, landmark_id2class_val, class_weights, allowed_classes = setup()

    tr_ds = GLRDataset(train, normalization=args.normalization, aug=args.test_aug, suffix='.png')
    test_ds = GLRDataset(valid, normalization=args.normalization, aug=args.test_aug)

    tr_dl = DataLoader(dataset=tr_ds, batch_size=args.test_batch_size, sampler=SequentialSampler(tr_ds),
                       collate_fn=collate_fn,
                       num_workers=args.num_workers, drop_last=True, pin_memory=True)

    test_dl = DataLoader(dataset=test_ds, batch_size=args.test_batch_size, sampler=SequentialSampler(test_ds),
                         collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    #
    # tr_filter_ds = GLRDataset(train_filter, normalization=args.normalization, aug=args.val_aug)
    # tr_filter_dl = DataLoader(dataset=tr_filter_ds, batch_size=args.batch_size, sampler=SequentialSampler(tr_filter_ds),
    #                           collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)

    checkpoints_dir = os.path.join(args.model_path, args.experiment_name, 'ckpt')

    checkpoints_paths = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))

    device = 'cuda:0'

    model = Net(args).to(device)

    for checkpoint_path in checkpoints_paths:
        current_checkpoint = torch.load(checkpoint_path, map_location=device)
        model_epoch = current_checkpoint['epoch']

        model_weights = current_checkpoint['state_dict']
        model_weights = preprocess_weights(model_weights)

        model.load_state_dict(model_weights, strict=True)
        model.eval()

        model_with_embeddings = functools.partial(model, get_embeddings=True)

        outputs_train = get_embeddings(model_with_embeddings, tr_dl)
        # outputs_train = None
        # outputs_test = None
        outputs_test = get_embeddings(model_with_embeddings, test_dl)

        process_visualization(outputs_train, outputs_test, model_epoch)

    # visualizations_folder_path = os.path.join(os.path.dirname(args.model_path), 'visualizations')
    # output_archive_name = os.path.join(os.path.dirname(args.model_path), 'visualizations.zip')

    # zip -r visualizations.zip visualizations
    # subprocess.call(['zip', '-r', f'{output_archive_name}', f'{visualizations_folder_path}'])





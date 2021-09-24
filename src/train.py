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


class Model(pl.LightningModule):

    def __init__(self, hparams, tr_dl, val_dl, tr_filter_dl, train_filter, metric_crit, metric_crit_val,
                 allowed_classes):
        super(Model, self).__init__()

        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.tr_filter_dl = tr_filter_dl
        self.train_filter = train_filter
        self.metric_crit = metric_crit
        self.metric_crit_val = metric_crit_val
        self.allowed_classes = torch.Tensor(allowed_classes).long()

        self.params = hparams
        if args.distributed_backend == "ddp":
            self.num_train_steps = math.ceil(
                len(self.tr_dl) / (len(args.gpus.split(',')) * args.gradient_accumulation_steps))
        else:
            self.num_train_steps = math.ceil(len(self.tr_dl) / args.gradient_accumulation_steps)

        self.model = Net(args)

    def forward(self, x, get_embeddings=False):
        return self.model(x, get_embeddings)

    def configure_optimizers(self):

        if args.optimizer == "adamw":
            self.optimizer = AdamW([{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}],
                                   lr=self.params.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                [{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}], lr=self.params.lr,
                momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

        elif args.optimizer == "fused_sgd":
            import apex
            self.optimizer = apex.optimizers.FusedSGD(
                [{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}], lr=self.params.lr,
                momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

        if args.scheduler["method"] == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.num_train_steps * args.scheduler[
                                                                 "warmup_epochs"],
                                                             num_training_steps=int(
                                                                 self.num_train_steps * (args.max_epochs)))
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]
        elif args.scheduler["method"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.scheduler["step_size"],
                                                             gamma=args.scheduler["gamma"], last_epoch=-1)
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'epoch'}]
        elif args.scheduler["method"] == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, mode="max",
                                                                        patience=1, verbose=True)
            return [self.optimizer], [
                {'scheduler': self.scheduler, 'interval': 'epoch', 'reduce_on_plateau': True, 'monitor': 'val_gap_pp'}]
        else:
            self.scheduler = None
            return [self.optimizer]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

    def train_dataloader(self):
        return self.tr_dl

    def training_step(self, batch, batcn_num):
        input_dict, target_dict = batch

        output_dict = self.forward(input_dict)
        loss = loss_fn(self.metric_crit, target_dict, output_dict)

        if args.arcface_s is None:
            s = self.metric_crit.s.detach().cpu().numpy()
        elif args.arcface_s == -1:
            s = 0
        else:
            s = self.metric_crit.s

        if args.distributed_backend == "ddp":
            step = self.global_step * args.batch_size * len(args.gpus.split(',')) * args.gradient_accumulation_steps
        else:
            step = self.global_step * args.batch_size * args.gradient_accumulation_steps

        tb_dict = {'train_loss': loss, 'arcface_s': s, 'step': step}

        for i, param_group in enumerate(self.optimizer.param_groups):
            tb_dict[f'lr/lr{i}'] = param_group['lr']

        output = OrderedDict({
            'loss': loss,
            'log': tb_dict,

        })

        return output

    def training_epoch_end(self, outputs):

        tqdm_dict = {
        }

        results = {'progress_bar': tqdm_dict,
                   'log': tqdm_dict,
                   }

        return results

    def val_dataloader(self):
        return [self.val_dl, self.tr_filter_dl]

    def validation_step(self, batch, batch_nb, dataset_idx):
        if dataset_idx == 0:
            input_dict, target_dict = batch
            output_dict = self.forward(input_dict, get_embeddings=True)
            loss = loss_fn(self.metric_crit_val, target_dict, output_dict, val=True)  # .data.cpu().numpy()

            logits = output_dict['logits']
            embeddings = output_dict['embeddings']

            preds_conf, preds = torch.max(logits.softmax(1), 1)

            allowed_classes = self.allowed_classes.to(logits.device)

            preds_conf_pp, preds_pp = torch.max(logits.gather(1, allowed_classes.repeat(logits.size(0), 1)).softmax(1),
                                                1)
            preds_pp = allowed_classes[preds_pp]

            targets = target_dict['target']

            output = dict({
                'idx': input_dict['idx'],
                'embeddings': embeddings,
                'val_loss': loss.view(1),
                'preds': preds,
                'preds_conf': preds_conf,
                'preds_pp': preds_pp,
                'preds_conf_pp': preds_conf_pp,
                'targets': targets,

            })

            return output
        else:
            input_dict, target_dict = batch
            targets = target_dict['target']
            output_dict = self.forward(input_dict, get_embeddings=True)
            embeddings = output_dict["embeddings"]
            output = dict({
                'idx': input_dict['idx'],
                'embeddings': embeddings,
                'targets': targets,
            })
            return output

    def sync_across_gpus(self, t):

        gather_t_tensor = [torch.ones_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)

    def validation_epoch_end(self, outputs):

        val_outputs = outputs[0]
        tr_filter_outputs = outputs[1]

        out_val = {}
        for key in val_outputs[0].keys():
            out_val[key] = torch.cat([o[key] for o in val_outputs])

        out_tr_filter = {}
        for key in tr_filter_outputs[0].keys():
            out_tr_filter[key] = torch.cat([o[key] for o in tr_filter_outputs])

        if args.distributed_backend == "ddp":
            for key in out_val.keys():
                out_val[key] = self.sync_across_gpus(out_val[key])
            for key in out_tr_filter.keys():
                out_tr_filter[key] = self.sync_across_gpus(out_tr_filter[key])

        rank = self.global_rank
        device = out_val["targets"].device

        for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)
        for key in out_tr_filter.keys():
            out_tr_filter[key] = out_tr_filter[key].detach().cpu().numpy().astype(np.float32)

        ## HERE MUST BE SLY CODE

        # out_val['idx'] = [0, 1, 2, ...]
        # out_val['targets'] = [1, 2, ...]
        # out_val['embeddings'] = [[512-dim emb1], ...]

        # out_tr_filter['idx'] = [0, 1, 2, ...]
        # out_tr_filter['targets'] = [1, 2, ...]
        # out_tr_filter['embeddings'] = [[512-dim emb1], ...]

        # output_path = os.path.join(args.model_path, args.experiment_name, 'visualizations', str(self.current_epoch))
        #
        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)
        #
        # os.makedirs(output_path, exist_ok=True)
        #
        # for index, val_image_index_in_ds in enumerate(out_val['idx'][:30]):
        #     pred_index_of_labels, pred_dist = functions.calculate_top_n_cosine_sim(out_tr_filter['embeddings'],
        #                                                                            [out_val['embeddings'][index]],
        #                                                                            top_n=8)
        #
        #     save_path = os.path.join(output_path, str(index))
        #
        #     query_img = val_ds.get_original_item(int(val_image_index_in_ds))['input'].permute(1, 2, 0).numpy()
        #     functions.save_tensors_by_indexes([query_img], tr_ds, pred_index_of_labels, pred_dist, save_path)

        ## HERE MUST BE END OF SLY CODE

        if rank == 0:
            experiment_path = args.model_path + args.experiment_name + '/'
            with open(experiment_path + '/' + 'out_val.p', 'wb') as handle:
                pickle.dump(out_val, handle)
            with open(experiment_path + '/' + 'out_tr_filter.p', 'wb') as handle:
                pickle.dump(out_tr_filter, handle)

        val_score = comp_metric(out_val["targets"], [out_val["preds"], out_val["preds_conf"]])
        val_score_landmarks = comp_metric(out_val["targets"], [out_val["preds"], out_val["preds_conf"]],
                                          ignore_non_landmarks=True)

        val_score_pp = comp_metric(out_val["targets"], [out_val["preds_pp"], out_val["preds_conf_pp"]])
        val_score_landmarks_pp = comp_metric(out_val["targets"], [out_val["preds_pp"], out_val["preds_conf_pp"]],
                                             ignore_non_landmarks=True)

        val_loss_mean = np.sum(out_val["val_loss"])

        vals, inds = get_topk_cossim(out_val["embeddings"], out_tr_filter["embeddings"], k=1, device=device)
        vals = vals.data.cpu().numpy().reshape(-1)
        inds = inds.data.cpu().numpy().reshape(-1)
        labels = pd.Series(out_tr_filter["targets"][inds])

        val_score_cosine = comp_metric(out_val["targets"], [labels, vals])
        val_score_landmarks_cosine = comp_metric(out_val["targets"], [labels, vals], ignore_non_landmarks=True)

        f = StandardScaler()
        f.fit(np.concatenate([out_val["embeddings"]], axis=0))
        out_tr_filter["embeddings"] = f.transform(out_tr_filter["embeddings"])
        out_val["embeddings"] = f.transform(out_val["embeddings"], )

        vals, inds = get_topk_cossim(out_val["embeddings"], out_tr_filter["embeddings"], k=1, device=device)
        vals = vals.data.cpu().numpy().reshape(-1)
        inds = inds.data.cpu().numpy().reshape(-1)
        labels = pd.Series(out_tr_filter["targets"][inds])

        val_score_sc_cosine = comp_metric(out_val["targets"], [labels, vals])
        val_score_landmarks_sc_cosine = comp_metric(out_val["targets"], [labels, vals], ignore_non_landmarks=True)

        tqdm_dict = {'val_loss': val_loss_mean,
                     'val_gap': val_score,
                     'val_gap_landmarks': val_score_landmarks,
                     'val_gap_pp': val_score_pp,
                     'val_gap_landmarks_pp': val_score_landmarks_pp,
                     'val_gap_cosine': val_score_cosine,
                     'val_gap_landmarks_cosine': val_score_landmarks_cosine,
                     'val_gap_sc_cosine': val_score_sc_cosine,
                     'val_gap_landmarks_sc_cosine': val_score_landmarks_sc_cosine,
                     'step': self.current_epoch
                     }

        results = {'progress_bar': tqdm_dict,
                   'log': tqdm_dict
                   }

        return results

    def test_step(self, batch, batch_nb, dataset_idx):
        return self.validation_step(batch, batch_nb, dataset_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


if __name__ == '__main__':

    train, valid, train_filter, landmark_ids, landmark_id2class, landmark_id2class_val, class_weights, allowed_classes = setup()

    if args.filter_warnings:
        warnings.filterwarnings("ignore")

    if args.data_frac < 1.:
        train = train.sample(frac=args.data_frac)

    if args.loss == 'bce':
        metric_crit = nn.CrossEntropyLoss()
        metric_crit_val = nn.CrossEntropyLoss(weight=None, reduction="sum")
    else:
        metric_crit = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=class_weights)
        metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit="bce", weight=None, reduction="sum")

    tr_ds = GLRDataset(train, normalization=args.normalization, aug=args.tr_aug)

    print("ds len", len(tr_ds))

    val_ds = GLRDataset(valid, normalization=args.normalization, aug=args.val_aug)

    tr_dl = DataLoader(dataset=tr_ds, batch_size=args.batch_size, sampler=RandomSampler(tr_ds), collate_fn=collate_fn,
                       num_workers=args.num_workers, drop_last=True, pin_memory=False)

    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size, sampler=SequentialSampler(val_ds),
                        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)

    tr_filter_ds = GLRDataset(train_filter, normalization=args.normalization, aug=args.val_aug)
    tr_filter_dl = DataLoader(dataset=tr_filter_ds, batch_size=args.batch_size, sampler=SequentialSampler(tr_filter_ds),
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)

    experiment_path = args.model_path + args.experiment_name + '/'

    if args.logger == 'neptune':
        logger = NeptuneLogger(
            project_name=args.neptune_project,
            experiment_name=args.experiment_name,
            params=args.__dict__,
            upload_source_files=["train.py", "models.py", "loss.py", f"../configs/{args.experiment_name}.py"],

        )
    elif args.logger == 'tensorboard':
        logger = TensorBoardLogger(save_dir=args.model_path + args.experiment_name)
    else:
        logger = None

    ckpt_save_path = experiment_path + 'ckpt/'
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    ckpt = ModelCheckpoint(ckpt_save_path, monitor='val_gap_pp', verbose=False, mode='max', period=1, save_top_k=5,
                           save_last=True, save_weights_only=args.save_weights_only)

    trainer = Trainer(gpus=args.gpus,
                      logger=logger,
                      resume_from_checkpoint=args.resume_from_checkpoint,
                      max_epochs=args.max_epochs,
                      accumulate_grad_batches=args.gradient_accumulation_steps,
                      default_root_dir=experiment_path,
                      checkpoint_callback=ckpt,
                      precision=args.precision,
                      early_stop_callback=None,
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      gradient_clip_val=5.0,
                      distributed_backend=args.distributed_backend,
                      sync_batchnorm=args.sync_batchnorm,
                      # fast_dev_run=True
                      )

    model = Model(args, tr_dl, val_dl, tr_filter_dl, train_filter=train_filter, metric_crit=metric_crit,
                  metric_crit_val=metric_crit_val, allowed_classes=allowed_classes)

    trainer.fit(model)

    torch.save(model.model.state_dict(), experiment_path + '/' + f'{args.experiment_name}_ckpt_{args.max_epochs}.pth')

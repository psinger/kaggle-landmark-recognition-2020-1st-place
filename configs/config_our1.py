import os
import albumentations as A

abs_path = os.path.dirname(__file__)

args = {
    'model_path': '/root/gld_pd/models/',
    'data_path': '/root/snacks_data/',
    'data_path_2019': '/root/snacks_data/',
    'valid_csv_fn': 'test_filtered.csv',
    'train_csv_fn': 'train_filtered.csv',

    'gpus': '0',
    'filter_warnings': True,
    'logger': 'tensorboard',
    'num_sanity_val_steps': 0,

    'distributed_backend': 'ddp',
    'channels_last': False,

    'gradient_accumulation_steps': 2,
    'precision': 16,
    'sync_batchnorm': False,

    'seed': 1138,
    'num_workers': 0,
    'save_weights_only': True,

    'p_trainable': True,

    'resume_from_checkpoint': None,
    'pretrained_weights': None,

    'normalization': 'imagenet',
    'crop_size': 448,

    'backbone': 'gluon_seresnext101_32x4d',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.4,

    'neck': 'option-D',
    'head': 'arc_margin',

    'crit': "bce",
    'loss': 'arcface',
    # 'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm': 'batch',

    'optimizer': "sgd",
    'weight_decay': 1e-4,
    'lr': 0.05,
    'batch_size': 46,
    'max_epochs': 10,
    'scheduler': {"method": "cosine", "warmup_epochs": 1},

    'n_classes': 83,
    'data_frac': 1.,

    'neptune_project': 'xx/kaggle-landmark',
}


args['tr_aug'] = A.Compose([
    # A.Resize(height=448, width=448),
    A.SmallestMaxSize(512),
    A.RandomCrop(height=args['crop_size'], width=args['crop_size'], p=1.),
    A.HorizontalFlip(p=0.5),
])

args['val_aug'] = A.Compose([
    # A.Resize(height=448, width=448),
    A.SmallestMaxSize(512),
    A.CenterCrop(height=args['crop_size'], width=args['crop_size'], p=1.)
])

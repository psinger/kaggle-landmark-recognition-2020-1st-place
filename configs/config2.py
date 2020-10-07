
import os
import albumentations as A
abs_path = os.path.dirname(__file__)

# os.environ["CUDA_VISIBLE_DEVICES"]= "0"

args = {
    'model_path':'../models/',
    'data_path':'/ssd/kaggle_landmark/input/',
    'data_path_2019':'/ssd/kaggle_landmark/2019/',
    'valid_csv_fn':'recognition_solution_v2.1.csv',
    'train_csv_fn':'train.csv',

    'gpus':'0,1',
    'filter_warnings':True,
    'logger': 'neptune',
    'num_sanity_val_steps': 0,

    'distributed_backend': 'ddp',
    'channels_last':False,

    'gradient_accumulation_steps':4,
    'precision':16,
    'sync_batchnorm':False,
    
    'seed':1138,
    'num_workers':4,
    'save_weights_only':True,

    'resume_from_checkpoint': None,
    'pretrained_weights':None,

    'normalization':'imagenet',
    'crop_size':448,

    'backbone':'gluon_seresnext101_32x4d',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.3,

    'p_trainable': True,

    'neck': 'option-D',
    'head':'arc_margin',

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.05,
    'batch_size': 24,
    'max_epochs': 10,
    'scheduler': {"method":"cosine","warmup_epochs": 1},
    

    'n_classes':81313,
    'data_frac':1.,

    'neptune_project':'Zoo/kaggle-landmark',
}

args['tr_aug'] = A.Compose([
    A.SmallestMaxSize(512),
    A.RandomCrop(height=args['crop_size'],width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([
    A.SmallestMaxSize(512),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])

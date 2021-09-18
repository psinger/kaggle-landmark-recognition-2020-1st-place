from conf import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch 
import albumentations as A
import multiprocessing as mp 

import numpy as np
import cv2

def collate_fn(batch):
    
    input_dict = {}
    target_dict = {}
    
    for key in ['input']:
        input_dict[key] = torch.stack([b[key] for b in batch])
    for key in ['idx']:
        input_dict[key] = torch.stack([b[key] for b in batch]).long()   
    
    for key in ['target']:
        target_dict[key] = torch.stack([b[key] for b in batch]).long()
        
    return input_dict, target_dict

class GLRDataset(Dataset):

    def __init__(self, df, suffix='.jpg', preload=False, aug = None, normalization='simple'):

        self.df = df
        self.aug = aug
        self.normalization = normalization
        self.labels = self.df.target.values
        self.img_folder = self.df.img_folder.values
        self.suffix = suffix
        self.image_names = self.df.id.values
        # self.image_names = self.df.images.values  # SLY CODE
        self.images_cache = {}
        self.images_in_cache = False

        if preload:
            self.preload()
            self.images_in_cache = True
        self.eps = 1e-6

    def __getitem__(self, idx):
        id_ = self.image_names[idx]
        img_folder_ = self.img_folder[idx]
        
        if self.images_in_cache:
            img = self.images_cache[id_]
        else:
            img = self.load_one(id_, img_folder_)
            
        if self.aug:
            img = self.augment(img)
                
        img = img.astype(np.float32)       
        if self.normalization:
            img = self.normalize_img(img)
    
        tensor = self.to_torch_tensor(img)
        
        target = torch.tensor(self.labels[idx])
        feature_dict = {'idx':torch.tensor(idx).long(),
                        'input':tensor,
                       'target':target.float()}
        return feature_dict

    def __len__(self):
        return len(self.image_names)


    def preload(self):
        if self.n_threads > 1:
            with mp.Pool(self.n_threads) as p:
                imgs = p.map(self.load_one,self.id)
            self.images_cache = dict(zip(self.id, imgs))
        else:
            for i in tqdm(self.id):
                self.images_cache[i] = self.load_one(i)

    def load_one(self, id_, img_folder_):
        try:
            img = cv2.imread(img_folder_ + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        except:
            print("FAIL READING IMG", img_folder_ + f'{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.suffix}')
            img = np.zeros((512,512,3), dtype=np.int8)
        return img

    def augment(self,img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def normalize_img(self,img):
        
        if self.normalization == 'channel':
            pixel_mean = img.mean((0,1))
            pixel_std = img.std((0,1)) + self.eps
            img = (img - pixel_mean[None,None,:]) / pixel_std[None,None,:]
            img = img.clip(-20,20)

        elif self.normalization == 'channel_mean':
            pixel_mean = img.mean((0,1))
            img = (img - pixel_mean[None,None,:])
            img = img.clip(-20,20)
            
        elif self.normalization == 'image':
            img = (img - img.mean()) / img.std() + self.eps
            img = img.clip(-20,20)
            
        elif self.normalization == 'simple':
            img = img/255
            
        elif self.normalization == 'inception':
            
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'imagenet':
            
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        else:
            pass
        
        return img
    
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))
    

import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import data.wh_datautils as wh_datautils
import os.path as osp
import h5py

class LQGT_dataset(data.Dataset):

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.LQ_hdf5_path = opt['dataroot_LQ']
        self.GT_hdf5_path = opt['dataroot_GT']
        self.f_LQ_hdf5 = h5py.File(self.LQ_hdf5_path)
        self.f_GT_hdf5 = h5py.File(self.GT_hdf5_path)
        self.length =self.f_GT_hdf5['train_hdr'].shape[0]
        
    def __getitem__(self, index):
        GT_path, LQ_path = "hhhh", None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
       
        img_LQ = self.f_LQ_hdf5['train_hdr'][index,:,:,:]
        img_GT = self.f_GT_hdf5['train_hdr'][index,:,:,:]
        

        if self.opt['phase'] == 'train':
            
            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size // scale

            if GT_size != 0:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # condition
        if self.opt['condition'] == 'image':
            cond = wh_datautils.mask(img_LQ,threshold=0.83) # 返回一个0-1mask H,W,C 
        
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            cond = cond[:, :, [2, 1, 0]]

        H, W, _ = img_LQ.shape
        img_GT_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        cond_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()
        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ_tensor, 'GT': img_GT_tensor, 'cond': cond_tensor, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return self.length

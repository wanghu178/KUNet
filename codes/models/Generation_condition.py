import logging
from collections import OrderedDict
from os import SEEK_CUR, makedirs

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.customize_loss import tanh_L1Loss, tanh_L2Loss,artifical_Loss,bright_L1Loss,mask_loss,mask_l1_loss,mask3,mask4

logger = logging.getLogger('base')

class GenerationModel(BaseModel):
    def __init__(self, opt):
        super(GenerationModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training bn
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device) #基础损失函数
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == "maskl1_l1":
                self.cri_pix = nn.L1Loss().to(self.device)
                self.mask_pix= mask_l1_loss().to(self.device) #重建mask
            elif loss_type == "ganzhi_mask":
                self.cri_pix = artifical_Loss().to(self.device) # hdrtv加感知损失函数u
                self.mask_pix= mask4().to(self.device) 
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        self.var_cond = data['cond'].to(self.device) # mask
        
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # 折中考虑。实际上是我太懒。
        # 通过这种措施对不同的损失函数进行计算。 
        output_netG = self.netG((self.var_L, self.var_cond))
        if torch.is_tensor(output_netG):
            self.fake_H = output_netG
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        elif isinstance(output_netG,tuple):
            self.fake_H = output_netG[0]
            self.up_KIB = output_netG[1]
            mask = self.var_cond
            if len(self.up_KIB)==4:
                KIB1,KIB2,KIB3,KIB4 = self.up_KIB 
                KIB1_loss = self.mask_pix(KIB1,self.real_H,mask)
                KIB2_loss = self.mask_pix(KIB2,self.real_H,mask)
                KIB3_loss = self.mask_pix(KIB3,self.real_H,mask)
                KIB4_loss = self.mask_pix(KIB4,self.real_H,mask)
                KIB_loss = self.l_pix_w*KIB1_loss+self.l_pix_w*KIB2_loss+self.l_pix_w*KIB3_loss+self.l_pix_w*KIB4_loss
                l_pix = self.cri_pix(self.fake_H,self.real_H) + KIB_loss
            elif len(self.up_KIB)==1:
                img_mask = self.up_KIB[0]
                img_mask_loss=self.mask_pix(img_mask,self.real_H,mask)
                l_pix = self.cri_pix(self.fake_H,self.real_H)+self.l_pix_w*img_mask_loss
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        
        self.netG.eval()
        with torch.no_grad():
            output_netG = self.netG((self.var_L, self.var_cond))
            if torch.is_tensor(output_netG):
                self.fake_H = output_netG
            elif isinstance(output_netG,tuple):
                self.fake_H = output_netG[0]
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):

        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)


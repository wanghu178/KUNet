'''
Author: your name
Date: 2021-07-07 10:32:59
LastEditTime: 2021-11-18 17:00:34
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \code\hdrunetplus\codes\models\networks.py
'''

import logging


import models.modules.KIB_DM_F_1x1_mask as KIB_DM_F_1x1_mask
import models.modules.KIB_DM_F_1X1 as KIB_DM_F_1x1


logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
  
    #----------------------------------------------------
    #KIB 对nitre数据集快速实验可以用该模型 对应于文中不加L_mask
    if which_model == "KIB_DM_F_1x1":
        netG = KIB_DM_F_1x1.HDRUNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], act_type=opt_net['act_type'])
    #-----------------------------------------------------

    #-----------------------------------------------------
    #KIB with mask 
    #hdrtv以及ntire最终实验使用该模型
    elif which_model == "KIB_DM_F_1x1_mask":
        netG = KIB_DM_F_1x1_mask.HDRUNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], act_type=opt_net['act_type'])
    #-----------------------------------------------------
    
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG
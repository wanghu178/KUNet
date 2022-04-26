import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.arch_util as arch_util
import models.modules.wh_utils as wh_util

class HDRUNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.conv_2 = nn.Conv2d(nf,nf,3,1,1)
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)

        ''' 
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)
        '''
        self.recon_trunk1 = wh_util.mulRDBx4(nf,nf,2)
        self.recon_trunk2 = wh_util.mulRDBx6(nf,nf,2)
        self.recon_trunk3 = wh_util.mulRDBx4(nf,nf,2)


        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)


        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 自适应分支
        self.alpha = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        self.belta = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
    
        self.gammma1 = wh_util.DM_F_1X1(nf)
        self.gammma2 = wh_util.DM_F_1X1(nf)
        self.gammma3 = wh_util.DM_F_1X1(nf)


    def forward(self, x):
        # x[0]: img; x[1]: cond
   

        fea0 = self.act(self.conv_first(x[0]))

        #fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.conv_2(fea0)
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1= self.recon_trunk1(fea1)*self.alpha(fea1)+self.belta(fea1)

        fea2 = self.act(self.down_conv2(fea1))

        out = self.recon_trunk2(fea2) *self.alpha(fea2) +self.belta(fea2)
        out = self.recon_trunk2(out)*self.alpha(out)+self.belta(out)
        out = out + self.gammma3(fea2,out)

        out = self.act(self.up_conv1(out)) 
        out = out +self.gammma2(fea1,out)

        out = self.recon_trunk3(out)*self.alpha(out)+self.belta(out)

        out = self.act(self.up_conv2(out)) 
        out = out + self.gammma1(fea0,out)
        
        out = self.conv_2(out)
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = out
        return out
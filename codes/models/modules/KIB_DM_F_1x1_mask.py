import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.arch_util as arch_util
import models.modules.wh_utils as wh_util

class HDRUNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        #self.SFT_layer1 = arch_util.SFTLayer()
        self.conv_2 = nn.Conv2d(nf,nf,3,1,1)
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)

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
        
        # 对不同的块进行上采样进行恢复.
        self.up_KIB1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        
        self.up_KIB2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2),
                                    nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_KIB3 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2),
                                    nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        
        self.up_KIB4 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        
        self.restoration1 = nn.Conv2d(nf,3,3,1,1,bias=True)
        self.restoration2 = nn.Conv2d(nf,3,3,1,1,bias=True)
        self.restoration3 = nn.Conv2d(nf,3,3,1,1,bias=True)
        self.restoration4 = nn.Conv2d(nf,3,3,1,1,bias=True)

        # 自适应分支
        self.alpha = nn.Sequential(nn.Conv2d(nf,nf,3,1,1),nn.Conv2d(nf,nf,3,1,1))
        self.belta = nn.Sequential(nn.Conv2d(nf,nf,3,1,1),nn.Conv2d(nf,nf,3,1,1))

        # 融入KIB_KIC与过曝来修正mupsnr下降。
        self.gammma1 = wh_util.DM_F_1X1(nf)
        self.gammma2 = wh_util.DM_F_1X1(nf)
        self.gammma3 = wh_util.DM_F_1X1(nf)

    def forward(self, x):
        # x[0]: img; x[1]: cond
   
        up_KIB=[]
        
        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.conv_2(fea0)
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1= self.recon_trunk1(fea1)*self.alpha(fea1)+self.belta(fea1)
        up_KIB1 =self.restoration1(self.up_KIB1(fea1)) # 进行一次上采样
        up_KIB.append(up_KIB1)

        fea2 = self.act(self.down_conv2(fea1))
        out = self.recon_trunk2(fea2) *self.alpha(fea2) +self.belta(fea2)
        up_KIB2 = self.restoration2(self.up_KIB2(out)) # 进行二次上采样
        up_KIB.append(up_KIB2)
      
        out = self.recon_trunk2(out)*self.alpha(out)+self.belta(out)+out
        up_KIB3 =self.restoration3(self.up_KIB3(out))
        up_KIB.append(up_KIB3)
       
        out = out + self.gammma3(fea2,out) # 底层KIC
        
        out = self.act(self.up_conv1(out))
        out = out + self.gammma2(fea1,out) # 中层KIC
        out = self.recon_trunk3(out)*self.alpha(out)+self.belta(out)
        
        up_KIB4 =self.restoration4(self.up_KIB4(out))
        up_KIB.append(up_KIB4)
        out = self.act(self.up_conv2(out)) 
        out =out + self.gammma1(fea0,out) # 顶层KIC
        out = self.conv_2(out)
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = out
        return out,up_KIB
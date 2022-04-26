from numpy.lib.function_base import select
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import loss
from models.vgg import VGGLoss
from models.ssim import SSIM
class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss

class tanh_L2Loss(nn.Module):
    def __init__(self):
        super(tanh_L2Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        return loss

#L = R*0.299+G*0.587+B*0.114
class bright_L1Loss(nn.Module):
    def __init__(self):
        super(bright_L1Loss,self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,x,y):
        loss1 = self.l1(tensor_rgbtgray(x),tensor_rgbtgray(y))
        loss2 = self.l1(x,y)
        loss = loss1+loss2
        return loss

class artifical_Loss(nn.Module):
    def __init__(self):
        super(artifical_Loss,self).__init__()
        self.vggloss = VGGLoss()
        self.l1 = nn.L1Loss()
        #self.ssim_loss = SSIM()
        self.l2 = nn.MSELoss()
    def forward(self,x,y):
        loss1 = self.l1(x,y)
        loss2 = self.l1(mu_tonemap(x),mu_tonemap(y))
        #loss_ssim = self.ssim_loss(x,y)
        loss3 = self.vggloss(x,y)
    
        #loss4 = self.l1(tensor_rgbtgray(x),tensor_rgbtgray(y))
        #loss = 6*loss1 + 0.1*loss2  + loss3+0.1*loss4
        loss = loss1+loss3
        return loss

class mask_loss(nn.Module):
    def __init__(self):
        super(mask_loss,self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self,x,y,mask):
        loss = self.l1(x*mask,y*mask)
        return loss

class mask_l1_loss(nn.Module):
    def __init__(self):
        super(mask_l1_loss,self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,x,y,mask):
        loss = self.l1(x*mask,y*mask)+0.1*self.l1(x,y)
        return loss

class mask3(nn.Module):
    def __init__(self):
        super(mask3,self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,x,y,mask):
        loss = self.l1(x*mask,y*mask)+self.l1(x*(1-mask),y*(1-mask))
        return loss

class mask4(nn.Module):
    def __init__(self):
        super(mask4,self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,x,y,mask):
        #loss = self.l1(x,y)*mask+self.l1(x,y)*(1-mask)
        loss = self.l1(x*mask,y*mask)+0.1*self.l1(x,y)
        return loss

def mu_tonemap(img):
    """ tonemapping HDR images using μ-law before computing loss """

    MU = 5000.0
    return torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)


def tensor_rgbtgray(tensor_rgb):
    """
    只能用于转换tensor 
    输入格式：B,C,H,W
    输出格式：B,C,H,W
    参考cvtcolor源码：L = R*0.299+G*0.587+B*0.114
    """
    tensor_gray = tensor_rgb[:,0,:,:]*0.299 + tensor_rgb[:,1,:,:]*0.587+ tensor_rgb[:,1,:,:]*0.114
    return tensor_gray
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
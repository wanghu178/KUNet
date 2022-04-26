import torch
import torch.nn as nn

#----------------------------------------------------------------------
# KIC

class DM_F_1X1(nn.Module):
    def __init__(self,ch):
        super(DM_F_1X1,self).__init__()
        self.tmo = nn.Sequential(conv2d(ch,ch,3,1,1,acti='relu'),conv2d(ch,ch,3,1,1,acti='relu'))
        self.denosing = res_block(ch,3,1,1,acti="relu")
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.refin = nn.Conv2d(ch,ch,1,1,0)
    
    def forward(self,x,guidance):
        denos_x = self.denosing(x)
        tmo_x_hat = self.tmo(denos_x)
        guidance_map = self.cos_sim(tmo_x_hat,guidance).unsqueeze(dim=1)
        out =self.refin(guidance_map*x)
        return out 





#----------------------------------------------------------------------     
#----------------------------------------------------------------------
# 曝光分支实验

#1*1激活块
class conv_1x1(nn.Module):
    def __init__(self,ch):
        super(conv_1x1,self).__init__()
        self.conv = nn.Conv2d(ch,ch,1,1,0)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.act(self.conv(x))
        return out 

# 该部分对mask恢复。 
class maskRB(nn.Module):
    def __init__(self,ch):
        super(maskRB,self).__init__()
        self.mask_restoration=nn.Sequential(conv_1x1(ch),conv_1x1(ch),
                                            conv_1x1(ch),conv_1x1(ch)) # 用1*1卷积来解决问题。
    def forward(self,x):
        out = self.mask_restoration(x)
        return out

#----------------------------------------------------------------------
class mulRDBx2(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx2,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*2,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        f_c = torch.cat((rdb1,rdb2),dim=1)
        out = self.conv1x1(f_c)
        return out

class mulRDBx4(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx4,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*4,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        rdb3 = self.rdb(rdb2)
        rdb4 = self.rdb(rdb3)
        f_c = torch.cat((rdb1,rdb2,rdb3,rdb4),dim=1)
        out = self.conv1x1(f_c)
        return out


#############################################################
#################HDR生成模块##################################
class mulRDBx6(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx6,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*6,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        rdb3 = self.rdb(rdb2)
        rdb4 = self.rdb(rdb3)
        rdb5 = self.rdb(rdb4)
        rdb6 = self.rdb(rdb5)
        f_c = torch.cat((rdb1,rdb2,rdb3,rdb4,rdb5,rdb6),dim=1)
        out = self.conv1x1(f_c)
        return out

class mulRDBx2(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx2,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*2,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        f_c = torch.cat((rdb1,rdb2),dim=1)
        out = self.conv1x1(f_c)
        return out





class mulRDB_attention(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDB_attention,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*4,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        rdb3 = self.rdb(rdb2)
        rdb4 = self.rdb(rdb3)
        f_c = torch.cat((rdb1,rdb2,rdb3,rdb4),dim=1)
        out = self.conv1x1(f_c)
        return out










#######################################
#############基础模块###################
class BasicBLock(nn.Module):
    def __init__(self,intput_dim,output_dim):
        super(BasicBLock,self).__init__()
        self.conv = conv2d(intput_dim,output_dim,3,1,1,acti='relu')
    def forward(self,x):
        out = self.conv(x)
        #return torch.cat((x,out),dim=1)
        return x+out # 由于cat计算量过大调整为add

class RDB(nn.Module):
    def __init__(self,input_dim,growth_rate,nb_layers):
        super(RDB,self).__init__()
        self.layer = self._makekayer(nb_layers,input_dim,growth_rate)
        #self.conv1x1 = conv2d(input_dim+growth_rate*nb_layers,growth_rate,1,1,0)
        self.conv1x1 = conv2d(input_dim,growth_rate,1,1,0) # 修改为add
        

    
    def _makekayer(self,nb_layers,intput_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            #layers.append(BasicBLock(intput_dim+i*growth_rate,growth_rate))
            layers.append(BasicBLock(intput_dim,growth_rate)) # 修改为add
        return nn.Sequential(*layers)
    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out+x




#定义残差块RB模块
class res_block(nn.Module):
    def __init__(self,c,k_s=3,s=1,p=1,
                d=1,g=1,b=True,acti=None,norm=None):
        super(res_block,self).__init__()
        self.conv = conv2d(c,c,k_s,s,p,acti=acti,norm=norm)
    
    def forward(self,x):
        n = self.conv(x)
        n = self.conv(n)
        out = x+n
        return out

class conv2d(nn.Module):
    def __init__(self,in_c,out_c,k_s,s,p,
                d=1,g=1,b=True,acti=None,norm=None):
        super(conv2d,self).__init__()
        if acti == 'relu':
            self.acti = nn.ReLU()
        elif acti == 'leak':
            self.acti = nn.LeakyReLU(0.1,inplace=True)
        elif acti == 'selu':
            self.acti = nn.SELU()
        elif acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'sigmod':
            self.acti = nn.Sigmoid()
        elif acti == None:
            self.acti = None
        else:
            raise RuntimeError("no activation function {}".format(acti))
        
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_c)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_c)
        elif norm == None:
            self.norm = None
        else:
            raise RuntimeError("no norm layer:{}".format(norm))
        self.conv = nn.Conv2d(in_c,out_c,k_s,s,p,d,g,b)
    def forward(self,x):
        out = self.conv(x)
        if self.norm != None:
            out = self.norm(out)
        if self.acti != None:
            out = self.acti(out)
        return out



def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


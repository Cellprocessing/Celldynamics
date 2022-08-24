
import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime
from cellpose.transformers import Transformer,CONFIGS

#from . import transforms, io, dynamics, utils

sz = 3

def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )  

def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj  = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase)-1):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], sz))
            
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
    
class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(in_channels*2, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels*2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels)
        
    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj  = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x
    
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        
    def forward(self, x, y, style, mkldnn=False):
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x
    
class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        #style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2],x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d'%(n-1), 
                    resup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d'%(n-1), 
                    convup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation))
        config_vit = CONFIGS['R50-ViT-B_16']
        config_vit.n_skip = 3
        img_size=224
        vit_patches_size=16
        if 'R50-ViT-B_16'.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.TFS = Transformer(config_vit, img_size=224)
    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)


        ################################################
        #print("x tfs input",x.size())
        x=self.TFS(x)
        #print("x tfs output",x.size())
        ######################

        for n in range(len(self.up)-2,-1,-1):

            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
                #print("N:", n,x.size())
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x
    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz,
                residual_on=True, style_on=True, 
                concatenation=False, mkldnn=False,
                diam_mean=30.):
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on
        
    def forward(self, data):
        if self.mkldnn:
            print(21534534563645)
            data = data.to_mkldnn()
        #print("1datato:",data.size())
        T0    = self.downsample(data)
        # for i in T0:
        #     print("2T0:", i.size())

        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense()) 
        else:
            style = self.make_style(T0[-1])
        # print("3style:", style.size())
        style0 = style
        if not self.style_on:
            style = style * 0
        #print("4style:", style.size())
        T0 = self.upsample(style, T0, self.mkldnn)
        #print("5T0:", T0.size())
        T0    = self.output(T0)
        #print("6T0:", T0.size())
        if self.mkldnn:
            T0 = T0.to_dense()    
            #T1 = T1.to_dense()
        #print("7T0:", T0.size())
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        print("load_pretrain_model")
        # model.load_state_dict(torch.load('./model_pth/STDC2-Seg/model_maxmIOU75.pth'), strict=False)
        filename="F:/spyder/project/cellpose_seg/train/models/CP_tissuenet"#"F:/spyder/project/cellpose_seg/train/models/CP_tissuenet_tfs"
        weight_dict = torch.load(filename, map_location='cpu')  # 读取预训练网络的权重键值。
        self.load_state_dict(weight_dict)
        # x = torch.load(self.weight)
        # del weight_dict['upsample.up.res_up_0.conv.conv_2.full.weight']
        # del weight_dict['upsample.up.res_up_0.conv.conv_1.full.weight']
        # del weight_dict['upsample.up.res_up_0.conv.conv_3.full.weight']
        # del weight_dict['upsample.up.res_up_1.conv.conv_1.full.weight']
        # del weight_dict['upsample.up.res_up_1.conv.conv_2.full.weight']
        # del weight_dict['upsample.up.res_up_1.conv.conv_3.full.weight']
        # del weight_dict['upsample.up.res_up_2.conv.conv_1.full.weight']
        # del weight_dict['upsample.up.res_up_2.conv.conv_2.full.weight']
        # del weight_dict['upsample.up.res_up_2.conv.conv_3.full.weight']
        #
        # del weight_dict['upsample.up.res_up_3.conv.conv_0.0.weight']
        # del weight_dict['upsample.up.res_up_3.conv.conv_0.0.bias']
        # del weight_dict['upsample.up.res_up_3.conv.conv_0.0.running_mean']
        # del weight_dict['upsample.up.res_up_3.conv.conv_0.0.running_var']
        # del weight_dict['upsample.up.res_up_3.conv.conv_0.2.weight']
        # del weight_dict['upsample.up.res_up_3.conv.conv_1.full.weight']
        # del weight_dict['upsample.up.res_up_3.conv.conv_3.full.weight']
        # del weight_dict['upsample.up.res_up_3.conv.conv_2.full.weight']
        # del weight_dict['upsample.up.res_up_3.proj.0.weight']
        # del weight_dict['upsample.up.res_up_3.proj.0.bias']
        # del weight_dict['upsample.up.res_up_3.proj.0.running_mean']
        # del weight_dict['upsample.up.res_up_3.proj.0.running_var']
        # del weight_dict['upsample.up.res_up_3.proj.1.weight']
        #
        # # del weight_dict['conv_out16.conv_out.weight']
        # # del weight_dict['conv_out32.conv_out.weight']
        # # 然后获取当前网络的权重键值
        # model_dict = self.state_dict()  # model为当前定义的网络
        # # print(model_dict)  # 查看网络结构
        # # 最关键一步，根据键命名筛选出需要载入的部分权重。当前网络中要载入权重的部分，命名要与预训练网络相同
        # weight_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
        #
        # # 更新当前网络的键值字典
        # model_dict.update(weight_dict)

        # 最后载入该键值字典到网络中
        #self.load_state_dict(model_dict)


        #self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)

#
if __name__ == '__main__':
    nbase = [2, 32, 64, 128, 256,512]
    x=torch.ones((4,2,224,224))
    print(x.size())
    net=CPnet(nbase,nout=3,sz=3,
                        residual_on=False,
                        style_on=False,
                        concatenation=True,
                        mkldnn=False,
                        diam_mean=30.)

    y=net(x)
    for i in y:
        print(i.size())

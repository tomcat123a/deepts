# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:14:11 2020

@author: Administrator
"""
import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,LSTM,GRU ,Sigmoid
from torch.nn import AdaptiveMaxPool1d,AdaptiveAvgPool1d,MSELoss,Linear,BCELoss,BCEWithLogitsLoss,MaxPool1d
from torch.nn import ReLU,Dropout,AvgPool1d
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import torch.nn
import os
os.chdir('E:/deepts/pysrc')
from seq import testmlp
class cnn(torch.nn.Module):  #resnet
    def __init__(self,in_channels,out_channels,kernel_size,stride,dilation):
        super(cnn, self).__init__()
        self.Conv=Conv1d(in_channels=in_channels,out_channels=out_channels,\
        kernel_size=kernel_size,\
        stride=stride,padding=0,dilation=dilation)
        self.pool=AvgPool1d(kernel_size=5,stride=3)
        self.Conv2=Conv1d(in_channels=out_channels,\
    out_channels=out_channels*2,kernel_size=3,stride=stride,dilation=dilation)
        self.act=ReLU()
        self.pool2=AdaptiveAvgPool1d(1)
        self.linear=Linear(out_channels*2,1)
        self.sigm=Sigmoid()
        self.bn1=BatchNorm1d(out_channels)
        self.bn2=BatchNorm1d(2*out_channels)
    def forward(self, x ):
        
        x=self.bn1(self.act( self.Conv(x) ) )
        x=self.pool(x)
        x= self.bn2(self.act( self.Conv2(x)  ))
        x=self.pool2(x)
        x= self.linear(x) 
         
        return  x
    
res=testmlp(n_features=36,ker_size =[7,3,3],stride=[1,1,1],channels_list=[4,10,20],dnn_list=[3,1],di=[1,1],res_on=True,poolsize=[12,3],\
                 ppoolstride=[0,0],pooltype=3,use_rnn=False,extra=False   ) 
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:14:20 2020

@author: Administrator
"""
import torch
from torch.nn import  MSELoss
from loader import myDataset
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
from seq import testmlp
import time
def evaluate1(mod,x,y,device):
    mod=mod.to(device)
    loss=MSELoss(reduction='mean')
    mod=mod.eval()
    out=mod(x)
    m=loss(out,y)
    print('mse={:.4f}'.format(m.cpu().detach().numpy()))
    
def evaluate_loader(mod,loader,device):
    mod=mod.to(device)
    mod=mod.eval()
    loss=MSELoss(reduction='sum')
    m=0
    for x,y in loader:
        x=x.to(device)
        y=y.to(device)
        out=mod(x)
        m+=loss(out,y).cpu().detach().numpy()
    print('mse={:.4f}'.format(m/loader.__len__()))
    
def train1(mod,trainloader,testloader,device,epoch):
    CEL=MSELoss(reduction='mean')
    mod=mod.to(device)
    optimizer=torch.optim.Adam(mod.parameters(), lr=0.01,amsgrad=False)
    for it in range(epoch):
        mod=mod.train()
    
        optimizer.zero_grad() 
        for x,y in trainloader:
            t0=time.time()
            x=x.to(device)
            y=y.to(device)
            out=mod(x).squeeze(-1)
            loss=CEL(out,y   )
            loss.backward()
            optimizer.step() 
            print('iter takes {}s'.format(time.time()-t0))
        print('epoch{:.0f} done loss {:.4f}'.format(it,loss.cpu().data.numpy()))
        mod=mod.eval()
        print('train')
        evaluate_loader(mod,testloader,device )
        print('test')
        evaluate_loader(mod,testloader,device )

dset=myDataset(freq='daily',trainlen=36,predlen=1,T='1d',\
 columns=['close'],testdays=800,train=True)
    
testset=myDataset(freq='daily',trainlen=36,predlen=1,T='1d',\
 columns=['close'],testdays=800,train=False)
  
trainloader=DataLoader(dset,batch_size=64,shuffle=True)
testloader=DataLoader(testset,batch_size=64,shuffle=True)

a=iter(testloader )
x,y=a.next()       

mod=testmlp(n_features=36,ker_size =[7,3,3],stride=[1,1,1],\
            channels_list=[4,10,20],dnn_list=[3,1],di=[1,1],\
            res_on=True,poolsize=[12,3],\
                 ppoolstride=[0,0],pooltype=3,use_rnn=False,extra=False   ) 
mod(x).size()
train1(mod,trainloader,testloader,device=\
       torch.device('cuda' if torch.cuda.is_available() else 'cpu'),epoch=20)

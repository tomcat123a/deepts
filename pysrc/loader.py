# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:09:11 2020

@author: Administrator
"""

import torch
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import pandas as pd
import numpy as np
import os

class myDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, freq='daily',trainlen=5,predlen=1,T='1d',\
                columns=['close'],testdays=800,train=True):
        if freq=='daily':
            file_path='E:/1m_18/stockdata/us_stockdata' if os.name=='nt' else\
            '/scratch/deepnet/yxz346/selection/us_day'
        if freq=='minute':
            file_path='E:/1m_18/stockdata/us_1min/20200807'
        os.chdir(file_path)
        files_list=os.listdir()
        files_list.remove('spy.csv')
        files_list.insert(0,'spy.csv')
        self.predlen=predlen
        self.trainlen=trainlen
        """
        Args: used
            l days
            columns :['open', 'high', 'low', 'close']
        """
        dflist=[]
        dfidx=[]
        for i in range(len(files_list)):
            d=pd.read_csv(files_list[i],index_col=0).iloc[::-1]
            d.index=pd.to_datetime(d.index)
            d.columns=['open', 'high', 'low', 'close','volume']
            if T!='1D' and T!='1d':
                d=d.resample(T,base=0,label='left',closed='right').agg({'open': 'first', 
                     'high': 'max', 
                     'low': 'min', 
                     'close': 'last','volume':'sum'}).dropna()
            if files_list[i]=='spy.csv':
                self.spy=d.copy() 
                date1= self.spy.index[-testdays]
            if (d.loc[d.index<date1].shape[0]<trainlen+predlen and train==True)\
            or (train==False and d.loc[d.index>=date1].shape[0]<trainlen+predlen):
                continue
            else:
                if train==True:
                   dflist.append(\
        d.loc[d.index<date1] )
                   dfidx+=[len(dflist)-1]*(dflist[-1].shape[0]\
                   -trainlen-predlen)
                else:
                    dflist.append(\
        d.loc[d.index>=date1] )
                    dfidx+=[len(dflist)-1]*(dflist[-1].shape[0]\
                   -trainlen-predlen)
        self.dflist=dflist
        self.date1=date1
        self.dfidx=dfidx
         
    def __len__(self):
        return len(self.dfidx)

    def __getitem__(self, idx,debug=0):
        fileid=self.dfidx[idx]
        rowid=len( np.where(self.dfidx[idx]==self.dfidx[:idx])[0] )
        assert fileid<len(self.dflist)
        assert rowid+self.predlen+self.trainlen<self.dflist[fileid].shape[0]
        seq=self.dflist[fileid].iloc[rowid:rowid+self.predlen+self.trainlen]
        seq_spy=self.spy.loc[seq.index]
        x,sx,y= seq.iloc[:self.trainlen,3:5].values,seq_spy.iloc[:self.trainlen,3:5].values,\
    100*(\
   seq.values[self.trainlen+self.predlen-1,3]/seq.values[self.trainlen-1,3]-1)
        x=x.astype(np.float32)
        x=x/x.max(axis=0)
        sx=sx/sx.max(axis=0)
        sx=sx.astype(np.float32)
        y=y.astype(np.float32)
        return torch.cat(\
   (torch.from_numpy(x),torch.from_numpy(sx)),-1).transpose(0,1),\
        torch.tensor(y)

'''
 
dset=myDataset(freq='daily',trainlen=5,predlen=1,T='1d',\
 columns=['close'],testdays=800,train=True)    

loader=DataLoader(dset,batch_size=4,shuffle=True)
'''

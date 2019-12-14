#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:10:40 2019

@author: liusiyu
"""
import numpy as np


def load(filename):
    
    dataset=[]
    lable=[]
    index=0
    file=open(filename,'r')
    f=file.readlines()
    for line in f:
        line=line.strip('\n')
        line=line.split(',')
        dataset.append([])
        dataset[index].append(1.0)
        for data in line[:-1]:
            dataset[index].append(float(data))
        lable.append(int(line[-1]))
        index+=1
    return dataset,lable


def train(train_data,train_lable,w,num):
    #num->迭代的次数
    for i in range(num):
        for index,data in enumerate(train_data):
            #print(len(w),len(data))
            res=np.dot(w,data)
            if res>=0 and train_lable[index]==0:
                for k in range(len(w)):
                    w[k]-=data[k]
            elif res<0 and train_lable[index]==1:
                for k in range(len(w)):
                    w[k]+=data[k]
            #print(w)
                
def test(w,test_data,test_lable):
    acc=0
    for index,data in enumerate(test_data):
        res=np.dot(data,w)
        if res>0 and lable[index]==1:
            acc+=1
        elif res<0 and lable[index]==0:
            acc+=1
    #print(acc/len(test_data))
    return  acc/len(test_data)      
    



if __name__ == "__main__":
    dataset,lable=load('train.csv')
    d=len(dataset[0])
    
    
    fold=5
    index=0
    row_num=len(dataset)/fold
    num=0
    
    new_dataset=[]
    new_lable=[]
    for i,data in enumerate(dataset):
        if num==0:
            new_dataset.append([])
            new_lable.append([])
        num+=1
        new_dataset[index].append(data)
        new_lable[index].append(lable[i])
        if num==row_num:
            num=0
            index+=1
    acc=0
    for num in range(20):
        acc=0
        for i in range(fold):
            w=[]
            for k in range(d):
                w.append(0)
            train_data=[]
            test_data=new_dataset[i].copy()
            train_lable=[]
            test_lable=new_lable[i].copy()
            for j in range(fold):
                if j==i:
                    continue
                for m,data in enumerate(new_dataset[j]):
                    train_data.append(data)
                    train_lable.append(new_lable[j][m])
            train(train_data,train_lable,w,num+1)
            acc+=test(w,test_data,test_lable)
        print(num+1,acc/fold)    
        #print(len(train_data),len(test_data),len(train_lable),len(test_lable))
        #print(train_lable)
       # print(test_lable)
#        for num in range(50):
#            train(train_data,train_lable,w,num)
#            acc+=test(w,test_data,test_lable)
#            print(num,acc/fold)  
    
             
     
    

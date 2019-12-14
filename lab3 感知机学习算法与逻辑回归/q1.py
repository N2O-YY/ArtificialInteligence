#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 23:46:47 2019

@author: liusiyu
"""

import numpy as np
import matplotlib.pyplot as plt
def load(filename):
    
    dataset=[]

    index=0
    file=open(filename,'r')
    f=file.readlines()
    for line in f:
        line=line.strip('\n')
        line=line.split(',')
        dataset.append([])
        dataset[index].append(1.0)
        for data in line[:]:
            dataset[index].append(float(data))
        index+=1
    return dataset


def train(train_data,w,num):
    #num->迭代的次数
    for i in range(num):
        for data in train_data:
            temp=data[:-1]
            res=np.dot(w,temp)
            if res>=0 and int(data[-1])==0:
                for k in range(len(w)):
                    w[k]-=data[k]
            elif res<0 and int(data[-1])==1:
                for k in range(len(w)):
                    w[k]+=data[k]
                    


def test(w,test_data):
    acc=0
    for data in test_data:
        temp=data[:-1]
        res=np.dot(temp,w)
        if res>=0 and int(data[-1])==1:
            acc+=1
        elif res<0 and int(data[-1])==0:
            acc+=1
    return  acc/len(test_data) 



if __name__ == "__main__":
    dataset=load('train.csv')
    #print(dataset)
    
    fold=5
    
    divide_dataset=[]
    
    index=0
    
    d=len(dataset)/fold
    
    num=0
    for data in dataset:
        if num==0:
            divide_dataset.append([])
        divide_dataset[index].append(data)
        num+=1
        if num==d:
            num=0
            index+=1
    num=len(dataset[0])
    res_list=[]
    for time in range(20):
        acc=0
        for i in range(fold):
            w=np.zeros(num-1)
            test_data=divide_dataset[i].copy()
            train_data=[]
            for j in range(fold):
                if j==i:
                    continue
                for data in divide_dataset[j]:
                    train_data.append(data)
            
            train(train_data,w,time)
            acc+=test(w,test_data)
        print('迭代',time,'次，准确率为',acc/fold)
        res_list.append(acc/fold)
    x = range(20)
    plt.plot(x[1:], res_list[1:], 'ro-')
    plt.xlabel("times of recursion") #X轴标签
    plt.ylabel("accuracy") #Y轴标签
    
    
        
#    train_data=dataset[:6000]
#    
#    test_data=dataset[6000:]
#    
#    num=len(dataset[0])
#    
#    w=np.zeros(num-1)
#    
#    train(train_data,w,2)
#    
#    test(w,test_data)
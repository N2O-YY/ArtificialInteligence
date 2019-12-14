#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:34:18 2019

@author: liusiyu
"""
import math
import numpy as np

import random
Filename="car_train.csv"



attribute=['buying','maint','door','person','lug_boot','safety']

attribute_map={'buying':0,'maint':1,'door':2,'person':3,'lug_boot':4,'safety':5}

def data_collet(FILENAME):
    data_set=[]
    file=open(FILENAME,'r')
    f=file.readlines()
    
    
    for line in f:
        line=line.strip('\n')
        
        line=line.split(',')
        
        data_set.append(line)
    
    #print(data_set)
    return data_set

      
       
 #ID3       
def cal_information_gain(data_set,attribute_index):
    """
    作用：求出某个特征的信息增益
    输入：data_set->数据集
    输出：信息增益
    """
    HD=0#经验熵
    HDA=0#条件熵
    
    label={}
    
    num=len(data_set)
    
    attribute_map={}
    for data in data_set:
        temp=data[-1]
        if temp not in label.keys():
            label[temp]=0
        label[temp]+=1
        attribute=data[attribute_index]
        if attribute not in attribute_map.keys():
            attribute_map[attribute]=0
        attribute_map[attribute]+=1
        
    #print(label)   
    for temp in label.items():
        HD+=-temp[1]/num*math.log2(temp[1]/num)
    
    for temp in attribute_map.items():
        temp_map={}
        length=0
        for data in data_set:
            if data[attribute_index]==temp[0]:
                length+=1
                if data[-1] not in temp_map.keys():
                    temp_map[data[-1]]=0
                temp_map[data[-1]]+=1
        #print(temp_map,length)
        for f in temp_map.items():
            HDA+=temp[1]/num*(-f[1]/length*math.log2(f[1]/length))
    #print(HDA)
    return HD-HDA
#C4.5
def  cal_infor_gian_ratio(data_set,attribute_index):
    information_gain=cal_information_gain(data_set,attribute_index)
    
    splitinfo=0
    
    attribute_map={}
    for data in data_set:
        attribute=data[attribute_index]
        if attribute not in attribute_map.keys():
            attribute_map[attribute]=1
        else:
            attribute_map[attribute]+=1
    
    num=len(dataset)
    for attribute in attribute_map.keys():
        splitinfo+=-(attribute_map[attribute]/num)*math.log2(attribute_map[attribute]/num)
    return information_gain/splitinfo
    

#cart
def cal_GINI(data_set,attribute_index):
    attribute_list=[line[attribute_index] for line in data_set]
    attribute_set=set(attribute_list)
    num=len(data_set)
    
    gini=0
    
    for attribute in attribute_set:
        attribute_number=0
        
        key_map={}
        for line in data_set:
            if line[attribute_index]==attribute:
                attribute_number+=1
                if line[-1] not in key_map.keys():
                    key_map[line[-1]]=1
                else:
                    key_map[line[-1]]+=1
        temp_pos=1
        for k in key_map.keys():
            temp_pos-=np.power(key_map[k]/attribute_number,2)
        gini+=attribute_number/num*temp_pos
            
    return gini
        
    
    
def renew_dataset(dataset,index,value):
    res=[]
    for line in dataset:
        if line[index]==value:
            temp=line[:index]
            temp.extend(line[index+1:])
            res.append(temp)
    return res

def get_themost(labellist):
    temp_map={}
    for lable in labellist:
        if lable not in temp_map.keys():
            temp_map[lable]=1
        else:
            temp_map[lable]+=1
    temp_map=sorted(temp_map.items(),key=lambda item:item[1],reverse=True)
    return temp_map[0][0]
    

      
def create_tree(dataset,attribute):
#    print(attribute,len(dataset[0]))
    #假设当前结点的数据集为D，特征集为A
    labellist=[line[-1] for line in dataset]
    #D中的样本属于同一类别C
    if labellist.count(labellist[0])==len(labellist):
        return labellist[0]
    #A为空集,或D中所有样本在A中所有特征上取值相同
    if len(dataset[0])==1:
        return get_themost(labellist)
    
    num=len(dataset[0])
    
#    index=0
#    temp=0
#
#    
#    for i in range(num-1):
#        data=cal_information_gain(dataset,i)
#        data=cal_infor_gian_ratio(dataset,i)
#        if temp<data:
#            index=i
#            temp=data
#    
    
    
    
    
    index=7
    temp=1000000  
    for i in range(num-1):
        data=cal_GINI(dataset,i)

        if temp>data:
            index=i
            temp=data
    
    bestattribute=attribute[index]
    
    Tree={bestattribute:{}}
    
    attribute_values=[line[index] for line in dataset]
    
    attribute_values=set(attribute_values)
    Tree[bestattribute]['majority']=get_themost(labellist)
    #print(bestattribute,attribute_values)
    del(attribute[index])
    for value in attribute_values:
        newattribute=attribute[:]
        Tree[bestattribute][value]=create_tree(renew_dataset(dataset,index,value),newattribute)   
    return Tree    
# 
 
    
def create_tree_new(dataset,attribute,testset):
#    print(attribute,len(dataset[0]))
    #假设当前结点的数据集为D，特征集为A
    labellist=[line[-1] for line in dataset]
    #D中的样本属于同一类别C
    if labellist.count(labellist[0])==len(labellist):
        return labellist[0]
    #A为空集,或D中所有样本在A中所有特征上取值相同
    if len(dataset[0])==1:
        return get_themost(labellist)
    
    num=len(dataset[0])
#    index=0
#    temp=0
#    
#    for i in range(num-1):
#        data=cal_information_gain(dataset,i)
#        data=cal_infor_gian_ratio(dataset,i)
#        if temp<data:
#            index=i
#            temp=data
# 
            
    index=7
    temp=1000000
    
    for i in range(num-1):
        data=cal_GINI(dataset,i)

        if temp>data:
            index=i
            temp=data
    
    bestattribute=attribute[index]
    
    Tree={bestattribute:{}}
    
    attribute_values=[line[index] for line in dataset]
    
    attribute_values=set(attribute_values)
    Tree[bestattribute]['majority']=get_themost(labellist)
    #print(bestattribute,attribute_values)
    tempattribute=attribute[:]
    del(attribute[index])
    for value in attribute_values:
        newattribute=attribute[:]
        Tree[bestattribute][value]=create_tree_new(renew_dataset(dataset,index,value),newattribute,renew_dataset(testset,index,value))
    acc=0
    test_list=[line[-1] for line in testset]
    if(len(test_list)==0 or len(testset[0])<=1):
        return Tree
    
    for line in testset:
         if new_test(Tree,line,tempattribute)==line[-1]:
             acc+=1
    
    acc2=test_list.count(get_themost(test_list))
    #print(acc2)
    if acc<acc2:
        for value in attribute_values:
            Tree[bestattribute][value]=get_themost(labellist)
    return Tree    





def new_test(Tree,test_sample,attribute_list):
    attribute=list(Tree.keys())[0]
    #print(list(Tree.keys()))
    
    NextTree=Tree[attribute]
    
    index=attribute_list.index(attribute)
    
    sample_attr=test_sample[index]
    
    if sample_attr not in NextTree.keys():
        #return str(random.randint(0,1))
        return NextTree['majority']
    temp=NextTree[sample_attr]
    
    if type(temp).__name__ == 'dict':
    #if isinstance(temp,dict):
        res=new_test(temp,test_sample,attribute_list)
    else:
        res=temp
        
    return res

def test(Tree,test_sample,attribute_map):
    attribute=list(Tree.keys())[0]
    
    #print(list(Tree.keys()))
    
    NextTree=Tree[attribute]
    
    index=attribute_map[attribute]
    
    sample_attr=test_sample[index]
    
    if sample_attr not in NextTree.keys():
        #return str(random.randint(0,1))
        return NextTree['majority']
    temp=NextTree[sample_attr]
    
    if type(temp).__name__ == 'dict':
    #if isinstance(temp,dict):
        res=test(temp,test_sample,attribute_map)
    else:
        res=temp
        
    return res

#def test(Tree):
#    print(Tree.keys())   



    
    
    
    
    
    
#    
#    if type(Tree).__name__!='dict':
#        return Tree
#    attribute=list(Tree.keys())[0]
#    
#    NextTree=Tree[attribute]
#    
#    attribute_index=attribute_list.index(attribute)
#    
#    cut_Tree={attribute:{}}
#    
#    lable_list=[line[-1] for line in dataset]
#    
#    res=get_themost(lable_list)
#    
#    attribute_value=[line[attribute_index] for line in dataset]
#    attribute_value_set=set(attribute_value)
#    for value in attribute_value_set:
#        cut_Tree[attribute][value]=res
#        
#    acc1=0
#    acc2=0
#    for line in test_dataset:
#        guess1=test(Tree,data,attribute_map)
#        
#        guess2=test(cut_Tree,data,attribute_map)
#        
#        acc1+=guess1==line[-1]
#        
#        acc2+=guess2==line[-1]
#        
#    if acc1>=acc2:
#        
#        return cut_Tree
#    
#    else:
#        print('*****')
#        del(attribute_list[attribute_index])
#        for value in attribute_value_set:
#            newattribute=attribute_list[:]
#            cut_Tree[attribute][value]=cut_leaf(NextTree[value],newattribute,renew_dataset(dataset,attribute_index,value),test_dataset,attribute_map)
#            
#    return cut_Tree
       
    
    
        
    
if __name__ == "__main__":
#    dataset=data_collet(Filename)
#    fun_list=[cal_information_gain,cal_infor_gian_ratio,cal_GINI]
#    res1=0
#    res2=0
#    dataset_list=[]
#    index=0
#    num=0
#    for data in dataset:
#        if num==0:
#            dataset_list.append([])
#        dataset_list[index].append(data)
#        num+=1
#        if num==250:
#            num=0
#            index+=1
#    numlen=len(dataset_list)
#    #print(dataset_list[0])
#    
#    
#    #Tree=create_tree(dataset_list[0],attribute)
#    for i in range(numlen):
#        #print("test ",i)
#        attribute=['buying','maint','door','person','lug_boot','safety']
#        test_data=dataset_list[i].copy()
#        temp_data_set=[]
#        for j in range(numlen):
#            if j==i:
#                continue
#            else:
#                for data in dataset_list[j]:
#                    temp_data_set.append(data)
#                    
#        data_set=temp_data_set.copy()
#        
#        Tree=create_tree(data_set,attribute)
#        
#        
#        index=0
#        acc=0
#        test_data=dataset_list[i].copy()
#        for data in test_data:
#            index+=1
#            guess=test(Tree,data,attribute_map)
#           # print(guess,data[-1])
#            if(guess==data[-1]):
#                acc+=1
#        res1+=acc/index
#        #print("剪枝前accuracy:",acc,index,acc/index)
#        data_set=temp_data_set.copy()
#        attribute=['buying','maint','door','person','lug_boot','safety']
#        
#        test_data=dataset_list[i].copy()
#        #print(len(test_data))
#        cut_tree=create_tree_new(data_set,attribute,test_data)
#        test_data=dataset_list[i].copy()
#        index=0
#        acc=0
#        for data in test_data:
#            index+=1
#            guess=test(cut_tree,data,attribute_map)
#           # print(guess,data[-1])
#            if(guess==data[-1]):
#                acc+=1
#        #print("剪枝后:accuracy",acc,index,acc/index)
#        res2+=acc/index
#       
#    print("剪枝前:accuracy",res1/numlen)    
#    print("剪枝后:accuracy",res2/numlen)
    print('剪枝前')   
    dataset=data_collet('train.csv')
    attribute=['buying','maint','door','person','lug_boot','safety']
    train_data=dataset.copy()
    Tree=create_tree(train_data,attribute)
    print(Tree)
    test_data=data_collet('test.csv')
    index=0
    acc=0
    for data in test_data:
        index+=1
        guess=test(Tree,data,attribute_map)
        print(guess)
    
     
    
    print('剪枝后')  
    dataset=data_collet('train.csv')
    attribute=['buying','maint','door','person','lug_boot','safety']
    train_data=dataset.copy()
    test_data=data_collet('test.csv')
    cut_tree=create_tree_new(train_data,attribute,test_data)
    print(cut_tree)
    test_data=data_collet('test.csv')
    index=0
    acc=0
    for data in test_data:
        index+=1
        guess=test(Tree,data,attribute_map)
        print(guess)
    
    #print("剪枝后accuracy:",res2)    
    
    
    
    
    
    

#    index=0
#    acc=0
#    for data in dataset[:100]:
#        index+=1
#        guess=test(Tree,data,attribute_map)
#      
#        if(guess==data[-1]):
#            acc+=1
#            
#    dataset=data_collet(Filename)
#    
#    attribute=['buying','maint','door','person','lug_boot','safety']
#    
#    test_dataset=dataset[:100]
#    
#    
#   
#    
#    
#    
#    print(index,acc)                 
#    print(acc/index)
#    index=0
#    acc=0
#    
#    
#    cut_tree=create_tree_new(dataset[100:],attribute,dataset[:100])
#    for data in dataset[:100]:
#        index+=1
#        guess=test(cut_tree,data,attribute_map)
#       # print(guess,data[-1])
#        if(guess==data[-1]):
#            acc+=1
#    print(index,acc)
#    print(acc/index)
    
    
    
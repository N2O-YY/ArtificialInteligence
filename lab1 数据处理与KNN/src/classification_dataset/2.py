#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:08:22 2019

@author: liusiyu
"""

import numpy as np
import math
import numpy
TRAINNAME = "train_set1.csv"
VALINAME="validation_set.csv"
TESTNAME="test_set1.csv"

#TRAINNAME = "train_set.csv"
#VALINAME="validation_set.csv"
#TESTNAME="test_set.csv"
def get_word_list(word_list,FILENAME,word_map):
    index=len(word_list)
    file=open(FILENAME,'r')
    f=file.readlines()
    for line in f[1:]:
        line=line.strip()
        line_list=line.split(',')
        line_word=line_list[0].split(' ')
        for word in line_word:
            if word not in word_list:
                word_list.append(word)
                word_map[word]=index
                index+=1

def get_mood_map(FILENAME):
    index=0
    file=open(FILENAME,'r')
    f=file.readlines()
    line=len(f)-1

    mood_map={}
    for line in f[1:]:
        line=line.strip()
        line_list=line.split(',')
        mood_map[index]=line_list[1]
        index+=1
    return mood_map
    
def get_onehot(FILENAME,word_list,word_map):
    file=open(FILENAME,'r')
    f=file.readlines()
    line=len(f)-1
    col=len(word_list)
    onehot=np.zeros((line,col),dtype=np.int)
    
    index=0
    for line in f[1:]:
        line=line.strip()
        line_list=line.split(',')
        line_word=line_list[0].split(' ')
        for word in line_word:
            if word in word_map:
                j=word_map[word]
                onehot[index][j]+=1
        index+=1
        
    return onehot
            

def cal_distance(onehot,val_onehot,mood_map,val_mood_map,K):
    line=len(val_onehot)
    col=len(val_onehot[0])
    
    num=len(onehot)
    acc=0
    for i in range(line):
        distance={}
        for j in range(num):
            res=0
            for k in range(col):
                res+=np.power(val_onehot[i][k]-onehot[j][k],2)
            distance[j]=res
        temp_distance=sorted(distance.items(),key=lambda item:item[1])
        res_map={}
        for m in range(K):
            mood=mood_map[temp_distance[m][0]]
            if mood not in res_map.keys():
                res_map[mood]=1
            else:
                res_map[mood]+=1
        
        res_map=sorted(res_map.items(),key=lambda item:item[1] )
        
        if res_map[-1][0] == val_mood_map[i]:
            acc+=1
        print(res_map,val_mood_map[i],acc)
        
    print(1.0*acc/line)



def get_TF(line,col,FILENAME,word_list,word_map):
    TF=np.zeros((line,col))
    file=open(FILENAME,'r')
    f=file.readlines()
    index=0
    for line in f[1:]:
        line=line.strip()
        line_list=line.split(',')
        line_word=line_list[0].split(' ')
        for word in line_word:
            if word in word_map:
                j=word_map[word]
                TF[index][j]+=1/len(line_word)
        index+=1
    return TF
  

      
def get_IDF(col,FILENAME,line,word_list):
    IDF=np.zeros(col)
    index=0
    file=open(FILENAME,'r')
    f=file.readlines()
    D=len(f)-1
#    print(D)
    for word in word_list:
        s=0
        for line in f[1:]:
            line=line.strip()
            line_list=line.split(',')
            line_word=line_list[0].split(' ')
            if word in line_word:
                s+=1
        IDF[index]=math.log((D+1)/(1+s))
        index+=1
    return IDF
        
        

def get_TFIDF(TF,IDF,line,col):
    TFIDF=np.zeros((line,col))
    for i in range(line):
        for j in range(col):
            TFIDF[i][j]=TF[i][j]*IDF[j]
    return TFIDF





def calcos(VALTFIDF,TFIDF,K,mood_map,val_mood_map):
    line=len(VALTFIDF)
    col=len(VALTFIDF[0])
    
    num=len(TFIDF)
    
    acc=0
    for i in range(line):
        distance={}
        for j in range(num):
            res=0
            len1=0
            len2=0
            for k in range(col):
                len1+=np.power(VALTFIDF[i][k],2)
                len2+=np.power(TFIDF[j][k],2)
                res+=VALTFIDF[i][k]*TFIDF[j][k]
            len1=np.power(len1,0.5)
            len2=np.power(len2,0.5)
            distance[j]=res/(len1*len2)
        temp_distance=sorted(distance.items(),key=lambda item:item[1],reverse=True)
        print(temp_distance)
        res_map={}
        for m in range(K):
            mood=mood_map[temp_distance[m][0]]
            if mood not in res_map.keys():
                res_map[mood]=1
            else:
                res_map[mood]+=1
        
        res_map=sorted(res_map.items(),key=lambda item:(item[1],-ord(item[0][0])))
        
        if res_map[-1][0] == val_mood_map[i]:
            acc+=1
        print(i,res_map,val_mood_map[i],acc)
        
    print(1.0*acc/line)
  
    
    
def get_res(RESTFIDF,TFIDF,K,mood_map):
    f=open('res.csv','w')
    
    
    line=len(RESTFIDF)
    col=len(RESTFIDF[0])
    
    num=len(TFIDF)
    
    for i in range(line):
        distance={}
        for j in range(num):
            res=0
            for k in range(col):
                res+=numpy.power(RESTFIDF[i][k]-TFIDF[j][k],2)
            distance[j]=res
            print(distance[j])
        temp_distance=sorted(distance.items(),key=lambda item:item[1])
       
        res_map={}
        for m in range(K):
            mood=mood_map[temp_distance[m][0]]
            if mood not in res_map.keys():
                res_map[mood]=1
            else:
                res_map[mood]+=1
        
        res_map=sorted(res_map.items(),key=lambda item:(item[1],-ord(item[0][0])))
        
        print(res_map)
        f.write(res_map[-1][0])
        f.write(',')
        f.write('\n')
        
        
#def get_res(RESTFIDF,TFIDF,K,mood_map):
#    f=open('res.csv','w')
#
#    line=len(RESTFIDF)
#    col=len(RESTFIDF[0])
#    
#    num=len(TFIDF)
#    
#    for i in range(line):
#        distance={}
#        for j in range(num):
#            res=0
#            len1=0
#            len2=0
#            for k in range(col):
#                len1+=np.power(RESTFIDF[i][k],2)
#                len2+=np.power(TFIDF[j][k],2)
#                res+=RESTFIDF[i][k]*TFIDF[j][k]
#            len1=np.power(len1,0.5)
#            len2=np.power(len2,0.5)
#            distance[j]=res/(len1*len2+0.00001)
#        temp_distance=sorted(distance.items(),key=lambda item:item[1],reverse=True)
#        res_map={}
#        for m in range(K):
#            mood=mood_map[temp_distance[m][0]]
#            if mood not in res_map.keys():
#                res_map[mood]=1
#            else:
#                res_map[mood]+=1
#        
#        res_map=sorted(res_map.items(),key=lambda item:(item[1],-ord(item[0][0])))
#        
#        print(res_map)
#        f.write(res_map[-1][0])
#        f.write(',')
#        f.write('\n')
        
    
            
if __name__ == "__main__":
    word_list=[]
    
    word_map={}
    
    get_word_list(word_list,TRAINNAME,word_map)
    
    get_word_list(word_list,VALINAME,word_map)
    
    get_word_list(word_list,TESTNAME,word_map)
#    print(word_list)
#    print(word_map)
    
    onehot=get_onehot(TRAINNAME,word_list,word_map)
    
    val_onehot=get_onehot(VALINAME,word_list,word_map)
    
    res_one_hot=get_onehot(TESTNAME,word_list,word_map)
   
    mood_map=get_mood_map(TRAINNAME)
    
    val_mood_map=get_mood_map(VALINAME)
    
    line=len(onehot)
    
    col=len(onehot[0])
    
#    cal_distance(onehot,val_onehot,mood_map,val_mood_map,int(np.power(line,0.5)))
    
    TF=get_TF(line,col,TRAINNAME,word_list,word_map)
    
    IDF=get_IDF(col,TRAINNAME,line,word_list)
    
    TFIDF=get_TFIDF(TF,IDF,line,col)
    
#    print(TFIDF)
    
    line=len(val_onehot)
    
    VALTF=get_TF(line,col,VALINAME,word_list,word_map)
    
    VALIDF=get_IDF(col,VALINAME,line,word_list)
    
    VALTFIDF=get_TFIDF(VALTF,VALIDF,line,col)
    
#    calcos(VALTFIDF,TFIDF,14,mood_map,val_mood_map)
    
#    print(onehot[0][5])
    
    line=len(res_one_hot)
    
#    print(word_map['can'])
#    print(res_one_hot)
    
    RESVALTF=get_TF(line,col,TESTNAME,word_list,word_map)
    
    RESVALIDF=get_IDF(col,TESTNAME,line,word_list)
    
    RESVALTFIDF=get_TFIDF(RESVALTF,RESVALIDF,line,col)
    
#    print(TFIDF)
    
#    print(TFIDF)
    
#    print(RESVALTFIDF)
    
    get_res(RESVALTFIDF,TFIDF,1,mood_map)
    
    
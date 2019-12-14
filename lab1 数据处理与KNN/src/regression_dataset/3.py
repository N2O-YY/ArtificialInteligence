#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:09:13 2019

@author: liusiyu
"""
import numpy as np
import math
import scipy.stats as stats
#TRAINNAME = "train_set.csv"
#VALINAME="validation_set.csv"
#TESTNAME="test_set.csv"
#
#VALINAME="test_set.csv"

TRAINNAME = "train.csv"
VALINAME="test.csv"
TESTNAME="test_set.csv"


mood_map={'anger':0,'disgust':1,'fear':2,'joy':3,'sad':4,'surprise':5}
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

def get_mood_po(FILENAME):
    file=open(FILENAME,'r')
    f=file.readlines()
    mood_po=np.zeros((len(f)-1,6))
    index1=0
    for line in f[1:]:
        line=line.strip()
        line_list=line.split(',')
        index2=0
        for num in line_list[1:]:
            mood_po[index1][index2]=float(num)
            index2+=1
        index1+=1
    return mood_po


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
            j=word_map[word]
            onehot[index][j]+=1
        index+=1
        
    return onehot



def get_topK(K,val_onehot,train_onehot):
    line=len(val_onehot)
    col=len(val_onehot[0])
    
    num=len(train_onehot)
    
    topK=np.zeros((line,K),dtype=np.int)
    
    topK_dis=np.zeros((line,K),dtype=np.int)
    for i in range(line):
        distance={}
        for j in range(num):
            res=0
            for k in range(col):
                res+=np.power(val_onehot[i][k]-train_onehot[j][k],2)
            distance[j]=res
        temp_distance=sorted(distance.items(),key=lambda item:item[1])
        for m in range(K):
            topK[i][m]=temp_distance[m][0]
            topK_dis[i][m]=temp_distance[m][1]
        print(topK[i])
    return topK,topK_dis
    
def get_pos(topK,train_mood_po,topK_dis):
    line=len(topK)
    
    col=len(train_mood_po[0])
    
    num=len(topK[0])
    
    posibility=np.zeros((line,col))
    
    for i in range(line):
        for j in range(col):
            posi=0
            for k in range(num):
                index=topK[i][k]
                posi+=train_mood_po[index][j]/(topK_dis[i][k]+0.001)
            posibility[i][j]=posi
    for i in range(line):
        t=sum(posibility[i][:])
        for j in range(col):
            posibility[i][j]/=t
            print(sum(posibility[i][:]))
    
    f=open('res.csv','w')
    for i in range(line):
        for j in range(col):
            f.write(str(posibility[i][j]))
            f.write(',')
        f.write('\n')
            
            
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



#def calcos(VALTFIDF,TFIDF,K,train_mood_po):
#    line=len(VALTFIDF)
#    col=len(VALTFIDF[0])
#    
#    num=len(TFIDF)
#    
#    posibility=np.zeros((line,6))
#
#    for i in range(line):
#        distance={}
#        for j in range(num):
#            res=0
#            len1=0
#            len2=0
#            for k in range(col):
#                len1+=np.power(VALTFIDF[i][k],2)
#                len2+=np.power(TFIDF[j][k],2)
#                res+=VALTFIDF[i][k]*TFIDF[j][k]
#            len1=np.power(len1,0.5)
#            len2=np.power(len2,0.5)
#            distance[j]=res/(len1*len2)
#        distance=sorted(distance.items(),key=lambda item:item[1],reverse=True)
#        for k in range(K):
#            index=distance[k][0]
#            for m in range(6):
#                posibility[i][m]+=train_mood_po[index][m]/(1-distance[k][1])
#        print(posibility[i])
#        
#    for i in range(line):
#        t=sum(posibility[i][:])
#        for j in range(6):
#            posibility[i][j]/=t
#        print(sum(posibility[i][:]))
#    f=open('res5.csv','w')
#    for i in range(line):
#        for j in range(6):
#            f.write(str(posibility[i][j]))
#            f.write(',')
#        f.write('\n')


def calcos(VALTFIDF,TFIDF,K,train_mood_po):
    line=len(VALTFIDF)
    col=len(VALTFIDF[0])
    
    num=len(TFIDF)
    
    posibility=np.zeros((line,6))

    for i in range(line):
        distance={}
        for j in range(num):
            res=0
            for k in range(col):
                res+=np.power(VALTFIDF[i][k]-TFIDF[j][k],2)
            distance[j]=res
        distance=sorted(distance.items(),key=lambda item:item[1])
        print(distance)
        for k in range(K):
            index=distance[k][0]
            print(index)
            for m in range(6):
                posibility[i][m]+=train_mood_po[index][m]/(distance[k][1])
        
        
    for i in range(line):
        t=sum(posibility[i][:])
        for j in range(6):
            posibility[i][j]/=t
    print(posibility)
    f=open('res_test.csv','w')
    for i in range(line):
        for j in range(6):
            f.write(str(posibility[i][j]))
            f.write(',')
        f.write('\n')    
        
        
    

if __name__ == "__main__":
    word_list=[]
    
    word_map={}
    
    get_word_list(word_list,TRAINNAME,word_map)
    
    get_word_list(word_list,VALINAME,word_map)
    
#    print(word_list)
#    
#    print(word_map)
    
    train_mood_po=get_mood_po(TRAINNAME)
#    
    train_onehot=get_onehot(TRAINNAME,word_list,word_map)
#    
    val_onehot=get_onehot(VALINAME,word_list,word_map)
#    
#    topK,topK_dis=get_topK(4,val_onehot,train_onehot)
    
    #print(topK)
#    get_pos(topK,train_mood_po,topK_dis)
    
    line=len(train_onehot)
    
    col=len(train_onehot[0])
    
#    cal_distance(onehot,val_onehot,mood_map,val_mood_map,int(np.power(line,0.5)))
    
    TF=get_TF(line,col,TRAINNAME,word_list,word_map)
    
    IDF=get_IDF(col,TRAINNAME,line,word_list)
    
    TFIDF=get_TFIDF(TF,IDF,line,col)
    
#    print(TFIDF)
    
    line=len(val_onehot)
    
    VALTF=get_TF(line,col,VALINAME,word_list,word_map)
    
    VALIDF=get_IDF(col,VALINAME,line,word_list)
    
    VALTFIDF=get_TFIDF(VALTF,VALIDF,line,col)
    
    
    calcos(VALTFIDF,TFIDF,3,train_mood_po)
    
    

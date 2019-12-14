#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:02:19 2019

@author: liusiyu
"""

import numpy as np
import math
FILENAME = "semeval.txt"

word_list=[]

def get_word_list(FILENAME):
    word_list=[]
    word_map={}
    file=open(FILENAME,'r')
    f=file.readlines()
    index=0
    for line in f:
        line=line.strip()
        line_list=line.split('\t')
        line_word=line_list[2].split(' ')
        for word in line_word:
            if word not in word_list:
                word_list.append(word)
                word_map[word]=index
                index+=1
    return word_list,len(f),word_map 

            
def get_TF(line,col,FILENAME,word_list,word_map):
    TF=np.zeros((line,col))
    file=open(FILENAME,'r')
    f=file.readlines()
    index=0
    for line in f:
        line=line.strip()
        line_list=line.split('\t')
        line_word=line_list[2].split(' ')
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
    D=len(f)
    for word in word_list:
        s=0
        for line in f:
            line=line.strip()
            line_list=line.split('\t')
            line_word=line_list[2].split(' ')
            if word in line_word:
                s+=1
        IDF[index]=math.log(D/(1+s))
        index+=1
    return IDF
        
        

def get_TFIDF(TF,IDF,line,col):
    TFIDF=np.zeros((line,col))
    for i in range(line):
        for j in range(col):
            TFIDF[i][j]=TF[i][j]*IDF[j]
    return TFIDF


            
if __name__ == "__main__":
    word_list,line,word_map=get_word_list(FILENAME)
    col=len(word_list)
    TF=get_TF(line,col,FILENAME,word_list,word_map)
    IDF=get_IDF(col,FILENAME,line,word_list)
    TFIDF=get_TFIDF(TF,IDF,line,col)
    print(TFIDF)
    res=np.savetxt('res.txt',TFIDF)
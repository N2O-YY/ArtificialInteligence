import codecs
import string
import numpy as np
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Lambda,Dense,Dropout
import keras.backend as K
from keras.optimizers import Adadelta,SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model,Sequential
from keras.models import Model
from keras.callbacks import LearningRateScheduler
import math
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau

def step_decay():
    initial_lrate = 2
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def load_sts(train_file):
    with codecs.open(train_file, 'r', encoding='utf8') as f:
        train_left = []
        train_right = []
        ylabel = []
        for line in f:
            line = line.strip().split('\t')
            score = float(line[4])
            sa, sb = line[5], line[6]
            ylabel.append(score)
            train_left.append(sa)
            train_right.append(sb)
    print('left length: ',len(train_left))
    print('right length: ',len(train_right))
    return train_left,train_right,ylabel

def make_dic(train_left, train_right):
    dic = {}
    for data in [train_left,train_right]:
        for line in  data:
            words = line.split(" ")
            for i,word in enumerate(words):
                if word not in dic:
                    dic[word]=len(dic)
    return dic

def pad_seq(left,right):
    max_len = 0
    for line in left:
        if max_len < len(line):
            max_len = len(line)

    for line in right:
        if max_len < len(line):
            max_len = len(line)
    
    pad_sequences(left,maxlen=max_len)
    return pad_sequences(left,maxlen=max_len),pad_sequences(right,maxlen=max_len),max_len


# 采用w2v作为单词到多维空间的映射
def get_embedding(dic,size=256,path='w2v'):
    embedding_array = np.random.rand(len(dic)+1,size)
    embedding_array[0] = 0
    w2v = gensim.models.Word2Vec.load(path)
    for index,word in enumerate(dic):
        if word in w2v.wv.vocab:
            embedding_array[index]=w2v.wv.word_vec(word)

    del w2v
    return embedding_array

def word2int(lines, dic):
    out = []
    for line in lines:
        words = line.split(" ")
        for i, word in enumerate(words):
            words[i] = dic[word]
        out.append(words)
    return out

def exponent_neg_manhattan_distance(left, right):
    return 5*K.exp(-K.abs(left - right))

# def exponent_neg_manhattan_distance(left, right):
#     return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

def y2op(y):
    ma = max(y)
    for index in range(len(y)):
        y[index] = y[index]/ma
    return y

def get_model(maxlen,embedding_array,emb_size):
    
    hidden = 150
    norm = 1.25
    left_input = Input(shape=(maxlen,),dtype='int32')
    right_input = Input(shape=(maxlen,),dtype='int32')

    embedding_layer = Embedding(len(embedding_array),emb_size,weights=[embedding_array],input_length=maxlen,trainable = False)

    ecoded_left = embedding_layer(left_input)
    ecoded_right = embedding_layer(right_input)
    print(ecoded_left.shape)

    share_lstm = LSTM(hidden)

    left_out = share_lstm(ecoded_left)
    right_out = share_lstm(ecoded_right)

    # malstm_distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
    #                     output_shape=lambda x: (x[0][0], 1))([left_out, right_out])

    malstm_distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]))([left_out, right_out])

    print(malstm_distance.shape)

    d = Dense(128,input_shape=(50,),activation='relu')
    dout = d(malstm_distance)

    dro0 = Dropout(0.5)
    dout0 = dro0(dout)

    d1 = Dense(1,input_shape=(128,))
    dout1 = d1(dout0)

    # d2 = Dense(1,input_shape=(64,))
    # dout2 = d2(dout1)

    model = Model([left_input,right_input],[dout1])
    
    optimi = Adadelta(lr=2,clipnorm = norm)
    model.compile(loss='mean_squared_error',optimizer=optimi,metrics=['accuracy'])
    model.summary()
    return model
    
def drop_stop(lines, stop):
    out = []
    for line in lines:
        for word in stop:
            line = line.replace(" "+word," ")
        out.append(line)
    return out

def get_stop():
    out = []
    with open('百度停用词表.txt','r',encoding='utf-8') as f0:
        for line in f0:
            out = line.replace("\n",'')
    f0.close()
    return out

if __name__ == "__main__":
    path = 'Stsbenchmark/stsbenchmark/sts-test.csv'
    logpath = './keras_log'
    left,right,y = load_sts(path)
    stop = get_stop()
    # left = drop_stop(left,stop)
    # right = drop_stop(right,stop)
    # y = y2op(y)
    dic = make_dic(left,right)
    id2w={dic[w]:w for w in dic}
    emb = get_embedding(dic)
    left = word2int(left,dic)
    right = word2int(right,dic)
    left,right,maxlen = pad_seq(left,right)

    ll = int(0.8*len(left))

    left_train = left[:ll]
    right_train = right[:ll]
    y_train = y[:ll]

    va_left = left[ll:]
    va_right = right[ll:]
    va_y = y[ll:]
    
    batch_size = 64
    epoch = 1000
    model = get_model(maxlen,emb,256)
    re_de = ReduceLROnPlateau(patience=5,min_lr=0.1)
    model.fit(x=[np.asarray(left_train),np.asarray(right_train)],y=y_train,batch_size=batch_size,epochs=epoch,validation_data=([np.asarray(va_left), np.asarray(va_right)], va_y), callbacks=[TensorBoard(log_dir='./tmp/log_has_stop_1000', histogram_freq=1,write_grads=True)])
    # tensorboard --logdir=./tmp/log

from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
from keras.layers.normalization import BatchNormalization
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
savedStdout = sys.stdout  #保存标准输出流
with open('out_CNN.txt', 'a+') as file:
    sys.stdout = file  #标准输出重定向至文件
    print('This message is for file!')
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_train_data(filename):

        for index, name in enumerate(filename):
            res = unpickle(name)
            if index == 0:
                x_train = res[b'data']
                y_train = res[b'labels']
            else:
                x_train = np.vstack((x_train, res[b'data']))
                y_train.extend(res[b'labels'])
            print(type(x_train))
            print(x_train.shape)
            print(type(y_train))
            print(len(y_train))
        return (x_train, y_train)

    if __name__ == "__main__":
        batch_size = 128
        epochs = 50
        num_classes = 10
        img_rows = 32
        img_cols = 32
        input_shape = (img_rows, img_cols, 3)

        filenames = ["data_batch_{}".format(str(i)) for i in range(1, 6)]
        (x_train, y_train) = get_train_data(filenames)

        test_file = ["test_batch"]
        (x_test, y_test) = get_train_data(test_file)

        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_train = np.array([x.transpose(1, 2, 0) for x in x_train])
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        x_test = np.array([x.transpose(1, 2, 0) for x in x_test])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model = Sequential()
        model.add(
            Conv2D(32,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        model.fit(x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
        #model.predict()
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

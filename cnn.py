from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras import backend as K

def k_f1_score(y_true,y_pred):
    y_pred = K.argmax(y_pred,axis=1)
    y_true = K.argmax(y_true,axis=1)
    pass



def larger_model(num_classes,shape):
    model = Sequential()
    print('input_shape:{}'.format(shape))
    model.add(Conv2D(128,(5,5),input_shape=shape,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
    #model.compile(loss='sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
    return model

"""
def main():
    seed=1
    np.random.seed(seed)
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print(f'X_train shape:{X_train.shape}\nX_test shape:{X_test.shape}\ny_train shape:{y_train.shape}\ny_test shape:{y_test.shape}')
    #build model
    model = larger_model()
    #fit the model
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=15,batch_size=32)
    #Final evaluation of the model
    scores = model.evaluate(X_test,y_test,verbose=0)
    print(scores)
"""

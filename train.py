from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from cnn import larger_model
import pickle
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import numpy as np
def load_data(file_path):
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    return data






def train():
    X,y = load_data('data/ecg_data.pkl')
    print(X.shape)
#    X = X.reshape(X.shape[0],X.shape[1],X.shape[2]).astype('float32')
    print(X.shape)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    num_classes = 4
    shape = X[0].shape
    print('x_train:{0}\nX_test:{1}\ny_train:{2}\ny_test{3}\n'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
    model = larger_model(num_classes,shape)
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=32)
    model.save('cnn.model')
    scores = model.evaluate(X_test,y_test,verbose=0)
    print('Accuracy:{}'.format(scores))
    y_predict = model.predict(X_test)
    y_predict = np.argmax(y_predict,axis=1).tolist()
    y_actual = np.argmax(y_test,axis=1).tolist()
    print('confusion matrix:')
    print(confusion_matrix(y_actual,y_predict))



if __name__ == "__main__":
    train()



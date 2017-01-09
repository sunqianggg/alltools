#encoding:utf8
'''
compare NN to another classify approaches like (svm,regression).
'''

from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from sklearn import svm
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

nb_classes = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def clf_svm():
    Y_train_list=[list(item).index(1.0) for item in Y_train]
    Y_test_list=[list(item).index(1.0) for item in Y_test]
    clf=svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train,Y_train_list)
    print(clf.score(X_test,Y_test_list))

def clf_nn():
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD, Adam, RMSprop

    batch_size = 128
    nb_epoch = 20

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__=="__main__":
    for clf in (clf_nn,clf_svm):
        print(clf.__name__)
        from time import time
        start=time()
        clf()
        end=time()
        print(end-start)
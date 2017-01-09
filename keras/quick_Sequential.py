# encoding:utf8

from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential([Dense(32, input_dim=784), Activation(
    'relu'), Dense(10), Activation('softmax'), ])
# 也可以通过.add()方法一个个的将layer假如模型中
"""
model=Sequential()
model.add(Dense(32,input_dim=784))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('softmax'))
"""

# 指定输入数据的shape

# 下面三种方法等价
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, batch_input_shape=(None, 784)))

model = Sequential()
model.add(Dense(32, input_dim=784))

# Merge层
# 多个Sequential可经由一个Merge层合并到一个输出。
from keras.layers import Merge

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')
# 也可以为Merge层提供关键字参数'mode'，以实现任意的变换
merged = Merge([left_branch, right_branch], model=lambda x: x[0] - x[1])

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='softmax'))

# 编译
# compile接受三个参数,优化器:optimizer,损失函数:loss,指标列表:metircs

# for a multi-class classification problem
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# for a binary classification problem
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# for a mean squared error regression problem
model.compile(optimizer='rmsprop', loss='mse')

# for a custom metircs
import keras.backend as K


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def false_rates(y_true, y_pred):
    # false_neg = ...
    # false_pos = ...
    return {
        'false_neg': false_neg,
        'false_pos': false_pos,
    }
model.compile(
    optimizer='rmprop',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        mean_pred,
         false_rates])

#训练
#for a single-input model with 2 classes
model=Sequential()
model.add(Dense(1,input_dim=784,activation='sigmod'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#generate dummy data
import numpy as np
data=np.random.random((1000,784))
labels=np.random.randint(2,size=(1000,1))

#train the model,iterating on the data in batches
model.fit(data,labels,nb_epoch=10,batch_size=32)
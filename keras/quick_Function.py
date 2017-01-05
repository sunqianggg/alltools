#encoding:utf8

#全连接模型
from keras.layers import Input,Dense
from keras.models import Model

#this returns a tensor
inputs=Input(shape=(784,))

#a layer instance is callable on a tensor, and returns a tensor
x=Dense(64,activation='relu')(inputs)
x=Dense(64,activation='relu')(x)
predictions=Dense(10,activation='softmax')(x)

#this creates a model that includes
#the Input layer and three Dense layers

model=Model(input=inputs,output=predictions)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(data,labels)


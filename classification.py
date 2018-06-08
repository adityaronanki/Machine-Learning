# Machine-Learning  classification of apple logo and apple fruit

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
from parser import load_data

training_data=load_data('data/training')
validate_data=load_data('data/validation')

model=Sequential()
model.add(Convolution2D(32,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flattern())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy ,optimizer='rmsprop',metrics=['acuracy'])

model.fit_generator(training_data,nb_epoch=30,validate_data=validate_data,nb_val_samples=32)
img=cv2.imread('test/apple.jpg')
img=img.resize((224, 224), Image.NEAREST)
prediction=model.predict(img)
print prediction;

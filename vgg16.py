from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Data loading and preprocessing
#import tflearn.datasets.oxflower17 as oxflower17
#X, Y = oxflower17.load_data(one_hot=True)
# (1) Importing dependency
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)



from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt 
%matplotlib inline

#train_path='C:/Users/hector/Desktop/machine_learning/alexnetflowers/train'
train_path='../jesusespinoza/Downloads/alexnetflowers/train'
train_batches =ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['daisy','dandelion','rose','sunflower','tulip'],batch_size= 10)

validation_path='../jesusespinoza/Downloads/alexnetflowers/test'
validation_batches =ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['daisy','dandelion','rose','sunflower','tulip'],batch_size= 10)

imgs,labels = next(train_batches)
# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
#model.fit(X, Y, n_epoch=500, shuffle=True,
 #         show_metric=True, batch_size=32, snapshot_step=500,
  #        snapshot_epoch=False, run_id='vgg_oxflowers17')


model.fit_generator(train_batches, steps_per_epoch=12,validation_data= validation_batches,validation_steps=1,epochs = 1, verbose=1)

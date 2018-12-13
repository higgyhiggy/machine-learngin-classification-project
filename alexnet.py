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

train_path='C:/Users/hector/Desktop/machine_learning/alexnetflowers/train'
train_batches =ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['daisy','dandelion','rose','sunflower','tulip'],batch_size= 10)

validation_path='C:/Users/hector/Desktop/machine_learning/alexnetflowers/test'
validation_batches =ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['daisy','dandelion','rose','sunflower','tulip'],batch_size= 10)

imgs,labels = next(train_batches)


# (2) Get Data
import tflearn.datasets.oxflower17 as oxflower17
#x, y = oxflower17.load_data(one_hot=True)

# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=128, input_shape=(224,224,3), kernel_size=(3,3),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=2048, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer17
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()

# (4) Compile /keras definition youtub/ adam is our optimizer "sgd rms or several other "
#loss how its calculated mean square error 
#metrics to judege prefromance of model 
model.compile(loss='categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

# (5) Train/ 
#fit defined in youtube /split percent of data that is saved for validation/ verbose how much info displayed /shuffle between each epochs images shuffled 
#different levels of verbose depending on what i want to see or not 
#model.fit(x, y, batch_size=64, epochs=1, verbose=1, \
#validation_split=0.2, shuffle=True)
model.fit_generator (train_batches, steps_per_epoch=12,validation_data= validation_batches,validation_steps=1,epochs = 1, verbose=1)


# saved our model 
#model.save('mlflowers.h5')


#from keras.models import load_model
#new_model = load_model('mlflowers.h5')

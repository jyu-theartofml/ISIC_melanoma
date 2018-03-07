
#### traning segnet autoencoder to generate mask for image and saves it in masks folder

import numpy as np
import cv2
import pandas as pd
import os

import tensorflow

import keras
keras.__version__
##if necessary, upgrade keras pip install keras --upgrade
from keras.models import Model,load_model
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History

K.set_image_dim_ordering('tf')

ROWS=200
COLS=200

#### the image numpy array was generated from running segmentation_01_preprocessing.py
img_train = np.load('train_imgs_array.npy')
mask_train= np.load('train_mask_array.npy')
img_validation = np.load('val_imgs_array.npy')
mask_validation= np.load('val_mask_array.npy')
##### normalize image arrays with mean and std
mean_val=np.mean(img_train)
sd_val=np.std(img_train)

img_train-=mean_val
img_train/=sd_val
img_validation-=mean_val
img_validation/=sd_val

training_img_set=img_train
training_mask_set=np.expand_dims(mask_train, axis=3)

validation_img_set=img_validation
validation_mask_set=np.expand_dims(mask_validation, axis=3)

print('image shape:', training_img_set.shape)
print('mask shape:', training_mask_set.shape)
print('val image shape:', validation_img_set.shape)
print('val mask shape:', validation_mask_set.shape)


#### define Dice loss function
def dice_loss(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


### Segnet autoencoder model

def segnet():
    input_img = Input(shape=(ROWS, COLS,3))
    ############ encoder ###########
    conv1 = Conv2D(16, (3, 3), padding='same')(input_img) #originally 16
    conv1=BatchNormalization()(conv1)
    conv1=Activation('relu')(conv1)
    conv1 = Conv2D(16, (3, 3), padding='same')(conv1)
    conv1=BatchNormalization()(conv1)
    conv1=Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(32, (3, 3), padding='same')(pool1) #originaly 32
    conv2=BatchNormalization()(conv2)
    conv2=Activation('relu')(conv2)
    conv2=Conv2D(32, (3, 3), padding='same')(conv2)
    conv2=BatchNormalization()(conv2)
    conv2=Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(64, (3, 3), padding='same')(pool2)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

    ########## decoder #############

    up4=UpSampling2D((2,2))(pool3)
    up4=Conv2D(64, (3,3), padding='same',activation='relu')(up4)
    up4=BatchNormalization()(up4)
    up4=Activation('relu')(up4)
    up4=Conv2D(64, (3,3), padding='same',activation='relu')(up4)
    up4=BatchNormalization()(up4)
    up4=Activation('relu')(up4)
    up4=Conv2D(64, (3,3), padding='same',activation='relu')(up4)
    up4=BatchNormalization()(up4)
    up4=Activation('relu')(up4)

    up5=UpSampling2D((2,2))(up4)
    up5=Conv2D(32, (3,3), padding='same',activation='relu')(up5)
    up5=BatchNormalization()(up5)
    up5=Activation('relu')(up5)
    up5=Conv2D(32, (3,3), padding='same',activation='relu')(up5)
    up5=BatchNormalization()(up5)
    up5=Activation('relu')(up5)

    up6=UpSampling2D((2,2))(up5)
    up6=Conv2D(16, (3,3), padding='same',activation='relu')(up6)
    up6=BatchNormalization()(up6)
    up6=Activation('relu')(up6)
    up6=Conv2D(16, (3,3), padding='same',activation='relu')(up6)
    up6=BatchNormalization()(up6)
    up6=Activation('relu')(up6)

    decoded=Conv2D(1, (1, 1), activation='sigmoid')(up6)
    autoencoder = Model(inputs=[input_img], outputs=[decoded])

    return autoencoder

model=segnet()
model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss=dice_loss)

early_stopping =EarlyStopping(monitor='val_loss', patience=4)
model_checkpoint = ModelCheckpoint('segnet_training_weights9_sgd.h5', monitor='val_loss', save_best_only=True,save_weights_only=True)


from sklearn.utils import shuffle
train_shuffled, train_mask_shuffled = shuffle(training_img_set, training_mask_set, random_state=12)
val_shuffled, val_mask_shuffled = shuffle(validation_img_set, validation_mask_set, random_state=12)


seg_model=model.fit(train_shuffled, train_mask_shuffled, batch_size=16, epochs=40, verbose=1, shuffle=True,
              validation_data=(val_shuffled, val_mask_shuffled),callbacks=[model_checkpoint,early_stopping])

################### evaluate on validation set ##################
model.load_weights('segnet_training_weights9_sgd.h5')
test_pred = model.predict(validation_img_set, batch_size=1, verbose=1)

test_masks=test_pred
true_masks=validation_mask_set

########## calculate total Dice Coefficient as a measure of similarity between predicted mask and true mask
true_mask_f = true_masks.flatten()
test_masks_f = np.around(test_masks.flatten())
smooth=1
intersection = np.sum((true_mask_f) *(test_masks_f))
dice=((2. * intersection + smooth) / (np.sum((true_mask_f)) + np.sum((test_masks_f)) + smooth))
print(dice)

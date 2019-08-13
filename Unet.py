import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from itertools import izip
import os
from data_preprocess import image_processing
from sklearn.cross_validation import train_test_split


import keras
keras.__version__

from keras.models import Model,load_model
from keras.layers import Input, concatenate,Activation, Conv2D, MaxPooling2D,UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History

K.set_image_dim_ordering('tf')
K.image_dim_ordering()


#### get array from image and mask
lesion_arr, mask_arr = image_processing(img_path, mask_path)

print(lesion_arr.shape)
print(mask_arr.shape)

############## optional: take a look at the image ################
plt.subplot(1,2,1)
plt.imshow(lesion_arr[2][:,:,:],cmap='gray')
plt.subplot(1,2,2)
plt.imshow(mask_arr[2][:,:,:],cmap='gray')



r_mean=np.mean(lesion[:,:,0])
g_mean=np.mean(lesion[:,:,1])
b_mean=np.mean(lesion[:,:,2])

lesion_arr[:,:,0] -= r_mean
lesion_arr[:,:,1] -= g_mean
lesion_arr[:,:,2] -= b_mean
arr_std=np.std(lesion_arr)
lesion_arr /= arr_std

########### split into training and test ##################
train_img_set=np.expand_dims(lesion_arr, 4)

X_train, X_test, y_train, y_test = train_test_split(lesion_arr, mask_arr, test_size=0.2, random_state=22)

################################### Build 3D model architecture (memory intensive!) ##############################
def isic_unet():
    inputs = Input((ROWS, COLS,3,1))
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4=Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4=Conv2D(512, (3,3), activation='relu', padding='same')(conv4)

        #expansive/synthesis path
    up5 = concatenate([Conv2D(512, (3,3), activation='relu', padding='same')(UpSampling2D((2,2))(conv4)), conv3], axis=4)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2D(256, (3,3), activation='relu', padding='same')(UpSampling2D((2,2))(conv5)), conv2], axis=4)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2D(128, (3,3), activation='relu', padding='same')(UpSampling2D((2,2))(conv6)), conv1], axis=4)
    conv7= Conv2D(64, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(1, (1,1),activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model

model=isic_unet()
model.compile(optimizer=Adam(lr=1e-3), loss=dice_loss, metrics=[f1])
early_stopping =EarlyStopping(monitor='val_loss', patience=4)
model_checkpoint = ModelCheckpoint('Unet_training_weights.h5', monitor='val_loss', save_best_only=True,save_weights_only=True)
hist=model.fit(X_train, y_train , batch_size=1, epochs=10, verbose=1, shuffle=True,
              validation_data=(X_test, y_test),callbacks=[model_checkpoint,early_stopping])

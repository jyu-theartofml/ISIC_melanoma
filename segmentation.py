import numpy as np
import cv2
import pandas as pd
import os
import glob
import tensorflow
import matplotlib.pyplot as plt
import keras
import sys

from keras.models import Model,load_model
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_dim_ordering('tf')

###image_file is string of image path###########


ROWS=200
COLS=200

image_path=sys.argv[1]
image_files=glob.glob(image_path+'*.jpg')
count=len(image_files)
resized_array= np.ndarray((count, ROWS, COLS, 3 ))
base_names=[]
img_train = np.load('train_imgs_array.npy')
mean_val=np.mean(img_train)
sd_val=np.std(img_train)

for i, image_file in enumerate(image_files):
    base_name=image_file.split("/")[1].split(".")[0]
    base_names.append(base_name)
    img_fullsize=cv2.imread(image_file)
    img_resized = cv2.resize(img_fullsize, dsize=(ROWS, COLS), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img_resized-=mean_val
    img_resized/=sd_val
    resized_array[i] = img_resized
#return resized_array



def dice_loss(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def segnet():
    input_img = Input(shape=(ROWS, COLS,3))
    ############ encoder ###########
    conv1 = Conv2D(16, (3, 3), padding='same')(input_img)
    conv1=BatchNormalization()(conv1)
    conv1=Activation('relu')(conv1)
    conv1 = Conv2D(16, (3, 3), padding='same')(conv1)
    conv1=BatchNormalization()(conv1)
    conv1=Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(32, (3, 3), padding='same')(pool1)
    conv2=BatchNormalization()(conv2)
    conv2=Activation('relu')(conv2)
    conv2=Conv2D(32, (3, 3), padding='same')(conv2)
    conv2=BatchNormalization()(conv2)
    conv2=Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3=BatchNormalization()(conv3)
    conv3=Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

    ########## decoder #############

    up4=UpSampling2D((2,2))(pool3)
    up4=Conv2D(128, (3,3), padding='same',activation='relu')(up4)
    up4=BatchNormalization()(up4)
    up4=Activation('relu')(up4)
    up4=Conv2D(128, (3,3), padding='same',activation='relu')(up4)
    up4=BatchNormalization()(up4)
    up4=Activation('relu')(up4)
    up4=Conv2D(128, (3,3), padding='same',activation='relu')(up4)
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
model.compile(optimizer=SGD(lr=5e-4, momentum=0.9), loss=dice_loss)
model.load_weights('segnet_training_weights8_sgd.h5')
predict_mask = model.predict(resized_array, verbose=1)

def resize_mask(test_pred):
    for i in range(len(test_pred)):
        test_mask=np.resize(test_pred[i], (test_pred.shape[1], test_pred.shape[2]))
        img_fullsize=cv2.imread(image_files[i])
        test_mask_fullsize=cv2.resize(test_mask, (img_fullsize.shape[1],img_fullsize.shape[0]))
                ##################### find contours and filter ############
        t,binary_mask=cv2.threshold(test_mask_fullsize, 0.5,1,cv2.THRESH_BINARY)
        (_,contours,_) = cv2.findContours(np.uint8(binary_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                ########## in case of 'noisy' mask, this finds the right one for skin lesion by selecting the largest contour #####
        max_cnt= sorted(contours, key = cv2.contourArea, reverse = True)[:1]
        filtered_mask=np.uint8(np.zeros((img_fullsize.shape[0],img_fullsize.shape[1])))
        h=cv2.drawContours(filtered_mask, max_cnt, -1, (255, 0,0), -1)
                #### save the mask image in folder masks/ ########
        save_mask_name='masks/'+base_names[i]+'_segmentation.png'
        #print(save_mask_name)
        plt.imsave(save_mask_name, filtered_mask, cmap='gray')


resize_mask(predict_mask)


import pandas as pd
import numpy as np
import os
import cv2
import glob
from sklearn.utils import shuffle

import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Model,load_model,Sequential
from keras.layers import Input,Activation, Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from skimage import exposure, img_as_ubyte

from keras import backend as K

################# Getting image numpy arrays ###################
train_data_dir='train_masked/'
validation_data_dir='validation_masked/'

train_benign=glob.glob(train_data_dir+'/benign_masked/*')
train_malignant=glob.glob(train_data_dir+'/malignant_masked/*')
train_images=train_benign+train_malignant

validation_benign=glob.glob(validation_data_dir+'/benign_masked/*')
validation_malignant=glob.glob(validation_data_dir+'/malignant_masked/*')
validation_images=validation_benign+validation_malignant

ROWS=192
COLS=192

def image_array(file_path, count):
    data = np.ndarray((count, ROWS, COLS, 3 ))
    for i, image_file in enumerate(file_path):
        img=cv2.resize(cv2.imread(image_file), (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        data[i] = img

    return data

train_array=image_array(train_images, len(train_images))
validation_array=image_array(validation_images, len(validation_images))

######################  Load pretrained model, then extract bottleneck ######################
base_model = MobileNet(input_shape=(192,192,3), include_top=False,weights='imagenet')

def bottleneck(processed_array, save_file_name): ## save_file_name is in string format
    scaled_img=preprocess_input(processed_array)
    bottleneck_features=base_model.predict(scaled_img)
    np.save(save_file_name+'.npy', bottleneck_features)
    
    return bottleneck_features
   
bottleneck_features_train = bottleneck(train_array, 'bottleneck_features_train')
bottleneck_features_validation = bottleneck(validation_array, 'bottleneck_features_val')

######################   Use the bottleneck features to train top model ########################
train_data = np.load('bottleneck_features_train.npy')
train_labels = np.array([0] * (1626) + [1] * (1122))

validation_data = np.load('bottleneck_features_val.npy')
validation_labels = np.array([0] * (120) + [1] * (60))

### Define F1 score as a metric (this is the F1 function used in older version of keras) ###
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.

        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

###########  Define top layer fully connected model #############
def Top_model(input_dim):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_dim))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.6))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.6))
    top_model.add(Dense(1, activation='sigmoid'))

    return top_model

model2=Top_model(train_data.shape[1:])
model2.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy',f1])


model2.fit(train_data, train_labels,
              epochs=5, batch_size=32,verbose=1,
              validation_data=(validation_data, validation_labels))

model2.save_weights('bottleneck_fc_model(3layer)_512size_mobilenet.h5')

################### Assemble full model #########################

top_model2 = Top_model(base_model.output_shape[1:])
top_model2.load_weights('bottleneck_fc_model(3layer)_512size_mobilenet.h5')
model = Model(input=base_model.input, output=top_model2(base_model.output))

for layer in model.layers[:4]:
    layer.trainable=False

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy',f1])

############### Data augmentation for model training ##################
train_datagen = ImageDataGenerator(
        height_shift_range=0.2,
        shear_range=0.4,
        rotation_range=45,
        preprocessing_function=preprocess_input)
validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            classes=['benign_masked','malignant_masked'],
            target_size=(ROWS, COLS),
            batch_size=32,
            class_mode='binary',
            shuffle='True')

validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            classes=['benign_masked','malignant_masked'],
            target_size=(ROWS, COLS),
            batch_size=32,
            class_mode='binary',
            shuffle='True')

################# Monitor training using callbacks ##########################
early_stopping =EarlyStopping(monitor='val_loss', patience=4)
model_checkpoint = ModelCheckpoint('training_weights_mobilenet_bottlneck3.h5', monitor='val_loss', save_best_only=True,save_weights_only=True)
model.load_weights('training_weights_mobilenet_bottlneck3.h5') #optional, load previous weights
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy',f1])

model_full=model.fit_generator(train_generator,
                           steps_per_epoch=86,
                           validation_data=validation_generator,
                           validation_steps=1,
                           callbacks=[early_stopping, model_checkpoint],
                           epochs=20,verbose=1)

model.evaluate_generator(validation_generator)

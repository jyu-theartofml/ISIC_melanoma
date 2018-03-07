
# # Training densenet bottleneck for 3 class classification: benign, melanoma,and seborrheic keratosis
# dataset is divided from total images
import pandas as pd
import numpy as np
import os
import glob
from sklearn.utils import shuffle
import keras

keras.__version__
#need current version of keras to run newer models

from keras.applications.densenet import DenseNet121
from keras.models import Model,load_model,Sequential
from keras.layers import Input,Activation, Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img, array_to_img
from keras.regularizers import l2
from keras.applications.densenet import preprocess_input
from keras.utils import plot_model,to_categorical

from keras import backend as K
K.image_dim_ordering()
K.tensorflow_backend._get_available_gpus()

### list data directory names
train_data_dir='train_total/'
validation_data_dir='val_total/'
test_data_dir='test_total/'


####  Process image files
train_benign=glob.glob(train_data_dir+'benign/*')
train_melanoma=glob.glob(train_data_dir+'melanoma/*')
train_sk=glob.glob(train_data_dir+'sk/*')
train_images=train_benign+train_melanoma+train_sk

print('train_benign: %s' %len(train_benign))
print('train_melanoma: %s' %len(train_melanoma))
print('train_sk: %s' %len(train_sk))

validation_benign=glob.glob(validation_data_dir+'benign/*')
validation_melanoma=glob.glob(validation_data_dir+'melanoma/*')
validation_sk=glob.glob(validation_data_dir+'sk/*')
validation_images=validation_benign+validation_melanoma+validation_sk

print('validation_benign: %s' %len(validation_benign))
print('validation_melanoma: %s' %len(validation_melanoma))
print('validaiton_sk: %s' %len(validation_sk))

test_benign=glob.glob(test_data_dir+'benign/*')
test_melanoma=glob.glob(test_data_dir+'melanoma/*')
test_sk=glob.glob(test_data_dir+'sk/*')
test_images=test_benign+test_melanoma+test_sk

print('test_benign: %s' %len(test_benign))
print('test_melanoma: %s' %len(test_melanoma))
print('test_sk: %s' %len(test_sk))


ROWS=256
COLS=256

def image_array(file_path, count):
    data = np.ndarray((count, ROWS, COLS, 3 ))
    for i, image_file in enumerate(file_path):
        raw_img=load_img(image_file, target_size=(ROWS, COLS))
        img=img_to_array(raw_img)
        data[i] = img

    return data


train_array=image_array(train_images, len(train_images))
validation_array=image_array(validation_images, len(validation_images))
test_array=image_array(test_images, len(test_images))

np.save('train_array_size256.npy', train_array)
np.save('validation_array_size256.npy', validation_array)
np.save('test_array_size256.npy', test_array)

train_preprocessed=preprocess_input(train_array)
validation_preprocessed=preprocess_input(validation_array)
test_preprocessed=preprocess_input(test_array)

base_model = DenseNet121(input_shape=(256,256,3), include_top=False,weights='imagenet')

### Train Bottlenecks

bottleneck_features_train = base_model.predict(train_preprocessed, verbose=1)
#save as numpy array,
np.save('bottleneck_features_train_256size_densenet.npy', bottleneck_features_train)

bottleneck_features_validation = base_model.predict(validation_preprocessed, verbose=1)
#save as numpy array,
np.save('bottleneck_features_val_256size_densenet.npy', bottleneck_features_validation)

bottleneck_features_test = base_model.predict(test_preprocessed, verbose=1)
#save as numpy array,
np.save('bottleneck_features_test_256size_densenet.npy', bottleneck_features_test)


train_data = np.load('bottleneck_features_train_256size_densenet.npy')
train_labels = np.array([0] * (1320) + [1] * (1168)+[2]*(890)) #label indices corresponds to alphanumeric order

validation_data = np.load('bottleneck_features_val_256size_densenet.npy')
validation_labels = np.array([0] * (327) + [1] * (130)+[2]*(88))

test_data = np.load('bottleneck_features_test_256size_densenet.npy')
test_labels = np.array([0] * (196) + [1] * (59)+[2]*(45)) #label indices corresponds to alphanumeric order

### one-hot encoding of y-labels, first column is benign, second is melanoma, 3rd is sk
train_target=to_categorical(train_labels, num_classes=3)
validation_target=to_categorical(validation_labels, num_classes=3)

X_train, y_train = shuffle(train_data, train_target, random_state=10)
X_val, y_val = shuffle(validation_data, validation_target, random_state=10)


def Top_model(input_dim):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_dim))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.6))

    top_model.add(Dense(3, activation='softmax'))

    return top_model

model2=Top_model(base_model.output_shape[1:])
model2.compile(optimizer=Adam(lr=0.00001),loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model2.fit(X_train, y_train, batch_size=32,
              epochs=50,verbose=1,shuffle=True,
              validation_data=(X_val, y_val))

model2.save_weights('top_model_1024size256_densenet.h5') #best yet

#### Assemble whole model
top_model2 = Top_model(base_model.output_shape[1:])
top_model2.load_weights('top_model_1024size256_densenet.h5')
model = Model(inputs=base_model.input, outputs=top_model2(base_model.output))

model.compile(optimizer=Adam(lr=0.00001),loss='categorical_crossentropy', metrics=['categorical_accuracy'])

### Evaluate performance on validation set
prediction=model.predict(validation_preprocessed, verbose=1)
validation_cat2=y_val[:,1]
prediction_cat2=prediction[:,1]

from sklearn.metrics import roc_auc_score,f1_score, roc_curve, auc, accuracy_score, confusion_matrix
print('AUC score: %f' %roc_auc_score(validation_cat2, prediction_cat2))
print('Accuracy score: %f' %accuracy_score(validation_cat2, np.round(prediction_cat2)))
print('F1 score: %f' %f1_score(validation_cat2, np.round(prediction_cat2)))

fpr, tpr, _ = roc_curve(validation_cat2, prediction_cat2)

#the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
#fpr=fp/(fp+tn)
#tpr=tp/(tp+fn)
confusion_matrix(validation_cat2, np.round(prediction_cat2))

### this script shows image augmentation preprocessing prior to training segmentation model
import numpy as np
import pandas as pd
import cv2
import os
import glob

img_row=200
img_col=200

def image_processing(img_path, mask_path):

    img_dataset=[]
    mask_dataset=[]


    for img_name in glob.glob(img_path+'/*.JPG'):
        base_name=os.path.basename(img_name).split('.')[0]
        img_fullsize=cv2.imread(img_name)
        img_resized = cv2.resize(img_fullsize, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img_dataset.append(img_resized)
        #perform same image augmentation (use same parameters for mask for deterministic augmentation)

        #Rotation
        M1 = cv2.getRotationMatrix2D((img_row/2,img_col/2),90,1)
        img_rotate= cv2.warpAffine(img_fullsize,M1,(img_row,img_col))
        img_dataset.append(img_rotate)

        #shearing
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M2 = cv2.getAffineTransform(pts1,pts2)
        img_shear= cv2.warpAffine(img_fullsize,M2,(img_fullsize.shape[1],img_fullsize.shape[0]))
        img_shear=cv2.resize(img_shear, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img_dataset.append(img_shear)


     # perform same transformation on mask files
        mask_name=mask_path+base_name+'_segmentation'+'.png'
        mask_fullsize=cv2.imread(mask_name,0)
        mask_resized=cv2.resize(mask_fullsize, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC)
        ret, mask_binary=cv2.threshold(mask_resized,0.5,1,cv2.THRESH_BINARY)
        mask_dataset.append(mask_binary)

        #need to run binary threshold again after augmentation
        mask_rotate= cv2.warpAffine(mask_binary,M1,(img_row,img_col))
        ret, mask_rotate=cv2.threshold(mask_rotate,0.6,1,cv2.THRESH_BINARY)
        mask_dataset.append(mask_rotate)

        ## shearing
        mask_shear= cv2.warpAffine(mask_fullsize,M2,(mask_fullsize.shape[1], mask_fullsize.shape[0]))
        mask_shear=cv2.resize(mask_shear, dsize=(img_row, img_col), interpolation=cv2.INTER_CUBIC)
        ret, mask_shear=cv2.threshold(mask_shear,0.5,1,cv2.THRESH_BINARY)
        mask_dataset.append(mask_shear)


    img_array=np.array(img_dataset)
    mask_array=np.array(mask_dataset)
    return img_array, mask_array

img_dir='ISIC-2017_Training_Data/ISIC-2017_Training_Data/'
mask_dir ='ISIC-2017_Training_Data_mask/ISIC-2017_Training_Part1_GroundTruth/'
val_img_dir='ISIC-2017_Validation_Data/ISIC-2017_Validation_Data/'
val_mask_dir='ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth/'

img_array, mask_array=image_processing(img_dir, mask_dir)
val_img_array, val_mask_array=image_processing(val_img_dir, val_mask_dir)

np.save('train_imgs_array.npy', img_array)
np.save('train_mask_array.npy', mask_array)
np.save('val_imgs_array.npy', val_img_array)
np.save('val_mask_array.npy', val_mask_array)

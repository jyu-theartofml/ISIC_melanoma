
import cv2
import numpy as np
import glob

def image_ROI(file_path, mask_dir, save_path):

    for i, image_file in enumerate(file_path):
        base_file=image_file.split("\\")[1].split(".")[0]
        mask_file=mask_dir+base_file+'_segmentation.png'

        img_color=cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

        #get mask
        img_gray=cv2.imread(mask_file,0) #use 0 to import as gray
        img_masked=cv2.bitwise_and(img_color,img_color, mask = img_gray)

        #save full sized masked image as jpg in folder
        cv2.imwrite(save_path+base_file+'_masked.jpg',  cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR))

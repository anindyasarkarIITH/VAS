# create two csv files with image name and label at each row
import os
import cv2
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import csv
import pandas as pd
#%matplotlib inline
import torchvision.transforms as transforms

#Load the class number -> class string label map
labels = {}
with open('/home/research/Visual_Active_Search_Project/EfficientObjectDetection/xview_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]


data_path = "/storage1/Active/aml/Visual_Active_Search/xview_dataset/"
input_path_list = os.listdir(data_path)
chip_name_list = list(); label_list = list()

def parse_data(csv_fname,input_path_list_):
  for img in input_path_list_: 
    #Load an image
    target_found_flag = False
    chip_name = data_path + img
    chip_name_ = img
    arr = wv.get_image(chip_name)
    arr = cv2.resize(arr, dsize=(3372, 2713), interpolation=cv2.INTER_CUBIC) 

    #We only want to coordinates and classes that are within our chip
    coords, chips, classes = wv.get_labels('/home/research/Visual_Active_Search_Project/EfficientObjectDetection/xView_train.geojson')
    coords = coords[chips==chip_name_]
    classes = classes[chips==chip_name_].astype(np.int64)

    #We can chip the image into 500x500 chips
    c_img, c_box, c_cls = wv.chip_image(img = arr, coords= coords, classes=classes, shape=(500,500))
    print("Num Chips: %d" % c_img.shape[0], c_img.shape[1], c_img.shape[2], c_img.shape[3])
    #Assign label to each chip (our target is small car(18) class id)
    label_vector = np.zeros(int(c_img.shape[0]))
    for idx,val in enumerate(c_cls.values()):
      if 18 in val:
        target_found_flag = True
        label_vector[idx] = 1  
    # we exclude the input if it doesn't contains small car target and if its result to wrong chip number
    if target_found_flag == True and int(label_vector.shape[0]) == 30: 
        chip_name_list.append(chip_name)
        label_list.append(label_vector)
    else:
        pass
  
  
  with open(csv_fname, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')

    for (img_name, img_label) in zip(chip_name_list, label_list):
        filewriter.writerow([img_name, img_label])
        


# creating train csv file 
parse_data('train.csv',input_path_list[0:])
out_array = np.array(label_list)
# creating the label file
np.save('label.npy', out_array)


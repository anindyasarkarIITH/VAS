import pandas as pd
import numpy as np
import warnings
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

####FixMatch
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

    
def augment_list():  
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95)
    ]
    return l

    
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # Second column is the image paths
        self.image_arr = np.asarray(data_info.iloc[0:450 , 0]) #300 for dota
        # First column is the image IDs
        self.label_arr = np.load('/home/research/Visual_Active_Search_Project/EfficientObjectDetection/label.npy') 
        self.label_arr = self.label_arr[0:450 ] 
        
        # Calculate len
        self.data_len = len(self.image_arr) #data_info

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name) #.convert('RGB') for dota
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        
        single_image_label = torch.from_numpy(single_image_label)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
    
class CustomDatasetFromImagesTest(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # Second column is the image paths
        
        self.image_arr = np.asarray(data_info.iloc[450: , 0]) 
        # First column is the image IDs
        
        self.label_arr = np.load('/home/research/Visual_Active_Search_Project/EfficientObjectDetection/label.npy') 
        self.label_arr = self.label_arr[450:] #450 for xview #300 for dota
        
        # Calculate len
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name) #.convert('RGB') for dota
        
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        
        single_image_label = torch.from_numpy(single_image_label)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesTestFM(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # Second column is the image paths
        
        self.image_arr = np.asarray(data_info.iloc[0: , 0]) 
        # First column is the image IDs
        
        self.label_arr = np.load('/home/research/Visual_Active_Search_Project/label.npy') 
        self.label_arr = self.label_arr[0:] 
        
        # Calculate len
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name) #.convert('RGB') for dota
        
        randaug = RandAugment(3,5)
        x_aug = randaug(img_as_img)
        x_aug_tensor = self.transforms(x_aug)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        #print (self.label_arr[index])
        #label_arr_conversion = np.fromstring(self.label_arr[index], dtype=int, sep=' ')
        #print (type(label_arr_conversion))
        single_image_label = torch.from_numpy(single_image_label)
        #print (single_image_label)

        return (img_as_tensor,x_aug_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesTestVIS(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # Second column is the image paths
        self.image_arr = np.asarray(data_info.iloc[0:450 , 0])   # 0 for tta
        # First column is the image IDs
        
        self.label_arr = np.load('/home/research/Visual_Active_Search_Project/label.npy') 
        
        self.label_arr = self.label_arr[0:450 ]   
        
        # Calculate len
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        #print (index)
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name) #.convert('RGB') for dota
        img_as = Image.open(self.image_arr[0])
        
        # save a image using extension
        #im1 = img_as.save("img.jpg")
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        
        single_image_label = torch.from_numpy(single_image_label)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

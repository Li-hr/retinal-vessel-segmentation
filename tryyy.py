import numpy as np
import cv2
import os
from PIL import Image
#import matplotlib as plt

def read_image_and_name(path):
    imgdir = os.listdir(path)
    imglst = []
    imgs = []
    for v in imgdir:
        imglst.append(path + v)
        imgs.append(cv2.imread(path + v))
    print(imglst)
    print('original images shape: ' + str(np.array(imgs).shape))

    return imglst,imgs
'''
#函数1：read_image_and_name测试(BGR图)
imglst,imgs=read_image_and_name('D:\\UNET-try\\DRIVE\\test\\images\\')
for i in range(20):
    cv2.imshow('pic', imgs[i])
    cv2.waitKey(0)
    i=i+1
'''
def read_label_and_name(path):
    labeldir = os.listdir(path)
    labellst = []
    labels = []
    for v in labeldir:
        labellst.append(path + v)
        labels.append(np.asarray(Image.open(path + v)))
    print(labellst)
    print('original labels shape: ' + str(np.array(labels).shape))
    return labellst,labels


#函数2：read_label_and_name测试(RGB图)

labellst,labels=read_label_and_name('D:\\UNET-try\\DRIVE\\test\\images\\')
''' 
for i in range(20):
    cv2.imshow('pic', labels[i])
    cv2.waitKey(0)
    i=i+1
'''
print('I\'m DONE')


#将图片调整成nxn的
def resize(imgs,resize_height, resize_width):
   img_resize = []
   for file in imgs:
       img_resize.append(cv2.resize(file,(resize_height,resize_width)))
   return img_resize

img_resize=resize(labels,576,576)
print('the size of resize_pic is:',np.array(img_resize).shape)

#将N张576x576的图片裁剪成48x48
def crop(image,dx):
    list = []
    for i in range(np.array(image).shape[0]):
        for x in range(np.array(image).shape[1] // dx):
            for y in range(np.array(image).shape[2] // dx):
                list.append(image[i][y*dx : (y+1)*dx,  x*dx : (x+1)*dx]) #这里的list一共append了20x12x12=2880次所以返回的shape是(2880,48,48)
    return np.array(list)
'''
 image1=cv2.imread('D:\\UNET-try\\DRIVE\\test\\images\\01_test.tif')
image1=image1[221:291,221:291]
cv2.imshow('try',image1)
cv2.waitKey(0)
'''
cut_num=48
crop_list=[]
crop_list=crop(img_resize,cut_num)



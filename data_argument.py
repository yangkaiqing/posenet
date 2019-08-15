# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
   author: XiJun.Gong
   date:2016-11-29
"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
import cv2
from scipy import misc
import math
logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated







def randomColor(image):
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def randomGaussian(image, mean=0.2, sigma=0.3):

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im


    img = np.array(image)
    #img.flags.writeable = True
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def motion_blur(image, angle=45):

    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    degree = random.randint(12, 20)
    angle = random.randint(10, 45)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return Image.fromarray(np.uint8(blurred))


def crop_image(image, crop_size):
    w = 960
    h = 480
    d = 32
    hang = (w - d) // d
    lie = (h - d) // d
    num = 0
    for i in range(lie):
        for j in range(hang):
            if num > 8:
                break
            cropped_img = image[i * d:i * d + 224, j * d:j * d + 224, :]
            cv2.imwrite(directory + '/' + fname + '/thumbnail_crop_' + str(num) + '_.jpg', cropped_img)
            num = num + 1


def saveImage(image, path):
    image.save(path)


directory = './data2/scene8_jiading_hualou_training'
dataset = '/jiading_hualou_training_coordinates .csv'
with open(directory + dataset) as f:
    next(f)  # skip the 3 header lines
    for line in f:
        fname, _,_,_ = line.split(',')
        print(fname)
        image = Image.open(directory + '/' +fname + '/thumbnail.jpg').convert('RGB')

        image_motionblur = motion_blur(image)
        saveImage(image_motionblur, directory + '/' + fname + '/thumbnail_motion.jpg')

        image_color = randomColor(image)
        saveImage(image_color, directory + '/' + fname + '/thumbnail_color.jpg')






        # pca jittering
        #print(imag
        img = np.array(image)
        img = img / 255.0
        img_size = img.size // 3
        img1 = img.reshape(img_size, 3)
        img1 = np.transpose(img1)
        img_cov = np.cov([img1[0], img1[1], img1[2]])
        lamda, p = np.linalg.eig(img_cov)
        p = np.transpose(p)
        alpha1 = random.normalvariate(0, 3)
        alpha2 = random.normalvariate(0, 3)
        alpha3 = random.normalvariate(0, 3)
        v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))
        add_num = np.dot(p, v)
        img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])*255
        img2 = np.swapaxes(img2, 0, 2)
        img2 = np.swapaxes(img2, 0, 1)
        #print(img2)
        #image_pca = Image.fromarray(np.uint8(img2))
        misc.imsave(directory + '/' + fname + '/thumbnail_pca.jpg',img2)

        level = 2
        image = cv2.imread(directory + '/' +fname + '/thumbnail.jpg')
        temp = image.copy()
        pyramid_images = []
        for i in range(level):
            dst = cv2.pyrDown(temp)
            # misc.imsave(directory + '/' + fname + '/thumbnail_res_'+str(i)+'_.jpg',dst[::-1])
            cv2.imwrite(directory + '/' + fname + '/thumbnail_res_'+str(i)+'_.jpg',dst)
            temp = dst.copy()
        # crop
        image = cv2.imread(directory + '/' + fname + '/thumbnail.jpg')
        resize_image = cv2.resize(image, (480, 240))
        crop_image(resize_image, 224)
        # rota
        image = cv2.imread(directory + '/' + fname + '/thumbnail.jpg')
        angel = random.randint(-30, 30)
        rote_image = rotate(image, angel)
        cv2.imwrite(directory + '/' + fname + '/thumbnail_roate' + '.jpg', rote_image)







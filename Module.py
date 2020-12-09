# important modules
import numpy as np
import cv2

from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

class Module:
    @staticmethod
    # function later used to find the angle of skweness
    def Skewed_angle(x1, y1, x2, y2):
        try:
            y = (y2 - y1)  # numerator of arctan
            x = (x2 - x1)  # denominator of arctan
            return (np.rad2deg(np.arctan2(y, x)))

        except ZeroDivisionError:
            return 0

    @staticmethod
    # load the image we want to de-skew
    # path: 输入图片的地址
    # 返回img, binary：
    #   img：返回原图矩阵数据
    #   binary：返回0~255灰度图片矩阵数据
    def load_image(path):
        try:
            img = cv2.imread(path)
            # perform BRG to gray scale conversion,we need only gray scale information
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # covert the gray scale image to binary image; here, 0 is the min-thresold for binarization (adjustable, but usually small)
            # returns the binarized image vector to "binary"
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # dilation = cv2.dilate(binary,(5,5),iterations = 2) # dilate the white pixels of image
            # binary = dilation
            return img, binary
        except:
            print("Could not load the image; provide correct path")
            return None, None

    @staticmethod
    def magnitude_spectrum(binary_image):
        # perform fast fourier transform of the image to find frequency information
        f = np.fft.fft2(binary_image)  # 二维快速傅里叶变换
        fshift = np.fft.fftshift(f)  # 将图像中的低频部分移动到图像的中心
        data = 20 * np.log(np.abs(fshift))  # 傅里叶变换得到的复数取模 取对数 得到频域能量谱
        # 我们只关心频域在不同方向的强度，以此表现该方向像素点的密集程度。下面将谱归一化到 0~255 的灰度图像
        data = data.astype(np.uint8)  # convert to 8 bit channel, back to like binary image
        # 黑白反色，处理后得到最终的频谱图
        data = cv2.bitwise_not(data)  # invert black and white pixels
        return data

    @staticmethod
    def canny_edge(magnitude_spectrum,low_th,high_th):




# important modules
import numpy as np
import cv2

from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from numpy import fabs,sin,cos,radians

# fabs=np.fabs
# sin=np.sin
# cos=np.cos
# function later used to find the angle of skweness
def Skewed_angle(x1, y1, x2, y2):
    try:
        y = (y2 - y1)  # numerator of arctan
        x = (x2 - x1)  # denominator of arctan
        return (np.rad2deg(np.arctan2(y, x)))
    except ZeroDivisionError:
        return 0

class Module:
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
    # 灰度图频谱化
    def magnitude_spectrum(gray_img):
        # perform fast fourier transform of the image to find frequency information
        f = np.fft.fft2(gray_img)  # 二维快速傅里叶变换
        fshift = np.fft.fftshift(f)  # 将图像中的低频部分移动到图像的中心
        data = 20*np.log(np.abs(fshift)) # 傅里叶变换得到的复数取模 取对数 得到频域能量谱
        # data = np.log(np.abs(fshift))  # 傅里叶变换得到的复数取模 取对数 得到频域能量谱
        # 我们只关心频域在不同方向的强度，以此表现该方向像素点的密集程度。下面将谱归一化到 0~255 的灰度图像
        # data = (250/np.max(data))*data
        data = data.astype(np.uint8)  # convert to 8 bit channel, back to like binary image
        # 黑白反色，处理后得到最终的频谱图
        data = cv2.bitwise_not(data)  # invert black and white pixels
        return data

    @staticmethod
    # 频谱特征线锐化处理
    def magnitude_spectrum0(gray_img):
        f = np.fft.fft2(gray_img)  # 二维快速傅里叶变换
        fshift = np.fft.fftshift(f)  # 将图像中的低频部分移动到图像的中心
        data = 20*np.log(np.abs(fshift)) # 傅里叶变换得到的复数取模 取对数 得到频域能量谱
        # 我们只关心频域在不同方向的强度，以此表现该方向像素点的密集程度。下面将谱归一化到 0~255 的灰度图像
        data = (255/np.max(data))*data
        data = data.astype(np.uint8)  # convert to 8 bit channel, back to like binary image
        return data

    @staticmethod
    def hough_lines(binary_image):
        # Classic straight-line Hough transform
        h, theta, d = hough_line(binary_image )
        lines = hough_line_peaks(h, theta, d, 10)  # detect only one highest voted line
        line_angles = []
        weigths= []
        xxyys=[]
        H,W = binary_image.shape
        centre=np.array([W,H])/2
        if lines[1].size > 0:
            for w, angle, dist in zip(*lines):
                # line_angles.append(np.rad2deg(angle+np.pi/2)%180.0)
                line=np.array([cos(angle),sin(angle),dist])
                borders=np.array([1,0,0,
                                  0,1,0,
                                  1,0,W,
                                  0,1,H]).reshape((4,3))
                xys=[]
                for border in borders:
                    Ab=np.array([line,border])
                    A,b=Ab[:,:-1],Ab[:,-1]
                    if np.linalg.matrix_rank(A)==A.shape[0]:
                        xys.append(np.linalg.solve(A,b))
                # 到中心点的一范数距离从小到大排序
                xys.sort(key=lambda xy: np.sum(abs(xy - centre)) )
                # xx,yy=(xys[0][0],xys[1][0]),(xys[0][1],xys[1][1])
                xys=np.array(xys)
                xx=xys[:2,0]
                yy=xys[:2,1]
                # print(xys)
                
                # if np.fabs(np.sin(angle))<0.1:
                #     yy=np.array([0,binary_image.shape[0]])
                #     xx=(dist-yy*np.sin(angle))/np.cos(angle)
                # else:
                #     xx=np.array([0,binary_image.shape[1]])
                #     yy=(dist-xx*np.cos(angle))/np.sin(angle)
                weigths.append(w)
                xxyys.append((xx,yy))
                line_angles.append(Skewed_angle(xx[0],yy[0],xx[1],yy[1]))
        return line_angles,weigths,xxyys

    format_list = (('3', '-'),('3', '--'),('3', ':'),('2', '-'),('2', '--'),('2', ':'),('1', '-'),('1', ':'))
    def show_lines(self,img_block,weights,xxyys):
        full_rate = int(max(weights)) ** 2

        for i in range(min(len(self.format_list),len(weights))):
            lw, ls = self.format_list[i]
            img_block.plot(*xxyys[i], linewidth=lw, linestyle=ls)

    # de-skew the image
    @staticmethod
    def de_skew(img, angle):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), round(angle), 1)
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
        return dst

    @staticmethod
    def spectrum_compare(source_path,output_path,name):
        if source_path[-1]!='/':source_path=source_path+'/'
        if output_path[-1]!='/':output_path=source_path+'/'
        img, binary = Module.load_image(source_path+name)
        if binary is None:return None
        spectrum = Module.magnitude_spectrum(binary)
        # 采用直方图二值化
        hist = cv2.equalizeHist(spectrum)
        plt.subplot(121)
        plt.hist(spectrum.ravel(), 256)
        plt.subplot(122)
        plt.hist(hist.ravel(), 256)
        plt.show()
        #输出拼接
        mdata = np.hstack((binary, hist))
        plt.imshow(mdata, cmap=cm.gray)
        cv2.imwrite(output_path+name, mdata)
        return mdata






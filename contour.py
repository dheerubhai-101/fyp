#import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# from shapedetector import ShapeDetector
import imutils

#load images
img_path = 'D:\FYP\Images\Imgdirty_1057_1.jpg.'
#color version
cimg = cv2.imread(img_path)
#grey scale image
img = cv2.imread(img_path,0)

cv2.imwrite('original_image.png', cimg)

plt_image = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
plt.imshow(plt_image)
plt.title('Image'), plt.xticks([]), plt.yticks([])
plt.show()

#Apply Global Threshold
m = np.mean(img, dtype=int)
global_thresh = cv2.threshold(img,int(m/1.2),255,cv2.THRESH_BINARY_INV)[1]

#Show and Save Global Threshold Image
cv2.imwrite('global_threshold.png', global_thresh) 

plt.imshow(global_thresh, 'gray')
plt.title('Global Threshold'), plt.xticks([]), plt.yticks([])
plt.show()

#Perform Adaptive Threshold
#adaptive_thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,27,15)
adaptive_thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,19,10)
#adaptive_thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)
#adaptive_thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)


#Show and Save Adaptive Threshold Image
cv2.imwrite('adapt_mean_threshold.png', adaptive_thresh_img)  

plt.imshow(adaptive_thresh_img, 'gray')
plt.title('Adaptive Mean Threshold'), plt.xticks([]), plt.yticks([])
plt.show()

#Image Magnification Filter Kernel
KERNEL = np.ones((10,10), dtype=int)*2  

# print(KERNEL)

#Filter the thresholded images*
img_filt = cv2.filter2D(adaptive_thresh_img,-1,KERNEL)
#global_thresh = cv2.filter2D(global_thresh,-1,KERNEL)

#Apply multiple times
# for i in range(2):
#     KERNEL_i = np.ones((int(10),int(10)), dtype=int)*10
#     img_filt = cv2.filter2D(img_filt,-1,KERNEL_i)

# #Show and Save Magnification Image
# cv2.imwrite('mag_filt.png', img_filt)    

# plt.imshow(img_filt, 'gray')
# plt.title('Magnification Filtered'), plt.xticks([]), plt.yticks([])
# plt.show()

# #Combine Thresholds
# comb = img_filt + global_thresh

# #Show and Save Combined Threshold Image
# cv2.imwrite('comb_threshold.png', comb)

# plt.imshow(comb, 'gray')
# plt.title('Combined Threshold'), plt.xticks([]), plt.yticks([])
# plt.show()



from PIL import Image
import numpy as np
from scipy.ndimage import filters			#引入滤波模块
import matplotlib.pyplot as plt 
im = np.array(Image.open('D:\\wfz.jpg'))
index = 141 			 #画1行4列的图，与 1,4,1 同
plt.subplot(index)
plt.imshow(im)
for sigma in (2, 5, 10): 		#模糊参数，值越大越模糊
    im_blur = np.zeros(im.shape, dtype=np.uint8)
    for i in range(3):  		#对图像的每一个通道都应用高斯滤波
        im_blur[:,:,i] = filters.gaussian_filter(im[:,:,i], sigma)
    index += 1
    plt.subplot(index)
    plt.imshow(im_blur)
plt.show()

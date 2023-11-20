from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')			#必须显式指明matplotlib的后端
import matplotlib.pyplot as plt
img=np.array(Image.open('d:\\tt1.jpg'))
plt.imshow(img)   	#显示图像
print('请单击图片中的物品')
x=plt.ginput(1)		 #等待用户单击1次
print('你单击了',x)
plt.show()

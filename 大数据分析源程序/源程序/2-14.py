import numpy as np  		 		#导入numpy库
import matplotlib.pyplot as plt 		#导入pyplot库
plt.figure() 		#创建一个绘图对象
x=np.arange(-5,5,0.01)  		#x值
y=x*x     			#y值
plt.plot(x,y,'b--')  		#进行绘图，第3个参数表示蓝色虚线
plt.show()  			#显示图像

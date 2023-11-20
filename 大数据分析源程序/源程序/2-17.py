import matplotlib.pyplot as plt #导入pyplot库
import numpy as np   #导入numpy库
plt.figure()		 #创建一个绘图对象
ax1=plt.subplot(121)  		#在1行2列的第1个区域创建轴对象ax1
ax2=plt.subplot(122)  		#在1行2列的第2个区域创建轴对象ax2
x=np.arange(-5,5,0.01)  #x值
y=x*x     #y值
plt.sca(ax1)   #选择子图1
plt.plot(x,y,'b--',label="x^2")  		#在子图1进行绘图
x2=np.arange(-5,5,0.01)  #x值
y2=2*x2+9
plt.sca(ax2)  #选择子图2
plt.plot(x2,y2,'r-.',label="2x+9")  	#绘制第2条曲线
plt.legend()	#显示图例
plt.show()  #显示图像

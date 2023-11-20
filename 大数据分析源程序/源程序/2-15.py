import matplotlib.pyplot as plt #导入pyplot库
import numpy as np   #导入numpy库
plt.figure()		 #创建一个绘图对象
plt.xlabel('x')	   	#x轴标签
plt.ylabel('y')  	#y轴标签
plt.title('Simple Diagram') #图的标题
x=np.arange(-5,5,0.01)  #x值
y,y2=x*x,2*x2+9     #同时给y和y2赋值
plt.plot(x,y,'b--',label="x^2")  		#进行绘图
plt.plot(x,y2,'r-.',label="2x+9")  	#绘制第2条曲线
plt.legend()	#显示图例
plt.show()  #显示图像

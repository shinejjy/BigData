import matplotlib.pyplot as plt 		#导入pyplot库
import numpy as np  		 #导入numpy库
plt.figure() 				#创建一个绘图对象
x=np.arange(-5,5,0.01)  	#y值
y=x*x     				#x值
plt.xlim(-8,8)  			#定义x轴的范围
plt.ylim(0,10)
plt.plot(x,y,'b--',label="x^2")  		#进行绘图
plt.xlabel('x')	 			 	 #y轴标签
plt.ylabel('y')  				#x轴标签
plt.title('Simple Diagram') 		#图的标题
x2=np.arange(-5,5,0.01) 		 #x值
y2=2*x2+9
plt.plot(x2,y2,'r-.',label="2x+9")  	#绘制第2条曲线
plt.legend()					#显示图例
plt.show()  					#显示图像

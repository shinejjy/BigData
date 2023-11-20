import matplotlib.pyplot as plt 		#导入pyplot库
plt.figure()		 	#创建一个绘图对象
x=[2,1,2,3,4,5] 	 		#设置6个散点的x坐标值
y=[1,2,2,5,4,3]
plt.scatter(x,y,s=60,c='r',marker='o')   #绘制散点图
plt.show()  #显示图像

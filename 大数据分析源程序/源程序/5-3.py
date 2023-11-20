import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus']=False		#正常显示-号
n = 3
x = np.linspace(-10,10,5)  		#做点
y = np.linspace(-10,10,n)
#构造网格点
X,Y = np.meshgrid(x,y)
plt.scatter(X,Y,s=60,c='r',marker='x') 		#用散点图绘制网格点
plt.show()			#输出图5-11所示的图
Z = np.array([[1,2,3,4],[2,1,4,3]]) 	#设置类别号
#print(Z.shape)
#作色块图
cm_light = mpl.colors.ListedColormap(['y', 'r', 'g', 'b'])
# pcolormesh()中，X，Y是坐标值，Z是类别值，cmap是颜色值
plt.pcolormesh(X,Y, Z, cmap=cm_light, alpha=0.5) 
plt.show()		#输出图5-12所示的图

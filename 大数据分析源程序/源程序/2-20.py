import matplotlib.pyplot as plt 		#导入pyplot库
import numpy as np  			 #导入numpy库
mu,sigma=100,20
x=mu+sigma*np.random.randn(20000)
plt.hist(x,bins=100,color='r',normed=True)
plt.show() 						 #显示图像

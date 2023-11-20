import numpy as np
import matplotlib.pyplot as plt
y = [15, 30, 25, 10,20,34,33,18]
x=np.arange(8)  #0-7个条形
plt.bar(x, y, color='r', width=0.5)	 #绘制条形图
plt.plot(x, y, "b", marker='*') 	 #此例同时绘制线形图
for x1, yy in zip(x, y):			#在条形上添加文本
  plt.text(x1, yy + 1, str(yy), ha='center', va='bottom')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering		#导入层次聚类模块
X1,X2 = [],[]
fr = open('C:\\km.txt') 			#打开数据文件
for line in fr.readlines():
    lineArr = line.strip().split()       
    X1.append([int(lineArr[0])])		#第1列读取到X1中
X2.append([int(lineArr[1])])
#把X1和X2合成成一个有两列的数组X并调整维度，此处X的维度为[10,2]
X = np.array(list(zip(X1, X2))).reshape(len(X1), 2)
#print(X)  		#X的值为[[2 1] [1 2] [2 2] [3 2] [2 3] [3 3] [2 4] [3 5] [4 4] [5 3]]
#model = AgglomerativeClustering(3).fit(X)
model=AgglomerativeClustering(n_clusters=3) 	#设置聚类数目为3
labels = model.fit_predict(X)
print(labels)
colors = ['b', 'g', 'r', 'c']
markers = ['o', 's', '<', 'v']
plt.axis([0,6,0,6])
for i, l in enumerate(model.labels_):
    plt.plot(X1[i], X2[i], color=colors[l],marker=markers[l],ls='None')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X1,X2= [],[]
fr = open('C:\\km.txt')
for line in fr.readlines():
    lineArr = line.strip().split()       
    X1.append([int(lineArr[0])])
    X2.append([int(lineArr[1])])
X = np.array(list(zip(X1, X2))).reshape(len(X1), 2)
model = KMeans(3).fit(X)	#调用估计器fit方法进行聚类，聚类数为3
colors = ['b', 'g', 'r', 'c']
markers = ['o', 's', 'x', 'v']
plt.axis([0,6,0,6])
for i, l in enumerate(model.labels_):
    plt.plot(X1[i], X2[i], color=colors[l],marker=markers[l],ls='None')
# 下面用倒三角形绘制均值点
centroids = model.cluster_centers_ 		#centroids保存了所有均值点
for i in range(3):			# 其中3表示聚类的类别数
    plt.plot(centroids[i][0], centroids[i][1], markers[3])
plt.show()

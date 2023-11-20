import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMedoids
X1,X2= [],[]
fr = open('C:\\ex.txt')
for line in fr.readlines():
    lineArr = line.strip().split()       
    X1.append([int(lineArr[0])])
    X2.append([int(lineArr[1])])
X = np.array(list(zip(X1, X2))).reshape(len(X1), 2)
model = KMedoids(2).fit(X)		#调用估计器fit方法进行聚类，聚类数为2
colors = ['b', 'g', 'r', 'c']
markers = ['o', 's', 'x', 'o']
for i, l in enumerate(model.labels_):
    plt.plot(X1[i], X2[i], color=colors[l],marker=markers[l],ls='None')
# 下面用X形绘制中心点
centroids =  model.cluster_centers_ 		#centroids保存了所有中心点
for i in range(2):
    plt.plot(centroids[i][0], centroids[i][1], markers[2])
plt.show()

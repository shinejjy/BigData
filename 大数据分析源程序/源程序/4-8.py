import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
X1, y1=datasets.make_circles(n_samples=1000, factor=.6, noise=.05)
X2, y2 = datasets.make_blobs(n_samples=100, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]], random_state=9)
X = np.concatenate((X1, X2))		 #将X1，X2两个数据集合并
y_pred = DBSCAN(eps = 0.11, min_samples = 10).fit_predict(X) # y_pred保存了类别值
#y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X) #用来和Kmeans聚类对比
plt.scatter(X[:, 0], X[:, 1], c=y_pred) #c=y_pred用来将不同聚类值用不同颜色表示
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist			#引入pdist计算距离
X1,X2 = [],[]
fr = open('C:\\km.txt')
for line in fr.readlines():
    lineArr = line.strip().split()       
    X1.append([int(lineArr[0])])
    X2.append([int(lineArr[1])])
X = np.array(list(zip(X1, X2))).reshape(len(X1), 2)
model=AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)
#print(labels)
#绘制层次聚类树
variables = ['X','Y']
df = pd.DataFrame(X,columns=variables,index=labels)
#print (df)		#df保存了样本点的坐标值和类别值，可打印出来看看
row_clusters = linkage(pdist(df,metric='euclidean'),method='complete')#使用完全距离矩阵
print (pd.DataFrame(row_clusters,columns=['row label1','row label2','distance','no. of items in clust.'],index=['cluster %d'%(i+1) for i in range(row_clusters.shape[0])]))
row_dendr = dendrogram(row_clusters,labels=labels) 	#绘制层次聚类树
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

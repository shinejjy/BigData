from sklearn.datasets import make_circles  
from sklearn.datasets import make_moons  
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt  
fig=plt.figure(figsize=(12, 4))  
plt.subplot(131)  
x1,y1=make_circles(n_samples=1000,factor=0.5,noise=0.1)
# factor表示里圈和外圈的距离之比.每圈共有n_samples/2个点
plt.scatter(x1[:,0],x1[:,1],marker='o',c=y1)  
plt.subplot(132)  
x1,y1=make_moons(n_samples=1000,noise=0.1)  
plt.scatter(x1[:,0],x1[:,1],marker='o',c=y1)  
plt.subplot(133)  
x1,y1=make_blobs(n_samples=100,n_features=2,centers=3)
plt.scatter(x1[:,0],x1[:,1],c=y1);
plt.show()

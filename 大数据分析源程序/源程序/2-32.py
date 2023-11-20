import matplotlib.pyplot as plt     #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA   #加载PCA算法包
from sklearn.datasets import load_iris 
data=load_iris()  #载入iris数据集
y=data.target
x=data.data
pca=PCA(n_components=2)  		#加载PCA算法，设置降维后维度为2
reduced_x=pca.fit_transform(x)		#对样本的特征属性集进行降维 
red_x,red_y=[],[]				#保存第0类样本
blue_x,blue_y=[],[]				#保存第1类样本
green_x,green_y=[],[] 			#保存第2类样本
for i in range(len(reduced_x)):
 if y[i] ==0:  #该数据集有3个类别，因此y[i]=0,1,2
  red_x.append(reduced_x[i][0]) 		#reduced_x[i]表示第i个样本降维后的
  red_y.append(reduced_x[i][1]) 
 elif y[i]==1:
  blue_x.append(reduced_x[i][0])
  blue_y.append(reduced_x[i][1]) 
 else:
  green_x.append(reduced_x[i][0])
  green_y.append(reduced_x[i][1]) 
plt.scatter(red_x,red_y,c='r',marker='x')		#可视化
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()

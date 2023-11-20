import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier		#KNN
from sklearn.model_selection import  train_test_split	#数据分割模块
from sklearn.model_selection import cross_val_score		#交叉验证模块
from sklearn.datasets import load_iris
iris=load_iris()
X=iris['data']
y=iris['target']
# 切分训练集和测试集
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.16)
k_range = range(1, 15)
k_error = []		#保存预测错误率
for k in k_range:		#循环，取k=1到14，查看Knn分类的预测准确率
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_error.append(1 - scores.mean())		#把每次的错误率添加到数组
#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()

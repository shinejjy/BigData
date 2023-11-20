import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier	#引入随机森林分类模块
X ,Y= [],[]     #读取数据
fr = open("D:\\knn.txt")
for line in fr.readlines():
    line = line.strip().split()
    X.append([int(line[0]),int(line[1])])
Y.append(int(line[-1]))			#将最后一列存入Y
X=np.array(X)  #转换成numpy数组,X是特征属性集
y=np.array(Y)  #y是类别标签集
#iris = datasets.load_iris()	 	#去掉这三行的注释符即可对鸢尾花数据集分类
#X = iris.data[:, [0, 2]]
#y = iris.target
# 训练随机森林模型，限制树的最大深度5，树的数目为10棵
clf = RandomForestClassifier(max_depth=5, n_estimators=10)
clf.fit(X, y)
# 画分类界面图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='D',alpha=1)
plt.show()

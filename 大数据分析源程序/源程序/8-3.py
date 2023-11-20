from sklearn.datasets import load_breast_cancer  # 引入肺癌数据集
from sklearn.svm import SVC  # 引入SVC类
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time  # 为了计算算法的耗时引入时间类
import datetime

data = load_breast_cancer()  # 载入肺癌数据集
X = data.data  # X为特征向量
y = data.target  # y为类别标签
# from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(X)
print(X.shape)  # f返回X的维度(569,30)，可见有30个特征
np.unique(y)  # 查看标签y中有几个分类值，将返回array([0,1])

plt.scatter(X[:, 0], X[:, 1], c=y)  # 取前两个特征向量值绘制散点图
plt.show()
# 分割训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
Kernel = ["linear", "poly", "rbf", "sigmoid"]  # 使用4种核函数
for kernel in Kernel:
    time0 = time()  # 为了计算耗时，获取当前时间的时间戳
    clf = SVC(kernel=kernel, gamma="auto"
              , degree=1  # 设置多项式核函数的d值为1次方
              , cache_size=6000  # 设置使用的内存为6000MB
              ).fit(Xtrain, Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    print("耗时：", datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
# （1）rbf核函数的参数调节
# rbf核函数只有一个参数γ的值可调节。下面来寻找rbf核函数的最优γ参数，将如下代码插入到程序8-3中分割训练集和测试集的代码下面。
score = []
gamma_range = np.logspace(-10, 1, 50)  # 返回在对数刻度上均匀间隔的数字
for i in gamma_range:
    clf = SVC(kernel="rbf", gamma=i, cache_size=5000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
# 输出最大分数及最大分数对应的γ值
print(max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range, score)
plt.show()
# （2）多项式核函数参数的调节
# 将如下代码插入到程序8-3中分割训练集和测试集的代码下面。
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

time0 = time()
gamma_range = np.logspace(-10, 1, 20)
coef0_range = np.linspace(0, 5, 10)
param_grid = dict(gamma=gamma_range, coef0=coef0_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=420)
grid = GridSearchCV(SVC(kernel="poly", degree=1, cache_size=5000),
                    param_grid=param_grid, cv=cv)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
# 4. 松弛系数惩罚项C的调整
# 将如下代码插入到程序8-3中分割训练集和测试集的代码下面
score = []
C_range = np.linspace(0.01, 30, 50)
# C_range = np.linspace(5,7,50)
for i in C_range:
    # 调节线性核函数的C值
    clf = SVC(kernel="linear", C=i, cache_size=5000).fit(Xtrain, Ytrain)
    # 调节rbf核函数的C值
    # clf = SVC(kernel="rbf",C=i, cache_size=5000, gamma = 0.01274) .fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest, Ytest))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()

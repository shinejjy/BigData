import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(0)
X = np.array([[3, 3], [4, 3], [1, 1]])  # X为特征向量
Y = np.array([1, 1, -1])

clf = svm.SVC(kernel='linear')  # 因为是线性分类，所以调用线性SVM核函数
clf.fit(X, Y)  # 拟合模型
# 绘制决策边界
w = clf.coef_[0]  # w为截距
a = -w[0] / w[1]  # a为斜率，即x0前面的参数
xx = np.linspace(-5, 5)
# 对比公式（8-12），可知yy就是x1，clf.intercept_[0]就是b
yy = a * xx - (clf.intercept_[0]) / w[1]
# 绘制支持向量经过的边界
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
# 绘制线
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
# 绘制散点
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.axis('tight')
plt.show()  # 显示图像
print(clf.decision_function(X))  # 打印决策函数

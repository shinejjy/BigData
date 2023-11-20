import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # 引入KNN模块
from sklearn.model_selection import train_test_split  # 引入数据集分割模块
from sklearn import metrics  # 引入机器学习的准确率评估模块

# 读取文本文件的数据，并分割成特征属性集X和类别集y
X1, y1 = [], []
fr = open('数据文件\\knn.txt')
for line in fr.readlines():
    lineArr = line.strip().split()
    X1.append([int(lineArr[0]), int(lineArr[1])])
    y1.append(int(lineArr[2]))
X = np.array(X1)  # 转换成numpy数组,X是特征属性集
y = np.array(y1)  # y是类别标签集
# 分割成训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.16)
knn = KNeighborsClassifier(3)  # 使用模型并训练
knn.fit(X, y)
# 分别绘制每个类中样本的散点
plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], color='red', marker='o')
plt.scatter(X_train[Y_train == 2, 0], X_train[Y_train == 2, 1], color='green', marker='x')
plt.scatter(X_train[Y_train == 3, 0], X_train[Y_train == 3, 1], color='blue', marker='d')
# 使用测试集对分类模型进行测试，测试集中有2个样本
y_pred = knn.predict(X_test)
# 输出测试结果
print(knn.score(X_test, Y_test))  # 输出整体预测结果的准确率，方法1
print(metrics.accuracy_score(y_true=Y_test, y_pred=y_pred))  # 输出准确率的方法2
# 输出混淆矩阵，如果为对角阵，则表示预测结果是正确的，准确度越大
print(metrics.confusion_matrix(y_true=Y_test, y_pred=y_pred))
# 输出更详细的分类测试报告
from sklearn.metrics import classification_report

target_names = ['labels_1', 'labels_2', 'labels_3']
print(classification_report(Y_test, y_pred))
# 预测新样本的类别
label = knn.predict([[7, 27], [2, 4]])
print(label)  # 输出[2 1],表示新样本分别属于2和1类

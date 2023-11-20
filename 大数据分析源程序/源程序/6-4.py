import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
def plot_decision_boundary(model, axis): 	#画分类界面的函数定义
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
# 读取数据
data = np.loadtxt('D:\\logi1.txt', delimiter=',')
data_X = data[:, 0:2]		#取数据的第0列和第1列
data_y = data[:, 2] 			#取数据的第2列
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=666)
# 训练模型
log_reg = LogisticRegression(solver='newton-cg')
log_reg.fit(X_train, y_train)
# 结果可视化
plot_decision_boundary(log_reg, axis=[0, 100, 0, 100])
plt.scatter(data_X[data_y == 0, 0], data_X[data_y == 0, 1], color='red')
plt.scatter(data_X[data_y == 1, 0], data_X[data_y == 1, 1], color='blue')
plt.xlabel('成绩1')
plt.ylabel('成绩2')
plt.title('课程成绩与是否录取的关系')
plt.show()
# 评估模型预测准确率
print(log_reg.score(X_train, y_train))
print(log_reg.score(X_test, y_test))

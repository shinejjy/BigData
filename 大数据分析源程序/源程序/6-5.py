import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline			#引入管道

from sklearn.preprocessing import PolynomialFeatures	#引入管道特征
from sklearn.preprocessing import StandardScaler	#引入标准化模块
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
def PolynomialLogisticRegression(degree):			#定义多项式逻辑回归
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)), 		#对特征添加多项式项
        ('std_scaler', StandardScaler()),		#对数据进行归一化处理
        ('log_reg', LogisticRegression(solver='newton-cg'))
    ])
# 读取数据
data = np.loadtxt('D:\\logi1.txt', delimiter=',')
data_X = data[:, 0:2]		#取第0和1列
data_y = data[:, 2] 			#取第2列
# 数据分割
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=666)
# 训练模型
poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)
# 结果可视化
plot_decision_boundary(poly_log_reg, axis=[0, 100, 0, 100])
plt.scatter(data_X[data_y == 0, 0], data_X[data_y == 0, 1], color='red')
plt.scatter(data_X[data_y == 1, 0], data_X[data_y == 1, 1], color='blue')
plt.xlabel('语文成绩')
plt.ylabel('数学成绩')
plt.show()

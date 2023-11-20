import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 	#导入MLPClassifier类
from sklearn.preprocessing import StandardScaler 	#导入数据预处理类
#二手房销售数据
data = [					#每个样本有2个特征属性，1个类别属性
 [-0.017612, 14.053064, 0],[-1.395634, 4.662541, 1],[-0.752157, 6.53862, 0],[-1.322371, 7.152853, 0],[0.423363, 11.054677, 0], [0.406704, 7.067335, 1],[0.667394, 12.741452, 0],[-2.46015, 6.866805, 1],[0.569411, 9.548755, 0],[-0.026632, 10.427743, 0], [0.850433, 6.920334, 1],[1.347183, 13.1755, 0],[1.176813, 3.16702, 1],[-1.781871, 9.097953, 0],[-0.566606, 5.749003, 1], [0.931635, 1.589505, 1],[-0.024205, 6.151823, 1],[-0.036453, 2.690988, 1],[-0.196949, 0.444165, 1],[1.014459, 5.754399, 1], [1.985298, 3.230619, 1],[-1.693453, -0.55754, 1],[-0.576525, 11.778922, 0],[-0.346811, -1.67873, 1],[-2.124484, 2.672471, 1], [1.217916, 9.597015, 0],[-0.733928, 9.098687, 0],[1.416614, 9.619232, 0],[1.38861, 9.341997, 0],[0.317029, 14.739025, 0] ]
dataMat = np.array(data)
X=dataMat[:,0:2]
y = dataMat[:,2]
#神经网络对数据尺度敏感，所以最好在训练前标准化，或者归一化
scaler = StandardScaler() 		# 对数据进行标准化
scaler.fit(X)  					# 训练标准化对象
X = scaler.transform(X)   		# 转换数据集
 # solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好
 # alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
 # hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就是3层神经网络
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1) 
clf.fit(X, y)				#拟合模型
print('每层网络层系数矩阵维度：\n',[coef.shape for coef in clf.coefs_])
y_pred = clf.predict([[0.317029, 14.739025]])
print('预测结果为：',y_pred)
y_pred_pro =clf.predict_proba([[0.317029, 14.739025]])
print('预测结果概率：\n',y_pred_pro)
cengindex = 0			#保存每层的层号
for wi in clf.coefs_:
 cengindex += 1  			# 表示第几层神经网络。
 print('第%d层网络层:' % cengindex)
 print('权重矩阵维度:',wi.shape)
 print('系数矩阵:\n',wi)
 # 绘制分割区域
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # 寻找每个维度的范围
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # 寻找每个维度的范围
# 在特征范围以0.01位步长预测每一个点的输出结果
xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max,0.01)) 
# 先形成待测样本的形式，再通过模型进行预测
Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape) 	# 将输出结果转换为和网格的矩阵形式，以便绘图
 # 绘制区域网格图
plt.rcParams['axes.unicode_minus']=False		#解决负号显示不正常的问题
plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.Paired)
 # 绘制样本点
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

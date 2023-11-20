from sklearn.linear_model import LinearRegression, SGDRegressor
sgd_reg = SGDRegressor(n_iter=100)
sgd_reg .fit(X_train_s, y_train)
score = sgd_reg .score(X_test, y_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
plt.rcParams['font.sans-serif']='SimHei'
X,y = [],[]
X=[[123],[150],[87],[102]]
y=[[250],[320],[160],[220]]
X=np.array(X)
y=np.array(y)
model=linear_model.SGDRegressor(loss="huber", penalty="l2", max_iter=5000)
model.fit(X, y.ravel())
y2=model.predict(X) 		#y2为预测值
print(y2)
plt.axis([80,160,150,350])
plt.scatter(X,y,color='y', marker='o')
plt.plot(X,y2,'g-')  		#画拟合线
plt.legend(['预测值','真实值'])
plt.show()
print("截距：",model.intercept_) 		#截距
print("斜率：",model.coef_) 		#斜率

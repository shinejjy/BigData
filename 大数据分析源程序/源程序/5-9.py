import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split	#数据分割模块
from sklearn.model_selection import cross_val_score		#交叉验证模块
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
X, y = datasets.make_classification(n_samples=1000,n_features=30,n_informative=15,flip_y=.5, weights=[.1, .9])
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.1)
mf_range = range(2, 28)
k_error = []		#保存预测错误率
for k in mf_range:		#循环，取k=2到27，查看RF分类的预测准确率
    rf=RandomForestClassifier(n_estimators=29,min_samples_leaf=5,max_features=k,n_jobs=2)
    #cv参数决定数据集划分比例，这里是按照9:1划分训练集和测试集
    scores = cross_val_score(rf, X, y, cv=9, scoring='accuracy')
    k_error.append(1 - scores.mean())		#把每次的错误率添加到数组
#画图，x轴为k值，y值为误差值
plt.plot(mf_range, k_error)
plt.xlabel('max_features for RF')
plt.ylabel('Error')
plt.show()

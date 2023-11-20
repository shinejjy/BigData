from sklearn.datasets import load_iris
dataSet = load_iris()	# 导入iris数据集
data = dataSet['data'] 			# data是特征属性集
label = dataSet['target']		 	# label是类别标签
feature = dataSet['feature_names'] 	# 特征的名称
target = dataSet['target_names']		 # 标签（类别）的名称
print(feature ,target)

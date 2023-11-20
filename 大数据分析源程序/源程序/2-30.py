from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
StandardScaler(copy=True, with_mean=True, with_std=True)
print(scaler.mean_)     			#输出均值
print(scaler.var_)     				#输出标准差
print(scaler.transform(data))      	 #标准化矩阵data
print(scaler.transform([[2, 2]]))  		 #标准化新数据

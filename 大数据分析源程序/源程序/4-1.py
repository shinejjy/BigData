import numpy as np
np.set_printoptions(suppress=True)		#不使用科学计数法输出结果
from sklearn.preprocessing import MinMaxScaler
data = [[36,50000,4,41000,1,1], 
        [42,45000,4,40000,2,1],
        [23,31000,2,35000,3,2],
        [61,70000,4,20000,4,3],
        [38,20000,3,10000,2,4]]
scaler = MinMaxScaler()
scaler.fit(data)			#fit()函数在此处是求最大值和最小值
MinMaxScaler(copy=True, feature_range=(0, 1))
print(scaler.transform(data)) 			# transform()在此处是进行数据归一化操作

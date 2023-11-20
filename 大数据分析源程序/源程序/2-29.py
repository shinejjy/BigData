from sklearn.preprocessing import MinMaxScaler
data = [[-1, 6], [-0.5, 2], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaler.fit(data)
MinMaxScaler(copy=True, feature_range=(0, 1))
print('range的最大值为：',scaler.data_max_)
print('range的最小值为：',scaler.data_min_)
print(scaler.transform(data))

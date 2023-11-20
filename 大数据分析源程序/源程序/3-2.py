import numpy as np
import pandas as pd		 #读取u.data文件
#数据文件格式用户id、商品id、评分、时间戳
header = ['user_id', 'item_id', 'rating', 'timestamp'] 
df = pd.read_csv('u.data', sep='\t', names=header) 	 #读取u.data文件
# 计算唯一用户和电影的数量
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)) 
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=21) 
# 协同过滤算法
# 第一步是创建uesr-item矩阵，这需创建训练和测试两个uesr-item矩阵
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
print(train_data_matrix.shape)
print(test_data_matrix.shape)
 #计算相似度 
# 本例使用cosine_similarity函数来计算余弦相似性
from sklearn.metrics.pairwise import cosine_similarity
 # 计算用户相似度
user_similarity = cosine_similarity(train_data_matrix)
# 计算物品相似度
item_similarity = cosine_similarity(train_data_matrix.T) 
print(u"用户相似度矩阵：", user_similarity.shape, u"  物品相似度矩阵：", item_similarity.shape)
print(u"用户相似度矩阵：", user_similarity)
print(u"物品相似度矩阵：", item_similarity) 
# 预测
def predict(ratings, similarity, type):
    # 基于用户相似度矩阵的
    if type == 'user':
#mean函数求取均值  axis=1 对各行求取均值，返回一个m*1的矩阵
        mean_user_rating = ratings.mean(axis=1)
# np.newaxis 给矩阵增加一个列，一维矩阵变为多维矩阵 mean_user_rating(n*1)
        ratings_diff = ( ratings - mean_user_rating[:, np.newaxis] )
        pred = mean_user_rating[:, np.newaxis] + np.dot(similarity, ratings_diff) / np.array( [np.abs(similarity).sum(axis=1)]).T
        # 基于物品相似度矩阵的
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    print(u"预测值: ", pred.shape)
    return pred 
# 预测结果
user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, item_similarity, type='item')
print(item_prediction)
print(user_prediction)
 # 评估指标，均方根误差
 # 使用sklearn的mean_square_error (MSE)函数，其中，RMSE仅仅是MSE的平方根
# 这里只是想要考虑测试数据集中的预测评分，
 #使用prediction[ground_truth.nonzero()]筛选出预测矩阵中的所有其他元素
from sklearn.metrics import mean_squared_error
from math import sqrt 
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
item_prediction = np.nan_to_num(item_prediction)
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

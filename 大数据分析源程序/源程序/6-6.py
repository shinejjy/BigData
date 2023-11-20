# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import train_test_split
from pandas import DataFrame as df


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2   #基于卡方的特征筛选
plt.rcParams['font.sans-serif']=['SimSun'] 		#将默认字体改成宋体，还可是SimHei
plt.rcParams['axes.unicode_minus']=False		#解决负号显示不正常的问题


def get_data(pathname='./loan/bankloan.xls'):
    # 读入数据
    try:
        bank_data = pd.read_excel(pathname)
        x = bank_data.iloc[:, :8]
        y = bank_data.iloc[:, 8]
        return x, y
    except:
        bank_data = pd.read_csv(pathname)
        return bank_data

# 筛选特征值
def screening(x, y):
    
#    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
#    model = SelectFromModel(lsvc,prefit=True)
#    X_new = model.transform(x)
#    rf = RandomForestRegressor()                              # 默认参数
#    rf.fit(x, y)
#    model = SelectFromModel(rf, prefit=True)
#    cols = model.transform(x)

 
    # 训练模型
#    rlr = RLR()
#    rlr.fit(x, y)
#    # 特征筛选
#    cols = x.columns[rlr.get_support()]
    selector = SelectKBest(chi2, k=4)
    selector.fit_transform(x, y)
    cols =x.columns[selector.get_support(indices=True)]
    print(cols)
    return cols

# 测试训练集的效果
def test(x, y):
    # 逻辑回归模型
    lr = LR(solver='liblinear')
    lr.fit(x, y)
    # 给出模型的正确率
    print('模型的正确率为{0}%'.format('%.2f'%(lr.score(x, y)*100)))
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    y_pred = lr.predict(x_test)
    
    # 绘制2种图形比较预测集与测试集
    # 散点图
    plt.figure(figsize=(14,12))
    # 调整子图间的距离
    plt.subplots_adjust(hspace=.3)
    plt.subplot(311)
    plt.scatter(range(len(x_test)), y_test+0.5, c='g', s=2, label='test')
    plt.scatter(range(len(x_test)), y_pred, c='r', s=2, label='pred')
    plt.title('测试结果')
    plt.yticks([0, 1], ['不违约', '违约'])
    plt.legend()
    plt.ylim([-0.5,2.5])
    
    # 小提琴图
    # 先合并数据
    data = pd.concat([df(y_pred, columns=['pred']),df(y_test.tolist(), columns=['test'])], axis=1)
    # 分类数据
    data = data.stack().reset_index()
    # 删除无用的数据
    data = data.drop(columns=[data.columns[0]])
    # 对每一列重命名
    data = data.rename(columns={data.columns[0]:'labels', data.columns[1]:'value' })
    data['xzhou'] = 1
    # 小提琴图
    plt.subplot(312)
    plt.title('测试结果')
    sns.violinplot(data=data, x='xzhou', y='value', split=True , hue='labels')
    plt.yticks([0, 1], ['不违约', '违约'])
    # 将模型返回
    return lr
    
def predicted(predicted_data, cols, lr):
    # 给每一条贷款数据插入编号
    predicted_data['sno'] = [i for i in range(len(predicted_data))]
    # 应用筛选出来的特征
    predicted_x = predicted_data[cols]
    # 预测
    predicted_result = lr.predict(predicted_x)
    plt.subplot(313, facecolor='k')
    plt.scatter(predicted_data['sno'], predicted_result, s=4, c='r')
    plt.title('预测可能违约情况分布')
    plt.xlabel('贷款人编号')
    plt.ylabel('违约情况')
    plt.xticks([i*10 for i in range(11)])
    plt.yticks([0, 1], ['不违约', '违约'])
    plt.grid(axis='x', alpha=.5)
    plt.ylim([-0.5, 1.5])
    
def main():
    # 获取过往贷款数据
    X, Y = get_data()
    # 筛选特征
    cols = screening(X, Y)
    X = X[cols].values
#    print(X)
    # 测试模型
    lr = test(X, Y)
    # 获取待预测数据
#    data = get_data('wait_pred_data.csv')
    # 绘制出预测的结果
#    predicted(data, cols, lr)
    
if __name__ == '__main__':
    main()
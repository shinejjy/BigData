import matplotlib.pyplot as plt
from sklearn import linear_model
plt.rcParams['font.sans-serif']='SimHei'
X,y = [],[]
fr = open('C:\\lr.txt')
for line in fr.readlines():
    lineArr = line.strip().split()       
    X.append([int(lineArr[0])])
    y.append(float(lineArr[1]))
X=[[123],[150],[87],[102]]
y=[[250],[320],[160],[220]]
model=linear_model.LinearRegression()
model.fit(X,y)
y2=model.predict(X)  #y2为预测值
plt.xlabel('面积')
plt.ylabel('房价')
#plt.title('房价和面积的回归分析')
plt.grid(True)
plt.axis([80,160,150,350])
#plt.plot(X,y,'k.')
plt.scatter(X,y,color='y', marker='o')
plt.plot(X,y2,'g-')  #画拟合线
plt.legend(['预测值','真实值'])
plt.show()
print("截距：",model.intercept_) #截距
print("斜率：",model.coef_) #斜率
a=model.predict([[200]]) #预测200的Y值
print("value is {:.2f}".format(a[0][0]))

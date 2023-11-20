import numpy as np
import random as random
import matplotlib as mpl
import matplotlib.pyplot as plt
datas = [[(1,2),-1],[(2,1),-1],[(2,2),-1],[(1,4),1],[(3,3),1],[(5,4),1],[(3, 3), 1], [(4, 3), 1], [(1, 1), -1],[(2, 3), -1], [(4, 2), 1]]						#训练数据
random.shuffle(datas)
fig = plt.figure('Input Figure')
plt.rcParams['font.sans-serif']=['SimHei'] 	#用来正常显示中文标签
xArr = np.array([x[0] for x in datas])
yArr = np.array([x[1] for x in datas])
xPlotx,xPlotx_,xPloty,xPloty_ = [],[],[],[]
for i in range(len(datas)):
    y = yArr[i]
    if y>0:				#正例
        xPlotx.append(xArr[i][0])
        xPloty.append(xArr[i][1])
    else: 				#负例
        xPlotx_.append(xArr[i][0])
        xPloty_.append(xArr[i][1])
plt.title('Perception 输入数据')
plt.grid(True)
pPlot1,pPlot2 = plt.plot(xPlotx,xPloty,'b+',xPlotx_,xPloty_,'rx') 		#绘制散点
plt.legend(handles = [pPlot1,pPlot2],labels=['Positive Sample','Negtive Sample'],loc='upper center')
plt.show()

w = np.array([1,1]) #权重初始值为1，1
b = 3               #偏置初始值为3
n = 1
while True:
    num = 0
    for i in range(len(datas)):
        num += 1
        x = xArr[i]
        y = yArr[i]
        z = y*(np.dot(w,x)+b) 	#np.dot()用于矩阵相乘，即计算向量w和x点积
        if z<=0 :   
            w = w+n*y*x    #修改权重值
            b = b+n*y
            break
    if num>=len(datas):
        break
fig = plt.figure('Output Figure')
x0 =np.linspace(0,5,100)
w0,w1 = w[0],w[1]
x1 = -(w0/w1)*x0-b/w1   		#计算预测值
plt.title("Perception 输出平面")
plt.xlabel('x0')
plt.ylabel('x1')
plt.annotate('输出分类界面',xy=(0.5,4.5),xytext=(1.7,3.5))
pPlot3, pPlot4= plt.plot(xPlotx,xPloty,'b+',xPlotx_,xPloty_,'rx')
plt.plot(x0,x1,'k', lw=1)  #绘制分类界面
plt.legend(handles = [pPlot3,pPlot4],labels=['Positive Sample','Negative Sample'],loc='upper right')
plt.show()
print(w0,w1,b)      #输出感知机模型的参数值

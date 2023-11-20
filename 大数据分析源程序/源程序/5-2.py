import matplotlib as mpl
N, M = 90, 90  # 网格采样点的个数，采样点越多，分类界面图越精细
t1 = np.linspace(0, 25, N)  		 #生成采样点的横坐标值
t2 = np.linspace(0,12, M)   		 #生成采样点的纵坐标值
x1, x2 = np.meshgrid(t1, t2)  		# 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1) 		 # 将采样点作为测试点
#print(X.shape)
y_show_hat = knn.predict(x_show)  	# 预测采样点的值
y_show_hat = y_show_hat.reshape(x1.shape)  #使之与输入的形状相同
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt.pcolormesh(x1, x2, y_show_hat,  cmap=cm_light,alpha=0.3)  # 预测值的显示

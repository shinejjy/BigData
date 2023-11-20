import numpy as np  		 #导入numpy库
a=np.matrix('1 2 3;4 5 6;7 8 9') 	 #用字符串创建矩阵
x=np.array([[1,3],[2,4]])
b=np.matrix(x) 			 #用数组创建矩阵
c=a.T    			#转置矩阵
d=a.H  #求共轭矩阵，仅对复数矩阵有用
e=a.I  #求逆矩阵
f=a.A  #返回该矩阵对应的二维数组
g=a[:,-1] #取矩阵一列，方法和数组完全相同
print(a,b,c,d,e,f)

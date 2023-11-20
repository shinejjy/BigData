import numpy as np   #导入numpy库
a=np.int32(100*np.random.random((3,4)))		#创建3*4的二维数组
print (a)
print(a.shape)   #输出(3, 4)
b=a.ravel()        #将数组a扁平化，转换为一维数组
print(b)
b.shape=(6,2)   #改变数组b的维度
print (b)
c=b.transpose()   #转置数组b
print (c)

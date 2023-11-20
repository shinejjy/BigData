import numpy as np  	 #导入numpy库
a = np.array([1,2,3])  #创建一维数组
print (a)            #输出[1 2 3]
b = np.array([[1,2], [3,4]])   #创建二维数组 
print (b)  		#输出[[1 2] [3 4]]
c = np.array([2,3,4,5], ndmin = 3)   #指定数组维度 
print (c)     		#输出[[[2 3 4 5]]]
d = np.array([1,2,3], dtype = complex)    #指定元素的数据类型 
print (d)		 	#输出[1.+0.j 2.+0.j 3.+0.j]
e = np.arange(0,1,0.2) 			#arrange()函数创建数组 
print (e) 			#输出[0 0.2 0.4 0.6 0.8 ]
f = np.linspace(0,10,5) 		
print (f) 			
g = np.zeros((3,4)) 			#zeros()函数创建全零数组
print (g)

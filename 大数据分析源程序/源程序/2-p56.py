def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('C:\\lr2.txt')		# 打开文本文件lr2.txt
    for line in fr.readlines():			# 依次读取文本文件中的一行
        lineArr = line.strip().split()	# 根据空格分割一行中的每列
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

import pandas as pd          #导入pandas库
data=pd.read_excel('D:\\18ds.xlsx') 			#读取excel文件
data2=pd.read_csv('C:\\lr2.csv') 			#该函数可读取csv或txt文件
print(data)

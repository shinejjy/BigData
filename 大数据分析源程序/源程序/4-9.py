# -*- coding: utf-8 -*-
import numpy as np
import PIL.Image as image #导入PIL包，用于打开和创建图片
from sklearn.cluster import KMeans   #导入Kmeans算法包

def loadData(filePath):             #加载图片
    f = open(filePath,'rb')     #以二进制形式打开图片
    data = []
    img = image.open(f)   #以列表形式返回图片像素值
    m,n = img.size      #获得图片大小
    for i in range(m):    #将每个像素点的颜色值压缩到0-1之间，目的是对特征属性作归一化处理
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])   #data保存了每个像素点颜色的RGB分量值
    f.close()
    return np.mat(data),m,n

imgData,row,col = loadData('./44.jpg')
#聚类获得每个像素点所属的类别
label = KMeans(n_clusters=3).fit_predict(imgData)
#将类别值转换成二维数组形式，以便和像素点相对应
label = label.reshape([row,col])
#创建一张新的灰度图像保存聚类后的结果
pic_new = image.new("L", (row, col))
#根据所属类别向图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
        #print(int(256/(label[i][j]+1)))
pic_new.save("che-2.jpg", "JPEG")   #以JPEG格式保存图像


import numpy as np
import matplotlib.pyplot as plt
X1,X2 = [],[]
fr = open('C:\\ex.txt')
for line in fr.readlines():
    lineArr = line.strip().split()       
    X1.append(float(lineArr[0]))
    X2.append(float(lineArr[1]))
txt = 'ABCDEFGHIJ'
plt.axis([0,7,0,7])
plt.scatter(X1, X2)
for i in range(len(txt)):
    #xy为被注释的点，xytext为注释文字的坐标位置
    plt.annotate(txt[i], xy = (X1[i], X2[i]), xytext = (X1[i]+0.1, X2[i]+0.2))
plt.show()

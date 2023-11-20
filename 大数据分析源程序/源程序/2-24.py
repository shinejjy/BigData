import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize			#引入scipy库的optimize模块
x=np.linspace(-5,3,100) 				#定义x值的范围为-5到3
def f(x):
    return 4*x**3+(x-2)**2+x**4
x_min_local=optimize.fmin_bfgs(f,2)		#采用fmin_bfgs()函数求f的最小值
print('f(x)极小值点:',x_min_local)
x_max_global=optimize.fminbound(f,-10,10)
print('取得极小值时的x值:',x_max_global)
plt.plot(x,f(x))
plt.show()

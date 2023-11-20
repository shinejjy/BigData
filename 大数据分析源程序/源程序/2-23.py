import numpy as np
import matplotlib.pyplot as plt
N = 20 					#绘制20个扇形
#设置每个标记所在射线与极径的夹角
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
ax = plt.subplot(111, projection='polar')
bars = ax.bar(theta, radii, width = width, bottom = 0.0)
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))			#绘制极坐标的扇形
    bar.set_alpha(0.5) 						#设置透明度
plt.show()

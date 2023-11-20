from sklearn.datasets import fetch_lfw_people		# 人脸照片数据集
import matplotlib.pyplot as plt
faces = fetch_lfw_people(min_faces_per_person=60) # 只识别最少有60张照片的人的人脸
print(faces.target_names)			#输出符合条件的人的姓名
print(faces.images.shape)			#输出照片尺寸
fig, ax = plt.subplots(3,5)				#以3行5列的形式显示照片
for i,axi in enumerate(ax.flat):
  axi.imshow(faces.images[i],cmap = 'bone')
  axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])
fig.show()

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=150,whiten=True, random_state=42) #PCA降维，维数降到150
svc = SVC(kernel='rbf',class_weight='balanced') 	# 使用rbf核函数
model = make_pipeline(pca, svc)	#流水线处理，先降维，再SVC分类

from sklearn.model_selection import train_test_split
Xtrain, Xtest,ytrain,ytest = train_test_split(faces.data, faces.target, random_state=40)
from sklearn.model_selection import GridSearchCV		#自动寻找最优参数
param_grid = {'svc__C': [1, 5, 10], 'svc__gamma': [0.0001, 0.0005, 0.001]}
grid = GridSearchCV(model, param_grid)
print(Xtrain.shape, ytrain.shape)	#特征集1011行，2914列，标签集1011行
grid.fit(Xtrain, ytrain) 	#建立模型
print(grid.best_params_) 	#输出模型的最优参数组合

model = grid.best_estimator_  #获得最好的模型
yfit = model.predict(Xtest)  #用当前最好的模型做预测
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone') 	#每个图像大小是62*47
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
fig.show()
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit, target_names=faces.target_names))

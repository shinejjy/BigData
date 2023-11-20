from sklearn.preprocessing import Binarizer
X = [[ 1., -1.,2.],[ 2.,0.,0.],[ 0.,1.,-1.]] #数据矩阵
binary = Binarizer() 
transformer =binary.fit(X) 	# fit does nothing.
transformer.transform(X)
Binarizer(copy=True, threshold=0.0)
print(transformer.transform(X))

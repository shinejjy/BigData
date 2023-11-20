from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
transformer = Normalizer().fit(X) 
print(transformer.transform(X))

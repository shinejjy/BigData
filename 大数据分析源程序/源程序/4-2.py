from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']

# Perform t-SNE to reduce the data to 2D
tsne = TSNE(n_components=2, random_state=123)
iris_data_2d = tsne.fit_transform(iris_data)

# Define the number of clusters to try
num_clusters_range = range(2, 7)
fmi_scores = []

# Perform K-Means clustering for different numbers of clusters
for num_clusters in num_clusters_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=123).fit(iris_data)

    # Calculate the FMI score
    score = fowlkes_mallows_score(iris_target, kmeans.labels_)
    fmi_scores.append(score)

    # Create a scatter plot for the current clustering result in 2D
    plt.scatter(iris_data_2d[:, 0], iris_data_2d[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title(f'K-Means Clustering (k={num_clusters}) - FMI: {score:.2f}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

# Plot the FMI scores against the number of clusters
plt.plot(num_clusters_range, fmi_scores, marker='o')
plt.title('Fowlkes-Mallows Index (FMI) vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Fowlkes-Mallows Index (FMI)')
plt.grid(True)
plt.show()

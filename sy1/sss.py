import matplotlib.pyplot as plt
from surprise import Dataset, SVD, KNNBasic, SVDpp
from surprise.model_selection import cross_validate
from surprise import accuracy

# 使用MovieLens数据集，加载数据集
data = Dataset.load_builtin('ml-100k')

# 初始化不同的推荐算法
algorithms = {
    'KNNBasic': KNNBasic(),
    'SVD': SVD(),
    'SVDpp': SVDpp()
}

# 用于存储不同算法的性能指标
results = {}

# 交叉验证并评估每个算法
for algo_name, algo in algorithms.items():
    print(f"Evaluating {algo_name}...")
    results[algo_name] = cross_validate(algo, data, measures=['RMSE'], cv=4, verbose=True)

# 可视化比较不同算法的性能
rmse_results = {algo_name: results[algo_name]['test_rmse'].mean() for algo_name in algorithms}
plt.bar(rmse_results.keys(), rmse_results.values())
plt.ylabel('RMSE (Lower is better)')
plt.title('Algorithm Comparison')
plt.show()

import numpy as np
from sklearn import cluster
if __name__ == '__main__':
    data = np.random.rand(100, 3)  # 生成一个随机数据，样本大小为100, 特征数为3
    k = 3  # 假如我要聚类为3个clusters
    [centroid, label, inertia] = cluster.k_means(data, k)
    print('data',data)
    print('centroid',centroid)
    print('label',label)
    print('inertia',inertia)



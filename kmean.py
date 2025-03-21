from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# Apply K-means clustering (k=4 since we have 4 programmes)
def apply_kmeans(X_scaled, y, pca=None, X_pca=None):
    # Find optimal number of clusters using the Elbow method
    inertia = []
    silhouette_scores = []
    # range_n_clusters = range(2, 10)
    #
    # for n_clusters in range_n_clusters:
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    #     kmeans.fit(X_scaled)
    #     inertia.append(kmeans.inertia_)
    #     silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    #
    # # Plot the Elbow method
    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(range_n_clusters, inertia, marker='o')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.grid(True)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range_n_clusters, silhouette_scores, marker='o')
    # plt.title('Silhouette Score Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Perform K-means with k=4 (matching number of programmes)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Map clusters to programmes based on majority class
    programme_mapping = {}
    for cluster in range(4):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            programmes = y.iloc[cluster_indices].values
            counts = pd.Series(programmes).value_counts()
            most_common = counts.index[0]
            programme_mapping[cluster] = most_common

    # Visualize clusters in PCA space
    if pca is not None and X_pca is not None:
        plt.figure(figsize=(10, 8))
        for i in range(4):
            indices = cluster_labels == i
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                        label=f'Cluster {i} → {programme_mapping.get(i, "?")}',
                        alpha=0.7)

        # Add centroids
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                    s=200, marker='X', c='black', label='Centroids')

        plt.title('K-means Clustering Results')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    # Create confusion matrix between clusters and programmes
    cluster_programme_matrix = np.zeros((4, len(y.unique())), dtype=int)
    programme_list = sorted(y.unique())

    for i, prog in enumerate(programme_list):
        for cluster in range(4):
            # Count students with this programme in this cluster
            count = np.sum((y == prog) & (cluster_labels == cluster))
            cluster_programme_matrix[cluster, i] = count

    # Display the matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cluster_programme_matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.xticks(np.arange(len(programme_list)), programme_list)
    plt.yticks(np.arange(4), [f'Cluster {i}' for i in range(4)])
    plt.xlabel('Programme')
    plt.ylabel('Cluster')

    for i in range(4):
        for j in range(len(programme_list)):
            plt.text(j, i, str(cluster_programme_matrix[i, j]),
                    ha="center", va="center",
                    color="white" if cluster_programme_matrix[i, j] > np.max(cluster_programme_matrix)/2 else "black")

    plt.title('Students per Programme in Each Cluster')
    plt.tight_layout()
    plt.show()

    return kmeans, cluster_labels, programme_mapping
plt.cla()  # 清除当前图形
# 加载数据
df = pd.read_csv("./student_data.csv")
df = df.drop(columns=["Index"])

# df['Total']=df['Q1']+df['Q2']+df['Q3']+df['Q4']+df['Q5']
# # 计算题目占总分比例
# df['Q1'] = df['Q1'] / 8   # Q1满分8分
# df['Q2'] = df['Q2'] / 8
# df['Q3'] = df['Q3'] / 14   # Q3满分14分
# df['Q4'] = df['Q4'] / 10
# df['Q5'] = df['Q5'] / 6
# df['Total'] = df['Total'] / df['Total'].max()  # 归一化


# 分离特征和目标变量
X = df.drop(columns=["Programme"])
y = df["Programme"]

poly = PolynomialFeatures(degree=10, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# 创建PCA + 分类器的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 定义参数网格
param_grid = {
    'pca__n_components': range(1, 40),
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 15]
}

# 使用网格搜索查找最佳参数
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# 打印最佳参数和分数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数进行PCA
best_components = grid_search.best_params_['pca__n_components']
best_pipeline = grid_search.best_estimator_
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用最佳管道中的PCA对数据进行降维
pca = best_pipeline.named_steps['pca']
X_pca = pca.transform(X_scaled)

# 调用K-means函数
kmeans, cluster_labels, programme_mapping = apply_kmeans(X_scaled, y, pca, X_pca)

# 打印评估指标
print(f"\nK-means Clustering Results:")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
print("\nCluster to Programme Mapping:")
for cluster, programme in programme_mapping.items():
    print(f"Cluster {cluster} → Programme {programme}")
plt.close()  # 关闭当前图形窗口
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 读取数据，这里假设文件名为 student_data.csv
df = pd.read_csv('./student_data.csv')

# 这里假定真实专业存储在 Programme 列中
if 'Programme' not in df.columns:
    print("数据中不存在 Programme 列")
    exit(1)

y_true = df['Programme'].values

# 选取用于PCA的特征，这里排除目标变量 Programme；根据实际情况选择合适的特征
X = df.drop('Programme', axis=1).select_dtypes(include=[np.number])  # 只选数值型特征

# 执行PCA降维，保留95%以上的方差
pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X)

# 使用KMeans进行聚类，聚类簇数设为真实专业的类别数
n_clusters = len(np.unique(y_true))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred = kmeans.fit_predict(X_reduced)

# 计算ARI指数
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index (ARI): {ari:.2f}")
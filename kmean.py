import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import silhouette_score

# 加载数据
df = pd.read_csv("./student_data.csv")
df = df.drop(columns=["Index"])

df['Total'] = df['Q1'] + df['Q2'] + df['Q3'] + df['Q4'] + df['Q5']
# 计算题目占总分比例
df['Q1'] = df['Q1'] / 8   # Q1满分8分
df['Q2'] = df['Q2'] / 8
df['Q3'] = df['Q3'] / 14   # Q3满分14分
df['Q4'] = df['Q4'] / 10
df['Q5'] = df['Q5'] / 6
df['Total'] = df['Total'] #/ df['Total'].max()  # 归一化

# 分离特征和目标变量
X = df.drop(columns=["Programme"])
y = df["Programme"]

# 生成高阶特征
poly = PolynomialFeatures(degree=20, interaction_only=False, include_bias=True)
X_poly = poly.fit_transform(X)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 使用 PCA 降维
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 可视化原始分类结果
plt.figure(figsize=(10, 8))
unique_programmes = y.unique()
for programme in unique_programmes:
    indices = y == programme
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Programme {programme}', alpha=0.5)

plt.title('Original Classification Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 计算 Silhouette Score
silhouette = silhouette_score(X_pca, y)
print(f"Silhouette Score for Original Classification: {silhouette:.4f}")
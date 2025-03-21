import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# 读取数据
df = pd.read_csv("./student_data.csv")
df = df.drop(columns=["Index"])

df['Total'] = df['Q1'] + df['Q2'] + df['Q3'] + df['Q4'] + df['Q5']

# 提取特征和目标变量
X = df.drop(columns=["Programme"])
y = df["Programme"]

# 生成高阶特征
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 尝试 PCA 降维（保留 95% 方差）
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 可视化原始分类结果
plt.figure(figsize=(10, 8))
unique_programmes = y.unique()
for programme in unique_programmes:
    plt.scatter(X_pca[y == programme, 0], X_pca[y == programme, 1], label=f'Programme {programme}', alpha=0.7)

plt.title('Original Classification Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
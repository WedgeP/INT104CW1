import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import itertools

esp = 0.0000001

# 自动生成特征：比值、乘积、差值、和、平方
def generate_interaction_features(X):
    new_features = pd.DataFrame()

    # 生成比值、乘积、差值、和等特征
    for col1, col2 in itertools.combinations(X.columns, 2):
        new_features[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + esp)  # 比值
        # new_features[f'{col1}_mul_{col2}'] = X[col1] * X[col2]  # 乘积
        new_features[f'{col1}_minus_{col2}'] = X[col1] - X[col2]  # 差值
        new_features[f'{col1}_plus_{col2}'] = X[col1] + X[col2]  # 和

    # 生成每个特征的平方
    for col in X.columns:
        new_features[f'{col}_squared'] = X[col] ** 2  # 平方

    return new_features


# 自动生成统计特征
def generate_statistical_features(X):
    stats_features = pd.DataFrame()
    stats_features['mean'] = X.mean(axis=1)  # 平均值
    stats_features['std'] = X.std(axis=1)  # 标准差
    stats_features['max'] = X.max(axis=1)  # 最大值
    stats_features['min'] = X.min(axis=1)  # 最小值
    return stats_features

# 读取数据
df = pd.read_csv("./student_data.csv")
df = df.drop(columns=["Index"])

# 分离特征和目标变量
X = df.drop(columns=["Programme"])

# 生成新的特征
interaction_features = generate_interaction_features(X)
statistical_features = generate_statistical_features(X)

# 合并原始特征和新生成的特征
X_combined = pd.concat([X, interaction_features, statistical_features], axis=1)
poly = PolynomialFeatures(degree=4, interaction_only=True, include_bias=False)
X_combined = poly.fit_transform(X_combined)
# 标准化所有特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# 使用PCA降维到2D
pca = PCA(n_components=2)

# 对标准化数据进行PCA降维
X_pca = pca.fit_transform(X_scaled)

# 将PCA降维结果转为DataFrame便于操作
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# 将原始的'Programme'列添加到降维结果中
df_pca['Programme'] = df['Programme']

# 可视化：根据'programme'列的值为数据点上色
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Programme', data=df_pca, palette='Set1', s=100, marker='o')

plt.title('PCA - Data Distribution after Different Operations')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Programme')
plt.show()

# 输出PCA方差解释比例
print("Explained variance ratio:", pca.explained_variance_ratio_)
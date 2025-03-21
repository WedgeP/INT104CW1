import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
plt.clf()  # 清除当前图形
# 加载数据
df = pd.read_csv("./student_data.csv")
df = df.drop(columns=["Index"])
# 计算题目占总分比例
df['Q1_percent'] = df['Q1'] / 8 * 100  # Q1满分8分
df['Q3_percent'] = df['Q3'] / 14 * 100  # Q3满分14分


# 分离特征和目标变量
X = df.drop(columns=["Programme"])
y = df["Programme"]

# 创建PCA + 分类器的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 定义参数网格
param_grid = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7],
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

# 获取PCA结果用于可视化
pca = best_pipeline.named_steps['pca']
X_scaled = best_pipeline.named_steps['scaler'].transform(X)
X_pca = pca.transform(X_scaled)

# 可视化PCA结果
plt.figure(figsize=(10, 8))
for programme in y.unique():
    indices = y == programme
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=programme, alpha=0.7)

plt.title(f'PCA Visualization (Components={best_components})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 查看混淆矩阵
y_pred = best_pipeline.predict(X)
print("\n混淆矩阵:")
print(confusion_matrix(y, y_pred))
print("\n分类报告:")
print(classification_report(y, y_pred))

# 可视化各主成分的解释方差
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, alpha=0.7)
plt.step(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), where='mid', color='red')
plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.5)
plt.xlabel('主成分数量')
plt.ylabel('解释方差比例')
plt.title('PCA解释方差')
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# 读取数据
df = pd.read_csv("./student_data.csv")
# 提取特征和目标变量
X = df.drop(columns=["Programme"])
y = df["Programme"]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 尝试 PCA 降维（保留 95% 方差）
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

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

# 查看混淆矩阵
y_pred = best_pipeline.predict(X)
print("\n混淆矩阵:")
print(confusion_matrix(y, y_pred))
print("\n分类报告:")
print(classification_report(y, y_pred))

# 读取测试数据
df_test = pd.read_csv("./test_data.csv")
# 提取特征和目标变量
X_test = pd.concat([df_test.drop(columns=["Programme", "MCQ", "Total"]), df_test["Total"]], axis=1)
mapping = {1: "A", 2: "B", 3: "C", 4: "D"}  # 你可以根据需要修改映射关系

df_test["Programme"] = df_test["Programme"].map(mapping)  # 或者使用 df_test["Programme"].replace(mapping)
y_test = df_test["Programme"]

y_pred_test = best_pipeline.predict(X_test)
print("\n测试集混淆矩阵:")
print(confusion_matrix(y_test, y_pred_test))
print("\n测试集分类报告:")
print(classification_report(y_test, y_pred_test))
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# 读取数据
df = pd.read_csv("./student_data.csv")

# 提取特征和目标变量
X = df.drop(columns=["Programme"])
y = df["Programme"]

# 创建PCA + 贝叶斯分类器的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', GaussianNB())
])

# 定义参数网格（这里只调节 PCA 维度）
param_grid = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7],
}

# 网格搜索找最佳参数
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# 输出最佳参数和分数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数构建模型
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X)

# 输出分类效果
print("\n混淆矩阵:")
print(confusion_matrix(y, y_pred))
print("\n分类报告:")
print(classification_report(y, y_pred))

# 读取测试数据
df_test = pd.read_csv("./test_data.csv")

# 提取测试特征和标签
X_test = pd.concat([df_test.drop(columns=["Programme", "MCQ", "Total"]), df_test["Total"]], axis=1)
mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
df_test["Programme"] = df_test["Programme"].map(mapping)
y_test = df_test["Programme"]

# 测试集预测与报告
y_pred_test = best_pipeline.predict(X_test)
print("\n测试集混淆矩阵:")
print(confusion_matrix(y_test, y_pred_test))
print("\n测试集分类报告:")
print(classification_report(y_test, y_pred_test))
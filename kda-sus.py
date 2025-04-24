from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np


# 自定义 KDE 分类器
class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        for cls in self.classes_:
            kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(X[y == cls])
            self.models_[cls] = kde
            self.priors_[cls] = np.mean(y == cls)
        return self

    def predict(self, X):
        log_probs = []
        for cls in self.classes_:
            log_prob = self.models_[cls].score_samples(X) + np.log(self.priors_[cls])
            log_probs.append(log_prob)
        return self.classes_[np.argmax(log_probs, axis=0)]


# 加载数据
df = pd.read_csv("student_data.csv")
X = df.drop(columns=["Programme"]).values
y = LabelEncoder().fit_transform(df["Programme"])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建 Pipeline 和 GridSearchCV
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kde", KDEClassifier())
])
param_grid = {
    "kde__bandwidth": [0.5, 0.75, 1.0, 1.18, 1.185, 1.19, 1.195, 1.2, 1.225, 1.25, 1.375, 1.5, 2.0],
    "kde__kernel": ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# 输出结果
print("最佳参数：", grid.best_params_)
print("交叉验证得分：", grid.best_score_)
print("\n测试集报告：")
y_pred = grid.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

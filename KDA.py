import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
# 导入缺失的库
from sklearn.decomposition import FastICA
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 设置随机种子和中文显示
np.random.seed(42)
# 设置中文显示
system = platform.system()
if system == 'Darwin':  # Mac系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif system == 'Windows':  # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
elif system == 'Linux':  # Linux系统
    # Linux系统可能需要安装中文字体，例如 Noto Sans CJK SC
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
else:
    # 默认字体（如果系统未识别）
    plt.rcParams['font.sans-serif'] = ['SimHei']


class KDE_NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.kde_models_ = {}
        self.class_priors_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            self.kde_models_[c] = [
                KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(X_c[:, i:i+1])
                for i in range(X.shape[1])
            ]
            self.class_priors_[c] = X_c.shape[0] / X.shape[0]

        return self

    def _joint_log_likelihood(self, X):
        n_samples, n_features = X.shape
        log_likelihoods = np.zeros((n_samples, len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            log_prob = np.log(self.class_priors_[c])
            for i in range(n_features):
                kde = self.kde_models_[c][i]
                log_density = kde.score_samples(X[:, i:i+1])
                log_prob += log_density
            log_likelihoods[:, idx] = log_prob

        return log_likelihoods

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        jll = self._joint_log_likelihood(X)
        log_prob_x = np.log(np.exp(jll).sum(axis=1)).reshape(-1, 1)
        return np.exp(jll - log_prob_x)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd

# 加载数据
df = pd.read_csv("./student_data.csv")
X = df.drop(columns=["Programme"])
y = df["Programme"]




from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np

# KDE 分类器（每个类训练一个 KDE，预测时用贝叶斯后验）
class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        for cls in self.classes_:
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X[y == cls])
            self.models_[cls] = kde
            self.priors_[cls] = np.mean(y == cls)
        return self

    def predict(self, X):
        log_probs = []
        for cls in self.classes_:
            log_prob = self.models_[cls].score_samples(X) + np.log(self.priors_[cls])
            log_probs.append(log_prob)
        return self.classes_[np.argmax(log_probs, axis=0)]

# 准备数据
X = df[["Total"]].values
y = LabelEncoder().fit_transform(df["Programme"])

# 构建 Pipeline 和参数网格
pipeline = Pipeline([
    ("kde", KDEClassifier())
])
param_grid = {
    "kde__bandwidth": [1.0, 2.0, 3.0, 4.0],
    "kde__kernel": ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
}

# 网格搜索
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

# 输出最佳参数
print("最佳参数:", grid.best_params_)
print("交叉验证准确率:", grid.best_score_)


import matplotlib.pyplot as plt

best_kernel = grid.best_params_["kde__kernel"]
best_bandwidth = grid.best_params_["kde__bandwidth"]

# 用最佳 KDE 绘图
kde = KernelDensity(kernel=best_kernel, bandwidth=best_bandwidth).fit(X)
x_d = np.linspace(X.min()-5, X.max()+5, 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_d)

plt.figure(figsize=(10, 6))
plt.plot(x_d[:, 0], np.exp(log_dens), label=f"{best_kernel}, bw={best_bandwidth}")
plt.hist(X, bins=30, density=True, alpha=0.5, color='gray', label="数据直方图")
plt.title("最佳 KDE 拟合分布")
plt.xlabel("Total 分数")
plt.ylabel("密度估计")
plt.legend()
plt.grid(True)
plt.show()
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats

class FlexibleBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_distribution_map=None, random_state=42):
        """
        初始化灵活贝叶斯分类器

        参数:
            feature_distribution_map: 字典，指定每个(类别,特征)对应的分布类型
                例如 {('A', 'Q1'): 'bimodal_gaussian', ('A', 'Q2'): 'log_normal'}
            random_state: 随机种子
        """
        self.feature_distribution_map = feature_distribution_map or {}
        self.random_state = random_state
        self.models = {}  # 存储每个类别每个特征的分布模型
        self.priors = {}  # 存储每个类别的先验概率
        self.classes_ = None
        self.features_ = None

    def _fit_distribution(self, X, feature_idx, feature_name, distribution_type):
        """为特定特征拟合指定的分布类型"""
        data = X[:, feature_idx]

        if distribution_type == 'bimodal_gaussian':
            model = GaussianMixture(
                n_components=2,
                covariance_type='full',
                random_state=self.random_state
            )
            # 重塑数据为2D数组(GMM需要)
            model.fit(data.reshape(-1, 1))
            return model

        elif distribution_type == 'log_normal':
            # 对于对数正态分布，我们需要确保数据为正
            pos_data = np.maximum(data, 1e-10)
            params = stats.lognorm.fit(pos_data)
            return {'distribution': 'log_normal', 'params': params}

        else:
            # 默认使用双峰高斯
            model = GaussianMixture(
                n_components=2,
                covariance_type='full',
                random_state=self.random_state
            )
            model.fit(data.reshape(-1, 1))
            return model

    def fit(self, X, y):
        """
        训练贝叶斯分类器

        参数:
            X: 特征矩阵
            y: 目标类别
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.features_ = np.arange(n_features)

        # 计算先验概率
        for cls in self.classes_:
            self.priors[cls] = np.mean(y == cls)

        # 为每个类别的每个特征拟合分布
        for cls in self.classes_:
            cls_samples = X[y == cls]
            if len(cls_samples) < 5:  # 样本太少，跳过
                continue

            self.models[cls] = {}

            for i, feature_idx in enumerate(self.features_):
                feature_name = f"feature_{i}"  # 默认特征名

                # 确定该(类别,特征)对应的分布类型
                dist_type = self.feature_distribution_map.get((cls, feature_name), 'bimodal_gaussian')

                # 拟合分布
                self.models[cls][feature_idx] = self._fit_distribution(
                    cls_samples, feature_idx, feature_name, dist_type
                )

        return self

    def _score_sample(self, x, cls):
        """计算样本在给定类别下的对数似然"""
        log_likelihood = 0.0

        for feature_idx in self.features_:
            if cls not in self.models or feature_idx not in self.models[cls]:
                continue

            model = self.models[cls][feature_idx]
            value = x[feature_idx]

            # 根据分布类型计算似然
            if isinstance(model, GaussianMixture):  # 双峰高斯
                # GMM需要2D数据
                ll = model.score_samples(np.array([[value]]))
                log_likelihood += ll[0]

            elif isinstance(model, dict) and model['distribution'] == 'log_normal':
                # 对��正态分布
                params = model['params']
                if value <= 0:  # 对数正态要求值为正
                    log_likelihood += np.log(1e-10)  # 极小概率
                else:
                    ll = stats.lognorm.logpdf(value, *params)
                    log_likelihood += ll

            else:  # 默认双峰高斯
                ll = 0  # 这里应该不会执行到，但为了安全起见
                log_likelihood += ll

        return log_likelihood

    def predict_proba(self, X):
        """
        预测每个样本属于各个类别的概率

        参数:
            X: 特征矩阵

        返回:
            概率矩阵，形状为(n_samples, n_classes)
        """
        if not self.models:
            raise ValueError("模型尚未训练")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # 初始化对数似然矩阵
        log_likelihoods = np.zeros((n_samples, n_classes))

        # 计算每个样本在每个类别下的对数似然
        for i, x in enumerate(X):
            for j, cls in enumerate(self.classes_):
                if cls in self.models:
                    # 计算对数似然
                    log_likelihood = self._score_sample(x, cls)
                    # 贝叶斯公式: P(C|X) ∝ P(X|C) * P(C)
                    log_likelihoods[i, j] = log_likelihood + np.log(self.priors[cls])
                else:
                    log_likelihoods[i, j] = -np.inf

        # 处理数值问题：从每一行减去最大值以防溢出
        max_log_probs = np.max(log_likelihoods, axis=1, keepdims=True)
        log_probs_stable = log_likelihoods - max_log_probs

        # 取指数并归一化
        probs = np.exp(log_probs_stable)
        row_sums = np.sum(probs, axis=1, keepdims=True)

        # 处理零概率行
        zero_rows = (row_sums == 0).ravel()
        if np.any(zero_rows):
            probs[zero_rows, :] = 1.0 / n_classes
            row_sums[zero_rows] = 1.0

        # 归一化概率
        probs = probs / row_sums

        # 检查并处理NaN
        if np.any(np.isnan(probs)):
            nan_rows = np.any(np.isnan(probs), axis=1)
            probs[nan_rows, :] = 1.0 / n_classes

        return probs

    def predict(self, X):
        """预测类别"""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
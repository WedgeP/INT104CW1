import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone


# 自定义Transformer：处理Dummy Class逻辑
class DummyClassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, confidence_threshold=0.5, margin_threshold=0.1):
        self.estimator = estimator
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.classes_ = None
        self.dummy_class = 'E'

    def fit(self, X, y):
        # 首先用原始分类器拟合数据
        self.estimator.fit(X, y)
        self.classes_ = np.unique(np.append(self.estimator.classes_, self.dummy_class))
        return self

    def transform(self, X, y=None):
        # 对验证集做预测，并标记难分类样本为dummy class
        if y is not None:
            # 获取预测概率
            y_proba = self.estimator.predict_proba(X)
            max_proba = np.max(y_proba, axis=1)
            second_proba = np.sort(y_proba, axis=1)[:, -2]
            margin = max_proba - second_proba

            # 将置信度低或边界模糊的样本标记为dummy class
            y_trans = y.copy()
            dummy_mask = (max_proba < self.confidence_threshold) | (margin < self.margin_threshold)
            if hasattr(y_trans, 'loc'):  # 如果是pandas Series
                y_trans.loc[dummy_mask] = self.dummy_class
            else:  # 如果是numpy array
                y_trans = np.array(y_trans, dtype=object)
                y_trans[dummy_mask] = self.dummy_class

            print(f"创建了 {dummy_mask.sum()} 个dummy class样本，占比 {dummy_mask.sum() / len(y):.2%}")
            return X, y_trans
        return X


# 自定义Estimator：集成Dummy Class和原始分类器
class DummyClassEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, confidence_threshold=0.5, margin_threshold=0.1):
        self.base_estimator = base_estimator
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.dummy_class = 'E'
        self.classes_ = None

    def fit(self, X, y):
        # 查找是否已有dummy class
        if self.dummy_class in np.unique(y):
            self.classes_ = np.unique(y)
        else:
            # 创建具有初始估计器的transformer
            transformer = DummyClassTransformer(
                clone(self.base_estimator),
                self.confidence_threshold,
                self.margin_threshold
            )
            transformer.fit(X, y)
            X_trans, y_trans = transformer.transform(X, y)
            self.classes_ = np.unique(y_trans)

            # 用包含dummy class的数据训练最终模型
            self.base_estimator.fit(X, y_trans)
        return self

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)
        return proba

    def predict(self, X):
        y_pred_raw = self.base_estimator.predict(X)

        # 如果预测中有dummy class，则改为最高概率的原始类别
        if self.dummy_class in y_pred_raw:
            y_proba = self.predict_proba(X)
            y_pred = []
            for i, pred in enumerate(y_pred_raw):
                if pred == self.dummy_class:
                    # 获取原始类别的概率
                    original_classes = [c for c in self.classes_ if c != self.dummy_class]
                    original_indices = [list(self.classes_).index(c) for c in original_classes]
                    original_proba = y_proba[i, original_indices]

                    # 选择最高概率的原始类别
                    best_class_idx = np.argmax(original_proba)
                    y_pred.append(original_classes[best_class_idx])
                else:
                    y_pred.append(pred)
            return np.array(y_pred)
        else:
            return y_pred_raw


# 主代码
df = pd.read_csv('./student_data.csv')
df = df[df['Programme'].notna()]

mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', '1': 'A', '2': 'B', '3': 'C', '4': 'D'}
df['Programme'] = df['Programme'].map(mapping).fillna(df['Programme'])

X = df.drop('Programme', axis=1)
y = df['Programme']

categorical_features = ['Gender', 'Grade']
numerical_features = X.columns.difference(categorical_features).tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

from sklearn.model_selection import GridSearchCV

# 使用原有的Pipeline设置
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),
    ('classifier', DummyClassEstimator(
        GaussianNB(),
        confidence_threshold=0.5,
        margin_threshold=0.1
    ))
])

# 设置参数搜索网格
param_grid = {
    'classifier__confidence_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
    'classifier__margin_threshold': [0.05, 0.1, 0.15, 0.2]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 训练模型并自动搜索最佳参数
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("\n最佳参数:")
print(grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 评估最佳模型
y_pred = best_model.predict(X_test)
print("\n测试集表现:")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:\n", classification_report(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
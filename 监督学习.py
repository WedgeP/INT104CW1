import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. 读取数据
df = pd.read_csv('./student_data.csv')
# 去除Programme为NaN的行
df = df[df['Programme'].notna()]
df = df.drop(columns=['Index'])
test_df = pd.read_csv('./unique_test_data.csv')

# 统一标签映射
mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', 1: 'A', 2: 'B', 3: 'C', 4: 'D','A':'A','B':'B','C':'C','D':'D'}
df['Programme'] = df['Programme'].map(mapping)
test_df['Programme'] = test_df['Programme'].map(mapping)

X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender']]
y = df['Programme']

# 2. 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 交叉验证和参数搜索
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# 决策树参数
dt_params = [{'criterion': 'gini'}, {'criterion': 'entropy'}]
for p in dt_params:
    clf = DecisionTreeClassifier(**p, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv)
    results.append({'Model': 'DecisionTree', 'Params': p, 'CV_Accuracy': np.mean(scores)})

# 朴素贝叶斯参数
nb_params = [{'var_smoothing': 1e-9}, {'var_smoothing': 1e-8}]
for p in nb_params:
    clf = GaussianNB(**p)
    scores = cross_val_score(clf, X_scaled, y, cv=cv)
    results.append({'Model': 'NaiveBayes', 'Params': p, 'CV_Accuracy': np.mean(scores)})

# kNN参数
knn_params = [{'n_neighbors': 3}, {'n_neighbors': 5}]
for p in knn_params:
    clf = KNeighborsClassifier(**p)
    scores = cross_val_score(clf, X_scaled, y, cv=cv)
    results.append({'Model': 'kNN', 'Params': p, 'CV_Accuracy': np.mean(scores)})

# 选出最佳参数
best_dt = max([r for r in results if r['Model'] == 'DecisionTree'], key=lambda x: x['CV_Accuracy'])
best_nb = max([r for r in results if r['Model'] == 'NaiveBayes'], key=lambda x: x['CV_Accuracy'])
best_knn = max([r for r in results if r['Model'] == 'kNN'], key=lambda x: x['CV_Accuracy'])

# 4. 处理测试集
X_test = scaler.transform(test_df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender']])
y_test = test_df['Programme']

# 5. 用最佳参数分别训练每个模型
dt_clf = DecisionTreeClassifier(**best_dt['Params'], random_state=42)
dt_clf.fit(X_scaled, y)

nb_clf = GaussianNB(**best_nb['Params'])
nb_clf.fit(X_scaled, y)

knn_clf = KNeighborsClassifier(**best_knn['Params'])
knn_clf.fit(X_scaled, y)

voting_clf = VotingClassifier(
    estimators=[
        ('dt', dt_clf),
        ('nb', nb_clf),
        ('knn', knn_clf)
    ],
    voting='soft'
)
voting_clf.fit(X_scaled, y)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_scaled, y)

# 6. 依次评估每个模型
models = {
    'DecisionTree': dt_clf,
    'NaiveBayes': nb_clf,
    'kNN': knn_clf,
    'Voting': voting_clf,
    'RandomForest': rf_clf
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).astype(str)
    print(f'\n{name} 测试集准确率: {accuracy_score(y_test, y_pred):.4f}')
    print('分类报告:\n', classification_report(y_test, y_pred))
    print('混淆矩阵:\n', confusion_matrix(y_test, y_pred))
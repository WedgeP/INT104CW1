import platform
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 设置随机种子
np.random.seed(42)

# 读取数据
df = pd.read_csv('./student_data.csv')

# 处理 Programme 字段
mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
if df['Programme'].dtype == 'int64' or df['Programme'].iloc[0] in [1, 2, 3, 4]:
    df['Programme'] = df['Programme'].map(mapping)

# 特征工程
def process_data(df, mode='train', preprocessors=None):
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    feature_sets = {}

    if mode == 'train':
        preprocessors = {'scalers': {}, 'columns': {}}

    if mode == 'train':
        exam_cols = [col for col in numeric_df.columns if 'Q' in col]
        if not exam_cols:
            exam_cols = numeric_df.columns[-5:].tolist()
        preprocessors['columns']['考试分数'] = exam_cols
    else:
        exam_cols = preprocessors['columns']['考试分数']
    for col in exam_cols:
        if col not in numeric_df.columns:
            numeric_df[col] = 0
    feature_sets['考试分数'] = numeric_df[exam_cols].values

    basic_patterns = ['性别', 'Gender', 'sex', 'Total', '总分', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    if mode == 'train':
        basic_cols = []
        for pat in basic_patterns:
            basic_cols += [col for col in numeric_df.columns if pat.lower() in col.lower()]
        basic_cols = list(dict.fromkeys(basic_cols))
        if not basic_cols:
            basic_cols = numeric_df.columns[:2].tolist()
        preprocessors['columns']['去除年级'] = basic_cols
    else:
        basic_cols = preprocessors['columns']['去除年级']
    for col in basic_cols:
        if col not in numeric_df.columns:
            numeric_df[col] = 0
    feature_sets['去除年级'] = numeric_df[basic_cols].values

    if mode == 'train':
        programme_cols = [col for col in numeric_df.columns if 'programme' in col.lower() or 'program' in col.lower()]
        all_cols = [col for col in numeric_df.columns if col not in programme_cols]
        preprocessors['columns']['全部特征'] = all_cols
    else:
        all_cols = preprocessors['columns']['全部特征']
    for col in all_cols:
        if col not in numeric_df.columns:
            numeric_df[col] = 0
    feature_sets['全部特征'] = numeric_df[all_cols].values

    for name, data in feature_sets.items():
        if mode == 'train':
            scaler = StandardScaler()
            feature_sets[name] = scaler.fit_transform(data)
            preprocessors['scalers'][name] = scaler
        else:
            feature_sets[name] = preprocessors['scalers'][name].transform(data)

    if mode == 'train':
        return feature_sets, preprocessors
    else:
        return feature_sets

feature_sets, preprocessors = process_data(df, mode='train')

# 特征转换
transformed_sets = {}
for name, X in feature_sets.items():
    minmax = MinMaxScaler().fit_transform(X)
    transformed_sets['归一化_' + name] = minmax

    standard = StandardScaler().fit_transform(X)
    transformed_sets['标准化_' + name] = standard

    normalized = Normalizer().fit_transform(X)
    transformed_sets['正则缩放_' + name] = normalized

# 降维处理
final_sets = {}
for name, X_scaled in transformed_sets.items():
    pca = PCA(n_components=min(X_scaled.shape[1], 10))
    final_sets['PCA_' + name] = pca.fit_transform(X_scaled)

    ica = FastICA(n_components=min(X_scaled.shape[1], 10), random_state=42)
    final_sets['ICA_' + name] = ica.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
    final_sets['TSNE_' + name] = tsne.fit_transform(X_scaled)

# 聚类评估
def evaluate_clustering(X, labels):
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = -1
    try:
        db_score = davies_bouldin_score(X, labels)
    except:
        db_score = float('inf')
    try:
        ch_score = calinski_harabasz_score(X, labels)
    except:
        ch_score = -1
    return silhouette, db_score, ch_score

def run_kmeans(X, n_clusters=[4]):
    results = []
    for n in n_clusters:
        model = KMeans(n_clusters=n, init='k-means++', random_state=42)
        labels = model.fit_predict(X)
        silhouette, db, ch = evaluate_clustering(X, labels)
        results.append({'method': 'kmeans', 'n_clusters': n, 'silhouette': silhouette, 'db': db, 'ch': ch})
    return results

def run_gmm(X, n_components=[4]):
    results = []
    for n in n_components:
        model = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        labels = model.fit_predict(X)
        silhouette, db, ch = evaluate_clustering(X, labels)
        results.append({'method': 'gmm', 'n_components': n, 'silhouette': silhouette, 'db': db, 'ch': ch})
    return results

def run_hierarchical(X, n_clusters=[4]):
    results = []
    for n in n_clusters:
        model = AgglomerativeClustering(n_clusters=n, linkage='ward', metric='euclidean')
        labels = model.fit_predict(X)
        silhouette, db, ch = evaluate_clustering(X, labels)
        results.append({'method': 'hierarchical', 'n_clusters': n, 'silhouette': silhouette, 'db': db, 'ch': ch})
    return results

# 运行所有实验
all_results = {}
for feature_name, X in final_sets.items():
    kmeans_res = run_kmeans(X)
    gmm_res = run_gmm(X)
    hc_res = run_hierarchical(X)
    all_results[feature_name] = kmeans_res + gmm_res + hc_res

# 提取最佳结果
best_results = {}
for feature_name, results in all_results.items():
    best = max(results, key=lambda x: x['silhouette'])
    best_results[feature_name] = best

# 结果表格
import pandas as pd
results_table = pd.DataFrame.from_dict(best_results, orient='index')
print(results_table)
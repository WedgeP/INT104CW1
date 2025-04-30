# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from evaluate import evaluate_clustering  # 自定义函数，将结果保存为CSV
import warnings
from sklearn.exceptions import ConvergenceWarning

np.random.seed(42)

# 读取数据
df = pd.read_csv('student_data.csv')
mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
if df['Programme'].dtype == 'int64' or df['Programme'].iloc[0] in [1, 2, 3, 4]:
    df['Programme'] = df['Programme'].map(mapping)

# 预处理函数
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
        basic_cols = list(dict.fromkeys(basic_cols)) or numeric_df.columns[:2].tolist()
        preprocessors['columns']['去除年级'] = basic_cols
    else:
        basic_cols = preprocessors['columns']['去除年级']
    for col in basic_cols:
        if col not in numeric_df.columns:
            numeric_df[col] = 0
    feature_sets['去除年级'] = numeric_df[basic_cols].values
    if mode == 'train':
        programme_cols = [col for col in numeric_df.columns if 'programme' in col.lower()]
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
    return (feature_sets, preprocessors) if mode == 'train' else feature_sets

feature_sets, preprocessors = process_data(df, mode='train')

# 特征转换
transformed_sets = {}
for name, X in feature_sets.items():
    transformed_sets[f'归一化_{name}'] = MinMaxScaler().fit_transform(X)
    transformed_sets[f'标准化_{name}'] = StandardScaler().fit_transform(X)
    transformed_sets[f'正则缩放_{name}'] = Normalizer().fit_transform(X)

# 降维
final_sets = {}
for name, X in transformed_sets.items():
    final_sets[f'PCA_{name}'] = PCA(n_components=min(X.shape[1], 10)).fit_transform(X)
    final_sets[f'ICA_{name}'] = FastICA(n_components=min(X.shape[1], 10), random_state=42).fit_transform(X)
    final_sets[f'TSNE_{name}'] = TSNE(n_components=2, random_state=42, init='random').fit_transform(X)


# 聚类方法
def eval(X, labels):
    try:
        return silhouette_score(X, labels), davies_bouldin_score(X, labels), calinski_harabasz_score(X, labels)
    except:
        return -1, float('inf'), -1


def run_kmeans(X, n=4):
    results = []
    for init in ['k-means++', 'random']:
        for max_iter in [100, 300, 500]:
            for tol in [1e-4, 1e-6]:
                for algorithm in ['lloyd', 'elkan']:
                    model = KMeans(n_clusters=n, init=init, max_iter=max_iter,
                                   tol=tol, algorithm=algorithm, random_state=42)
                    labels = model.fit_predict(X)
                    silhouette = eval(X, labels)[0]
                    results.append({
                        'method': 'kmeans',
                        'n_clusters': n,
                        'init': init,
                        'max_iter': max_iter,
                        'tol': tol,
                        'algorithm': algorithm,
                        'silhouette': silhouette
                    })
    return results


def run_gmm(X, n=4):
    # Suppress ConvergenceWarning specifically for GMM fitting
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    results = []
    for cov in ['full', 'tied', 'diag', 'spherical']:
        for max_iter in [100, 200]:
            for tol in [1e-3, 1e-6]:
                for init_params in ['kmeans', 'random']:
                    try: # Add try block for robustness
                        model = GaussianMixture(n_components=n, covariance_type=cov, max_iter=max_iter,
                                                tol=tol, init_params=init_params, random_state=42)
                        labels = model.fit_predict(X)
                        silhouette = eval(X, labels)[0]
                        results.append({
                            'method': 'gmm',
                            'n_components': n,
                            'cov_type': cov,
                            'max_iter': max_iter,
                            'tol': tol,
                            'init_params': init_params,
                            'silhouette': silhouette
                        })
                    except Exception as e: # Add except block to handle potential errors during fitting
                        print(f"Skipping GMM combination due to error: {e}")
                        results.append({
                            'method': 'gmm',
                            'n_components': n,
                            'cov_type': cov,
                            'max_iter': max_iter,
                            'tol': tol,
                            'init_params': init_params,
                            'silhouette': -1 # Indicate failure
                        })
    # Restore default warning behavior if needed elsewhere
    # warnings.filterwarnings("default", category=ConvergenceWarning)
    return results


def run_hc(X, n=4):
    results = []
    for linkage in ['ward', 'complete', 'average', 'single']:
        if linkage == 'ward':
            # ward只能使用euclidean距离，并且不需要指定affinity参数
            model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
            labels = model.fit_predict(X)
            silhouette = eval(X, labels)[0]
            results.append({
                'method': 'hierarchical',
                'n_clusters': n,
                'linkage': linkage,
                'affinity': 'euclidean',  # 记录用于后续参考，但不作为参数传递
                'silhouette': silhouette
            })
        else:
            # 其他linkage可以使用多种距离度量
            for affinity in ['euclidean', 'manhattan', 'cosine']:
                try:
                    model = AgglomerativeClustering(n_clusters=n, linkage=linkage, affinity=affinity)
                    labels = model.fit_predict(X)
                    silhouette = eval(X, labels)[0]
                    results.append({
                        'method': 'hierarchical',
                        'n_clusters': n,
                        'linkage': linkage,
                        'affinity': affinity,
                        'silhouette': silhouette
                    })
                except Exception as e:
                    print(f"跳过无效组合: linkage={linkage}, affinity={affinity}, 错误: {e}")
    return results
# 运行实验
all_results = {}
for name, X in final_sets.items():
    all_results[name] = run_kmeans(X) + run_gmm(X) + run_hc(X)

# 选择最佳模型
best_results = {name: max(results, key=lambda r: r['silhouette']) for name, results in all_results.items()}
best_feature_name, best_model_info = max(best_results.items(), key=lambda x: x[1]['silhouette'])

print("✅ 最佳特征:", best_feature_name)
print("✅ 最佳模型:", best_model_info)

# 训练最佳模型
X_train_best = final_sets[best_feature_name]
method = best_model_info['method']
n_clusters = best_model_info.get('n_clusters') or best_model_info.get('n_components')

if method == 'kmeans':
    best_model = KMeans(
        n_clusters=n_clusters,
        init=best_model_info.get('init', 'k-means++'),
        max_iter=best_model_info.get('max_iter', 300),
        tol=best_model_info.get('tol', 1e-4),
        algorithm=best_model_info.get('algorithm', 'lloyd'),
        random_state=42
    )
elif method == 'gmm':
    best_model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=best_model_info.get('cov_type', 'full'),
        max_iter=best_model_info.get('max_iter', 100),
        tol=best_model_info.get('tol', 1e-3),
        init_params=best_model_info.get('init_params', 'kmeans'),
        random_state=42
    )
elif method == 'hierarchical':
    linkage = best_model_info.get('linkage', 'ward')
    if linkage == 'ward':
        best_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
    else:
        best_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=best_model_info.get('affinity', 'euclidean')
        )
best_model.fit(X_train_best)

# 预测测试数据
df_test = pd.read_csv('test_data.csv')
if 'Programme' in df_test.columns and (df_test['Programme'].dtype == 'int64' or df_test['Programme'].iloc[0] in [1, 2, 3, 4]):
    df_test['Programme'] = df_test['Programme'].map(mapping)

test_feature_sets = process_data(df_test, preprocessors=preprocessors, mode='test')
base_name = best_feature_name.split('_')[-1]
X_test_base = test_feature_sets[base_name]

if best_feature_name.startswith('PCA'):
    X_test_final = PCA(n_components=min(X_train_best.shape[1], 10)).fit(X_train_best).transform(X_test_base)
elif best_feature_name.startswith('ICA'):
    X_test_final = FastICA(n_components=min(X_train_best.shape[1], 10), random_state=42).fit(X_train_best).transform(X_test_base)
elif best_feature_name.startswith('TSNE'):
    print("⚠️ Warning: Applying TSNE to test data separately. Embedding might not align with training data.")
    tsne_transformer = TSNE(n_components=2, random_state=42, init='random')
    X_test_final = tsne_transformer.fit_transform(X_test_base)
else:
    X_test_final = X_test_base

# 预测并保存
if method == 'hierarchical':
    linkage = best_model_info.get('linkage', 'ward')
    affinity = best_model_info.get('affinity', 'euclidean') if linkage != 'ward' else 'euclidean'
    hc_predict_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    predicted_labels = hc_predict_model.fit_predict(X_test_final)
elif best_feature_name.startswith('TSNE'):
    predicted_labels = best_model.predict(X_test_final)
else:
    predicted_labels = best_model.predict(X_test_final)

# 保存预测
evaluate_clustering(X_test_final, predicted_labels, output_file='predicted_labels.csv')
print("🎉 结果已保存到 predicted_labels.csv")
# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import itertools
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示 (Mac系统)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统支持的字体
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("./student_data.csv")

# 数据探索
print("数据基本信息:")
print(df.info())
print("\n数据统计摘要:")
print(df.describe())

# 检查缺失值
print("\n缺失值数量:")
print(df.isnull().sum())

# 查看各系别、年级、性别的学生分布
print("\n各系别学生数量:")
print(df['Programme'].value_counts())

print("\n各年级学生数量:")
print(df['Grade'].value_counts())

print("\n各性别学生数量:")
print(df['Gender'].value_counts())

print("\n各系别年级性别组合的学生数量:")
print(df.groupby(['Programme', 'Grade', 'Gender']).size())

# 数据可视化：各系别总分分布
plt.figure(figsize=(12, 6))
sns.boxplot(x='Programme', y='Total', data=df)
plt.title('各系别总分分布')
plt.xlabel('系别')
plt.ylabel('总分')
plt.show()

# 各系别各题目得分分布
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Total']):
    sns.boxplot(x='Programme', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'各系别{col}题目得分分布')
    axes[i].set_xlabel('系别')
    axes[i].set_ylabel('得分')

plt.tight_layout()
plt.show()

# 添加双峰分布分析（高斯混合模型）
results_gmm = []

for grade in df['Grade'].unique():
    grade_data = df[df['Grade'] == grade]

    for programme in grade_data['Programme'].unique():
        prog_data = grade_data[grade_data['Programme'] == programme]

        for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Total']:
            data = prog_data[col].dropna().values.reshape(-1, 1)

            if len(data) < 5:  # 样本太少，跳过
                continue

            # 拟合单峰正态分布
            params_norm = stats.norm.fit(data)
            aic_norm = 2 * 2 + 2 * (-np.sum(stats.norm.logpdf(data, *params_norm)))

            # 拟合双峰高斯混合模型
            try:
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(data)
                aic_gmm = 2 * 6 + 2 * (-gmm.score(data) * len(data))  # 6个参数：2均值+2方差+2权重

                # 比较AIC选择更好的模型
                best_model = "双峰高斯混合" if aic_gmm < aic_norm else "单峰正态"
                best_aic = min(aic_norm, aic_gmm)

                results_gmm.append([grade, programme, col, best_model, best_aic])
                print(f"年级 {grade}, 系别 {programme}, {col}: 最佳拟合分布 {best_model}, AIC: {best_aic:.4f}")
            except:
                continue

# 将拟合结果转换为DataFrame
results_df = pd.DataFrame(results_gmm, columns=['Grade', 'Programme', 'Question', 'Best_Model', 'AIC'])

# 可视化拟合结果: 哪些题目更适合双峰分布
plt.figure(figsize=(12, 6))
best_models = results_df.pivot_table(index=['Programme', 'Grade'], columns='Question', values='Best_Model',
                                     aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None)
sns.heatmap(best_models == '双峰高斯混合', cmap='YlGnBu', cbar_kws={'label': '是否为双峰分布'})
plt.title('各系别各年级各题目的最佳拟合分布类型')
plt.xlabel('题目')
plt.ylabel('系别-年级')
plt.show()

# 重点分析年级3系别C的Q1分布（根据前面已知是双峰）
grade3_C = df[(df['Grade'] == 3) & (df['Programme'] == 'C')]
data_q1 = grade3_C['Q1'].dropna().values.reshape(-1, 1)

if len(data_q1) > 0:
    plt.figure(figsize=(10, 6))

    # 绘制原始数据直方图
    plt.hist(data_q1, bins=8, density=True, alpha=0.6, color='g', label='原始数据')

    # 拟合单峰正态分布
    mu, sigma = stats.norm.fit(data_q1)
    x = np.linspace(min(data_q1), max(data_q1), 100).reshape(-1, 1)
    pdf_norm = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf_norm, 'b--', label=f'正态分布 (μ={mu:.2f}, σ={sigma:.2f})')

    # 拟合双峰高斯混合模型
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data_q1)
    pdf_gmm = np.exp(gmm.score_samples(x))
    plt.plot(x, pdf_gmm, 'r-', label='双峰高斯混合模型')

    plt.title('系别C年级3的Q1题目得分分布拟合')
    plt.xlabel('得分')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 显示GMM参数
    print("\nGMM模型参数:")
    for i, (mean, covar, weight) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        print(f"组件 {i + 1}: 均值={mean[0]:.2f}, 方差={covar[0][0]:.2f}, 权重={weight:.2f}")


# 基于贝叶斯构建不同年级的预测模型
# 创建特征工程函数
def create_features(df_input, with_interactions=True):
    X = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Gender']]

    # 添加交互特征
    if with_interactions:
        # 添加比例特征
        esp = 0.0000001
        for i, j in itertools.combinations(range(1, 6), 2):
            X[f'Q{i}_to_Q{j}'] = df_input[f'Q{i}'] / (df_input[f'Q{j}'] + esp)

        # 添加统计特征
        X['mean'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)
        X['std'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].std(axis=1)
        X['max_min_diff'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].max(axis=1) - df_input[
            ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].min(axis=1)

    # 将Gender转为数值型
    X['Gender'] = X['Gender'].astype(int)

    return X


# 按年级分别建模
results_by_grade = {}
feature_importances = {}
confusion_matrices = {}

for grade in df['Grade'].unique():
    grade_df = df[df['Grade'] == grade]
    if grade_df.size < 30:  # 数据不足，跳过
        continue

    # 特征工程
    X = create_features(grade_df)
    y = grade_df['Programme']

    # 过滤掉样本数量太少的类别
    programme_counts = y.value_counts()
    valid_programmes = programme_counts[programme_counts >= 3].index  # 至少需要3个样本
    if len(valid_programmes) < 2:  # 需要至少2个有效类别
        print(f"年级 {grade} 没有足够的样本数量，跳过")
        continue

    mask = y.isin(valid_programmes)
    X = X[mask]
    y = y[mask]

    # 使用交叉验证代替简单的训练测试集分割
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=min(10, X.shape[1]))),
        ('classifier', GaussianNB())
    ])

    # 训练与评估
    cv_scores = []
    all_y_pred = []
    all_y_true = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cv_scores.append((y_pred == y_test).mean())

        all_y_pred.extend(y_pred)
        all_y_true.extend(y_test)

    # 计算平均准确率
    accuracy = np.mean(cv_scores)
    results_by_grade[grade] = accuracy

    # 保存混淆矩阵
    confusion_matrices[grade] = confusion_matrix(all_y_true, all_y_pred)

    # 训练完整模型获取特征重要性
    pipeline.fit(X, y)
    selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)
    selected_features = X.columns[selected_indices]
    feature_scores = pipeline.named_steps['feature_selection'].scores_[selected_indices]
    feature_importances[grade] = dict(zip(selected_features, feature_scores))

    print(f"\n年级 {grade} 的系别预测准确率: {accuracy:.4f}")
    print(f"年级 {grade} 的分类报告:")
    print(classification_report(all_y_true, all_y_pred))
    print(f"最重要的特征: {selected_features.tolist()}")

# 可视化特征重要性
plt.figure(figsize=(14, 8))
for grade, features in feature_importances.items():
    plt.subplot(1, len(feature_importances), list(feature_importances.keys()).index(grade) + 1)
    features_df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Importance': list(features.values())
    }).sort_values('Importance', ascending=False).head(10)

    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title(f'年级 {grade} 最重要特征')
    plt.tight_layout()

plt.show()

# 可视化混淆矩阵
plt.figure(figsize=(14, 5 * len(confusion_matrices)))
for i, (grade, cm) in enumerate(confusion_matrices.items()):
    plt.subplot(len(confusion_matrices), 1, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(df['Programme'].unique()),
                yticklabels=sorted(df['Programme'].unique()))
    plt.title(f'年级 {grade} 预测混淆矩阵')
    plt.xlabel('预测系别')
    plt.ylabel('实际系别')

plt.tight_layout()
plt.show()

# 三维可视化：不同系别在Q1-Q3空间的分布
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for programme in df['Programme'].unique():
    prog_data = df[df['Programme'] == programme]
    ax.scatter(prog_data['Q1'], prog_data['Q2'], prog_data['Q3'],
               label=f'系别 {programme}', alpha=0.7)

ax.set_xlabel('Q1分数')
ax.set_ylabel('Q2分数')
ax.set_zlabel('Q3分数')
ax.set_title('不同系别在Q1-Q3空间的分布')
ax.legend()
plt.show()

# 绘制成对特征之间的关系
sns.pairplot(df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Programme']], hue='Programme')
plt.suptitle('各系别在不同题目上的得分分布关系', y=1.02)
plt.show()


# 基于贝叶斯概率更新的学生系别预测演示
def predict_programme_with_bayes(new_scores, grade, gender):
    # 先验概率：各系别在指定年级中的比例
    grade_df = df[df['Grade'] == grade]
    prior_probs = grade_df['Programme'].value_counts(normalize=True).to_dict()

    # 似然函数：各系别在各题目上的得分分布
    likelihoods = {}
    posterior_probs = {}

    for prog in prior_probs.keys():
        prog_data = grade_df[grade_df['Programme'] == prog]

        # 初始化后验概率为先验概率
        posterior_probs[prog] = prior_probs[prog]

        # 计算各题目的似然
        for i, score in enumerate(new_scores, start=1):
            col = f'Q{i}'
            # 过滤性别
            gender_prog_data = prog_data[prog_data['Gender'] == gender]

            if len(gender_prog_data) < 3:  # 数据不足，使用全部数据
                gender_prog_data = prog_data

            data = gender_prog_data[col].dropna()

            if len(data) < 3:  # 数据仍然不足
                continue

            # 检查是单峰还是双峰
            is_bimodal = False
            for row in results_gmm:
                if row[0] == grade and row[1] == prog and row[2] == col and row[3] == '双峰高斯混合':
                    is_bimodal = True
                    break

            if is_bimodal and len(data) >= 5:
                # 使用双峰高斯混合模型
                gmm = GaussianMixture(n_components=2).fit(data.values.reshape(-1, 1))
                likelihood = np.exp(gmm.score_samples(np.array([[score]])))[0]
            else:
                # 使用单峰正态分布
                mu, sigma = data.mean(), data.std()
                if sigma < 0.1:  # 防止sigma过小
                    sigma = 0.1
                likelihood = stats.norm.pdf(score, mu, sigma)

            # 更新后验概率
            posterior_probs[prog] *= likelihood

    # 归一化后验概率
    total = sum(posterior_probs.values())
    if total > 0:
        for prog in posterior_probs:
            posterior_probs[prog] /= total

    return posterior_probs


# 测试贝叶斯预测：选择几个典型案例
test_cases = [
    {'scores': [8, 6, 14, 10, 2], 'grade': 3, 'gender': 1, 'expected': 'C'},  # 高分C系学生
    {'scores': [4, 2, 8, 8, 0], 'grade': 3, 'gender': 2, 'expected': 'C'},  # 中分C系学生
    {'scores': [2, 4, 4, 0, 0], 'grade': 3, 'gender': 2, 'expected': 'C'},  # 低分C系学生
    {'scores': [8, 6, 14, 10, 3], 'grade': 2, 'gender': 2, 'expected': 'A'}  # A系学生
]

print("\n贝叶斯预测演示:")
for case in test_cases:
    probs = predict_programme_with_bayes(case['scores'], case['grade'], case['gender'])
    predicted = max(probs, key=probs.get)
    print(f"学生得分: {case['scores']}, 年级: {case['grade']}, 性别: {case['gender']}")
    print(f"预测系别: {predicted}, 预期系别: {case['expected']}")
    print(f"各系别概率: {', '.join([f'{k}: {v:.4f}' for k, v in probs.items()])}")
    print()
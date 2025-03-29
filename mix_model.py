# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# 设置随机种子和中文显示
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("./student_data.csv")

# 系别映射（如果需要）
mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
if df['Programme'].dtype == 'int64' or df['Programme'].iloc[0] in [1, 2, 3, 4]:
    df['Programme'] = df['Programme'].map(mapping)

# ============ PCA模型训练 ============
print("\n开始训练PCA全局模型...")

# 提取特征和目标变量用于PCA模型
X_pca = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender']]  # 确保只使用这些特征
y_pca = df['Programme']

# PCA + 随机森林管道
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 定义参数网格
pca_param_grid = {
    'pca__n_components': [2, 3, 4, 5],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 15]
}

# 使用网格搜索查找最佳参数
pca_grid_search = GridSearchCV(pca_pipeline, pca_param_grid, cv=5, scoring='accuracy')
pca_grid_search.fit(X_pca, y_pca)
print(f"PCA模型最佳参数: {pca_grid_search.best_params_}")
print(f"PCA模型交叉验证分数: {pca_grid_search.best_score_:.4f}")

# 获取最佳PCA模型
pca_best_model = pca_grid_search.best_estimator_

# ============ 贝叶斯模型（按年级训练） ============
print("\n开始按年级训练贝叶斯模型...")

# 按年级划分训练集
grade_models = {}


# 定义高级特征工程函数
def create_advanced_features(df_input):
    eps = 0.0000001
    # 基础特征
    X = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Gender']].copy()

    # 题目得分比例特征
    for i in range(1, 6):
        for j in range(i + 1, 6):
            X[f'Q{i}_to_Q{j}'] = df_input[f'Q{i}'] / (df_input[f'Q{j}'] + eps)

    # 统计特征
    X['mean_score'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)
    X['std_score'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].std(axis=1)
    X['range_score'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].max(axis=1) - df_input[
        ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].min(axis=1)
    X['cv_score'] = X['std_score'] / (X['mean_score'] + eps)  # 变异系数

    # 归一化题目分数
    max_scores = [8, 8, 14, 10, 6]  # 每题满分
    for i in range(1, 6):
        X[f'Q{i}_norm'] = df_input[f'Q{i}'] / max_scores[i - 1]

    return X


# 按年级分别训练模型
for grade in df['Grade'].unique():
    print(f"开始训练年级 {grade} 的模型")
    grade_df = df[df['Grade'] == grade]

    X_grade = create_advanced_features(grade_df)
    y_grade = grade_df['Programme']

    # 检查样本量
    programme_counts = y_grade.value_counts()
    valid_programmes = programme_counts[programme_counts >= 5].index
    if len(valid_programmes) < 2:
        print(f"年级 {grade} 的有效系别数量不足，将使用PCA全局模型")
        continue

    # 过滤有效系别
    mask = y_grade.isin(valid_programmes)
    X_grade = X_grade[mask]
    y_grade = y_grade[mask]

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 创建模型
    rf = RandomForestClassifier(random_state=42)

    # 超参数网格
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 15],
        'min_samples_split': [2, 5]
    }

    # 网格搜索
    grid_rf = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='accuracy')
    grid_rf.fit(X_grade, y_grade)

    # 保存最佳模型
    grade_models[grade] = {
        'model': grid_rf.best_estimator_,
        'features': list(X_grade.columns),
        'accuracy': grid_rf.best_score_,
        'best_params': grid_rf.best_params_,
        'valid_programmes': list(valid_programmes)
    }

    print(f"年级 {grade} 模型准确率: {grid_rf.best_score_:.4f}")


# =========== 融合预测函数 ===========
def hybrid_predict(scores, grade, gender):
    """
    结合贝叶斯和PCA模型进行预测

    参数:
        scores: 学生的各科成绩
        grade: 学生年级
        gender: 学生性别

    返回:
        预测的系别
        各系别概率字典
        使用的模型类型 ('bayes', 'pca', 'ensemble')
    """
    bayes_probs = {}

    # 尝试使用贝叶斯模型
    if grade in grade_models:
        model_info = grade_models[grade]
        model = model_info['model']

        # 构建特征
        eps = 0.0000001
        feature_dict = {
            'Q1': scores[0], 'Q2': scores[1], 'Q3': scores[2],
            'Q4': scores[3], 'Q5': scores[4], 'Gender': gender
        }

        # 计算比例特征
        for i in range(1, 6):
            for j in range(i + 1, 6):
                feature_dict[f'Q{i}_to_Q{j}'] = scores[i - 1] / (scores[j - 1] + eps)

        # 计算统计特征
        feature_dict['mean_score'] = np.mean(scores)
        feature_dict['std_score'] = np.std(scores)
        feature_dict['range_score'] = max(scores) - min(scores)
        feature_dict['cv_score'] = feature_dict['std_score'] / (feature_dict['mean_score'] + eps)

        # 归一化分数
        max_scores = [8, 8, 14, 10, 6]
        for i in range(1, 6):
            feature_dict[f'Q{i}_norm'] = scores[i - 1] / max_scores[i - 1]

        # 转换为DataFrame
        X_input = pd.DataFrame([feature_dict])

        # 确保特征列顺序与训练一致
        X_input = X_input[model_info['features']]

        # 预测概率
        probs_array = model.predict_proba(X_input)
        classes = model.classes_
        bayes_probs = {cls: prob for cls, prob in zip(classes, probs_array[0])}

    # 使用PCA模型进行预测
    pca_input = pd.DataFrame([[scores[0], scores[1], scores[2], scores[3], scores[4],
                               grade, gender]],
                             columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender'])

    pca_probs_array = pca_best_model.predict_proba(pca_input)
    pca_classes = pca_best_model.classes_
    pca_probs = {cls: prob for cls, prob in zip(pca_classes, pca_probs_array[0])}

    # 决定使用哪种模型
    if not bayes_probs:
        # 贝叶斯模型不可用，使用PCA模型
        final_probs = pca_probs
        model_type = 'pca'
    else:
        # 按系别融合两个模型的结果
        # PCA模型对C类敏感，其他类别使用贝叶斯
        final_probs = {}
        all_classes = set(list(bayes_probs.keys()) + list(pca_probs.keys()))

        for cls in all_classes:
            if cls == 'C':
                # 对C类，给PCA更大权重
                bayes_weight = 0.3
                pca_weight = 0.7
            else:
                # 其他类型，贝叶斯模型权重更大
                bayes_weight = 0.7
                pca_weight = 0.3

            # 计算加权概率
            bayes_prob = bayes_probs.get(cls, 0.0)
            pca_prob = pca_probs.get(cls, 0.0)
            final_probs[cls] = (bayes_weight * bayes_prob + pca_weight * pca_prob) / (bayes_weight + pca_weight)

        model_type = 'ensemble'

    # 获取最大概率的系别
    predicted = max(final_probs, key=final_probs.get)
    return predicted, final_probs, model_type


# =========== 测试模型效果 ===========

# 读取测试数据
test_df = pd.read_csv("./test_data.csv")

# 系别映射（如果需要）
if test_df['Programme'].dtype == 'int64' or test_df['Programme'].iloc[0] in [1, 2, 3, 4]:
    test_df['Programme'] = test_df['Programme'].map(mapping)

# 提取测试集特征和标签
X_test = test_df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender']]  # 确保与训练特征一致
y_test = test_df['Programme']

# 单独使用PCA模型进行预测
pca_predictions = pca_best_model.predict(X_test)
print("\nPCA模型测试集结果:")
print("混淆矩阵:")
print(confusion_matrix(y_test, pca_predictions))
print("分类报告:")
print(classification_report(y_test, pca_predictions))
print(f"准确率: {accuracy_score(y_test, pca_predictions):.4f}")

# 进行混合模型预测
predictions = []
probabilities = []
model_types = []

for _, row in X_test.iterrows():
    scores = [row['Q1'], row['Q2'], row['Q3'], row['Q4'], row['Q5']]
    grade = row['Grade']
    gender = row['Gender']

    predicted, probs, model_type = hybrid_predict(scores, grade, gender)
    predictions.append(predicted)
    probabilities.append(probs)
    model_types.append(model_type)

# 评估融合模型性能
print("\n融合模型测试集结果:")
print("混淆矩阵:")
print(confusion_matrix(y_test, predictions))
print("分类报告:")
print(classification_report(y_test, predictions))
print(f"准确率: {accuracy_score(y_test, predictions):.4f}")

# 分析不同模型类型的使用情况
model_type_counts = pd.Series(model_types).value_counts()
print("\n模型使用情况:")
print(model_type_counts)

# 分析各种模型对不同系别的预测性能
model_type_df = pd.DataFrame({
    'True': y_test,
    'Predicted': predictions,
    'Model': model_types
})

# 统计各系别的预测情况
print("\n各系别预测结果统计:")
for programme in sorted(y_test.unique()):
    mask = y_test == programme
    if mask.sum() > 0:
        correct = sum(y_test[mask] == np.array(predictions)[mask])
        total = mask.sum()
        print(f"系别 {programme}: 准确率 {correct / total:.4f} ({correct}/{total})")
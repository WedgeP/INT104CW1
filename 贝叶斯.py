# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# 设置随机种子和中文显示
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 按年级划分训练集
grade_models = {}

# 读取数据
df = pd.read_csv("./student_data.csv")


# 定义高级特征工程函数
def create_advanced_features(df_input):
    eps = 0.0000001
    # 基础特征
    X = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Gender']]

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
    print(f"\n开始训练年级 {grade} 的模型")
    grade_df = df[df['Grade'] == grade]

    X_grade = create_advanced_features(grade_df)
    y_grade = grade_df['Programme']

    # 检查样本量
    programme_counts = y_grade.value_counts()
    valid_programmes = programme_counts[programme_counts >= 5].index
    if len(valid_programmes) < 2:
        print(f"年级 {grade} 的有效系别数量不足，跳过")
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
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # 网格搜索
    grid_rf = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='accuracy')
    grid_rf.fit(X_grade, y_grade)

    # 保存最佳模型
    grade_models[grade] = {
        'model': grid_rf.best_estimator_,
        'features': X_grade.columns,
        'accuracy': grid_rf.best_score_,
        'best_params': grid_rf.best_params_
    }

    print(f"年级 {grade} 模型训练完成，最佳准确率: {grid_rf.best_score_:.4f}")
    print(f"最佳参数: {grid_rf.best_params_}")


# 预测函数
def predict_programme_with_bayes(scores, grade, gender):
    """
    使用训练好的模型预测某个学生的专业归属概率。

    参数:
        scores (list): 学生的各科成绩
        grade (int): 学生所在年级
        gender (int): 学生性别
    返回:
        dict: 各个专业的概率分布
    """
    # 检查给定年级是否有模型
    if grade not in grade_models:
        print(f"警告：年级 {grade} 没有训练模型")
        return {}

    # 获取对应年级的模型
    model_info = grade_models[grade]
    model = model_info['model']

    # 构建输入特征
    eps = 0.0000001  # 防止除零错误

    # 创建特征字典
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

    # 计算归一化分数
    max_scores = [8, 8, 14, 10, 6]  # 每题满分
    for i in range(1, 6):
        feature_dict[f'Q{i}_norm'] = scores[i - 1] / max_scores[i - 1]

    # 转换为模型可用的特征向量
    X_input = pd.DataFrame([feature_dict])

    # 确保特征顺序与训练时一致
    feature_cols = model_info['features']
    X_input = X_input[feature_cols]

    # 获取预测概率
    probs_array = model.predict_proba(X_input)

    # 将概率与对应类别组合
    classes = model.classes_
    probs_dict = {cls: prob for cls, prob in zip(classes, probs_array[0])}

    return probs_dict


# 测试案例
test_cases = [
    {'scores': [4, 0, 0, 0, 2], 'grade': 2, 'gender': 2, 'expected': 'A'},  # 低分A系学生
    {'scores': [8, 2, 2, 0, 0], 'grade': 2, 'gender': 2, 'expected': 'D'},  # 高分D系学生
    {'scores': [8, 2, 2, 7, 1], 'grade': 2, 'gender': 2, 'expected': 'A'},  # 中分A系学生
    {'scores': [8, 6, 14, 10, 3], 'grade': 2, 'gender': 2, 'expected': 'A'},  # A系学生
    {'scores': [8, 6, 14, 8, 0], 'grade': 2, 'gender': 2, 'expected': 'A'},
    {'scores': [8, 6, 12, 5, 2], 'grade': 2, 'gender': 2, 'expected': 'C'}
]

# 读取测试数据
test_df = pd.read_csv("./test_data.csv")

# 系别映射
mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}  # 可以根据需要修改映射关系

# 转换系别编码（如果需要）
if test_df['Programme'].dtype == 'int64' or test_df['Programme'].iloc[0] in [1, 2, 3, 4]:
    test_df['Programme'] = test_df['Programme'].map(mapping)

# 提取测试集特征和标签
X_test = test_df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender']]
y_test = test_df['Programme']

# 按年级分别进行预测
predictions = []
probabilities = []

for _, row in X_test.iterrows():
    scores = [row['Q1'], row['Q2'], row['Q3'], row['Q4'], row['Q5']]
    grade = row['Grade']
    gender = row['Gender']

    # 获取预测概率
    probs = predict_programme_with_bayes(scores, grade, gender)

    if not probs:
        # 如果预测失败，使用最常见的系别作为默认预测
        predictions.append(y_test.mode()[0])
        prob_dict = {prog: 0.0 for prog in set(y_test)}
        probabilities.append(prob_dict)
    else:
        # 获取概率最大的系别作为预测结果
        predicted = max(probs, key=probs.get)
        predictions.append(predicted)
        probabilities.append(probs)

# 评估模型性能
print("\n测试集混淆矩阵:")
print(confusion_matrix(y_test, predictions))
print("\n测试集分类报告:")
print(classification_report(y_test, predictions))

# 计算总体准确率
accuracy = accuracy_score(y_test, predictions)
print(f"测试集准确率: {accuracy:.4f}")
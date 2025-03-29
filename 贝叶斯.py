import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

# 按年级划分训练集
grade_models = {}


# 定义特征工程函数
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

    # 归一化题目分数（考虑题目满分不同）
    max_scores = [8, 8, 14, 10, 6]  # 每题满分
    for i in range(1, 6):
        X[f'Q{i}_norm'] = df_input[f'Q{i}'] / max_scores[i - 1]

    return X


# 读取数据
df = pd.read_csv("./student_data.csv")

# 按年级分别训练模型
for grade in df['Grade'].unique():
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

    # 创建模型
    rf = RandomForestClassifier(random_state=42)

    # 简化参数
    rf_params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 2
    }

    # 设置模型参数
    rf.set_params(**rf_params)

    # 训练模型
    rf.fit(X_grade, y_grade)

    # 保存模型
    grade_models[grade] = {
        'model': rf,
        'features': X_grade.columns,
        'accuracy': None,
        'best_params': rf_params
    }

    print(f"年级 {grade} 模型训练完成")


# 预测函数
def predict_programme_with_bayes(scores, grade, gender):
    """
    使用训练好的模型预测某个学生的专业归属概率。

    参数:
        scores (list): 学生的各科成绩
        grade (int): 学生所在年级
        gender (int): 学生性别（1=男，2=女）

    返回:
        dict: 各个专业的概率分布，例如 {'A': 0.6, 'B': 0.3, 'C': 0.1}
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


# 测试集验证
def validate_on_test_set(test_file_path):
    # 导入测试数据
    test_df = pd.read_csv(test_file_path)
    print(f"验证数据集样本数量: {len(test_df)}")

    # 系别映射（如果需要）
    mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

    # 检查是否需要转换系别编码
    if test_df['Programme'].dtype == 'int64' or test_df['Programme'].iloc[0] in ['1', '2', '3', '4']:
        test_df['Programme'] = test_df['Programme'].astype(str).map(mapping)

    # 初始化结果
    results = {
        'actual': [],
        'predicted': [],
        'correct': []
    }

    # 按年级分组进行验证
    for grade in sorted(test_df['Grade'].unique()):
        grade_test = test_df[test_df['Grade'] == grade]
        print(f"\n年级 {grade} 验证样本数: {len(grade_test)}")

        # 如果该年级没有训练模型，则跳过
        if grade not in grade_models:
            print(f"⚠️ 年级 {grade} 没有训练好的模型，跳过")
            continue

        correct_count = 0
        total_count = 0

        for _, student in grade_test.iterrows():
            scores = [student['Q1'], student['Q2'], student['Q3'], student['Q4'], student['Q5']]
            gender = student['Gender']
            actual_programme = student['Programme']

            # 预测系别
            probs = predict_programme_with_bayes(scores, grade, gender)
            if not probs:
                continue

            predicted = max(probs, key=probs.get)
            is_correct = predicted == actual_programme

            # 保存结果
            results['actual'].append(actual_programme)
            results['predicted'].append(predicted)
            results['correct'].append(is_correct)

            correct_count += is_correct
            total_count += 1

        # 计算该年级的准确率
        grade_accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"年级 {grade} 预测准确率: {grade_accuracy:.4f} ({correct_count}/{total_count})")

    # 计算总体准确率
    overall_accuracy = sum(results['correct']) / len(results['correct']) if results['correct'] else 0
    print(f"\n总体准确率: {overall_accuracy:.4f} ({sum(results['correct'])}/{len(results['correct'])})")

    return results


# 简单调用示例
if __name__ == "__main__":
    # 训练模型
    # 注意：grade_models变量已经在上面的代码中填充了训练好的模型

    # 在测试集上验证
    validate_on_test_set("./test_data.csv")

    # 预测单个学生示例
    test_student = {'scores': [8, 6, 14, 10, 3], 'grade': 2, 'gender': 2}
    prediction = predict_programme_with_bayes(
        test_student['scores'],
        test_student['grade'],
        test_student['gender']
    )

    if prediction:
        predicted_programme = max(prediction, key=prediction.get)
        print(f"预测系别: {predicted_programme}")
        print(f"各系别概率: {prediction}")
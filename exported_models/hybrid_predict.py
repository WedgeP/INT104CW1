
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from scipy import stats
from flexBYS import FlexibleBayesClassifier

def load_models(models_dir='../exported_models'):
    models = {}
    models['pca'] = joblib.load(os.path.join(models_dir, 'pca_model.joblib'))
    models['random_forest'] = joblib.load(os.path.join(models_dir, 'random_forest_models.joblib'))
    models['bayes'] = joblib.load(os.path.join(models_dir, 'bayes_models.joblib'))
    return models

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
    X['range_score'] = df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].max(axis=1) - df_input[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].min(axis=1)
    X['cv_score'] = X['std_score'] / (X['mean_score'] + eps)

    # 归一化题目分数
    max_scores = [8, 8, 14, 10, 6]
    for i in range(1, 6):
        X[f'Q{i}_norm'] = df_input[f'Q{i}'] / max_scores[i-1]

    return X

def _score_bayes_sample(x, model_dict, feature_indices):
    """计算样本在贝叶斯模型中的对数似然"""
    log_likelihood = 0.0

    for feature_idx in feature_indices:
        if feature_idx not in model_dict:
            continue

        model = model_dict[feature_idx]
        value = x[feature_idx]

        # 根据分布类型计算似然
        if hasattr(model, 'score_samples'):  # 如果模型有score_samples方法（如GaussianMixture）
            ll = model.score_samples(np.array([[value]]))
            log_likelihood += ll[0]

        elif isinstance(model, dict) and model.get('distribution') == 'log_normal':
            # 对数正态分布
            params = model['params']
            if value <= 0:
                log_likelihood += np.log(1e-10)
            else:
                ll = stats.lognorm.logpdf(value, *params)
                log_likelihood += ll

    return log_likelihood

def hybrid_predict(scores, grade, gender, models=None):
    """
    使用混合模型进行预测

    参数:
        scores: 列表，包含5个成绩 [Q1, Q2, Q3, Q4, Q5]
        grade: 整数，年级
        gender: 整数，性别
        models: 可选，已加载的模型字典

    返回:
        predicted: 预测的系别
        probabilities: 各系别概率
        model_info: 模型使用信息
    """
    if models is None:
        models = load_models()

    pca_model = models['pca']
    rf_models = models['random_forest']
    bayes_models = models['bayes']

    # 模型预测结果
    predictions = {}

    # 1. PCA模型预测
    pca_input = pd.DataFrame([[scores[0], scores[1], scores[2], scores[3], scores[4], grade, gender]],
                           columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade', 'Gender'])

    try:
        pca_probs = pca_model.predict_proba(pca_input)[0]
        pca_classes = pca_model.classes_
        pca_probs_dict = {cls: prob for cls, prob in zip(pca_classes, pca_probs)}
        predictions['pca'] = {
            'predicted': pca_model.predict(pca_input)[0],
            'probabilities': pca_probs_dict
        }
    except Exception as e:
        print(f"PCA预测错误: {e}")
        predictions['pca'] = {'predicted': None, 'probabilities': {}}

    # 2. 随机森林模型预测
    if grade in rf_models:
        model_info = rf_models[grade]
        rf_model = model_info['model']

        # 创建特征
        single_df = pd.DataFrame([[scores[0], scores[1], scores[2], scores[3], scores[4], gender]],
                               columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Gender'])
        X_features = create_advanced_features(single_df)

        try:
            # 确保特征列顺序一致
            X_input = X_features[model_info['features']]

            rf_probs = rf_model.predict_proba(X_input)[0]
            rf_classes = rf_model.classes_
            rf_probs_dict = {cls: prob for cls, prob in zip(rf_classes, rf_probs)}
            predictions['rf'] = {
                'predicted': rf_model.predict(X_input)[0],
                'probabilities': rf_probs_dict
            }
        except Exception as e:
            print(f"随机森林预测错误: {e}")
            predictions['rf'] = {'predicted': None, 'probabilities': {}}
    else:
        predictions['rf'] = {'predicted': None, 'probabilities': {}}

    # 3. 贝叶斯模型预测
    if grade in bayes_models:
        bayes_model_info = bayes_models[grade]

        try:
            # 创建特征向量
            features_array = np.array(scores + [gender])  # 将���有特征连接成一维数组

            # 获取类别列表
            classes = bayes_model_info.get('classes', [])

            if len(classes)>0:
                # 计算每个类别的对数似然
                log_probs = {}
                for cls in classes:
                    if cls in bayes_model_info['models']:
                        # 获取该类别的模型和先验
                        models_dict = bayes_model_info['models'][cls]
                        prior = bayes_model_info['priors'].get(cls, 1.0/len(classes))

                        # 计算对数似然
                        log_likelihood = _score_bayes_sample(
                            features_array,
                            models_dict,
                            bayes_model_info['features']
                        )

                        # 贝叶斯公式: P(C|X) ∝ P(X|C) * P(C)
                        log_probs[cls] = log_likelihood + np.log(prior)

                # 标准化概率
                if log_probs:
                    max_log_prob = max(log_probs.values())
                    probs = {
                        cls: np.exp(log_prob - max_log_prob)
                        for cls, log_prob in log_probs.items()
                    }

                    # 归一化
                    total = sum(probs.values())
                    if total > 0:
                        probs = {cls: p/total for cls, p in probs.items()}

                        # 最高概率类别
                        predicted = max(probs.items(), key=lambda x: x[1])[0]

                        predictions['bayes'] = {
                            'predicted': predicted,
                            'probabilities': probs
                        }
                    else:
                        predictions['bayes'] = {'predicted': None, 'probabilities': {}}
                else:
                    predictions['bayes'] = {'predicted': None, 'probabilities': {}}
            else:
                predictions['bayes'] = {'predicted': None, 'probabilities': {}}
        except Exception as e:
            print(f"贝叶斯预测错误: {e}")
            predictions['bayes'] = {'predicted': None, 'probabilities': {}}
    else:
        predictions['bayes'] = {'predicted': None, 'probabilities': {}}

    # 融合预测结果
    # 按系别合并概率
    all_programmes = set()
    for model_type in predictions.keys():
        all_programmes.update(predictions[model_type]['probabilities'].keys())

    # 设置模型权重
    weights = {
        'pca': 0.3,
        'rf': 0.6,
        'bayes': 0.1
    }

    # 对C类别的特殊处理
    predictions_contain_c = False
    for model_type in predictions.keys():
        if 'C' in predictions[model_type]['probabilities']:
            c_prob = predictions[model_type]['probabilities']['C']
            if c_prob > 0.4:  # 如果某个模型对C类预测概率较高
                predictions_contain_c = True

    if predictions_contain_c:
        weights['pca'] = 0.7  # 提高PCA模型权重
        weights['rf'] = 0.2
        weights['bayes'] = 0.1

    # 计算加权概率
    final_probs = {}
    for prog in all_programmes:
        weighted_sum = 0
        total_weight = 0

        for model_type, model_result in predictions.items():
            if prog in model_result['probabilities']:
                weighted_sum += model_result['probabilities'][prog] * weights[model_type]
                total_weight += weights[model_type]

        if total_weight > 0:
            final_probs[prog] = weighted_sum / total_weight

    # 获取最高概率的系别
    if final_probs:
        predicted = max(final_probs.items(), key=lambda x: x[1])[0]
    else:
        predicted = None

    return predicted, final_probs, predictions

# 测试函数
# 测试函数
if __name__ == '__main__':
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    # 加载模型
    models = load_models()

    # 加载测试数据
    test_data_path = '../unique_test_data.csv'
    if not os.path.exists(test_data_path):
        test_data_path = './unique_test_data.csv'

    test_df = pd.read_csv(test_data_path)
    print(f"加载测试数���: {test_data_path}, 样本数: {len(test_df)}")

    # 检查是否需要转换系别编码
    mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    if test_df['Programme'].dtype == 'int64' or str(test_df['Programme'].iloc[0]) in ['1', '2', '3', '4']:
        print("检测到系别使用数字编码，进行转换...")
        test_df['Programme'] = test_df['Programme'].astype(str).map(mapping)
        print(f"系别转换后的分布: \n{test_df['Programme'].value_counts()}")

    # 初始化结果列表
    results = {
        'actual': [],
        'predicted': [],
        'correct': [],
        'probabilities': []
    }

    # 对测试数据进行预测
    print("\n开始对测试数据进行预测...")

    for _, student in test_df.iterrows():
        scores = [student['Q1'], student['Q2'], student['Q3'], student['Q4'], student['Q5']]
        grade = student['Grade']
        gender = student['Gender']
        actual_programme = student['Programme']

        # 预测系别
        predicted, probs, _ = hybrid_predict(scores, grade, gender, models)
        is_correct = predicted == actual_programme

        # 保存结果
        results['actual'].append(actual_programme)
        results['predicted'].append(predicted)
        results['correct'].append(is_correct)
        results['probabilities'].append(probs)

    # 计算总体准确率
    correct_count = sum(results['correct'])
    total_count = len(results['correct'])
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n总体准确率: {overall_accuracy:.4f} ({correct_count}/{total_count})")

    # 按年级计算准确率
    for grade in sorted(test_df['Grade'].unique()):
        grade_mask = test_df['Grade'] == grade
        grade_indices = test_df[grade_mask].index
        grade_correct = sum([results['correct'][i] for i in range(len(results['correct']))
                          if i in grade_indices])
        grade_total = len(grade_indices)
        grade_accuracy = grade_correct / grade_total if grade_total > 0 else 0
        print(f"年级 {grade} 预测准确率: {grade_accuracy:.4f} ({grade_correct}/{grade_total})")

    # 创建混淆矩阵
    all_programmes = sorted(list(set(results['actual']) | set(results['predicted'])))
    cm = confusion_matrix(results['actual'], results['predicted'], labels=all_programmes)

    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print(pd.DataFrame(cm, index=all_programmes, columns=all_programmes))

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(results['actual'], results['predicted']))

    # 绘制混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_programmes,
                yticklabels=all_programmes)
    plt.title('测试集预测混淆矩阵')
    plt.xlabel('预测系别')
    plt.ylabel('实际系别')
    plt.tight_layout()
    plt.savefig('./confusion_matrix.png')
    print("\n混淆矩阵热图已保存至 './confusion_matrix.png'")

    # 显示图形
    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图形: {e}")

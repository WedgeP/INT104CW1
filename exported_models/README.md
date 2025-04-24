# 专业预测混合模型

## 模型文件说明
- pca_model.joblib: PCA降��+分类器模型
- random_forest_models.joblib: 按年级训练的随机森林模型
- bayes_models.joblib: 按年级训练的贝叶斯模型
- hybrid_predict.py: 模型加载与混合预测功能

## 使用方法
```python
from hybrid_predict import hybrid_predict, load_models

# 加载模型
models = load_models()

# 预测示例
scores = [7, 5, 10, 8, 4]  # 示例成绩 [Q1, Q2, Q3, Q4, Q5]
grade = 2                  # 年级
gender = 1                 # 性别

# 进行预测
predicted, probabilities, model_info = hybrid_predict(scores, grade, gender, models)
print(f"预测系别: {predicted}")
print(f"各系别概率: {probabilities}")
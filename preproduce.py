import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, norm

# 读取数据
df = pd.read_csv("./student_data.csv")

# 确保列名正确
expected_columns = {"Programme", "Grade", "Total"}
missing_columns = expected_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# 获取所有 Programme + Grade 组合
grouped_data = df.groupby(["Programme", "Grade"])["Total"]

# 计算组合数量，确定子图布局
num_groups = grouped_data.ngroups
rows = (num_groups // 2) + (num_groups % 2)  # 2 列布局

# 创建图形
fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 5))  # 2列布局
axes = axes.flatten()  # 展平，方便遍历

# 遍历所有 Programme + Grade 组合
for i, ((programme, Grade), programme_data) in enumerate(grouped_data):
    # 计算均值和标准差
    mu, sigma = programme_data.mean(), programme_data.std()

    # 直方图 + KDE 曲线
    sns.histplot(programme_data, kde=True, bins=20, stat="density", color="blue", ax=axes[i])

    # 拟合正态分布
    x = np.linspace(min(programme_data), max(programme_data), 100)
    pdf = norm.pdf(x, mu, sigma)
    axes[i].plot(x, pdf, 'r', label=f'Normal Fit ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)')

    # 统计检验
    shapiro_p = shapiro(programme_data).pvalue
    ks_p = kstest(programme_data, 'norm', args=(mu, sigma)).pvalue

    # 图表信息
    axes[i].set_title(f"{programme} - Grade {Grade}\nShapiro p={shapiro_p:.4f}, KS p={ks_p:.4f}")
    axes[i].set_xlabel("Total Score")
    axes[i].set_ylabel("Density")
    axes[i].legend()

# 调整布局
plt.tight_layout()
plt.show()
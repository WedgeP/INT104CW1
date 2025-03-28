import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, norm, lognorm

# 读取数据
df = pd.read_csv("./student_data.csv")
pointString="Total"
# 确保列名正确
expected_columns = {"Programme", "Gender","Grade", pointString}
missing_columns = expected_columns - set(df.columns)
if missing_columns:
    raise ValueError("Missing columns in dataset: {missing_columns}")

# 获取所有 Programme + Gender 组合
grouped_data = df.groupby(["Programme", "Gender","Grade"])[pointString]

# 计算组合数量，确定子图布局
num_groups = grouped_data.ngroups
rows = (num_groups // 2) + (num_groups % 2)  # 2 列布局

# 创建图形
fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 5))  # 2列布局
axes = axes.flatten()  # 展平，方便遍历

# 遍历所有 Programme + Gender + Grade 组合
results = []
for i, ((programme, Gender, Grade), programme_data) in enumerate(grouped_data):
    if len(programme_data) < 3:  # 数据不足 3 直接跳过
        print(f"Skipping {programme} - Gender {Gender}: Not enough data")
        continue

    # 计算均值和标准差
    mu, sigma = programme_data.mean(), programme_data.std()

    # 直方图 + KDE 曲线
    sns.histplot(programme_data, kde=True, bins=20, stat="density", color="blue", ax=axes[i])

    # **拟合正态分布**
    x = np.linspace(min(programme_data), max(programme_data), 100)
    pdf_normal = norm.pdf(x, mu, sigma)
    axes[i].plot(x, pdf_normal, 'r', label='Normal Fit ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)')
    programme_data = programme_data[programme_data > 0]  # 过滤非正数
    if len(programme_data) < 3:
        print(f"Skipping {programme} - Gender {Gender}: Not enough positive data")
        continue
    # **拟合对数正态分布（修正参数错误）**
    shape, loc, scale = lognorm.fit(programme_data)  # 正确拟合 lognormal
    pdf_lognorm = lognorm.pdf(x, shape, loc, scale)
    axes[i].plot(x, pdf_lognorm, 'g--', label="Lognormal Fit")

    # 统计检验
    shapiro_p = shapiro(programme_data).pvalue if len(programme_data) >= 3 else np.nan
    ks_p = kstest(programme_data, 'norm', args=(mu, sigma)).pvalue

    # 图表信息
    axes[i].set_title(f"{programme} - Gender {Gender}\nShapiro p={shapiro_p:.4f}, KS p={ks_p:.4f}")
    axes[i].set_xlabel("Total Score")
    axes[i].set_ylabel("Density")
    axes[i].legend()

    # 保存结果
    results.append([programme, Gender, mu, sigma, shapiro_p, ks_p])

# 调整布局
plt.tight_layout()
plt.show()

# 结果表格
df_results = pd.DataFrame(results,
                          columns=["Programme", "Gender", "Mean (μ)", "Std (σ)", "Shapiro p-value", "KS p-value"])
print(df_results)
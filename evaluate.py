import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import os


def create_score_percentile_table(file_path='./student_data.csv', output_dir='./output_figures'):
    """Generate score percentile table with analysis by programme"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Figures will be saved to: {os.path.abspath(output_dir)}")

    # 设置中文字体支持
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei']
    elif platform.system() == 'Windows':  # Windows
        plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']

    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 读取数据
    df = pd.read_csv(file_path)

    # 确保Total列存在
    if 'Total' not in df.columns:
        score_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        if all(col in df.columns for col in score_cols):
            df['Total'] = df[score_cols].sum(axis=1)
        else:
            print("Error: Unable to find or calculate Total column")
            return

    # 总分一分一段表
    score_counts = df['Total'].value_counts().reset_index()
    score_counts.columns = ['Score', 'Count']
    score_counts = score_counts.sort_values(by='Score', ascending=False)
    score_counts['Cumulative Count'] = score_counts['Count'].cumsum()
    score_counts['Cumulative Percentage'] = 100 * score_counts['Cumulative Count'] / len(df)
    score_counts['Rank'] = score_counts['Cumulative Count'] - score_counts['Count'] + 1

    print(f"Total Students: {len(df)}")
    print(f"Score Range: {df['Total'].min():.1f}~{df['Total'].max():.1f}")
    print("\n=== Total Score Percentile Table ===")

    # 设置更好的显示格式
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(score_counts.to_string(index=False,
                                 float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))

    # 按专业分组创建一分一段表
    programmes = sorted(df['Programme'].unique())
    prog_stats = {}

    # 设置颜色映射 - 使用tab20提供更多颜色
    cmap = plt.cm.get_cmap('tab20', len(programmes))
    color_map = {prog: cmap(i) for i, prog in enumerate(programmes)}

    # ===== 1. 总分分析（按专业分组） =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 使用实际数据中的分数值而非整数范围
    all_scores = sorted(df['Total'].unique())

    # 按专业绘制堆叠直方图
    score_by_prog = {}
    for score in all_scores:
        score_by_prog[score] = {prog: 0 for prog in programmes}

    for prog in programmes:
        prog_df = df[df['Programme'] == prog]
        for score, count in prog_df['Total'].value_counts().items():
            score_by_prog[score][prog] = count

    # 绘制堆叠直方图
    bottom = np.zeros(len(all_scores))
    for prog in programmes:
        values = [score_by_prog[score][prog] for score in all_scores]
        ax1.bar(all_scores, values, bottom=bottom,
                label=f'Programme {prog}', color=color_map[prog], alpha=0.8)
        bottom += np.array(values)

    # 完善图形
    ax1.set_title('Total Score Distribution by Programme')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend()

    # 绘制累积百分比曲线
    ax2.plot(score_counts['Score'], score_counts['Cumulative Percentage'], 'k-',
             linewidth=3, label='All Programmes', alpha=0.7)

    for prog in programmes:
        prog_df = df[df['Programme'] == prog]
        prog_counts = prog_df['Total'].value_counts().reset_index()
        prog_counts.columns = ['Score', 'Count']
        prog_counts = prog_counts.sort_values(by='Score', ascending=False)
        prog_counts['Cumulative Count'] = prog_counts['Count'].cumsum()
        prog_counts['Cumulative Percentage'] = 100 * prog_counts['Cumulative Count'] / len(prog_df)

        ax2.plot(prog_counts['Score'], prog_counts['Cumulative Percentage'], '-',
                 linewidth=2, label=f'Programme {prog}', color=color_map[prog])

    ax2.set_title('Cumulative Percentage by Score')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')

    plt.tight_layout()

    # 保存总分析图
    total_fig_path = os.path.join(output_dir, 'total_score_analysis.png')
    fig.savefig(total_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {total_fig_path}")
    plt.show()

    # ===== 2. Q1-Q5各题目分析 =====
    score_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    fig, axs = plt.subplots(len(score_cols), 2, figsize=(14, 4 * len(score_cols)))

    for i, col in enumerate(score_cols):
        # 创建该题目的一分一段表
        q_counts = df[col].value_counts().reset_index()
        q_counts.columns = ['Score', 'Count']
        q_counts = q_counts.sort_values(by='Score', ascending=False)
        q_counts['Cumulative Count'] = q_counts['Count'].cumsum()
        q_counts['Cumulative Percentage'] = 100 * q_counts['Cumulative Count'] / len(df)

        # 获取该题目的所有实际分数
        all_scores_for_q = sorted(df[col].unique())

        # 准备按专业分组的分数分布数据
        score_by_prog_for_q = {}
        for score in all_scores_for_q:
            score_by_prog_for_q[score] = {prog: 0 for prog in programmes}

        for prog in programmes:
            prog_df = df[df['Programme'] == prog]
            for score, count in prog_df[col].value_counts().items():
                score_by_prog_for_q[score][prog] = count

        # 直方图 - 使用堆叠条形图
        ax1 = axs[i, 0]
        bottom = np.zeros(len(all_scores_for_q))

        for prog in programmes:
            values = [score_by_prog_for_q[score][prog] for score in all_scores_for_q]
            ax1.bar(all_scores_for_q, values, bottom=bottom,
                    label=f'Programme {prog}', color=color_map[prog], alpha=0.8)
            bottom += np.array(values)

        ax1.set_title(f'{col} Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        if i == 0:  # 只在第一行显示图例
            ax1.legend()

        # 累积百分比曲线
        ax2 = axs[i, 1]
        ax2.plot(q_counts['Score'], q_counts['Cumulative Percentage'], 'k-',
                 linewidth=3, label='All Programmes', alpha=0.7)

        # 各专业累积曲线
        for prog in programmes:
            prog_df = df[df['Programme'] == prog]
            prog_q_counts = prog_df[col].value_counts().reset_index()
            prog_q_counts.columns = ['Score', 'Count']
            prog_q_counts = prog_q_counts.sort_values(by='Score', ascending=False)
            prog_q_counts['Cumulative Count'] = prog_q_counts['Count'].cumsum()
            prog_q_counts['Cumulative Percentage'] = 100 * prog_q_counts['Cumulative Count'] / len(prog_df)

            ax2.plot(prog_q_counts['Score'], prog_q_counts['Cumulative Percentage'], '-',
                     linewidth=2, label=f'Programme {prog}', color=color_map[prog])

        ax2.set_title(f'{col} Cumulative Percentage')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        if i == 0:  # 只在第一行显示图例
            ax2.legend(loc='lower right')

    plt.tight_layout()

    # 保存Q1-Q5整体分析图
    q_all_fig_path = os.path.join(output_dir, 'q1_to_q5_analysis.png')
    fig.savefig(q_all_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {q_all_fig_path}")
    plt.show()

    # 为每个问题单独保存图表
    for i, col in enumerate(score_cols):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 获取该题目的所有实际分数
        all_scores_for_q = sorted(df[col].unique())

        # 为当前问题准备数据
        score_by_prog_for_q = {}
        for score in all_scores_for_q:
            score_by_prog_for_q[score] = {prog: 0 for prog in programmes}

        for prog in programmes:
            prog_df = df[df['Programme'] == prog]
            for score, count in prog_df[col].value_counts().items():
                score_by_prog_for_q[score][prog] = count

        # 绘制堆叠直方图
        bottom = np.zeros(len(all_scores_for_q))
        for prog in programmes:
            values = [score_by_prog_for_q[score][prog] for score in all_scores_for_q]
            ax1.bar(all_scores_for_q, values, bottom=bottom,
                    label=f'Programme {prog}', color=color_map[prog], alpha=0.8)
            bottom += np.array(values)

        ax1.set_title(f'{col} Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend()

        # 生成累积百分比数据
        q_counts = df[col].value_counts().reset_index()
        q_counts.columns = ['Score', 'Count']
        q_counts = q_counts.sort_values(by='Score', ascending=False)
        q_counts['Cumulative Count'] = q_counts['Count'].cumsum()
        q_counts['Cumulative Percentage'] = 100 * q_counts['Cumulative Count'] / len(df)

        # 绘制累积百分比曲线
        ax2.plot(q_counts['Score'], q_counts['Cumulative Percentage'], 'k-',
                 linewidth=3, label='All Programmes', alpha=0.7)

        for prog in programmes:
            prog_df = df[df['Programme'] == prog]
            prog_q_counts = prog_df[col].value_counts().reset_index()
            prog_q_counts.columns = ['Score', 'Count']
            prog_q_counts = prog_q_counts.sort_values(by='Score', ascending=False)
            prog_q_counts['Cumulative Count'] = prog_q_counts['Count'].cumsum()
            prog_q_counts['Cumulative Percentage'] = 100 * prog_q_counts['Cumulative Count'] / len(prog_df)

            ax2.plot(prog_q_counts['Score'], prog_q_counts['Cumulative Percentage'], '-',
                     linewidth=2, label=f'Programme {prog}', color=color_map[prog])

        ax2.set_title(f'{col} Cumulative Percentage')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower right')

        plt.tight_layout()

        # 保存单个题目分析图
        q_fig_path = os.path.join(output_dir, f'{col}_analysis.png')
        fig.savefig(q_fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {q_fig_path}")
        plt.close(fig)  # 关闭单题图以节省内存

    # 查询功能
    print("\n=== Score Query Function ===")
    print("Enter score type (Total/Q1/Q2/Q3/Q4/Q5) and score to check its percentile")

    while True:
        try:
            col_input = input("\nEnter score type (Total/Q1/Q2/Q3/Q4/Q5, 'q' to quit): ")
            if col_input.lower() == 'q':
                break

            if col_input not in ['Total', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                print("Invalid score type, please enter Total or Q1-Q5")
                continue

            score_input = input("Enter score: ")
            if score_input.lower() == 'q':
                break

            score = float(score_input)

            # 总体查询
            col_counts = df[col_input].value_counts().reset_index()
            col_counts.columns = ['Score', 'Count']
            col_counts = col_counts.sort_values(by='Score', ascending=False)
            col_counts['Cumulative Count'] = col_counts['Count'].cumsum()
            col_counts['Cumulative Percentage'] = 100 * col_counts['Cumulative Count'] / len(df)
            col_counts['Rank'] = col_counts['Cumulative Count'] - col_counts['Count'] + 1

            # 查找最接近的分数
            matching_rows = col_counts[col_counts['Score'] <= score]
            if matching_rows.empty:
                print(f"{col_input} score {score} is below the minimum, unable to query rank")
                continue

            closest_row = matching_rows.iloc[0]

            print(f"\n{col_input} Score: {closest_row['Score']}")
            print(f"Count for this score: {closest_row['Count']} students")
            print(f"Rank: {closest_row['Rank']}")
            print(f"Percentile: {100 - closest_row['Cumulative Percentage']:.2f}%")
            print(f"Cumulative Percentage: {closest_row['Cumulative Percentage']:.2f}%")

            # 按专业查询
            print("\nBy Programme:")
            for prog in programmes:
                prog_df = df[df['Programme'] == prog]
                if len(prog_df) == 0:
                    continue

                prog_counts = prog_df[col_input].value_counts().reset_index()
                prog_counts.columns = ['Score', 'Count']
                prog_counts = prog_counts.sort_values(by='Score', ascending=False)
                prog_counts['Cumulative Count'] = prog_counts['Count'].cumsum()
                prog_counts['Cumulative Percentage'] = 100 * prog_counts['Cumulative Count'] / len(prog_df)

                matching_rows = prog_counts[prog_counts['Score'] <= score]
                if matching_rows.empty:
                    print(f"Programme {prog}: Score below minimum")
                    continue

                closest_row = matching_rows.iloc[0]
                print(f"Programme {prog}: Rank {closest_row['Cumulative Count']}/{len(prog_df)}, " +
                      f"Percentile {100 - closest_row['Cumulative Percentage']:.2f}%")

        except ValueError:
            print("Please enter a valid score")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    # 默认保存到当前目录
    create_score_percentile_table(output_dir=".")
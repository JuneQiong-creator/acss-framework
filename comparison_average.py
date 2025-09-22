import pandas as pd
from pathlib import Path

# --- Step 1: 定义输入文件路径 ---
base_path = r"C:\Users\LIU Qiong\Desktop\PhD\research\present\article no.1\code"
datasets = ["5-FU", "Carbo", "CDDP", "ETO", "Carbo+A", "CDDP+A"]
files = [f"{base_path}\\{ds}_comparison_metrics.csv" for ds in datasets]

# --- Step 2: 读取所有文件到字典 ---
dfs = {}
for ds, file in zip(datasets, files):
    df = pd.read_csv(file)
    df["Dataset"] = ds
    dfs[ds] = df.set_index("Method")

# --- Step 3: 提取所有方法名称 ---
methods = dfs[datasets[0]].index.tolist()

# --- Step 4: 为每个方法创建新表，并保存每列的均值和标准差 ---
output_dir = Path(base_path) / "Methodwise_CSVs"
output_dir.mkdir(exist_ok=True)

summary_list = []  # 汇总均值和标准差

for method in methods:
    rows = []
    for ds in datasets:
        row = dfs[ds].loc[method].copy()
        row["Dataset"] = ds
        rows.append(row)

    new_df = pd.DataFrame(rows)
    cols = ["Dataset"] + [c for c in new_df.columns if c != "Dataset"]
    new_df = new_df[cols]

    # 计算均值和标准差
    mean_row = {"Dataset": "Mean"}
    std_row = {"Dataset": "Std"}
    for col in new_df.columns[1:]:
        numeric_col = pd.to_numeric(new_df[col], errors="coerce")
        mean_row[col] = numeric_col.mean()
        std_row[col] = numeric_col.std(ddof=1)

    # 保存到单独 CSV
    new_df = pd.concat([new_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    output_file = output_dir / f"{method.replace('/', '_')}.csv"
    new_df.to_csv(output_file, index=False)

    # 汇总均值和标准差，用于总表
    summary = {"Method": method}
    for col in new_df.columns[1:]:
        summary[f"{col}_mean"] = mean_row[col]
        summary[f"{col}_std"] = std_row[col]
    summary_list.append(summary)

# --- Step 5: 创建汇总表 ---
summary_df = pd.DataFrame(summary_list)

# 添加 Feature 列
summary_df["Feature"] = summary_df["Method"].apply(lambda x: "CRS" if "CRS" in x else "IC50")

# 按 Feature 排序：CRS 先，IC50 后
summary_df["Method"] = pd.Categorical(
    summary_df["Method"],
    categories=[m for m in methods if "CRS" in m] + [m for m in methods if "IC50" in m or "IC50" in m.replace('-', '_')],
    ordered=True
)
summary_df = summary_df.sort_values("Method")

# 保存总汇总表
summary_file = output_dir / "Summary_Mean_Std_with_Feature.csv"
summary_df.to_csv(summary_file, index=False)

# --- Step 6: 按 Feature 分组，再计算每列的均值和标准差 ---
grouped_stats = []
for feature, group in summary_df.groupby("Feature"):
    stats = {"Feature": feature}
    for col in summary_df.columns:
        if col not in ["Method", "Feature"]:
            numeric_col = pd.to_numeric(group[col], errors="coerce")
            stats[f"{col}_mean"] = numeric_col.mean()
            stats[f"{col}_std"] = numeric_col.std(ddof=1)
    grouped_stats.append(stats)

grouped_df = pd.DataFrame(grouped_stats)

# 保存分组统计结果
grouped_file = output_dir / "Feature_Grouped_Mean_Std.csv"
grouped_df.to_csv(grouped_file, index=False)

print("✅ 完成！方法汇总表和按 Feature 分组统计表均已生成。")

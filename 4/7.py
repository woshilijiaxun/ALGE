import pandas as pd
import os
import glob

# 1. 获取桌面路径
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# 2. 拼接你的目标文件夹路径（例如：桌面/成绩汇总）
folder_path = os.path.join(desktop, "课程成绩")

# 3. 获取所有 Excel 文件（.xlsx）
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
# 获取所有 .xlsx 文件，排除 ~ 开头的临时文件
excel_files = [f for f in glob.glob(os.path.join(folder_path, "*.xlsx")) if not os.path.basename(f).startswith("~$")]


# 存储所有结果
all_results = []

# 逐个文件处理
for file in excel_files:
    try:
        # 读取 Sheet0 表
        df = pd.read_excel(file, sheet_name="Sheet0")

        # 提取文件名
        name = os.path.splitext(os.path.basename(file))[0]

        # 成绩转数值
        df["成绩"] = pd.to_numeric(df["成绩"], errors='coerce')

        # 分组求平均成绩
        result = df.groupby("开课学年")["成绩"].mean().reset_index()

        # 添加文件名列
        result["文件名"] = name

        # 调整列顺序
        result = result[["文件名", "开课学年", "成绩"]]
        result.rename(columns={"成绩": "平均成绩"}, inplace=True)

        # 加入结果列表
        all_results.append(result)

        # 插入一行空白 DataFrame
        all_results.append(pd.DataFrame([["", "", ""]], columns=["文件名", "开课学年", "平均成绩"]))

    except Exception as e:
        print(f"⚠️ 读取文件 {file} 出错：{e}")

# 合并所有结果
final_df = pd.concat(all_results, ignore_index=True)

# 保存结果
output_path = os.path.join(desktop, "汇总平均成绩结果1.xlsx")
final_df.to_excel(output_path, index=False)

print(f"✅ 已完成，结果保存在：{output_path}")



import openpyxl
import pickle

with open('active_learning_ken.pkl', 'rb') as f:
    data = pickle.load(f)

# 创建新的Excel工作簿
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Network Data"

# 从A1单元格开始写入数据
row = 1

# 遍历每个para
for param in next(iter(data.values())).keys():  # 获取第一个网络的所有参数
    ws.append([f"Data for {param}"])  # 写入块标题
    row += 1

    # 写入表头
    ws.append(['Network', param])
    row += 1

    # 写入每个网络的数据
    for network, params in data.items():
        if param in params:
            ws.append([network, params[param]])
            row += 1

    # 增加空行作为分隔
    row += 1

# 保存到文件
wb.save("ken_for_active_learning.xlsx")

# import openpyxl
# import pickle
#
# with open('f-e-AND-prgc_ken(0.5-1.5).pkl', 'rb') as f:
#     data = pickle.load(f)
#
# # 创建新的Excel工作簿
# wb = openpyxl.Workbook()
# ws = wb.active
# ws.title = "Network Data"
#
# # 从A1单元格开始写入数据
# row = 1
#
# # 遍历每个para
# for param in next(iter(data.values())).keys():  # 获取第一个网络的所有参数
#     ws.append([f"Data for {param}"])  # 写入块标题
#     row += 1
#
#     # 写入表头
#     ws.append(['Network', param])
#     row += 1
#
#     # 写入每个网络的数据
#     for network, params in data.items():
#         if param in params:
#             ws.append([network, params[param]])
#             row += 1
#
#     # 增加空行作为分隔
#     row += 1
#
# # 保存到文件
# wb.save("f-eANDprgc.xlsx")
import openpyxl
import pickle

# 读取数据
with open('Ken_data/f-e-AND-prgc_ken(0.5-1.5).pkl', 'rb') as f:
    data = pickle.load(f)

# 创建 Excel 工作簿
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Network Data"

row = 1  # 初始行号

# 获取所有参数
params = next(iter(data.values())).keys()  # 例如 ["f-e", "prgc"]

for param in params:
    ws.append([f"Data for {param}"])  # 添加参数标题
    row += 1

    # 获取所有方法名称
    method_names = list(next(iter(data.values()))[param].keys())  # 例如 ["method1", "method2"]

    # 写入表头
    ws.append(["Network"] + method_names)
    row += 1

    # 遍历数据
    for network, param_dict in data.items():
        if param in param_dict:
            # 提取该网络在当前参数下的所有方法的值
            values = [param_dict[param].get(method, "N/A") for method in method_names]
            ws.append([network] + values)
            row += 1

    # 添加空行分隔不同的参数块
    row += 1

# 保存 Excel 文件
wb.save("f-eANDprgc.xlsx")

import torch
from Utils import get_dgl_g_input,load_multilayer_graph
import networkx as nx
import dgl
from scipy.stats import kendalltau
import random
import copy
from influence_evaluation import Model
def load_multilayer_graph(path):
    # 创建一个动态扩展的多层网络列表
    G = []
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            if len(vertices) != 4:
                raise ValueError(f"Invalid line format: {line}")

            layer = int(vertices[0])  # 网络层
            source = int(vertices[1])  # 源节点
            target = int(vertices[2])  # 目标节点
            weight = int(vertices[3])  # 权重

            # 动态扩展图列表
            while len(G) < layer:
                G.append(nx.Graph())

            # 添加边并存储权重
            G[layer - 1].add_edge(source, target, weight=weight)

    # 移除自环
    for graph in G:
        graph.remove_edges_from(nx.selfloop_edges(graph))

    # 返回多层网络和实际的层数
    return G, len(G)
def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[-1])
            G.add_edge(source, target)
    return G

def threshhold(G):
    # 计算网络的平均度
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    # 计算网络中每个节点的度数，求平方并相加
    squared_degree_sum = sum(deg ** 2 for node, deg in G.degree())
    # 计算度的平方的平均值
    average_squared_degree = squared_degree_sum / len(G)
    beita = avg_degree / (average_squared_degree - avg_degree)
    return round(beita, 4)


def IC_simulation(p, g, set, num_trials=1000):
    total_influence = 0  # 用来累加每次实验的影响力
    for _ in range(num_trials):
        # 深拷贝图，避免修改原始图
        g_copy = copy.deepcopy(g)

        # 初始化所有节点的状态为0
        for node in g_copy.nodes():
            g_copy.nodes[node]['state'] = 0

        # 如果集合为空，随机选择节点作为初始激活节点
        if not set:
            for i in list(g_copy.nodes()):
                g_copy.nodes[i]['state'] = 1 if random.random() < .01 else 0
                if g_copy.nodes[i]['state'] == 1:
                    set.append(i)

        # 如果集合不为空，设置集合中的节点为激活状态
        if set:
            for i in list(g_copy.nodes()):
                if i in set:
                    g_copy.nodes[i]['state'] = 1

        # 设置每条边的传播概率
        for j in list(g_copy.edges()):
            g_copy.edges[j]['p'] = p  # 传播概率为p

        # 复制图用于更新状态
        nextg = g_copy.copy()
        terminal = 0

        # 仿真开始
        while terminal == 0:
            for i in list(g_copy.nodes()):
                if g_copy.nodes[i]['state'] == 1:  # 如果节点激活
                    for j in g_copy.neighbors(i):
                        if g_copy.nodes[j]['state'] == 0:
                            nextg.nodes[j]['state'] = 1 if random.random() < nextg.edges[i, j]['p'] else 0
                    nextg.nodes[i]['state'] = 2  # 节点变为已传播状态
            g_copy = nextg
            nextg = g_copy.copy()
            terminal = 1
            for i in list(g_copy.nodes()):
                if g_copy.nodes[i]['state'] == 1:  # 还有未激活节点时继续仿真
                    terminal = 0

        # 统计最终的激活节点数
        count = 0
        for i in list(g_copy.nodes()):
            if g_copy.nodes[i]['state'] == 2:  # 已传播的节点
                count += 1

        # 只将从初始集合中成功传播的节点计入影响力
        successful_initial_nodes = sum(1 for node in set if node in g_copy.nodes() and g_copy.nodes[node]['state'] == 2)
        count -= len(set)  # 从总影响力中减去初始激活节点的数量
        count += successful_initial_nodes  # 加上成功传播的初始激活节点的数量

        total_influence += count  # 累加本次实验的影响力

    # 返回1000次实验的平均影响力
    return total_influence / num_trials

class InfluenceCalculator:
    def __init__(self, graph):
        self.G = graph

    def sir_model(self, beta, source_node):
        """
        SIR模型的简化实现，恢复概率为1
        """
        nodes = self.G.nodes()
        state = {node: 'S' for node in nodes}
        # 如果 source_node 是单个节点（非列表），将其转换为列表
        if isinstance(source_node, int):
            source_node = [source_node]

        # 将 source_node 列表中的节点设置为 'I'（感染状态）
        for node in source_node:
            state[node] = 'I'


        while 'I' in state.values():
            susceptible_nodes = [node for node in nodes if state[node] == 'S']
            for susceptible_node in susceptible_nodes:
                neighbors = list(self.G.neighbors(susceptible_node))
                for neighbor in neighbors:
                    if state[neighbor] == 'I' and random.random() < beta:
                        state[susceptible_node] = 'I'

                        break  # 一旦感染，中断内循环

            for node in nodes:
                    if state[node] == 'I':
                        state[node] = 'R'

        recovered_nodes = [node for node in nodes if state[node] == 'R']
        return len(recovered_nodes)

    def calculate_average_influence(self, beta, source_node, experiments=500, iterations_per_experiment=100):
        """
        计算节点的平均影响度量
        """
        total_influence = 0
        for _ in range(experiments):
            total_influence += self.sir_model(beta, source_node)

        average_influence = total_influence / experiments
        return average_influence

    def calculate_sorted_nodes(self, beta):
        """
        计算并返回按平均影响度量降序排序的节点列表
        """
        average_influence_measures = [(node, self.calculate_average_influence(beta, node)) for node in self.G.nodes()]
        sorted_nodes = sorted(average_influence_measures, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_nodes],[item[1] for item in sorted_nodes],dict(sorted_nodes)

    # def calculate_transmission_capacity(self, beta, initial_infected_nodes, experiments=100, iterations_per_experiment=30):
    #     """
    #     计算节点在30个不同时间步中的传染能力
    #     """
    #     transmission_capacity = []
    #     for _ in range(experiments):
    #         infected_nodes = initial_infected_nodes.copy()  # 复制初始感染节点列表
    #         recovered_nodes = []
    #         for _ in range(iterations_per_experiment):
    #             for node in infected_nodes[:]:  # 使用切片来复制列表，避免在循环中修改列表长度
    #                 neighbors = list(self.G.neighbors(node))
    #                 for neighbor in neighbors:
    #                     if random.random() < beta and neighbor not in infected_nodes and neighbor not in recovered_nodes:
    #                         infected_nodes.append(neighbor)
    #                 infected_nodes.remove(node)
    #                 recovered_nodes.append(node)
    #             transmission_capacity.append(len(infected_nodes) + len(recovered_nodes))
    #     return transmission_capacity[:30]  # 返回传染能力列表的前30个元素
import pickle

def process_network(index, name1, name2, path):
    """每个网络的并行任务"""
    try:
        # 1. 读取网络
        G_er = load_graph(path + name1 + str(index) + '.txt')
        G_ws = load_graph(path + name2 + str(index) + '.txt')

        # 2. 计算ER网络的影响力
        b_er = threshhold(G_er) * 1.5
        sir_dict_er = InfluenceCalculator(G_er).calculate_sorted_nodes(b_er)[-1]
        with open(f'{path}{name1}{index}Influence_p.txt', 'w') as f:
            for key, value in sir_dict_er.items():
                f.write(f'{key} {value}\n')
        print(f'er{index} success')

        # 3. 计算WS网络的影响力
        b_ws = threshhold(G_ws) * 1.5
        sir_dict_ws = InfluenceCalculator(G_ws).calculate_sorted_nodes(b_ws)[-1]
        with open(f'{path}{name2}{index}Influence_p.txt', 'w') as f:
            for key, value in sir_dict_ws.items():
                f.write(f'{key} {value}\n')
        print(f'ws{index} success')

    except Exception as e:
        print(f"Error processing network {index}: {e}")


import networkx as nx
import random
import multiprocessing as mp

import csv

import os


import networkx as nx
import random
import multiprocessing as mp
import pandas as pd
def load_graph1(file_path):
    """从CSV文件加载图，假设文件格式为：id,u,v"""
    try:
        # 使用 pandas 读取 csv 文件
        df = pd.read_csv(file_path)

        # 创建空的无向图
        G = nx.Graph()

        # 将每一行作为边加入图中
        for _, row in df.iterrows():
            u = int(row['u'])  # 假设 u 是节点1的 ID
            v = int(row['v'])  # 假设 v 是节点2的 ID
            G.add_edge(u, v)  # 将边加入图中

        # 如果你需要在图中保存节点的某些属性，下面可以添加更多的代码
        # 例如，节点的特征可以从文件中读取并设置
        return G
    except Exception as e:
        print(f"Error loading graph from {file_path}: {e}")
        return None


def process_network(index, name1, name2, path):
    """每个网络的并行任务"""
    try:
        # 1. 读取网络
        G_er = load_graph1(path + name1 + str(index) + '.csv')
        G_ws = load_graph1(path + name2 + str(index) + '.csv')

        # 2. 计算ER网络的影响力
        b_er = threshhold(G_er) * 1.5
        sir_dict_er = InfluenceCalculator(G_er).calculate_sorted_nodes(b_er)[-1]

        # 将影响力数据转换为 DataFrame 并保存为 CSV
        df_er = pd.DataFrame(list(sir_dict_er.items()), columns=['Node', 'Influence'])
        df_er.to_csv(f'{path}{name1}{index}Influence_p.csv', index=False)  # 保存为 CSV 文件
        print(f'er{index} success')

        # 3. 计算WS网络的影响力
        b_ws = threshhold(G_ws) * 1.5
        sir_dict_ws = InfluenceCalculator(G_ws).calculate_sorted_nodes(b_ws)[-1]

        # 将影响力数据转换为 DataFrame 并保存为 CSV
        df_ws = pd.DataFrame(list(sir_dict_ws.items()), columns=['Node', 'Influence'])
        df_ws.to_csv(f'{path}{name2}{index}Influence_p.csv', index=False)  # 保存为 CSV 文件
        print(f'ws{index} success')

    except Exception as e:
        print(f"Error processing network {index}: {e}")


# if __name__ == '__main__':
    # # 网络名称和路径
    # name1 = 'SyntheticNetEr'
    # name2 = 'SyntheticNetWs'
    # path = './dataset/synthetic/'
    #
    # # 创建 50 个索引任务
    # indices = list(range(50))
    #
    # # 使用多进程，设置进程数量为CPU核心数量
    # num_workers = mp.cpu_count()  # 自动获取CPU核心数
    # with mp.Pool(processes=num_workers) as pool:
    #     pool.starmap(process_network, [(i, name1, name2, path) for i in indices])
    #
    # print("All networks processed.")









# def txt_to_csv(txt_file, csv_file):
#     with open(txt_file, 'r') as infile:
#         # 读取txt文件中的每行
#         lines = infile.readlines()
#
#     with open(csv_file, 'w', newline='') as outfile:
#         writer = csv.writer(outfile)
#
#         # 写入CSV文件的表头
#         writer.writerow(['u', 'v'])
#
#         # 遍历每条边，写入id, u, v
#         for i, line in enumerate(lines):
#             nodes = line.strip().split()
#             writer.writerow([nodes[0], nodes[1]])
#     os.remove(txt_file)
#
# if __name__ == '__main__':
#     t = './dataset/synthetic/'  # 原始txt文件路径
#     c = './dataset/synthetic/'  # 目标csv文件路径
#     name1 = 'SyntheticNetEr'
#     name2 = 'SyntheticNetWs'
#     for i in range(50):
#         path = t+name1+str(i)+'.csv'
#         os.remove(path)


import pandas as pd
import multiprocessing as mp

def load_graph1(file_path):
    """从CSV文件加载图，假设文件格式为：id,u,v"""
    try:
        # 使用 pandas 读取 csv 文件
        df = pd.read_csv(file_path)

        # 创建空的无向图
        G = nx.Graph()

        # 将每一行作为边加入图中
        for _, row in df.iterrows():
            u = int(row['u'])  # 假设 u 是节点1的 ID
            v = int(row['v'])  # 假设 v 是节点2的 ID
            G.add_edge(u, v)  # 将边加入图中

        # 如果你需要在图中保存节点的某些属性，下面可以添加更多的代码
        # 例如，节点的特征可以从文件中读取并设置

        return G

    except Exception as e:
        print(f"Error loading graph from {file_path}: {e}")
        return None
def process_network(index, name1, path):
    """每个网络的并行任务"""
    try:
        # 1. 读取网络
        G_er = load_graph1(path + name1 + str(index) + '.csv')


        # 2. 计算ER网络的影响力
        b_er = threshhold(G_er) * 1.5
        sir_dict_er = InfluenceCalculator(G_er).calculate_sorted_nodes(b_er)[-1]

        # 将影响力数据转换为 DataFrame 并保存为 CSV
        df_er = pd.DataFrame(list(sir_dict_er.items()), columns=['Node', 'Influence'])
        df_er.to_csv(f'{path}{name1}{index}Influence_p.csv', index=False)  # 保存为 CSV 文件
        print(f'er{index} success')




        print(f'ws{index} success')

    except Exception as e:
        print(f"Error processing network {index}: {e}")


import networkx as nx
import random
import multiprocessing as mp





def worker(graph, beta, node_list, result_dict, process_id):
    """
    计算每个节点平均影响力的工作函数
    """
    calculator = InfluenceCalculator(graph)
    for node in node_list:
        result_dict[node] = calculator.calculate_average_influence(beta, node)
    print(f"进程 {process_id} 完成计算，处理 {len(node_list)} 个节点")


class InfluenceCalculator_multiprocess:
    def __init__(self, graph):
        self.G = graph

    def sir_model(self, beta, source_node):
        """
        SIR模型的简化实现，恢复概率为1
        """
        nodes = self.G.nodes()
        state = {node: 'S' for node in nodes}
        state[source_node] = 'I'

        while 'I' in state.values():
            susceptible_nodes = [node for node in nodes if state[node] == 'S']
            for susceptible_node in susceptible_nodes:
                neighbors = list(self.G.neighbors(susceptible_node))
                for neighbor in neighbors:
                    if state[neighbor] == 'I' and random.random() < beta:
                        state[susceptible_node] = 'I'
                        break  # 一旦感染，中断内循环
            for node in nodes:
                if state[node] == 'I':
                    state[node] = 'R'

        recovered_nodes = [node for node in nodes if state[node] == 'R']
        return len(recovered_nodes)

    def calculate_average_influence(self, beta, source_node, experiments=500):
        """
        计算节点的平均影响度量
        """
        total_influence = 0
        for _ in range(experiments):
            total_influence += self.sir_model(beta, source_node)

        average_influence = total_influence / experiments
        print(f"{source_node}已计算完成")
        return average_influence

    def calculate_sorted_nodes(self, beta, processes=4):
        """
        使用多进程计算并返回按平均影响度量降序排序的节点列表
        """
        nodes = list(self.G.nodes())
        # 分割节点列表，每个进程处理一部分
        chunk_size = len(nodes) // processes
        node_chunks = [nodes[i * chunk_size: (i + 1) * chunk_size] for i in range(processes)]
        if len(nodes) % processes != 0:
            node_chunks[-1].extend(nodes[processes * chunk_size:])

        # 定义一个共享的队列来存储结果
        manager = mp.Manager()
        result_dict = manager.dict()

        # 创建多个进程
        processes_list = []
        for i in range(processes):
            process = mp.Process(target=worker, args=(self.G, beta, node_chunks[i], result_dict, i))
            processes_list.append(process)
            process.start()

        # 等待所有进程完成
        for process in processes_list:
            process.join()

        # 排序结果
        sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_results], [item[1] for item in sorted_results], dict(sorted_results)





import  os
#if __name__ == '__main__':

    # nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980,
    #                                       'drosophila_genetic_multiplex': 8215,
    #                                      }
    #
    # # nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980, 'celegans_connectome_multiplex': 279,
    # #                                      'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
    # #                                      'cS-Aarhus_multiplex': 61, 'drosophila_genetic_multiplex': 8215,
    # #                                      'hepatitusC_genetic_multiplex': 105,
    # #                                      'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71,
    # #                                      'rattus_genetic_multiplex': 2640}
    # network = [key + '.edges' for key in nodes_num_from_multiplex_networks.keys()]
    # for name in network:
    #     path = 'MNdata/'+name
    #
    #     # path = 'MNdata/cS-Aarhus_multiplex.edges'
    #     Gs, total_layers = load_multilayer_graph(path)
    #     sir_dict = {}
    #     for i in range(total_layers):
    #         time = 1.5
    #         b = threshhold(Gs[i]) * time
    #         influence_calculator = InfluenceCalculator_multiprocess(Gs[i])
    #         _, _, sir = influence_calculator.calculate_sorted_nodes(b, processes=5)
    #
    #
    #         #sir = InfluenceCalculator(Gs[i]).calculate_sorted_nodes(b)[-1]
    #         sir_dict[i+1] = sir
    #     t = 1
    #     with open('./MN_SIR_'+str(time)+'beitac/'+name+'.txt','w') as f:
    #         for layer,sir in sir_dict.items():
    #             for k,v in sir.items():
    #                 f.write(f'{layer} {k} {v}\n')
    #                 t += 1
    #                 if t % 100 == 0:
    #                     print(f"已处理 {t} 个节点")

    # 存储数据的列表

    # # 存储数据的列表
    # data = []
    #
    # # 读取每个pkl文件并收集数据
    # for t in [0.5, 0.75, 1.25, 1.5]:
    #     # 在DataFrame中标注t的起点，添加一个标识行
    #     data.append({'Key': f'{t} '})  # 这是一个标志行
    #     with open('k_' + str(t) + 'b.pkl', 'rb') as pickle_file:
    #         loaded_dict = pickle.load(pickle_file)
    #         for key, value in loaded_dict.items():
    #             # 将 key 和 value(字典) 转换成行
    #             row = {'Key': key}
    #             row.update(value)  # 将 value 字典展开为多列
    #             data.append(row)  # 每一行添加到数据列表中
    #
    # # 将数据转为DataFrame
    # df = pd.DataFrame(data)
    #
    # # 将数据写入 Excel
    # df.to_excel('ken_[0.5, 0.75, 1.25, 1.5].xlsx', index=False)  # 不存储索引

    # # 网络名称和路径
    # name1 = 'SyntheticNet100Myba'
    #
    # path = './dataset/synthetic/'
    #
    # # 创建 50 个索引任务
    # indices = list(range(50))
    #
    # # 使用多进程，设置进程数量为CPU核心数量
    # num_workers = mp.cpu_count()  # 自动获取CPU核心数
    # with mp.Pool(processes=num_workers) as pool:
    #     pool.starmap(process_network, [(i, name1, path) for i in indices])
    #
    # print("All networks processed.")

    # for i in range(50):
    #     os.remove(path+name1+str(i)+'.csv')
    #     os.remove(path+name1+str(i)+'Influence_p'+'.csv')



# import networkx as nx
# import pandas as pd
# import random
# import os
#
#
# def generate_fixed_edge_ba_graph(n, m):
#     """
#     生成一个具有固定边数的 Barabási–Albert 网络。
#     参数：
#     - n: 节点数
#     - m: 每次新加入节点时连接的边数
#     """
#     G = nx.barabasi_albert_graph(n, m)
#     return G
#
#
# def generate_multiple_ba_graphs(n, m_list, num_graphs=50):
#     """
#     生成每种边数的 BA 网络，数量为 num_graphs。
#     参数：
#     - n: 节点数
#     - m_list: 不同的 m 值列表（m 表示新节点加入时要连的边数）
#     - num_graphs: 每种 m 生成的网络数量
#     """
#     ba_graphs = {m: [] for m in m_list}  # 存储每种边数的网络
#     for m in m_list:
#         for i in range(num_graphs):
#             G = generate_fixed_edge_ba_graph(n, m)
#             ba_graphs[m].append(G)
#     return ba_graphs
#
#
# def save_ba_networks_csv(ba_networks, folder='./dataset/synthetic/', name='SyntheticNet100Myba'):
#     """
#     将每种 m 值的 BA 网络保存到 CSV 文件中，文件包含 id, u, v。
#     """
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     j = 0
#     for m in ba_networks:
#         for i, G in enumerate(ba_networks[m]):
#             network_id = j * 10 + i  # 生成唯一网络ID
#             filename = f'{folder}{name}{network_id}.csv'
#
#             # 生成边列表 (u, v) 并加入 id 列
#             edges = list(G.edges())
#             df = pd.DataFrame(edges, columns=['u', 'v'])
#             df.insert(0, 'id', i)  # 在第一列插入网络ID
#
#             # 保存为 CSV 文件
#             df.to_csv(filename, index=False)
#             print(f"已保存网络：{filename}")
#         j += 1
#
#
# # 参数设置
# n = 100  # 节点数
# m_list = [1, 2, 3, 4, 5]  # 每次加入节点时的边数（类似于WS中的边数，但控制方式不同）
# num_graphs = 10  # 每种 m 生成 10 个网络
#
# # 生成网络
# ba_networks = generate_multiple_ba_graphs(n, m_list, num_graphs)
#
# # 保存网络
# save_ba_networks_csv(ba_networks)
#
import multiprocessing
# 并行化实验的主函数
def parallel_ic_simulation(p, g, set, num_trials=1000, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()  # 默认为系统的CPU核心数

    # 切分实验任务，每个任务执行 num_trials // num_processes 次实验
    trials_per_process = num_trials // num_processes

    # 使用 multiprocessing.Pool 来并行化任务
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(IC_simulation, [(p, g, set, trials_per_process) for _ in range(num_processes)])

    # 计算所有实验结果的平均值
    total_influence = sum(results)
    average_influence = total_influence / num_processes
    return average_influence

import pickle
import pandas as pd
import openpyxl
if __name__ == '__main__':
    # 读取pkl文件
    dict_list = []
    for t in [10,20,30,40,50]:
        file_path = './top-k influence propagation/ic_top-'+str(t)+'_1.5b.pkl'  # 替换为你的pkl文件路径
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            dict_list.append(data)
    # 使用 ExcelWriter 将多个字典写入同一个 sheet，并隔一行
    with pd.ExcelWriter('ic_[10-50]——1.5b.xlsx', engine='openpyxl') as writer:
        # 设置一个起始的行号
        start_row = 0

        for idx, data in enumerate(dict_list):
            # 将字典转换为 DataFrame，并转置
            df = pd.DataFrame(data).T

            # 为每个字典数据写入 Excel 文件
            df.to_excel(writer, sheet_name='Networks', startrow=start_row, index=True)

            # 更新起始行号，使下一个字典数据隔一行
            start_row += len(df) + 1  # 当前 DataFrame 的行数 + 1 行空白




    # file_path = 'sorted_nodes.pkl'  # 替换为你的pkl文件路径
    # with open(file_path, 'rb') as file:
    #     data = pickle.load(file)
    #
    # for n in [10,20,30,40,50]:
    #     Result = {}
    #     for network_name,v in data.items():
    #         Gs,total_layer = load_multilayer_graph('./MNdata/'+network_name)
    #         method_inf = {}
    #         for method, nodes in v.items():
    #             inf = 0
    #             for i in range(total_layer):
    #                 b = threshhold(Gs[i])
    #                 inf += parallel_ic_simulation(b,Gs[i],nodes[:n])
    #             method_inf[method] = inf
    #         Result[network_name] = method_inf
    #
    #     with open('./top-k influence propagation/ic_top-'+str(n)+'.pkl', 'wb') as f:
    #         pickle.dump(Result, f)






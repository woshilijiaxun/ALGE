import torch
from Utils import get_dgl_g_input
import networkx as nx
import dgl
from scipy.stats import kendalltau
import random
import copy
from influence_evaluation import Model
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

def IC_simulation(p, g, set):
    g = copy.deepcopy(g)
    # pos = nx.spring_layout(g)
    for node in g.nodes():
        g.nodes[node]['state'] = 0

    if set == []:
        for i in list(g.nodes()):
            g.nodes[i]['state'] = 1 if random.random() < .01 else 0
            if g.nodes[i]['state'] == 1:
                set.append(i)
    if set != []:
        for i in list(g.nodes()):
            if i in set:
                g.nodes[i]['state'] = 1


    for j in list(g.edges()):
        g.edges[j]['p'] = p  # rd.uniform(0,1)
    nextg = g.copy()
    terminal = 0
    # 仿真开始
    while (terminal == 0):
        for i in list(g.nodes()):
            if g.nodes[i]['state'] == 1:
                for j in g.neighbors(i):
                    if g.nodes[j]['state'] == 0:
                        nextg.nodes[j]['state'] = 1 if random.random() < nextg.edges[i, j]['p'] else 0
                nextg.nodes[i]['state'] = 2
        g = nextg
        nextg = g
        terminal = 1
        for i in list(g.nodes()):
            if g.nodes[i]['state'] == 1:
                terminal = 0
    count = -len(set)
    for i in list(g.nodes()):
        if g.nodes[i]['state'] == 2:
            count += 1
    return count

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

    def calculate_average_influence(self, beta, source_node, experiments=100, iterations_per_experiment=100):
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
if __name__ == '__main__':


    # 读取 .pkl 文件
    with open('influence_evaluation/ken_table.pkl', 'rb') as file:  # 'rb'表示以二进制方式读取
        data = pickle.load(file)

    print(data)


    model = torch.load('influence_evaluation/ALGE_B_11_20.pth')
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    G = load_graph(r'C:\Users\ADM\Desktop\CycleRatio-main\Dataset\NS_GC.txt')
    g = dgl.from_networkx(G)
    b = threshhold(G)
    IC_dict = {}




    _, _, IC_dict = InfluenceCalculator(G).calculate_sorted_nodes(b)

    print('ic_dict',IC_dict)
    IC_list = [key for key in IC_dict.keys()]
    node_rank_simu = list(range(1, len(IC_list) + 1))

    node_features_ = get_dgl_g_input(G)
    node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)

    model.eval()
    value = model(g, node_features)
    nodes_list = list(G.nodes())
    prediction_I = value.detach().numpy()
    prediction_I_with_node = [[nodes_list[i], prediction_I[i][0]] for i in range(len(prediction_I))]
    prediction_I_with_node.sort(key=lambda x: x[1], reverse=True)
    pre_value = dict(prediction_I_with_node)
    pre_sorted_node = [key for key in pre_value]
    print('sorted_node', pre_sorted_node)

    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in IC_list]
    k1 = kendalltau(node_rank_simu, node_rank_p)
    print(k1[0])
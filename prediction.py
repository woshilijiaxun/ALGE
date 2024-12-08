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

import random
import multiprocessing as mp

import csv

import os
def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as infile:
        # 读取txt文件中的每行
        lines = infile.readlines()

    with open(csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        # 写入CSV文件的表头
        writer.writerow(['node','labels'])

        # 遍历每条边，写入id, u, v
        for i, line in enumerate(lines):
            nodes = line.strip().split()
            writer.writerow([nodes[0], nodes[1]])

        # 删除原有的txt文件
    os.remove(txt_file)
    print(f'原文件 {txt_file} 已删除')



if __name__ == '__main__':
    # 网络名称和路径
    name1 = 'SyntheticNetEr'
    name2 = 'SyntheticNetWs'
    path = './dataset/synthetic/'

    txt_file = './dataset/synthetic/'  # 原始txt文件路径
    csv_file = './dataset/synthetic/'  # 目标csv文件路径

    for i in range(1):


        txt2 = txt_file + name2 + str(i) +'Influence_p'+ '.txt'
        csv2 = csv_file + name2 + str(i)+'Influence_p' + '.csv'


        txt_to_csv(txt2, csv2)

        print(f'转换完成，结果保存为 {csv2}')



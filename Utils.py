import matplotlib.pyplot as plt
import networkx as nx
import copy
import torch
import torch.nn as nn
import random as rd
import numpy as np
import pandas as pd
import torch.nn.functional as F
from dgl.nn.pytorch import  SAGEConv
def get_neigbors(g, node, depth):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output


def get_dgl_g_input(G0):
    G = copy.deepcopy(G0)
    input = torch.ones(len(G), 11)
    for i in G.nodes():
        input[i, 0] = G.degree()[i]
        input[i, 1] = sum([G.degree()[j] for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)  # 节点的邻居的平均度数
        input[i, 2] = sum([nx.clustering(G, j) for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1) # 节点邻居的平均聚类系数
        egonet = G.subgraph(list(G.neighbors(i)) + [i])
        input[i, 3] = len(egonet.edges())
        input[i, 4] = sum([G.degree()[j] for j in egonet.nodes()]) - 2 * input[i, 3]   #计算自我网络中所有节点的度数之和，减去自我网络中的两倍边数。
    for l in [1, 2, 3]:
        for i in G.nodes():
            ball = get_neigbors(G, i, l)
            input[i, 5 + l - 1] = (G.degree()[i] - 1) * sum([G.degree()[j] - 1 for j in ball[l]])
    v = nx.voterank(G)
    votescore = dict()
    for i in list(G.nodes()): votescore[i] = 0
    for i in range(len(v)):
        votescore[v[i]] = len(G) - i
    e = nx.eigenvector_centrality(G, max_iter=2000)
    k = nx.core_number(G)
    for i in G.nodes():
        input[i, 8] = votescore[i]
        input[i, 9] = e[i]
        input[i, 10] = k[i]
    for i in range(len(input[0])):
        if max(input[:, i]) != 0:
            input[:, i] = input[:, i] / max(input[:, i])
    return input
def get_dgl_g_input_test(G0):
    G = copy.deepcopy(G0)
    input = torch.ones(len(G), 5)
    for i in G.nodes():
        input[i, 0] = G.degree()[i]
        input[i, 1] = sum([G.degree()[j] for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)  # 节点的邻居的平均度数
        input[i, 2] = sum([nx.clustering(G, j) for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1) # 节点邻居的平均聚类系数
    #     egonet = G.subgraph(list(G.neighbors(i)) + [i])
    #     input[i, 3] = len(egonet.edges())
    #     #input[i, 4] = sum([G.degree()[j] for j in egonet.nodes()]) - 2 * input[i, 3]   #计算自我网络中所有节点的度数之和，减去自我网络中的两倍边数。
    # for l in [1, 2]:
    #     for i in G.nodes():
    #         ball = get_neigbors(G, i, l)
    #         input[i, 4 + l - 1] = (G.degree()[i] - 1) * sum([G.degree()[j] - 1 for j in ball[l]])

    e = nx.eigenvector_centrality(G, max_iter=10000)
    k = nx.core_number(G)
    for i in G.nodes():
        input[i, 3] = e[i]
        input[i, 4] = k[i]
    for i in range(len(input[0])):
        if max(input[:, i]) != 0:
            input[:, i] = input[:, i] / max(input[:, i])
    return input

def load_multilayer_sir_labels(path,total_nodes_num,total_layers):
    data ={}
    with open(path,'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            layer = int(vertices[0])  # 网络层
            node = int(vertices[1])  # 节点
            label = float(vertices[2])  # 值
            if layer not in data:
                data[layer] = {}
            data[layer][node] = label


    # 初始化每个节点的总和
    node_sums = {node: 0 for node in range(1, total_nodes_num + 1)}

    # 累加每个节点在所有层的值
    for layer, nodes in data.items():
        for node in range(1, total_nodes_num + 1):
            node_sums[node] += nodes.get(node, 0)  # 缺失节点补 0

    # 计算每个节点的平均值
    node_averages = {node: total / total_layers for node, total in node_sums.items()}
    node_averages = dict(sorted(node_averages.items(), key=lambda x:x[1],reverse=True))
    return node_averages,data

def cal_average_sir(data,total_layers,total_nodes_num):
    # 初始化每个节点的总和
    node_sums = {node: 0 for node in range(1, total_nodes_num + 1)}

    # 累加每个节点在所有层的值
    for layer, nodes in data.items():
        for node in range(1, total_nodes_num + 1):
            node_sums[node] += nodes.get(node, 0)  # 缺失节点补 0


    # 计算每个节点的平均值
    node_averages = {node: total / total_layers for node, total in node_sums.items()}
    node_averages = dict(sorted(node_averages.items(), key=lambda x: x[1], reverse=True))
    return node_averages

def IC_simulation(p, g, set):
    g = copy.deepcopy(g)
    # pos = nx.spring_layout(g)
    if set == []:
        for i in list(g.nodes()):
            g.nodes[i]['state'] = 1 if rd.random() < .01 else 0
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
                        nextg.nodes[j]['state'] = 1 if rd.random() < nextg.edges[i, j]['p'] else 0
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


def matrix_(G, L):
    # A=[]
    B = []
    labels = []
    Gu = []
    for node in list(G.nodes()):
        L_1_neigbors = []
        for depth in range(1, L):  # 1-7
            neighbors = get_neigbors(G, node, depth)
            neighbors_depth = neighbors[depth]
            if len(neighbors_depth) <= L - 1 - len(L_1_neigbors):
                L_1_neigbors = L_1_neigbors + neighbors_depth
            else:
                degree = [G.degree(x) for x in neighbors_depth]
                rank = sorted(range(len(degree)), key=lambda k: degree[k], reverse=True)
                neighbors_depth1 = [neighbors_depth[rank[i]] for i in range(len(rank))]
                L_1_neigbors = L_1_neigbors + neighbors_depth1[0:L - 1 - len(L_1_neigbors)]
            if len(L_1_neigbors) == L - 1:
                Gu.append([node] + L_1_neigbors)
                break
    for m in range(len(Gu)):
        u0 = list(G.nodes)[m]
        u = Gu[m]
        egonet = G.subgraph(Gu[m])
        Au = nx.adjacency_matrix(egonet)
        # A.append(Au)
        Bu = copy.deepcopy(Au)
        for i in range(len(u)):
            for j in range(len(u)):
                if i == 0 and j > 0:
                    Bu[i, j] = Au[i, j] * egonet.degree(u[j])
                if i > 0 and j == 0:
                    Bu[i, j] = Au[i, j] * egonet.degree(u[i])
                if i == j:
                    Bu[i, j] = egonet.degree(u[i])
                else:
                    Bu[i, j] = Au[i, j]
        Bu = Bu.toarray()
        Bu = torch.FloatTensor(Bu).reshape(1, 1, L, L)
        B.append(Bu)
        # 如果 B 为空，提供一个默认值（全零张量）
    if len(B) == 0:
        default_tensor = torch.zeros(1, 1, L, L)  # 默认值（全零矩阵）
        B.append(default_tensor)
    matrix = torch.concat(B)  # len(G)*1*len(egonet)*len(egonet)
    return matrix


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim=3, hidden_dim=128, dense_dim=32, target_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, dense_dim)
        self.l2 = nn.Linear(dense_dim, target_size)

    def forward(self, embedding):
        lstm_out, _ = self.lstm(embedding)
        l1_out = self.l1(lstm_out)
        l2_out = self.l2(l1_out)
        return l2_out


# def embedding_(G):
#     p = nx.degree_centrality(G)# degree centrality
#     q = H_index(G)
#     r = nx.core_number(G)
#     p = list(p.values())
#     q = list(q.values())
#     r = list(r.values())
#     p = [x/max(p) for x in p]
#     q = [x/max(q) for x in q]
#     r = [x/max(r) for x in r]
#     fmat = [torch.Tensor([p[i],q[i],r[i]]).reshape(1,3) for i in range(len(G))]
#     embedding = torch.concat(fmat)
#     return embedding
#
#
def H_index(g, node_set=-1):
    if node_set == -1:
        nodes = list(g.nodes())
        d = dict()
        H = dict()  # H-index
        for node in nodes:
            d[node] = g.degree(node)
        for node in nodes:
            neighbors = list(g.neighbors(node))
            neighbors_d = [d[x] for x in neighbors]
            for y in range(len(neighbors_d)):
                if y > len([x for x in neighbors_d if x >= y]):
                    break
            H[node] = y - 1

    if node_set in list(g.nodes()):  # 计算节点node的H-index
        neighbors = list(g.neighbors(node))
        neighbors_d = [d[x] for x in neighbors]
        for y in range(len(neighbors_d)):
            if y > len([x for x in neighbors_d if x >= y]):
                break
        H = y - 1
    return H


def calculate_h_index(G,node):
    degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
    degrees.sort(reverse=True)

    h_index = 0
    for i, degree in enumerate(degrees, 1):
        if degree >= i:
            h_index = i
        else:
            break
    return h_index

# 为图 G 中的每个节点计算 h-index，并返回结果字典
def calculate_all_h_indices(G):
    h_indices = {node: calculate_h_index(G, node) for node in G.nodes()}
    return h_indices

def embedding_(G):
    p = nx.degree_centrality(G)# degree centrality
    q = calculate_all_h_indices(G)
    r = nx.core_number(G)
    p = list(p.values())
    q = list(q.values())
    r = list(r.values())

    p = [x/max(p) for x in p]
    q = [x/max(q) for x in q]
    r = [x/max(r) for x in r]
    fmat = [torch.Tensor([p[i],q[i],r[i]]).reshape(1,3) for i in range(len(G))]
    embedding = torch.concat(fmat)
    return embedding


def load_csv_net_gcc(name):
    path = "..\\dataset\\real\\%s\\" % name
    nodes_pos = pd.read_csv(path + "nodes_gcc.csv")
    edges = pd.read_csv(path + "edges_gcc.csv")
    pos = dict()
    for i in range(len(nodes_pos)):
        x = nodes_pos[' _pos'][i]
        x = x.strip('[]')
        x = x.split(",")
        x = list(map(float, x))
        x = np.array(x)
        # nodes_pos[' _pos'][i] = x
        pos[nodes_pos["# index"][i]] = x
    G = nx.Graph()
    G.add_nodes_from(list(nodes_pos['# index']))
    edge_list = [(edges['# source'][i], edges[' target'][i]) for i in range(len(edges))]
    G.add_edges_from(edge_list)
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        G.nodes[i]['state'] = 0
    return G, pos


def cal_betac(G):
    # 计算渗流阈值近似值
    d = [G.degree()[node] for node in list(G.nodes())]
    d2 = [x ** 2 for x in d]
    d_ = sum(d) / len(d)
    d2_ = sum(d2) / len(d2)
    betac = d_ / (d2_ - d_)
    return betac
    return G,pos


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

class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.gcn1 = SAGEConv(5, 32, aggregator_type="lstm")
        self.gcn2 = SAGEConv(32, 32, aggregator_type="lstm")

        # Linear layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)

        # Activation function
        # activation_functions = {
        #     "relu": nn.ReLU(),
        #     "tanh": nn.Tanh(),
        #     "sigmoid": nn.Sigmoid(),
        #     "elu": nn.ELU(),
        #     "LeakyReLU": nn.LeakyReLU()
        # }
        self.activation = nn.LeakyReLU()

    def forward(self, graphs, node_features):
        gcn_outputs = []
        graphs = [graphs]
        node_features = [node_features]
        # 针对每个网络图独立使用GCN处理节点特征
        for i, g in enumerate(graphs):
            x = self.gcn1(g, node_features[i])  # GCN输出节点特征
            x = F.relu(x)
            x = self.gcn2(g, x)
            x = F.relu(x)
            gcn_outputs.append(x)
        combined_features = torch.stack(gcn_outputs, dim=0)  # Shape: [L, num_nodes, gat_out_dim]
        # print('combined_features',combined_features.shape)
        x = torch.mean(combined_features, dim=0)  # 或者使用 max 进行池化
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)


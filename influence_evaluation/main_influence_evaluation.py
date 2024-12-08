import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
from influence_evaluation.ALGE import ALGE_C
from influence_evaluation.other_algorithms import ci,k_shell,h_index,lid,dcl,ncvr,rcnn,glstm
import numpy as np
import math
import influence_evaluation.Model
from Utils import load_csv_net_gcc,LSTMModel
import pickle
import random
def calculate_test(G,data_memory):
    k_algec, r_algec, node_algec, node_rank_algec, _, train_nodes = ALGE_C(G, data_memory)
    k_c,r_c,node_c,node_rank_CI = ci(G,data_memory,train_nodes)
    k_k,r_k,node_k,node_rank_K  = k_shell(G, data_memory,train_nodes)
    k_h,r_h,node_h,node_rank_H  = h_index(G, data_memory,train_nodes)
    k_l,r_l,node_lid,node_rank_L  = lid(G, data_memory,train_nodes)
    k_d,r_d,node_dcl,node_rank_D  = dcl(G, data_memory,train_nodes)
    k_n,r_n,node_ncvr,node_rank_N  = ncvr(G, data_memory,train_nodes)
    k_r,r_r,node_rcnn,node_rank_R = rcnn(G, data_memory,train_nodes)
    k_gl,r_gl,node_glstm,node_rank_G  = glstm(G, data_memory,train_nodes)

    ken_list = [k_c,k_k,k_h,k_l,k_d,k_n,k_r,k_gl,k_algec]
    rank_list = [r_c,r_k,r_h,r_l,r_d,r_n,r_r,r_gl,r_algec]
    node_sort = [node_c,node_k,node_h,node_lid,node_dcl,node_ncvr,node_rcnn,node_glstm,node_algec]
    node_rank = [node_rank_CI,node_rank_K,node_rank_H,node_rank_L,node_rank_D ,node_rank_N ,node_rank_R,node_rank_G,node_rank_algec]
    return ken_list,rank_list,node_sort,node_rank,train_nodes


def calculate_all(G,data_memory):
    k_c,r_c,node_c,node_rank_CI = ci(G,data_memory)
    k_k,r_k,node_k,node_rank_K  = k_shell(G, data_memory)
    k_h,r_h,node_h,node_rank_H  = h_index(G, data_memory)
    k_l,r_l,node_lid,node_rank_L  = lid(G, data_memory)
    k_d,r_d,node_dcl,node_rank_D  = dcl(G, data_memory)
    k_n,r_n,node_ncvr,node_rank_N  = ncvr(G, data_memory)
    k_r,r_r,node_rcnn,node_rank_R = rcnn(G, data_memory)
    k_gl,r_gl,node_glstm,node_rank_G  = glstm(G, data_memory)
    k_algec,r_algec,node_algec,node_rank_algec,_,_  = ALGE_C(G, data_memory)
    ken_list = [k_c,k_k,k_h,k_l,k_d,k_n,k_r,k_gl,k_algec]
    rank_list = [r_c,r_k,r_h,r_l,r_d,r_n,r_r,r_gl,r_algec]
    node_sort = [node_c,node_k,node_h,node_lid,node_dcl,node_ncvr,node_rcnn,node_glstm,node_algec]
    node_rank = [node_rank_CI,node_rank_K,node_rank_H,node_rank_L,node_rank_D ,node_rank_N ,node_rank_R,node_rank_G,node_rank_algec]
    return ken_list,rank_list,node_sort,node_rank
def frequency_rank(rank_list,name,n=0):
    if n==0:n = len(rank_list[0])
    rank_list = [l[0:n] for l in rank_list]
    m= ['o','+','^','x','s','D','o','^','x','s']
    edgecolors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    colors = ['none','orange','none','red','none','none','none','none','olive','none']
    methods = ['CI','kshell','H-index','LID','DCL','NCVR','RCNN','GLSTM','ALGE','ALGE-C']
    plt.title(name)
    plt.figure(figsize=[8, 5])  # 设置画布
    plt.subplots_adjust(right=0.7,bottom=0.1)
    plt.xlabel('Rank')
    plt.ylabel('Frequency(ln)')
    for i in range(len(rank_list)):
        r = rank_list[i]
        x = list(set(rank_list[i]))
        y = [math.log(r.count(j)) for j in x]
        plt.scatter(x,y,marker=m[i],label =methods[i],s=20,alpha=0.8,c=colors[i],edgecolors=edgecolors[i])  # edgecolor
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=True, fontsize=10)
    plt.savefig('%s rank frequency_%s.png' % (name,n),dpi=300)

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
        average_influence_measures = [[node, self.calculate_average_influence(beta, node)] for node in self.G.nodes()]
        sorted_nodes = sorted(average_influence_measures, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_nodes],[item[1] for item in sorted_nodes],sorted_nodes

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
def threshhold(G):
    # 计算网络的平均度
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    # 计算网络中每个节点的度数，求平方并相加
    squared_degree_sum = sum(deg ** 2 for node, deg in G.degree())
    # 计算度的平方的平均值
    average_squared_degree = squared_degree_sum / len(G)
    beita = avg_degree / (average_squared_degree - avg_degree)
    return round(beita, 4)

def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[-1])
            G.add_edge(source, target)
    return G

if __name__=='__main__':
    #加载 .pkl 文件
    # with open('node_rank_record.pkl', "rb") as file:
    #     data = pickle.load(file)
    # # 查看内容
    # print(data)

    # 影响力label路径，网络数据路径，所有网络名称
    inpath  = '..\\dataset\\real\\'
    influence_path= '..\\dataset\\real_influence\\'
    datanames = set(os.listdir(inpath))
    net_names = []
    for i in datanames: net_names.append(i)
    net_names.sort()
    ken_table = []
    rank_record = []
    sort_record = []
    node_rank_record = []
    train_nodes_list =[]
    # with open('train_nodes_list.pkl', 'rb') as f:
    #     train_nodes_list = pickle.load(f)
    # with open('ken_table.pkl', 'rb') as f:
    #     ken_table = pickle.load(f)
    # with open('rank_record.pkl', 'rb') as f:
    #     rank_record= pickle.load(f)
    # with open('sort_record.pkl', 'rb') as f:
    #     sort_record = pickle.load(f)
    # with open('node_rank_record.pkl', 'rb') as f:
    #     node_rank_record = pickle.load(f)

    for x in range(15,16):
        name = net_names[x]  # 网络名
        print("net %s %s" % (x + 1, name))
        #data = pd.read_csv(influence_path + '%s_gcc_Influence_P=1.5betac_Run1000.csv' % (name))  # 仿真得到的真实值
        #data_memory = [list(data.loc[i]) for i in range(len(data))]
        # G, pos = load_csv_net_gcc(name)
        G = load_graph(r'C:\Users\ADM\Desktop\CycleRatio-main\Dataset\Jazz.txt')
        G = nx.convert_node_labels_to_integers(G)
        print(G)
        b = threshhold(G) * 1.5
        data_memory = InfluenceCalculator(G).calculate_sorted_nodes(b)[-1]
        print(data_memory)
        ken_list,rank_list,node_sort,node_rank,train_nodes=calculate_test(G,data_memory)

        for x in data_memory: x[0] = int(x[0])
        simu_I = data_memory.copy()
        simu_I.sort(key=lambda x: x[1], reverse=True)
        simu_sort = [x[0] for x in simu_I]
        node_rank=[simu_sort]+node_rank
        # frequency_rank(rank_list,name,n)
        train_nodes_list.append(train_nodes)
        ken_table.append(ken_list)
        rank_record.append(rank_list)
        sort_record.append((node_sort))
        node_rank_record.append(node_rank)
    print('finish')

    with open('train_nodes_list.pkl', 'wb') as f:
        pickle.dump(train_nodes_list, f)
    with open('ken_table.pkl', 'wb') as f:
        pickle.dump(ken_table, f)
    with open('rank_record.pkl', 'wb') as f:
        pickle.dump(rank_record, f)
    with open('sort_record.pkl', 'wb') as f:
        pickle.dump(sort_record, f)
    with open('node_rank_record.pkl', 'wb') as f:
        pickle.dump(node_rank_record, f)

    # with open('train_nodes', 'wb') as f:
    #     pickle.dump(train_nodes, f)
    # with open('ken_table.pkl', 'wb') as f:
    #     pickle.dump(ken_table, f)
    # with open('rank_record.pkl', 'wb') as f:
    #     pickle.dump(rank_record, f)
    # with open('sort_record.pkl', 'wb') as f:
    #     pickle.dump(sort_record, f)
    # with open('node_rank_record.pkl', 'wb') as f:
    #     pickle.dump(node_rank_record, f)
    # with open('data.pkl', 'rb') as f:
    #     loaded_a = pickle.load(f)
    # df = pd.DataFrame(columns =['name','train_size','CI','kshell','H-index','LID','DCL','NCVR','RCNN','GLSTM','ALGE-C'])
    # df['name'] = net_names
    # df['train_size'] = [len(x) for x in train_nodes_list]
    # for i,column in enumerate(['CI','kshell','H-index','LID','DCL','NCVR','RCNN','GLSTM','ALGE-C']):
    #     print(column)
    #     df[column] = [x[i] for x in ken_table]
    # df.to_csv('ken_table_test_data.csv')

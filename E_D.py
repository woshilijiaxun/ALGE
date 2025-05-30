from Utils import load_multilayer_graph,get_dgl_g_input_test,load_multilayer_sir_labels,cal_average_sir,embedding_,\
    get_dgl_g_input,LSTMModel,matrix_,MultiPR,graphs_to_tensor,compute_distance_tensor
import networkx as nx
import dgl
import torch
from scipy.stats import kendalltau
import pickle
from influence_evaluation.ALGE import ALGE_C
import influence_evaluation.Model
import numpy as np
import time
class ED_method:
    def __init__(self, graphs,num):
        self.graphs = graphs
        self.L = len(graphs)
        self.node_list = [i for i in range(1,num+1)]
        #self.N = len(self.node_list)
        self.N = num
        # 确保所有图都有 完整节点集
        for G in self.graphs:
            missing_nodes = set(self.node_list) - set(G.nodes())
            G.add_nodes_from(missing_nodes)



    def compute_metrics(self):
        con = {node: set() for node in self.node_list}
        BC = [{node: 0 for node in self.node_list} for _ in range(self.L)]

        # 计算单层BC
        for alpha in range(self.L):
            bc_dict = nx.betweenness_centrality(self.graphs[alpha], normalized=True)  # normalized=True 自动归一化
            for node in self.node_list:
                BC[alpha][node] = bc_dict.get(node, 0)  # 有些孤立点没值，补0

        # 层间交互部分
        for node in self.node_list:
            for alpha in range(self.L):
                for beta in range(self.L):
                    if alpha != beta:
                        D = BC[alpha][node] + BC[beta][node] + 1
                        con[node].add(D)

        # 熵计算 & 最终得分
        result = {}

        total_sum = np.sum([len(con[n]) + sum([np.sum(nx.to_numpy_array(self.graphs[a], nodelist=self.node_list)[self.node_list.index(n)]) for a in range(self.L)]) for n in self.node_list])

        for node in self.node_list:
            total_degree = 0
            for alpha in range(self.L):
                A = nx.to_numpy_array(self.graphs[alpha], nodelist=self.node_list)
                total_degree += np.sum(A[self.node_list.index(node)])

            p = (len(con[node]) + total_degree) / total_sum
            entropy_value = p * np.log(p + 1e-10) *(-1)
            result[node] = entropy_value

        return result

def ED(path,nodes_num,network_name):

    Gs, total_layers = load_multilayer_graph(path)

    ED_dict = ED_method(Gs, nodes_num).compute_metrics()
    ED_sorted_dict = dict(sorted(ED_dict.items(), key=lambda x: x[1], reverse=True))
    print(ED_sorted_dict)






    sir_dict, _ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))

    pre_sorted_node = [key for key in ED_sorted_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)

    print(k[0])
    print('ed_list', pre_sorted_node)
    print('sir_list', sir_list)



    return k[0],ED_sorted_dict

def GLSTM(path,nodes_num,network_name):
    model = torch.load('influence_evaluation/GLSTM_1k_4.pth')
    Gs, total_layers = load_multilayer_graph(path)
    node_feature_lsit = []
    maps = []
    for i in range(total_layers):  # 将每层网络的节点映射到0-n，保存映射关系。模型预测后转换节点映射。
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)

        mapping = {node: idx for idx, node in enumerate(Gs[i].nodes())}  # {node:id}
        maps.append(mapping)
        Gs[i] = nx.convert_node_labels_to_integers(Gs[i])

        node_feature = embedding_(Gs[i])
        node_feature_lsit.append(node_feature)


    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))

    g_list = [dgl.from_networkx(Gs[i]) for i in range(total_layers)]

    pre_sir_dict = {}
    for i in range(total_layers):
        model.eval()
        with torch.no_grad():
            predictions = model(node_feature_lsit[i])
        predictions = predictions.tolist()
        # 提取节点及影响力值
        nodes_list = list(Gs[i].nodes())
        node_influence = {nodes_list[j]: predictions[j][0] for j in range(len(predictions))}
        sorted_node_influence = dict(sorted(node_influence.items(), key=lambda x: x[1], reverse=True))
        pre_dict = sorted_node_influence
        pre_sir_dict[i + 1] = pre_dict
    # print(pre_sir_dict)
    # print(maps)
    result = {}
    for i in range(total_layers):
        dic = {}
        for node, id in maps[i].items():
            if id in pre_sir_dict[i + 1]:
                dic[node] = pre_sir_dict[i + 1][id]
        dic = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
        result[i + 1] = dic
    # print(result)

    pre_avg_sir_dict = cal_average_sir(result, total_layers, nodes_num)
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0],pre_avg_sir_dict

def RCNN(path,nodes_num,network_name):
    L = 28
    model = torch.load('influence_evaluation/RCNN.pth')
    Gs, total_layers = load_multilayer_graph(path)
    node_feature_lsit = []
    maps = []
    for i in range(total_layers):  # 将每层网络的节点映射到0-n，保存映射关系。模型预测后转换节点映射。
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)

        mapping = {node: idx for idx, node in enumerate(Gs[i].nodes())}  # {node:id}
        maps.append(mapping)
        Gs[i] = nx.convert_node_labels_to_integers(Gs[i])

        node_feature = matrix_(Gs[i],L=L)
        node_feature_lsit.append(node_feature)


    sir_dict, _ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))

    g_list = [dgl.from_networkx(Gs[i]) for i in range(total_layers)]

    pre_sir_dict = {}
    for i in range(total_layers):
        model.eval()
        with torch.no_grad():
            predictions = model(node_feature_lsit[i])
        predictions = predictions.tolist()
        # 提取节点及影响力值
        nodes_list = list(Gs[i].nodes())
        node_influence = {nodes_list[j]: predictions[j][0] for j in range(len(predictions))}
        sorted_node_influence = dict(sorted(node_influence.items(), key=lambda x: x[1], reverse=True))
        pre_dict = sorted_node_influence
        pre_sir_dict[i + 1] = pre_dict
    # print(pre_sir_dict)
    # print(maps)
    result = {}
    for i in range(total_layers):
        dic = {}
        for node, id in maps[i].items():
            if id in pre_sir_dict[i + 1]:
                dic[node] = pre_sir_dict[i + 1][id]
        dic = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
        result[i + 1] = dic
    # print(result)

    pre_avg_sir_dict = cal_average_sir(result, total_layers, nodes_num)
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0],pre_avg_sir_dict





def DC(path,nodes_num,NetworkName):
    Gs, total_layers = load_multilayer_graph(path)
    DegreeDict = {}
    for i in range(total_layers):
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)
        dc = nx.degree_centrality(Gs[i])
        DegreeDict[i+1] = dc

    multiplex_network = path.split('.')[-2].split('/')[-1]
    network_name = NetworkName
    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))


    pre_avg_sir_dict = cal_average_sir(DegreeDict, total_layers, nodes_num)

    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)

    return k[0],pre_avg_sir_dict

def k_shell(path,nodes_num,network_name):
    Gs, total_layers = load_multilayer_graph(path)
    KshellDict = {}
    for i in range(total_layers):
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)
        kshell = nx.core_number(Gs[i])
        KshellDict[i + 1] = kshell


    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))
    print(KshellDict)
    pre_avg_sir_dict = cal_average_sir(KshellDict, total_layers, nodes_num)
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0],pre_avg_sir_dict

def f_eigenvector_centrality(path,nodes_num,NetworkName):
    Gs, total_layers = load_multilayer_graph(path)

    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = NetworkName

    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))
    print('sir_list',sir_list)
    f_eigenvector_dict,layer_inf = MultiPR(Gs,nodes_num)

    f_eigenvector_sorted_node = [key for key in f_eigenvector_dict.keys()]
    node_rank_f_eigenvector = [f_eigenvector_sorted_node.index(x) if x in f_eigenvector_sorted_node else len(f_eigenvector_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_f_eigenvector)
    print('node_rank_f_eigenvector',node_rank_f_eigenvector)

    return k[0],f_eigenvector_dict


def PRGC(path,nodes_num,NetworkName):
    Gs, total_layers = load_multilayer_graph(path)
    adj_matrix, _ = graphs_to_tensor(Gs, nodes_num) # 获得多层网络的邻接矩阵，用于计算距离矩阵
    nodes_inf_dict, layer_inf = MultiPR(Gs, nodes_num)   #获得节点中心性值和层中心性值, x={node:influence}, y=[]
    print('nid',nodes_inf_dict)
    print(len(list(nodes_inf_dict)))


    distance_tensor = compute_distance_tensor(adj_matrix)
    # 确保层重要性的形状匹配
    if distance_tensor.shape[2] != len(layer_inf):
        raise ValueError("层重要性列表的长度必须与张量的层数相等")
    # 逐层加权
    weighted_distance = np.tensordot(distance_tensor, layer_inf, axes=([2], [0]))  # 将每层的节点距离乘上层重要性再加权求和
    print(weighted_distance)
    """
       计算节点的最终影响值，仅考虑二阶以内邻居。

       参数：
           weighted_distance (np.ndarray): 节点加权距离的二维矩阵，形状为 (N, N)。
           nodes_inf_dict (dict): 节点的中心性值，形如 {节点: 中心性值}。
           adj_matrix (np.ndarray): 邻接矩阵，形状为 (N, N)。

       返回：
           final_influence (dict): 节点的最终影响值，形如 {节点: 影响值}。
       """
    num_nodes = adj_matrix.shape[0]
    final_influence = {}
    #print(weighted_distance.shape)
    #print(np.sum((weighted_distance > 0) ))
    combined_adj = np.sum(adj_matrix, axis=2)  # 综合所有层的信息，得到形状 (N, N)

    for i in range(num_nodes):


        first_order_neighbors = set(np.where(combined_adj[i] > 0)[0])
        # 找到一阶邻居
        #first_order_neighbors = set(np.where(adj_matrix[i] > 0)[0])
        # 计算一阶邻居的影响
        first_order_influence = 0
        for j in first_order_neighbors:
             if weighted_distance[i][j] > 0:  # 防止除以零
                first_order_influence += (
                        nodes_inf_dict.get(i+1, 0) * nodes_inf_dict.get(j+1, 0) /
                        (weighted_distance[i][j] ** 2)
                )
        # 找到二阶邻居
        second_order_neighbors = set()
        for neighbor in first_order_neighbors:
            second_order_neighbors.update(np.where(combined_adj[neighbor] > 0)[0])

        # 排除自身
        second_order_neighbors.discard(i)

        # 计算二阶邻居的影响
        second_order_influence = 0
        # for k in second_order_neighbors:
        #     #if weighted_distance[i][j] > 0 :  # 防止除以零
        #         second_order_influence += (
        #                 nodes_inf_dict.get(i+1, 0) * nodes_inf_dict.get(k+1, 0) /
        #                 (weighted_distance[i][k] ** 2)
        #         )

        # 存储最终影响值
        final_influence[i+1] = first_order_influence + second_order_influence

    print('keys',final_influence.keys())
    prgc_dict = dict(sorted(final_influence.items(),key=lambda x:x[1],reverse=True))
    print('pd',prgc_dict)
    print(len(list(prgc_dict)))
    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = NetworkName

    sir_dict, _ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))
    print('sir_list', sir_list)

    prgc_sorted_node = [key for key in prgc_dict.keys()]
    print('prgc_sorted_node',prgc_sorted_node)
    node_rank_prgc = [
        prgc_sorted_node.index(x) if x in prgc_sorted_node else len(prgc_sorted_node) for x
        in sir_list]
    k = kendalltau(node_rank_simu, node_rank_prgc)
    print('node_rank_prgc', node_rank_prgc)


    return k[0], prgc_dict



def convert_sir(sir_dict, mapping):
    dic = {}
    for node, id in mapping.items():
        dic[id] = sir_dict[node]
    dic = dict(sorted(dic.items(),key=lambda x:x[1],reverse=True))
    data_memory = [[k,v] for k,v in dic.items()]
    return data_memory

def MGNN_AL(path,nodes_num,NetworkName):
    active_learning_epochs = {61:20,71:5,105:6,246:20,279:3,1005:15,2640:20,3879:10,6980:10,8215:5}
    Gs, total_layers = load_multilayer_graph(path)
    node_feature_lsit = []
    maps = []

    for i in range(total_layers):            #将每层网络的节点映射到0-n，保存映射关系。模型预测后转换节点映射。
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)

        mapping = {node: idx for idx, node in enumerate(Gs[i].nodes())}       #{node:id}
        maps.append(mapping)
        Gs[i]= nx.convert_node_labels_to_integers(Gs[i])

        node_features = get_dgl_g_input_test(Gs[i])
        # node_features_ = get_dgl_g_input(Gs[i])
        # node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)
        node_feature_lsit.append(node_features)
    #print(maps)



    network_name = NetworkName
    print(network_name)
    #nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
    sir_dict,sir_dict_each_layer = load_multilayer_sir_labels(network_name,nodes_num ,total_layers)
    sir_list = [key for key in sir_dict.keys()]
    #print('sir_dict',sir_dict)
    #data_memory = [[key,value] for key,value in sir_dict.items()]

    node_rank_simu = list(range(0, len(sir_list)))

    g_list = [dgl.from_networkx(Gs[i]) for i in range(total_layers)]

    pre_sir_dict = {}
    for i in range(total_layers):
        model = torch.load('influence_evaluation/GraphSAGE_GAT_concat.pth')
        # model.eval()
        # with torch.no_grad():
        #     predictions = model(g_list[i], node_feature_lsit[i])

        data_memory = convert_sir(sir_dict_each_layer[i+1],maps[i])
        epoch_num = active_learning_epochs[nodes_num]
        predictions = ALGE_C(model,Gs[i],data_memory,epoch_num)

        #print('1111111111111111',predictions)

        predictions = predictions.tolist()
        # 提取节点及影响力值
        nodes_list = list(Gs[i].nodes())
        node_influence = {nodes_list[j]: predictions[j][0] for j in range(len(predictions))}
        sorted_node_influence = dict(sorted(node_influence.items(), key=lambda x: x[1], reverse=True))
        pre_dict = sorted_node_influence
        pre_sir_dict[i + 1] = pre_dict
        # print(pre_sir_dict)
        # print(maps)
    result = {}
    for i in range(total_layers):
        dic = {}
        for node, id in maps[i].items():
            if id in pre_sir_dict[i+1]:
                dic[node] = pre_sir_dict[i+1][id]
        dic = dict(sorted(dic.items(),key=lambda x:x[1],reverse=True))
        result[i+1] = dic
    #print(result)

    pre_avg_sir_dict = cal_average_sir(result,total_layers, nodes_num)
    print('pre_avg_sir_dict:',pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0],pre_avg_sir_dict

if __name__ == '__main__':


    # nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980, 'celegans_connectome_multiplex': 279,
    #                                      'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
    #                                      'cS-Aarhus_multiplex': 61, 'drosophila_genetic_multiplex': 8215, 'hepatitusC_genetic_multiplex': 105,
    #                                      'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71, 'rattus_genetic_multiplex': 2640}
    nodes_num_from_multiplex_networks = {'celegans_connectome_multiplex': 279,
                                         'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
                                         'cS-Aarhus_multiplex': 61,
                                         'hepatitusC_genetic_multiplex': 105,
                                         'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71,
                                         'rattus_genetic_multiplex': 2640}

    network = [key+'.edges' for key in nodes_num_from_multiplex_networks.keys()]
    Result_sorted_dict= {}
    Result_kendall= {}

    for name in network:
        print(f'正在处理{name}')
        #path = 'MNdata/drosophila_genetic_multiplex.edges'
        relative_path = 'MNdata/' + name
        PATH = './dataset/real_multiplex_networks/MNdata/' + name
        print(relative_path)
        multiplex_network = relative_path.split('/')[1].split('.')[0]
        nodes_num = nodes_num_from_multiplex_networks[multiplex_network]

        nodes_ranking_dict = {}
        ken_dict = {}
        for time in [0.5,0.75,1,1.25,1.5]:
            network_name = './dataset/real-influence/MN_SIR_' + str(time) + 'beitac/' + multiplex_network + '.txt'
            dc_k,dc_dict = DC(PATH,nodes_num,network_name)
            kshell_k,kshell_dict = k_shell(PATH,nodes_num,network_name)
            glstm_k,glstm_dict = GLSTM(PATH,nodes_num,network_name)
            rcnn_k,rcnn_dict = RCNN(PATH,nodes_num,network_name)
            f_k,f_dict= f_eigenvector_centrality(PATH,nodes_num,network_name)
            prgc_k,prgc_dict= PRGC(PATH,nodes_num,network_name)
            ed_k,ed_dict = ED(PATH,nodes_num,network_name)
            mgnn_k,mgnn_dict = MGNN_AL(PATH,nodes_num,network_name)
            nodes_ranking_dict[time] = {'dc':dc_dict,'kshell':kshell_dict,'glstm':glstm_dict,'rcnn':rcnn_dict,'f-e':f_dict,'prgc':prgc_dict,
                         'ed':ed_dict,'mgnn-al':mgnn_dict}
            ken_dict[time] = {'dc': dc_k, 'kshell': kshell_k, 'glstm': glstm_k, 'rcnn': rcnn_k,
                                        'f-e': f_k, 'prgc': prgc_k,
                                        'ed': ed_k, 'mgnn-al': mgnn_k}
        Result_sorted_dict[name] = nodes_ranking_dict
        Result_kendall[name] = ken_dict


    with open('node_ranking_dict.pkl', 'wb') as f:
        pickle.dump(Result_sorted_dict, f)

    with open('kendall[0.5-1.5].pkl', 'wb') as f:
        pickle.dump(Result_kendall, f)


        #model = torch.load('influence_evaluation/ALGE_B_11_20.pth')

    # nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980, 'celegans_connectome_multiplex': 279,
    #                                      'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
    #                                      'cS-Aarhus_multiplex': 61, 'drosophila_genetic_multiplex': 8215, 'hepatitusC_genetic_multiplex': 105,
    #                                      'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71, 'rattus_genetic_multiplex': 2640}
    #
    #
    # epoch_for_multiplex_networks = {'arabidopsis_genetic_multiplex': 10, 'celegans_connectome_multiplex': 3,
    #                                      'celegans_genetic_multiplex': 10, 'cKM-Physicians-Innovation_multiplex': 20,
    #                                      'cS-Aarhus_multiplex': 20, 'drosophila_genetic_multiplex': 5,
    #                                      'hepatitusC_genetic_multiplex': 6,
    #                                      'humanHIV1_genetic_multiplex': 15, 'lazega-Law-Firm_multiplex': 5,
    #                                      'rattus_genetic_multiplex': 20}
    #
    # #path = 'MNdata/cS-Aarhus_multiplex.edges'
    # #multiplex_network = path.split('/')[1].split('.')[0]
    # #
    # # network = [key+'.edges' for key in nodes_num_from_multiplex_networks.keys()]
    # # Result_k = {}
    # # Result_sorted_nodes = {}
    # # Result_dict = {}
    # # Result_layer = {}
    # # for name in network:
    # #     path = 'MNdata/' + name
    # #     multiplex_network = path.split('/')[1].split('.')[0]
    # #     nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
    # #     prgc_k,prgc_eigenvector_sorted_node, = PRGC(path,nodes_num)
    # #     #print(k_f_eigenvector,f_eigenvector_sorted_nodes)
    # #     Result_k[name] = prgc_k
    # #     Result_sorted_nodes[name] = prgc_eigenvector_sorted_node
    # #
    # #
    # # with open('prgc_ken.pkl', 'wb') as f:
    # #     pickle.dump(Result_k, f)
    # #
    # # with open('f_eigenvector_sorted_nodes.pkl', 'wb') as f:
    # #     pickle.dump(Result_sorted_nodes, f)
    # #
    # # with open('f_eigenvector_nodes_dict.pkl', 'wb') as f:
    # #     pickle.dump(Result_dict, f)
    # #
    # # with open('f_eigenvector_layer_inf.pkl', 'wb') as f:
    # #     pickle.dump(Result_layer, f)
    #
    #
    #
    # network = [key+'.edges' for key in nodes_num_from_multiplex_networks.keys()]
    # Result = {}
    # ken_result = {}
    # for name in network:
    # #for _ in range(1):
    #
    #     epoch_num = epoch_for_multiplex_networks[name.split('.')[0]]
    #     print(epoch_num)
    #     path = 'MNdata/' + name
    #     #path = 'MNdata/hepatitusC_genetic_multiplex.edges'
    #     Gs, total_layers = load_multilayer_graph(path)
    #     multiplex_network = path.split('/')[1].split('.')[0]
    #
    #     node_feature_lsit = []
    #     maps = []
    #
    #     for i in range(total_layers):            #将每层网络的节点映射到0-n，保存映射关系。模型预测后转换节点映射。
    #         # 查找入度为0的节点
    #         zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
    #         Gs[i].remove_nodes_from(zero_degree_nodes)
    #
    #         mapping = {node: idx for idx, node in enumerate(Gs[i].nodes())}       #{node:id}
    #         maps.append(mapping)
    #         Gs[i]= nx.convert_node_labels_to_integers(Gs[i])
    #
    #         node_features = get_dgl_g_input_test(Gs[i])
    #         # node_features_ = get_dgl_g_input(Gs[i])
    #         # node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)
    #         node_feature_lsit.append(node_features)
    #     print(maps)
    #     ken = {}
    #     for time in [1]:
    #
    #        # network_name = 'MN_SIR_1beitac/' + multiplex_network + '.txt'
    #         network_name = 'MN_SIR_'+str(time)+'beitac/' + multiplex_network + '.txt'
    #         print(network_name)
    #         nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
    #         sir_dict,sir_dict_each_layer = load_multilayer_sir_labels(network_name,nodes_num ,total_layers)
    #         sir_list = [key for key in sir_dict.keys()]
    #         print('sir_dict',sir_dict)
    #         #data_memory = [[key,value] for key,value in sir_dict.items()]
    #
    #         node_rank_simu = list(range(0, len(sir_list)))
    #
    #         g_list = [dgl.from_networkx(Gs[i]) for i in range(total_layers)]
    #
    #         pre_sir_dict = {}
    #         for i in range(total_layers):
    #             model = torch.load('influence_evaluation/GraphSAGE_GAT_concat.pth')
    #             # model.eval()
    #             # with torch.no_grad():
    #             #     predictions = model(g_list[i], node_feature_lsit[i])
    #
    #             data_memory = convert_sir(sir_dict_each_layer[i+1],maps[i])
    #
    #             predictions = ALGE_C(model,Gs[i],data_memory,epoch_num)
    #
    #             #print('1111111111111111',predictions)
    #
    #             predictions = predictions.tolist()
    #             # 提取节点及影响力值
    #             nodes_list = list(Gs[i].nodes())
    #             node_influence = {nodes_list[j]: predictions[j][0] for j in range(len(predictions))}
    #             sorted_node_influence = dict(sorted(node_influence.items(), key=lambda x: x[1], reverse=True))
    #             pre_dict = sorted_node_influence
    #             pre_sir_dict[i + 1] = pre_dict
    #         print(pre_sir_dict)
    #         print(maps)
    #         result = {}
    #         for i in range(total_layers):
    #             dic = {}
    #             for node, id in maps[i].items():
    #                 if id in pre_sir_dict[i+1]:
    #                     dic[node] = pre_sir_dict[i+1][id]
    #             dic = dict(sorted(dic.items(),key=lambda x:x[1],reverse=True))
    #             result[i+1] = dic
    #         print(result)
    #
    #         pre_avg_sir_dict = cal_average_sir(result,total_layers, nodes_num_from_multiplex_networks[multiplex_network])
    #         print('pre_avg_sir_dict:',pre_avg_sir_dict)
    #         pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    #         node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    #         k = kendalltau(node_rank_simu, node_rank_p)
    #
    #
    #
    #         k_glstm,glstm_sorted_nodes = GLSTM(path,nodes_num)
    #
    #
    #         k_dc,dc_sorted_nodes = DC(path,nodes_num,network_name)
    #
    #
    #         k_kshell,kshell_sorted_nodes = k_shell(path, nodes_num)
    #
    #         #k_rcnn ,rcnn_sorted_nodes= RCNN(path,nodes_num)
    #         k123 = {}
    #         print('pre',k[0])
    #         print('GLSTM', k_glstm)
    #         print('DC', k_dc)
    #         print('Kshell', k_kshell)
    #         #print('rcnn', k_rcnn)
    #         print(g_list)
    #         k123={'pre':k[0]}
    #         #Result[name] = {'DC':dc_sorted_nodes,'Kshell':kshell_sorted_nodes,'GLSTM':glstm_sorted_nodes,
    #                         # 'rcnn':rcnn_sorted_nodes,'pre':pre_sorted_node}
    #         #Result[name] = k123
    #     # with open('graphsage_ken_.pkl', 'wb') as f:
    #     #     pickle.dump(Result, f)
    #
    #     ken_result[name] = k[0]
    #     Result[name] = pre_sorted_node
    # with open('Sorted_nodes_data/active_learning_sorted_nodes.pkl', 'wb') as f:
    #     pickle.dump(Result, f)
    # with open('Ken_data/active_learning_ken_b=1.pkl', 'wb') as f:
    #     pickle.dump(ken_result, f)
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

def GLSTM(path,nodes_num):
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

    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = 'MN_SIR_beitac/' + multiplex_network + '.txt'
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
    return k[0],pre_sorted_node

def RCNN(path,nodes_num):
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

    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = 'MN_SIR_beitac/' + multiplex_network + '.txt'
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
    return k[0],pre_sorted_node





def DC(path,nodes_num):
    Gs, total_layers = load_multilayer_graph(path)
    DegreeDict = {}
    for i in range(total_layers):
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)
        dc = nx.degree_centrality(Gs[i])
        DegreeDict[i+1] = dc

    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = 'MN_SIR_beitac/' + multiplex_network + '.txt'
    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))


    pre_avg_sir_dict = cal_average_sir(DegreeDict, total_layers, nodes_num)

    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)

    return k[0],pre_sorted_node

def k_shell(path,nodes_num):
    Gs, total_layers = load_multilayer_graph(path)
    KshellDict = {}
    for i in range(total_layers):
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)
        kshell = nx.core_number(Gs[i])
        KshellDict[i + 1] = kshell

    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = 'MN_SIR_beitac/' + multiplex_network + '.txt'
    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))
    print(KshellDict)
    pre_avg_sir_dict = cal_average_sir(KshellDict, total_layers, nodes_num)
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0],pre_sorted_node

def f_eigenvector_centrality(path,nodes_num):
    Gs, total_layers = load_multilayer_graph(path)

    multiplex_network = path.split('/')[1].split('.')[0]
    network_name = 'MN_SIR_1beitac/' + multiplex_network + '.txt'

    sir_dict,_ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))
    print('sir_list',sir_list)
    f_eigenvector_dict,layer_inf = MultiPR(Gs,nodes_num)

    f_eigenvector_sorted_node = [key for key in f_eigenvector_dict.keys()]
    node_rank_f_eigenvector = [f_eigenvector_sorted_node.index(x) if x in f_eigenvector_sorted_node else len(f_eigenvector_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_f_eigenvector)
    print('node_rank_f_eigenvector',node_rank_f_eigenvector)

    return k[0],f_eigenvector_sorted_node,f_eigenvector_dict,layer_inf


def PRGC(path,nodes_num):
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
    network_name = 'MN_SIR_1beitac/' + multiplex_network + '.txt'

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


    return k[0], prgc_sorted_node



def convert_sir(sir_dict, mapping):
    dic = {}
    for node, id in mapping.items():
        dic[id] = sir_dict[node]
    dic = dict(sorted(dic.items(),key=lambda x:x[1],reverse=True))
    data_memory = [[k,v] for k,v in dic.items()]
    return data_memory



if __name__ == '__main__':

    # with open('f_eigenvector_ken.pkl','rb') as f:
    #     data = pickle.load(f)
    # for k,v in data.items():
    #     print(k,v)
    # print('-'*50)
    # with open('f_eigenvector_ken.pkl','rb') as f:
    #     data = pickle.load(f)
    # for k,v in data.items():
    #     print(k,v)

    # nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980, 'celegans_connectome_multiplex': 279,
    #                                      'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
    #                                      'cS-Aarhus_multiplex': 61, 'drosophila_genetic_multiplex': 8215, 'hepatitusC_genetic_multiplex': 105,
    #                                      'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71, 'rattus_genetic_multiplex': 2640}
    #
    # network = [key+'.edges' for key in nodes_num_from_multiplex_networks.keys()]
    # Result_k= {}
    # Result_sorted_nodes={}
    # for name in network:
    #     #path = 'MNdata/drosophila_genetic_multiplex.edges'
    #     path = 'MNdata/' + name
    #     multiplex_network = path.split('/')[1].split('.')[0]
    #     nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
    #     #prgc,prgc_sorted_nodes= PRGC(path,nodes_num)
    #     f_k,f_eigenvector_sorted_node,f_eigenvector_dict,layer_inf= f_eigenvector_centrality(path,nodes_num)
    #     #print(prgc,f_k)
    #     Result_k[name] = f_k
    #     Result_sorted_nodes[name] = f_eigenvector_sorted_node
    #
    # with open('f_eigenvector_ken.pkl', 'wb') as f:
    #     pickle.dump(Result_k, f)
    #
    # with open('f_eigenvector_sorted_nodes.pkl', 'wb') as f:
    #     pickle.dump(Result_sorted_nodes, f)
    #     #model = torch.load('influence_evaluation/ALGE_B_11_20.pth')

    nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980, 'celegans_connectome_multiplex': 279,
                                         'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
                                         'cS-Aarhus_multiplex': 61, 'drosophila_genetic_multiplex': 8215, 'hepatitusC_genetic_multiplex': 105,
                                         'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71, 'rattus_genetic_multiplex': 2640}


    #path = 'MNdata/cS-Aarhus_multiplex.edges'
    #multiplex_network = path.split('/')[1].split('.')[0]
    #
    # network = [key+'.edges' for key in nodes_num_from_multiplex_networks.keys()]
    # Result_k = {}
    # Result_sorted_nodes = {}
    # Result_dict = {}
    # Result_layer = {}
    # for name in network:
    #     path = 'MNdata/' + name
    #     multiplex_network = path.split('/')[1].split('.')[0]
    #     nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
    #     prgc_k,prgc_eigenvector_sorted_node, = PRGC(path,nodes_num)
    #     #print(k_f_eigenvector,f_eigenvector_sorted_nodes)
    #     Result_k[name] = prgc_k
    #     Result_sorted_nodes[name] = prgc_eigenvector_sorted_node
    #
    #
    # with open('prgc_ken.pkl', 'wb') as f:
    #     pickle.dump(Result_k, f)
    #
    # with open('f_eigenvector_sorted_nodes.pkl', 'wb') as f:
    #     pickle.dump(Result_sorted_nodes, f)
    #
    # with open('f_eigenvector_nodes_dict.pkl', 'wb') as f:
    #     pickle.dump(Result_dict, f)
    #
    # with open('f_eigenvector_layer_inf.pkl', 'wb') as f:
    #     pickle.dump(Result_layer, f)



    network = [key+'.edges' for key in nodes_num_from_multiplex_networks.keys()]
    Result = {}
    for name in network:
    #for _ in range(1):
        path = 'MNdata/' + name
        #path = 'MNdata/hepatitusC_genetic_multiplex.edges'
        Gs, total_layers = load_multilayer_graph(path)
        multiplex_network = path.split('/')[1].split('.')[0]

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
        print(maps)
        ken = {}
        for time in [0.5, 0.75, 1.25, 1.5]:

           # network_name = 'MN_SIR_1beitac/' + multiplex_network + '.txt'
            network_name = 'MN_SIR_'+str(time)+'beitac/' + multiplex_network + '.txt'
            print(network_name)
            nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
            sir_dict,sir_dict_each_layer = load_multilayer_sir_labels(network_name,nodes_num ,total_layers)
            sir_list = [key for key in sir_dict.keys()]
            print('sir_dict',sir_dict)
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

                predictions = ALGE_C(model,Gs[i],data_memory)

                #print('1111111111111111',predictions)

                predictions = predictions.tolist()
                # 提取节点及影响力值
                nodes_list = list(Gs[i].nodes())
                node_influence = {nodes_list[j]: predictions[j][0] for j in range(len(predictions))}
                sorted_node_influence = dict(sorted(node_influence.items(), key=lambda x: x[1], reverse=True))
                pre_dict = sorted_node_influence
                pre_sir_dict[i + 1] = pre_dict
            print(pre_sir_dict)
            print(maps)
            result = {}
            for i in range(total_layers):
                dic = {}
                for node, id in maps[i].items():
                    if id in pre_sir_dict[i+1]:
                        dic[node] = pre_sir_dict[i+1][id]
                dic = dict(sorted(dic.items(),key=lambda x:x[1],reverse=True))
                result[i+1] = dic
            print(result)

            pre_avg_sir_dict = cal_average_sir(result,total_layers, nodes_num_from_multiplex_networks[multiplex_network])
            print('pre_avg_sir_dict:',pre_avg_sir_dict)
            pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
            node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
            k = kendalltau(node_rank_simu, node_rank_p)



            k_glstm,glstm_sorted_nodes = GLSTM(path,nodes_num)


            k_dc,dc_sorted_nodes = DC(path,nodes_num)


            k_kshell,kshell_sorted_nodes = k_shell(path, nodes_num)

            #k_rcnn ,rcnn_sorted_nodes= RCNN(path,nodes_num)
            k123 = {}
            print('pre',k[0])
            print('GLSTM', k_glstm)
            print('DC', k_dc)
            print('Kshell', k_kshell)
            #print('rcnn', k_rcnn)
            print(g_list)
            k123={'pre':k[0]}
            #Result[name] = {'DC':dc_sorted_nodes,'Kshell':kshell_sorted_nodes,'GLSTM':glstm_sorted_nodes,
                            # 'rcnn':rcnn_sorted_nodes,'pre':pre_sorted_node}
            #Result[name] = k123
        # with open('graphsage_ken_.pkl', 'wb') as f:
        #     pickle.dump(Result, f)

            ken[time] = k[0]
        Result[name] = ken
    with open('active_learning_ken.pkl', 'wb') as f:
        pickle.dump(Result, f)
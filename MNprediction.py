from Utils import load_multilayer_graph,get_dgl_g_input_test,load_multilayer_sir_labels,cal_average_sir,embedding_,get_dgl_g_input
import networkx as nx
import dgl
import torch
from scipy.stats import kendalltau

def GLSTM(path,nodes_num):
    model = torch.load('influence_evaluation/GLSTM.pth')
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

    sir_dict = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
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
    return k[0]

def DC(path,nodes_num):
    Gs, total_layers = load_multilayer_graph(path)
    DegreeDict = {}
    for i in range(total_layers):
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)
        dc = nx.degree_centrality(Gs[i])
        DegreeDict[i+1] = dc


    sir_dict = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))


    pre_avg_sir_dict = cal_average_sir(DegreeDict, total_layers, nodes_num)
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0]

def k_shell(path,nodes_num):
    Gs, total_layers = load_multilayer_graph(path)
    KshellDict = {}
    for i in range(total_layers):
        # 查找入度为0的节点
        zero_degree_nodes = [node for node, in_deg in Gs[i].degree() if in_deg == 0]
        Gs[i].remove_nodes_from(zero_degree_nodes)
        kshell = nx.core_number(Gs[i])
        KshellDict[i + 1] = kshell

    sir_dict = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))
    print(KshellDict)
    pre_avg_sir_dict = cal_average_sir(KshellDict, total_layers, nodes_num)
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)
    return k[0]



if __name__ == '__main__':
    #model = torch.load('influence_evaluation/ALGE_B_11_20.pth')
    model = torch.load('influence_evaluation/GraphSAGE.pth')

    nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex': 6980, 'celegans_connectome_multiplex': 279,
                                         'celegans_genetic_multiplex': 3879, 'cKM-Physicians-Innovation_multiplex': 246,
                                         'cS-Aarhus_multiplex': 61, 'drosophila_genetic_multiplex': 8215, 'hepatitusC_genetic_multiplex': 105,
                                         'humanHIV1_genetic_multiplex': 1005, 'lazega-Law-Firm_multiplex': 71, 'rattus_genetic_multiplex': 2640}
    path = 'MNdata/cKM-Physicians-Innovation_multiplex.edges'
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




    network_name = 'MN_SIR_beitac/' + multiplex_network + '.txt'
    print(network_name)
    nodes_num = nodes_num_from_multiplex_networks[multiplex_network]
    sir_dict = load_multilayer_sir_labels(network_name,nodes_num ,total_layers)
    sir_list = [key for key in sir_dict.keys()]
    print('sir_dict',sir_dict)
    node_rank_simu = list(range(0, len(sir_list)))

    g_list = [dgl.from_networkx(Gs[i]) for i in range(total_layers)]

    pre_sir_dict = {}
    for i in range(total_layers):
        model.eval()
        with torch.no_grad():
            predictions = model(g_list[i], node_feature_lsit[i])
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
    print(pre_avg_sir_dict)
    pre_sorted_node = [key for key in pre_avg_sir_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)



    k_glstm = GLSTM(path,nodes_num)


    k_dc = DC(path,nodes_num)


    k_kshell = k_shell(path, nodes_num)
    print('pre',k[0])
    print('GLSTM', k_glstm)
    print('DC', k_dc)
    print('Kshell', k_kshell)
    print(g_list)
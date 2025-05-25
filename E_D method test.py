import numpy as np
import networkx as nx

from Utils import load_multilayer_graph, load_multilayer_sir_labels

import numpy as np
import networkx as nx
from scipy.stats import kendalltau
class ED_method:
    def __init__(self, graphs):
        self.graphs = graphs
        self.L = len(graphs)
        self.node_list = [i for i in range(1,61+1)]
        #self.N = len(self.node_list)
        self.N = 61
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


if __name__ == '__main__':
    path ='./dataset/real_multiplex_networks/MNdata/cS-Aarhus_multiplex.edges'
    Gs, total_layers = load_multilayer_graph(path)

    ED_dict= ED_method(Gs).compute_metrics()
    ED_sorted_dict = dict(sorted(ED_dict.items(),key=lambda x:x[1],reverse=True))
    print(ED_sorted_dict)
    time =1
    network_name = './dataset/real-influence/MN_SIR_' + str(time) + 'beitac/' + 'cS-Aarhus_multiplex' + '.txt'
    nodes_num = 61
    sir_dict, _ = load_multilayer_sir_labels(network_name, nodes_num, total_layers)
    sir_list = [key for key in sir_dict.keys()]
    node_rank_simu = list(range(0, len(sir_list)))


    pre_sorted_node = [key for key in ED_sorted_dict.keys()]
    node_rank_p = [pre_sorted_node.index(x) if x in pre_sorted_node else len(pre_sorted_node) for x in sir_list]
    k = kendalltau(node_rank_simu, node_rank_p)

    print(k[0])
    print('ed_list',pre_sorted_node)
    print('sir_list',sir_list)
import copy
import networkx as nx
import random
import multiprocessing
import pickle

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

            while len(G) < layer:
                G.append(nx.Graph())

            G[layer - 1].add_edge(source, target, weight=weight)

    for graph in G:
        graph.remove_edges_from(nx.selfloop_edges(graph))

    return G, len(G)

def threshhold(G):
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    squared_degree_sum = sum(deg ** 2 for node, deg in G.degree())
    average_squared_degree = squared_degree_sum / len(G)
    beita = avg_degree / (average_squared_degree - avg_degree)
    return round(beita, 4)

def IC_simulation(p, g, set, num_trials=100):
    total_influence = 0
    for _ in range(num_trials):
        g_copy = copy.deepcopy(g)
        for node in g_copy.nodes():
            g_copy.nodes[node]['state'] = 0
        for i in list(g_copy.nodes()):
            if i in set:
                g_copy.nodes[i]['state'] = 1
        for j in list(g_copy.edges()):
            g_copy.edges[j]['p'] = p

        nextg = g_copy.copy()
        terminal = 0
        while terminal == 0:
            for i in list(g_copy.nodes()):
                if g_copy.nodes[i]['state'] == 1:
                    for j in g_copy.neighbors(i):
                        if g_copy.nodes[j]['state'] == 0:
                            nextg.nodes[j]['state'] = 1 if random.random() < nextg.edges[i, j]['p'] else 0
                    nextg.nodes[i]['state'] = 2
            g_copy = nextg
            nextg = g_copy.copy()
            terminal = 1
            for i in list(g_copy.nodes()):
                if g_copy.nodes[i]['state'] == 1:
                    terminal = 0
        count = 0
        for i in list(g_copy.nodes()):
            if g_copy.nodes[i]['state'] == 2:
                count += 1
        total_influence += count
    return total_influence / num_trials

def parallel_ic_simulation(p, g, set, num_trials=100, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    trials_per_process = num_trials // num_processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(IC_simulation, [(p, g, set, trials_per_process) for _ in range(num_processes)])
    total_influence = sum(results)
    average_influence = total_influence / num_processes
    return average_influence

class InfluenceCalculator:
    def __init__(self, graph):
        self.G = graph

    def sir_model(self, beta, source_nodes):
        nodes = self.G.nodes()
        state = {node: 'S' for node in nodes}
        if isinstance(source_nodes, int):
            source_nodes = [source_nodes]
        for node in source_nodes:
            if node in nodes:
                state[node] = 'I'
        while 'I' in state.values():
            susceptible_nodes = [node for node in nodes if state[node] == 'S']
            for susceptible_node in susceptible_nodes:
                neighbors = list(self.G.neighbors(susceptible_node))
                for neighbor in neighbors:
                    if state[neighbor] == 'I' and random.random() < beta:
                        state[susceptible_node] = 'I'
                        break
            for node in nodes:
                if state[node] == 'I':
                    state[node] = 'R'
        recovered_nodes = [node for node in nodes if state[node] == 'R']
        return len(recovered_nodes)

    def calculate_average_influence_single(self, beta, source_nodes):
        return self.sir_model(beta, source_nodes)

    def calculate_average_influence(self, beta, source_nodes, experiments=100):
        with multiprocessing.Pool(processes=32) as pool:
            results = pool.starmap(self.calculate_average_influence_single, [(beta, source_nodes)] * experiments)
        total_influence = sum(results)
        average_influence = total_influence / experiments
        return average_influence

if __name__ == '__main__':

    nodes_num_from_multiplex_networks = {'arabidopsis_genetic_multiplex.edges': 6980, 'celegans_connectome_multiplex.edges': 279,
                                         'celegans_genetic_multiplex.edges': 3879, 'cKM-Physicians-Innovation_multiplex.edges': 246,
                                         'cS-Aarhus_multiplex.edges': 61, 'drosophila_genetic_multiplex.edges': 8215, 'hepatitusC_genetic_multiplex.edges': 105,
                                         'humanHIV1_genetic_multiplex.edges': 1005, 'lazega-Law-Firm_multiplex.edges': 71, 'rattus_genetic_multiplex.edges': 2640}



    file_path = './Sorted_nodes_data/sorted_nodes.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(data)
    #Data = {}
    #Data['cKM-Physicians-Innovation_multiplex.edges'] = data['cKM-Physicians-Innovation_multiplex.edges']
    #Data['lazega-Law-Firm_multiplex.edges'] = data['lazega-Law-Firm_multiplex.edges']
    #Data['humanHIV1_genetic_multiplex.edges'] = data['humanHIV1_genetic_multiplex.edges']
    #Data['drosophila_genetic_multiplex.edges'] = data['drosophila_genetic_multiplex.edges']


    INF = []
    data = dict(sorted(data.items(), key=lambda x: (x[0][0].lower(), x[0][1].lower() if len(x[0]) > 1 else '')))
    for beita_times in [0.5,0.75,1,1.25,1.5]:
        for n in [i * 0.01 for i in range(1, 11)]:
            Result = {}
            #for network_name, method_dict in data.items():
            for network_name, nodes in data.items():

                Gs, total_layer = load_multilayer_graph('./dataset/real_multiplex_networks/MNdata/' + network_name)
                method_inf = {}
                #for method,nodes in method_dict.items():
                method = 'MGNN-AL'
                inf = 0
                network_nodes = 0
                for i in range(total_layer):
                    #b=0.2
                    b= 5*threshhold(Gs[i])*beita_times
                    scale = int(len(Gs[i].nodes()) * n)
                    if scale < 1:
                        scale = 1
                    inf += InfluenceCalculator(Gs[i]).calculate_average_influence(b, nodes[:scale])
                    network_nodes += len(Gs[i].nodes())
                    print(f'已计算{network_name} 第{i}/{total_layer}层 {method} {n} {scale}个节点 {network_nodes}, b={b},b_times={beita_times}')
                method_inf[method] = inf /   network_nodes            #network_nodes * total_layer
                Result[network_name] = method_inf
        #     INF.append(Result)
        # print(INF)
        # print('平均值:',sum(INF)/len(INF))
            with open('./temp_data/MGNN-AL-(total_nodes)INF_top-' +  str(n) + '_b='+str(beita_times)+'b(0.01-0.1)_500num.pkl', 'wb') as f:
                pickle.dump(Result, f)


    # inf =[{'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.1799406528189911}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.5205044510385757}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.5272848664688428}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.5676557863501484}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.6390059347181009}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.6827893175074183}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.6996290801186944}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.7958011869436202}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.8050296735905045}}, {'cKM-Physicians-Innovation_multiplex.edges': {'MGNN-AL': 0.8237982195845698}}]
    #
    #
    #
    # sum=0
    # for V in INF:
    #     for network,method in V.items():
    #         for m,v in method.items():
    #             sum+=v
    # print(sum/10)
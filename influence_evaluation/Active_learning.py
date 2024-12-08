import networkx as nx
from math import log
import numpy as np
from scipy.optimize import leastsq
import copy
def box_covering(G):
    print("diameter ",nx.diameter(G))
    lbmax = min(5,nx.diameter(G)+1)
    lb_list = [x for x in range(1,lbmax+1)]
    c = []
    for n in range(len(G)): c.append([-1]*lbmax)
    for lb in lb_list:
        used_c = set()
        c[0][lb - 1] = 0
        used_c.add(c[0][lb - 1])
        for i in range(1,len(G)):
            ban_c = set()
            for j in range(0,i):
                lij = nx.shortest_path_length(G,i,j)
                if lij>=lb:
                    ban_c.add(c[j][lb-1])
            avai_c = used_c-ban_c
            if avai_c==set():
                c[i][lb-1] = max(ban_c)+1
            else:
                c[i][lb - 1] = min(avai_c)
            used_c.add(c[i][lb - 1])
    return [x+1 for x in c[-1]]
def y_pre(p,x):
    f=np.poly1d(p)
    return f(x)
def error(p,x,y):
    return y-y_pre(p,x)
def least_square(x,y):
    p=np.random.rand(2)
    res = leastsq(error,p,args=(x,y))
    w1,w2 = res[0]
    # x_ = np.linspace(1, 10, 100)
    # y_p = w1 * x_ + w2
    # plt.scatter(x, y)
    # plt.plot(x_, y_p)
    # plt.show()
    # plt.close()
    return w1
def find_threshold_2_div(S,len_G):
    S_ = np.array(S)
    for i in range(len_G): S_[i][i] = 0
    S_f = S_.flatten()
    S_f =sorted(S_f,reverse=True)
    S_f = [x for x in S_f if x>0]
    m = len(S_f)
    while(True):
        InforG = nx.Graph()
        k = max(int(0.5*(m))-1,0)
        for i in range(len_G):
            for j in range(i+1,len_G):
                if S[i][j]>=S_f[k]:
                    InforG.add_edge(i,j,weight = S[i][j])
        if len(InforG)<len_G:
            S_f = S_f[k+1:m]
            m=len(S_f)
        if len(InforG)==len_G:
            S_f = S_f[0:k+1]
            m=len(S_f)
            if len(S_f)==1:
                break
    return InforG

def al_model(G):
    node_list = list(G.nodes())
    len_G = len(node_list)
    Ns = box_covering(G)
    s = list(range(1,len(Ns)+1))
    # 最小二乘法求斜率即分形维度
    y = [log(yi) for yi in Ns]
    x = [log(xi) for xi in s]
    df = -least_square(x,y)
    # 2 计算本地维度dli和概率集pi
    dl = []
    for node in node_list:
        dij = list(nx.shortest_path_length(G,node).values())
        r_list = list(range(1,max(dij)+1))
        Nir=[]
        for r in r_list:
            Nr = len([x for x in dij if x<=r])
            Nir.append(Nr)
        y = [log(yi) for yi in Nir]
        x = [log(xi) for xi in r_list]
        dli = -least_square(x, y)
        dl.append(dli)
    P=[]
    for i in range(len(G)):
        node = node_list[i]
        pi=[]
        neighbor = list(nx.neighbors(G,node))
        Gi_node = neighbor+[node]
        dlGi = [dl[x] for x in Gi_node]
        degree_list = [G.degree(x) for x in Gi_node]
        m = max(degree_list)+1
        for k in range(m):
            di = G.degree(node)
            if k+1<=di+1:
                pik = dlGi[k]/sum(dlGi)
            else:
                pik = 0
            pi.append(pik)
        P.append(pi)

    PRE=[]
    for i in range(len(G)):PRE.append([-1]*len(G))
    for i in range(len(G)):
        for j in range(len(G)):
            m_ = min(G.degree(i),G.degree(j))+1
            P_i = sorted(P[i],reverse=True)
            P_j = sorted(P[j],reverse=True)
            PRE[i][j]=0
            for k in range(m_):
                PRE[i][j] += ( (P_i[k]/P_j[k])**df-(P_i[k]/P_j[k])   )/(1-df)
    R = []
    for i in range(len(G)): R.append([-1] * len(G))
    for i in range(len(G)):
        for j in range(len(G)):
            R[i][j] = PRE[i][j]+PRE[j][i]
    S = []
    for i in range(len(G)):
        S.append([-1] * len(G))
    Rmin = np.min(R)
    for i in range(len(G)):
        for j in range(len(G)):
            S[i][j] =1-R[i][j]/ Rmin
    InforG = find_threshold_2_div(S,len_G)
    Representative=[]
    InforG_=copy.deepcopy(InforG)
    for i in range(10):
        degree_dict = dict(InforG_.degree(weight="weight"))
        node = max(degree_dict,key=degree_dict.get) #当前度最大的节点
        Representative.append(node)
        neighbor = list(nx.neighbors(InforG_,node))
        InforG_.remove_nodes_from([node]+neighbor)
        if len(InforG_)==0:
            break
    print(Representative)
    return Representative


class CycleRatioCalculator:
    def __init__(self, graph):
        """
        初始化周期比计算器
        :param graph: networkx.Graph 对象
        """
        self.Mygraph = graph.copy()
        self.DEF_IMPOSSLEN = self.Mygraph.number_of_nodes() + 1  # Impossible simple cycle length
        self.SmallestCycles = set()
        self.NodeGirth = {}
        self.SmallestCyclesOfNodes = {}
        self.Coreness = nx.core_number(self.Mygraph)
        self.CycleRatio = {}
        self.CycLenDict = {}
        self._preprocess_graph()

    def _preprocess_graph(self):
        """初始化图的节点属性并移除低度数节点。"""
        removeNodes = set()
        for node in self.Mygraph.nodes():
            self.SmallestCyclesOfNodes[node] = set()
            self.CycleRatio[node] = 0
            if self.Mygraph.degree(node) <= 1 or self.Coreness[node] <= 1:
                self.NodeGirth[node] = 0
                removeNodes.add(node)
            else:
                self.NodeGirth[node] = self.DEF_IMPOSSLEN

        self.Mygraph.remove_nodes_from(removeNodes)
        for i in range(3, self.Mygraph.number_of_nodes() + 2):
            self.CycLenDict[i] = 0

    def _my_all_shortest_paths(self, G, source, target):
        """找到图中两个节点之间的所有最短路径。"""
        pred = nx.predecessor(G, source)
        if target not in pred:
            raise nx.NetworkXNoPath(
                f"Target {target} cannot be reached from given sources"
            )
        sources = {source}
        seen = {target}
        stack = [[target, 0]]
        top = 0
        while top >= 0:
            node, i = stack[top]
            if node in sources:
                yield [p for p, n in reversed(stack[: top + 1])]
            if len(pred[node]) > i:
                stack[top][1] = i + 1
                next_node = pred[node][i]
                if next_node in seen:
                    continue
                else:
                    seen.add(next_node)
                top += 1
                if top == len(stack):
                    stack.append([next_node, 0])
                else:
                    stack[top][:] = [next_node, 0]
            else:
                seen.discard(node)
                top -= 1

    def get_smallest_cycles(self):
        """找到图中的所有最小环。"""
        NodeList = list(self.Mygraph.nodes())
        NodeList.sort()

        # Step 1: 找到三角形环
        for ix in NodeList[:-2]:
            if self.NodeGirth[ix] == 0:
                continue
            for jx in NodeList[NodeList.index(ix) + 1: -1]:
                if self.NodeGirth[jx] == 0:
                    continue
                if self.Mygraph.has_edge(ix, jx):
                    for kx in NodeList[NodeList.index(jx) + 1:]:
                        if self.NodeGirth[kx] == 0:
                            continue
                        if self.Mygraph.has_edge(kx, ix) and self.Mygraph.has_edge(kx, jx):
                            self.SmallestCycles.add(tuple(sorted([ix, jx, kx])))
                            for node in [ix, jx, kx]:
                                self.NodeGirth[node] = 3

        # Step 2: 找到其他环
        ResiNodeList = [
            node for node in NodeList if self.NodeGirth[node] == self.DEF_IMPOSSLEN
        ]
        visitedNodes = {node: set() for node in ResiNodeList}

        for nod in ResiNodeList:
            for nei in list(self.Mygraph.neighbors(nod)):
                if not nei in visitedNodes:
                    visitedNodes[nei] = set()
                if nod not in visitedNodes[nei]:
                    visitedNodes[nod].add(nei)
                    visitedNodes[nei].add(nod)
                    self.Mygraph.remove_edge(nod, nei)
                    if nx.has_path(self.Mygraph, nod, nei):
                        for path in self._my_all_shortest_paths(self.Mygraph, nod, nei):
                            path_len = len(path)
                            self.SmallestCycles.add(tuple(sorted(path)))
                            for node in path:
                                if self.NodeGirth[node] > path_len:
                                    self.NodeGirth[node] = path_len
                    self.Mygraph.add_edge(nod, nei)

    def calculate_cycle_ratios(self):
        """计算每个节点的周期比。"""
        for cyc in self.SmallestCycles:
            len_cyc = len(cyc)
            self.CycLenDict[len_cyc] += 1
            for node in cyc:
                self.SmallestCyclesOfNodes[node].add(cyc)

        for node, small_cycles in self.SmallestCyclesOfNodes.items():
            if not small_cycles:
                continue
            cycle_neighbors = set()
            NeiOccurTimes = {}
            for cyc in small_cycles:
                for n in cyc:
                    NeiOccurTimes[n] = NeiOccurTimes.get(n, 0) + 1
                cycle_neighbors.update(cyc)
            cycle_neighbors.discard(node)

            sum_ratio = sum(
                float(NeiOccurTimes[nei]) / len(self.SmallestCyclesOfNodes[nei])
                for nei in cycle_neighbors
            )
            self.CycleRatio[node] = sum_ratio + 1

    def get_cycle_ratios(self):
        """主函数：计算周期比并排序输出。"""
        self.get_smallest_cycles()
        self.calculate_cycle_ratios()
        return dict(sorted(self.CycleRatio.items(), key=lambda x: x[1], reverse=True))

def sample_nodes(G):
    cr_dict = CycleRatioCalculator(G).get_cycle_ratios()
    cr_list = [key for key in cr_dict.keys()]
    nodes_num = nx.number_of_nodes(G)
    n = int(nodes_num * 0.01)
    if n < 1: n=1
    cr_list_n = cr_list[:n]
    return cr_list_n

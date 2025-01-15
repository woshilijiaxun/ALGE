import networkx as nx
import numpy as np
def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[-1])
            G.add_edge(source, target)
    return G


def power_t2(A, x0, tol=1e-6):
    """
    Power method for 3rd order tensors.

    Parameters:
        A (np.ndarray): 3rd order tensor describing the multilayer network, shape (n, n, m).
        x0 (np.ndarray): Starting vector, shape (n,).
        alpha (float): Parameter alpha.
        beta (float): Parameter beta.
        tol (float): Convergence tolerance. Default is 1e-5.

    Returns:
        x (np.ndarray): Node centralities vector, shape (n,).
        y (np.ndarray): Layer centralities vector, shape (m,).
        it (int): Number of iterations.
    """
    # Normalize the starting vector
    x = x0 / np.linalg.norm(x0, 1)

    # Compute initial y
    y = np.einsum('ijt,i->jt', A, x)
    y = np.einsum('jt,j->t', y, x)
    y /= np.linalg.norm(y, 1)

    rex, rey = 1, 1
    it = 0
    converged = False

    while rex > tol or rey > tol:
        x_old = x
        y_old = y

        # Update x
        xx = np.einsum('ijt,j->it', A, x)
        xx = np.einsum('it,t->i', xx, y)
        xx = np.abs(xx)
        x = xx / np.linalg.norm(xx, 1)

        # Update y
        yy = np.einsum('ijt,i->jt', A, x)
        yy = np.einsum('jt,j->t', yy, x)
        yy = np.abs(yy)
        y = yy / np.linalg.norm(yy, 1)

        # Compute relative differences
        rex = np.linalg.norm(x_old - x) / np.linalg.norm(x)
        rey = np.linalg.norm(y_old - y) / np.linalg.norm(y)

        # Print convergence information
        if not converged and rex <= tol:
            print(f"\n * Node centrality vector converges first ({it + 1} iterations)")
            converged = True
        if not converged and rey <= tol:
            print(f"\n * Layer centrality vector converges first ({it + 1} iterations)")
            converged = True

        it += 1

    return x, y, it


def graphs_to_tensor(Gs, nodes_num):
    """
    将图列表的邻接矩阵填充到指定的节点数量大小。

    参数:
        Gs: list of networkx.Graph
            图列表，其中每个图可能具有不同的节点数量。
        nodes_num: int
            填充后的目标节点数量。

    返回:
        padded_adj_matrices: numpy.ndarray
            填充后的三维邻接矩阵张量，形状为 (len(Gs), nodes_num, nodes_num)。
        sorted_node_orders: list of list
            每个图的节点排序（保持原始图的节点顺序，填充的节点为 None）。
    """
    padded_adj_matrices = []
    sorted_node_orders = []

    for G in Gs:
        # 获取当前图的节点列表并排序
        nodes = sorted(G.nodes())
        num_nodes = len(nodes)

        # 记录节点的排序
        sorted_node_orders.append(nodes + [None] * (nodes_num - num_nodes))

        # 初始化零矩阵
        adj_matrix = np.zeros((nodes_num, nodes_num))

        # 填充原始图的邻接矩阵
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if G.has_edge(u, v):
                    adj_matrix[i, j] = G[u][v].get("weight", 1)  # 默认权重为1

        # 将填充的邻接矩阵添加到结果列表
        padded_adj_matrices.append(adj_matrix)

    # 转换为三维 numpy 张量
    padded_adj_matrices = np.stack(padded_adj_matrices)

    return padded_adj_matrices

def MultiPR(Gs,nodes_num):

    A = graphs_to_tensor(Gs,nodes_num)

    x0 = np.random.rand(nodes_num)  # 初始向量
    # 调用函数
    x, y, it = power_t2(A, x0)
    # print("Node centralities (x):", x)
    # print("Layer centralities (y):", y)
    # print("Iterations:", it)


    node_influence = {id: j for id,j in enumerate(x,start=1)}
    sorted_node_influence = dict(sorted(node_influence.items(), key=lambda x: x[1], reverse=True))
    return sorted_node_influence, y

if __name__ == '__main__':
    print(1)


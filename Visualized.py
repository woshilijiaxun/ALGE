import networkx as nx
import matplotlib.pyplot as plt
from Utils import load_multilayer_graph
# 创建一个随机图
path = 'dataset/real_multiplex_networks/MNdata/lazega-Law-Firm_multiplex.edges'
Gs, total_layers = load_multilayer_graph(path)
G = Gs[1]

pos = nx.spring_layout(G, seed=42)

# 绘制图形（不显示编号）
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=False, node_size=150, node_color='skyblue', edgecolors='black', edge_color='gray')



# 保存为高分辨率图片
plt.savefig("network_high_res3.png", dpi=300, bbox_inches='tight')  # dpi=300 适合论文/海报
plt.show()

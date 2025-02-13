import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv, GATv2Conv

device = torch.device("cpu")



class GATv3(nn.Module):
    def __init__(self, GATv3_P):
        super(GATv3, self).__init__()
        self.K = GATv3_P["num_layer"]
        if GATv3_P["activation"] == "relu":  self.AF = nn.ReLU()
        if GATv3_P["activation"] == "tanh":   self.AF = nn.Tanh()
        if GATv3_P["activation"] == "sigmod":  self.AF = nn.Sigmoid()
        if GATv3_P["activation"] == "elu":  self.AF = nn.ELU()

        self.in_dim,self.out_dim ,self.embed_dim, self.bias,self.dropout,self.heads = GATv3_P["in_dim"], GATv3_P["out_dim"], GATv3_P["embed_dim"], GATv3_P["bias"],GATv3_P["dropout"], GATv3_P['heads']
        self.layer = nn.ModuleList()
        self.layer.append(GATv2Conv(self.in_dim,32,num_heads = self.heads[0],bias = self.bias,activation = None,negative_slope=0.2,attn_drop=0.0))
        for i in range(1,self.K-1):
            self.layer.append(GATv2Conv(32*self.heads[i-1],32,num_heads = self.heads[i],bias = self.bias,activation = None,attn_drop= 0.0))
        self.layer.append(GATv2Conv(32*self.heads[-2],32, num_heads = self.heads[-1],bias = self.bias,activation = None,attn_drop= 0.0))
        self.dropout = nn.Dropout(self.dropout)

        self.linear_layer = nn.ModuleList()
        for i in range(0,self.K):
            self.linear_layer.append(nn.Linear(self.layer[i].fc_src.in_features,self.layer[i].fc_src.out_features))

    def forward(self, g, input_features):
        gs=[g.to(device)]*self.K
        h = input_features.to(device)
        for i,(layer,g) in enumerate(zip(self.layer,gs)):
            if i != self.K - 1:
                h = F.elu(layer(g,h).flatten(1)+self.linear_layer[i](h))
            else:
                h = layer(g,h).mean(1)+self.linear_layer[i](h)

        return h


class GATv3Net(nn.Module):
    def __init__(self,Net_P,GATv3_P):
        super(GATv3Net, self).__init__()
        self.fc1 = GATv3(GATv3_P).to(device)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.fc3.weight  = nn.init.normal_(self.fc3.weight,0.1,0.01)
        self.activation = Net_P["activation"]
    def forward(self, g,input_features):
        value = self.fc1(g,input_features)
        value = self.activation(value)
        value = self.fc2(value)
        value = self.activation(value)
        value = self.fc3(value)
        return value
    def reset_parameters(self):
        self.fc1.layer[0].reset_parameters()
        self.fc1.layer[1].reset_parameters()
        self.fc1.layer[2].reset_parameters()
        self.fc1.linear_layer[0].reset_parameters()
        self.fc1.linear_layer[1].reset_parameters()
        self.fc1.linear_layer[2].reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.weight  = nn.init.normal_(self.fc3.weight,0.1,0.01)




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

        return x

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)


class CombinedModel(nn.Module):
    def __init__(self, Net_P,GATv3_P):
        super(CombinedModel, self).__init__()

        # 初始化 GATv3 和 GraphSAGE
        self.gat_model = GATv3(GATv3_P).to(device)
        self.sage_model = GraphSAGE().to(device)

        # 定义拼接后输入的线性层
        self.fc1 = nn.Linear(32 + 32, 16)  # 32 是 GATv3 和 GraphSAGE 输出的维度
        self.fc2 = nn.Linear(16, 1)
        self.activation = Net_P["activation"]

        # 初始化权重
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)

    def forward(self, g, input_features):
        # 通过 GATv3 提取特征
        gat_features = self.gat_model(g, input_features)

        # 通过 GraphSAGE 提取特征
        sage_features = self.sage_model(g, input_features)

        # 拼接 GAT 和 SAGE 的特征
        combined_features = torch.cat([gat_features, sage_features], dim=1)

        # 通过全连接层预测
        x = self.fc1(combined_features)
        x = self.activation(x)
        x = self.fc2(x)

        return x

    def reset_parameters(self):
        self.gat_model.reset_parameters()
        self.sage_model.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn.pytorch import SAGEConv, GATv2Conv
#
# device = torch.device("cpu")
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn.pytorch import SAGEConv, GATv2Conv
#
#
# class SageGATModel(nn.Module):
#     def __init__(self, GATv3_P, SAGE_P):
#         super(SageGATModel, self).__init__()
#         # 定义 SAGEConv 层参数
#         self.sage1 = SAGEConv(SAGE_P["in_dim"], SAGE_P["hidden_dim"], aggregator_type="lstm")
#         self.sage2 = SAGEConv(SAGE_P["hidden_dim"], SAGE_P["out_dim"], aggregator_type="lstm")
#
#         # 定义 GATv2Conv 层参数
#         self.K = 2  # 两层GAT
#         if GATv3_P["activation"] == "relu":
#             self.AF = nn.ReLU()
#         if GATv3_P["activation"] == "tanh":
#             self.AF = nn.Tanh()
#         if GATv3_P["activation"] == "sigmod":
#             self.AF = nn.Sigmoid()
#         if GATv3_P["activation"] == "elu":
#             self.AF = nn.ELU()
#
#         self.heads = GATv3_P['heads']
#         self.bias = GATv3_P["bias"]
#         self.layer = nn.ModuleList()
#         self.layer.append(GATv2Conv(SAGE_P["out_dim"], 32, num_heads=self.heads[0], bias=self.bias, activation=None, negative_slope=0.2, attn_drop=0.0))
#         self.layer.append(GATv2Conv(32 * self.heads[0], 32, num_heads=self.heads[1], bias=self.bias, activation=None, attn_drop=0.0))
#
#         self.linear_layer = nn.ModuleList()
#         for i in range(self.K):
#             self.linear_layer.append(nn.Linear(self.layer[i].fc_src.in_features, self.layer[i].fc_src.out_features))
#
#         # 最终预测的线性层
#         self.fc1 = nn.Linear(32, 16)
#         self.fc2 = nn.Linear(16, 1)
#         self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)
#         self.activation = nn.LeakyReLU()
#
#     def forward(self, g, input_features):
#         # 通过 SAGEConv 处理
#         h = self.sage1(g, input_features)
#         h = F.relu(h)
#         h = self.sage2(g, h)
#         h = F.relu(h)
#
#         # 通过 GATv2Conv 处理
#         for i, layer in enumerate(self.layer):
#             if i != self.K - 1:
#                 h = F.elu(layer(g, h).flatten(1) + self.linear_layer[i](h))
#             else:
#                 h = layer(g, h).mean(1) + self.linear_layer[i](h)
#
#         # 通过线性层预测
#         h = self.fc1(h)
#         h = self.activation(h)
#         h = self.fc2(h)
#
#         return h
#
#     def reset_parameters(self):
#         # 重置 SAGEConv 层
#         self.sage1.reset_parameters()
#         self.sage2.reset_parameters()
#
#         # 重置 GATv2Conv 层
#         for layer in self.layer:
#             layer.reset_parameters()
#         for linear in self.linear_layer:
#             linear.reset_parameters()
#
#         # 重置最终线性层
#         self.fc1.reset_parameters()
#         self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)


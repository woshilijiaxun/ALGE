import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, SAGEConv

device = torch.device("cpu")


class CombinedSAGEGAT(nn.Module):
    def __init__(self, GATv3_P):
        super(CombinedSAGEGAT, self).__init__()

        # GraphSAGE部分
        self.sage_conv1 = SAGEConv(GATv3_P["sage_in_dim"], 32, aggregator_type="lstm")
        self.sage_conv2 = SAGEConv(32, 32, aggregator_type="lstm")

        # GATv3部分
        self.K = GATv3_P["num_layer"]
        self.heads = GATv3_P["heads"]
        self.bias = GATv3_P["bias"]
        self.dropout = GATv3_P["dropout"]

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATv2Conv(32, 32, num_heads=self.heads[0], bias=self.bias, attn_drop=self.dropout))
        for i in range(1, self.K - 1):
            self.gat_layers.append(
                GATv2Conv(32 * self.heads[i - 1], 32, num_heads=self.heads[i], bias=self.bias, attn_drop=self.dropout))
        self.gat_layers.append(
            GATv2Conv(32 * self.heads[-2], 32, num_heads=self.heads[-1], bias=self.bias, attn_drop=self.dropout))

        # 最后两层全连接网络
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)

        # 激活函数
        self.activation = nn.LeakyReLU()

    def forward(self, g, node_features):
        # GraphSAGE部分
        x = self.sage_conv1(g, node_features)
        x = F.relu(x)
        x = self.sage_conv2(g, x)
        x = F.relu(x)

        # GAT部分
        for i, layer in enumerate(self.gat_layers):
            if i != len(self.gat_layers) - 1:
                x = F.elu(layer(g, x).flatten(1))
            else:
                x = layer(g, x).mean(1)

        # 全连接部分
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

    def reset_parameters(self):
        self.sage_conv1.reset_parameters()
        self.sage_conv2.reset_parameters()

        for layer in self.gat_layers:
            layer.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)

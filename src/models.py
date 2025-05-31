import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv as BaseGINConv,
    GCNConv as BaseGCNConv,
    GATConv,
    global_add_pool, global_mean_pool, global_max_pool,
    GlobalAttention, Set2Set
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_edge, degree

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, VGAE

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGAEWithClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.vgae = VGAE(Encoder(in_channels, hidden_channels))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        z = self.vgae.encode(data.x, data.edge_index)
        if hasattr(data, "y") and data.y is not None:
            # during training
            loss = self.vgae.recon_loss(data.x, data.edge_index) + (1 / data.num_nodes) * self.vgae.kl_loss()
            class_out = self.classifier(z)
            return class_out, loss
        else:
            # during inference
            class_out = self.classifier(z)
            return class_out


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_encoder = torch.nn.Linear(7, emb_dim)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_emb))

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr="add")
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

class GNN_node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin'):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.node_encoder = torch.nn.Embedding(300, emb_dim)  # <-- modificato
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(f"GNN type {gnn_type} not supported")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_encoder(x.squeeze(-1).long())  # <-- modificato
        h_list = [h]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, self.training) if layer != self.num_layer - 1 else F.dropout(h, self.drop_ratio, self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        elif self.JK == "sum":
            return sum(h_list)
        else:
            raise ValueError("Invalid Jumping Knowledge strategy")

class GNN_node_Virtualnode(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin'):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.node_encoder = torch.nn.Embedding(300, emb_dim)  # <-- modificato
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(f"GNN type {gnn_type} not supported")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for _ in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU()
                )
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        virtualnode_emb = self.virtualnode_embedding(torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device))
        h = self.node_encoder(x.squeeze(-1).long())  # <-- modificato
        h_list = [h]

        for layer in range(self.num_layer):
            #print("virtualnode_emb.shape:", virtualnode_emb.shape)
            #print("batch.max():", batch.max().item())

            h_list[layer] = h_list[layer] + virtualnode_emb[batch]
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, self.training) if layer != self.num_layer - 1 else F.dropout(h, self.drop_ratio, self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            if layer < self.num_layer - 1:
                pooled = global_add_pool(h_list[layer], batch)
                virtualnode_tmp = virtualnode_emb + pooled
                if self.residual:
                    virtualnode_emb = virtualnode_emb + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_tmp), self.drop_ratio, self.training)
                else:
                    virtualnode_emb = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_tmp), self.drop_ratio, self.training)

        if self.JK == "last":
            return h_list[-1]
        elif self.JK == "sum":
            return sum(h_list)
        else:
            raise ValueError("Invalid Jumping Knowledge strategy")

class GNN(torch.nn.Module):
    def __init__(self, num_class, num_layer=5, emb_dim=300, gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
        super().__init__()
        self.graph_pooling = graph_pooling

        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, drop_ratio, JK, residual, gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, drop_ratio, JK, residual, gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1)
                )
            )
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type")

        out_dim = 2 * emb_dim if graph_pooling == "set2set" else emb_dim
        self.graph_pred_linear = torch.nn.Linear(out_dim, num_class)

    def forward(self, data):
        h_node = self.gnn_node(data)
        h_graph = self.pool(h_node, data.batch)
        return self.graph_pred_linear(h_graph)

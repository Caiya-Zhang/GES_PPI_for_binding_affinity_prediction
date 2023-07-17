import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.inits import uniform
from modules.conv import GCNConv, GINConv
from modules.utils import pad_batch

from utils import *
from multiprocessing import Pool
from layers import Gate


### GNN to generate nodse embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    @staticmethod
    def need_deg():
        return False

    N_atom_features=30
    def __init__(self, num_layer, emb_dim, node_encoder, edge_encoder_cls, drop_ratio=0.5, JK="last", residual=False, gnn_type="gin"):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers
        """

        super(GNN_node, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder
        self.dropout_rate = args.dropout_rate
        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([Gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)])        
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1]+3, d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.embede = nn.Linear(N_atom_features, d_graph_layer, bias = False)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim, edge_encoder_cls))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None

        ### computing input node embedding
        if self.node_encoder is not None:
            encoded_node = (
                self.node_encoder(x)
                if node_depth is None
                else self.node_encoder(
                    x,
                    node_depth.view(
                        -1,
                    ),
                )
            )
        else:
            encoded_node = x
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "cat":
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=-1)

        return node_representation
    def embede_graph(self, data):
        h_m,h_w, adj_m,adj_w,p_m,p_w = data
        h_m = self.embede(h_m) 
        h_w = self.embede(h_w) 
        n1 = h_m.shape[1]
        n2 = h_w.shape[1]
        bn1 = nn.BatchNorm1d(n1)
        bn2 = nn.BatchNorm1d(n2)
        bn1.cuda()
        bn2.cuda()           
        h_m_sc = h_m
        h_w_sc = h_w
        for k in range(len(self.gconv1)):
            h_m = self.gconv1[k](h_m, adj_m)
            h_m = bn1(h_m)
            h_w = self.gconv1[k](h_w, adj_w)
            h_w = bn2(h_w) 
        h_m = h_m+h_m_sc
        h_w = h_w+h_w_sc
        h_m = torch.cat((h_m,p_m),2)
        h_w = torch.cat((h_w,p_w),2)
        h_m = h_m.sum(1)
        h_w = h_w.sum(1)
        h = h_m-h_w
        return h

    def fully_connected(self, h):
        n=h.shape[0]
        fc_bn = nn.BatchNorm1d(n)
        fc_bn.cuda()
        for k in range(len(self.FC)):
            if k<len(self.FC)-1:
                h = self.FC[k](h)
                h = h.unsqueeze(0)
                h = fc_bn(h)
                h = h.squeeze(0)
                h = F.dropout(h, p=self.dropout_rate, training=self.training)
                h = F.leaky_relu(h)
            else:
                h = self.FC[k](h)
        return h

    def train_model(self, data):
        h = self.embede_graph(data)
        h = self.fully_connected(h)
        h = h.view(-1)
        return h
    
    def test_model(self,data):
        h = self.embede_graph(data)
        h = self.fully_connected(h)
        h = h.view(-1)
        return h

### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    @staticmethod
    def need_deg():
        return False

    def __init__(self, num_layer, emb_dim, node_encoder, edge_encoder_cls, drop_ratio=0.5, JK="last", residual=False, gnn_type="gin"):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim, edge_encoder_cls))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                )
            )

    def forward(self, batched_data, perturb=None):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None

        ### computing input node embedding
        if self.node_encoder is not None:
            encoded_node = (
                self.node_encoder(x)
                if node_depth is None
                else self.node_encoder(
                    x,
                    node_depth.view(
                        -1,
                    ),
                )
            )
        else:
            encoded_node = x
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training=self.training
                    )

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "cat":
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=-1)

        return node_representation


def GNNNodeEmbedding(virtual_node, *args, **kwargs):
    if virtual_node:
        return GNN_node_Virtualnode(*args, **kwargs)
    else:
        return GNN_node(*args, **kwargs)

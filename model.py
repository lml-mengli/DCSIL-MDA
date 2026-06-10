# -*- coding: utf-8 -*-
import torch.nn.functional as F
# from dgl.ops import edge_softmax
from utils_ import *
from torch_geometric.nn import GATConv, SAGEConv, to_hetero, HANConv, HeteroConv, GCNConv, GATv2Conv, GraphConv
from torch import Tensor, nn
from torch.nn import Linear
from torch_geometric.utils import negative_sampling


class Subgraph_Fusion(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Subgraph_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, z):
        w = self.project(z).mean(0)
        beta_ = torch.softmax(w, dim=0)
        beta = beta_.expand((z.shape[0],) + beta_.shape)
        return (beta * z).sum(1), beta_
class TransformerFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=2, num_layers=1
                 ,dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_list):
        # z_list: list of (N, D), e.g. [z_cause, z_rev]
        z = torch.stack(z_list, dim=1)   # (N, 2, D)
        h = self.encoder(z)              # (N, 2, D)

        # attention pooling
        α = torch.softmax(self.pool(h), dim=1)  # (N, 2, 1)
        z_fused = (α * h).sum(dim=1)             # (N, D)
        return z_fused, α

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv2 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv3 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv4 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv5 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x_src, x_dst = x
        x_dst = F.relu(self.conv1((x_src, x_dst),edge_index))
        x_dst = F.relu(self.conv2((x_src, x_dst), edge_index))
        x_dst = F.relu(self.conv3((x_src, x_dst), edge_index))
        x_dst = F.relu(self.conv4((x_src, x_dst), edge_index))
        x_dst = self.conv5((x_src, x_dst),edge_index)
        return x_dst
class GNN1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.conv5 = SAGEConv(hidden_channels, hidden_channels)


    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))

        x = self.conv5(x,edge_index)
        return x
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = GraphConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv2 = GraphConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv3 = GraphConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv4 = GraphConv((hidden_channels, hidden_channels), hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x_src, x_dst = x

        x_dst = F.relu(self.conv1((x_src, x_dst), edge_index))
        x_dst = F.relu(self.conv2((x_src, x_dst), edge_index))
        x_dst = F.relu(self.conv3((x_src, x_dst), edge_index))
        x_dst = self.conv4((x_src, x_dst), edge_index)

        return x_dst

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False,heads=1)
        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.conv2 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False,heads=1)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.conv3 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=1)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.conv4 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=1)
        self.lin4 = Linear(hidden_channels, hidden_channels)

    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()


        return x


class Causal_Layer(nn.Module):
    def __init__(self, meta_paths,  hidden_size, layer_num_heads, dropout, etypes, name,data):
        super(Causal_Layer, self).__init__()
        self.num_heads = layer_num_heads
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.fc_m = nn.Linear(data["micro"].num_nodes, hidden_size)
        self.micro_emb = torch.nn.Embedding(data["micro"].num_nodes, hidden_size)
        self.fc_d = nn.Linear(data["drug"].num_nodes, hidden_size)
        self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, hidden_size)
        self.etypes = etypes
        self.semantic_encoder_layer = nn.ModuleList()
        self.hidden_size = hidden_size
        for i in range(len(meta_paths)):
            self.semantic_encoder_layer.append(GNN(self.hidden_size))

        self.subgraph_fusion = Subgraph_Fusion(in_size=self.hidden_size * self.num_heads)
        self.transformer_fusion=TransformerFusion(hidden_dim=self.hidden_size * self.num_heads, num_heads=2, num_layers=3)


        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)
        nn.init.xavier_normal_(self.micro_emb.weight, gain=1.414)
        nn.init.xavier_normal_(self.drug_emb.weight, gain=1.414)


    def stack_embedding(self, embeddings):
        subgraph_num_nodes = [embeddings[i].size()[0] for i in range(len(embeddings))]
        if subgraph_num_nodes.count(subgraph_num_nodes[0]) == len(subgraph_num_nodes):
            embeddings = torch.stack(embeddings, dim=1)
        else:
            for i in range(0, len(embeddings)):
                index = max(subgraph_num_nodes) - subgraph_num_nodes[i]
                if index != 0:
                    h_ = torch.zeros(index, self.hidden_size * self.num_heads)
                    embeddings[i] = torch.cat((embeddings[i], h_), dim=0)
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings

    def forward(self, train_data):

        semantic_embeddings = {'m': [], 'd': []}
        nodes_embeddings = {}

        x_dict = {
            "micro": self.fc_m(train_data["micro"].x) + self.micro_emb(train_data["micro"].node_id),
            "drug": self.fc_d(train_data["drug"].x) + self.drug_emb(train_data["drug"].node_id),
        }
        edge_index_cause = train_data['micro', 'cause', 'drug'].edge_index
        edge_index_rev = train_data['drug', 'rev_cause', 'micro'].edge_index
        # 3. micro -> drug
        drug_from_micro = self.semantic_encoder_layer[0]((x_dict["micro"], x_dict["drug"]), edge_index_cause)
        # 4. drug -> micro
        micro_from_drug = self.semantic_encoder_layer[1]((x_dict["drug"], x_dict["micro"]),edge_index_rev)

        semantic_embeddings['d'] = [drug_from_micro,  x_dict["drug"]]
        semantic_embeddings['m'] = [micro_from_drug,  x_dict["micro"]]

        z_drug, alpha_d = self.transformer_fusion(semantic_embeddings['d'])
        z_micro, alpha_m = self.transformer_fusion(semantic_embeddings['m'])
        nodes_embeddings['d'] = z_drug
        nodes_embeddings['m'] = z_micro

        return nodes_embeddings,x_dict


class DCSILGNN(nn.Module):
    def __init__(self, meta_paths, hidden_size, n_in_features, n_drug,num_heads, dropout, args, data):
        super(DCSILGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.n_in_features = n_in_features
        self.n_drug = n_drug
        self.hid_features = 32
        self.hid_out = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.gcn_hetro1 = GCNConv(in_channels=self.n_in_features, out_channels=self.hid_out)
        self.hetero_transformerEncoder = Attention(d_model=self.n_in_features, n_heads=2, n_layers=1)

        self.weight_dict = self.init_weight(args)
        self.mess_dropout = [0.1, 0.1, 0.1]

        self.layers1 = Causal_Layer(meta_paths,  hidden_size,  num_heads, dropout, args.etypes, name=['m', 'd'],data=data)

        self.reconstructions = InnerProductDecoder( name='gan_decoder',input_dim=self.hid_features, num_d=self.n_drug, act=torch.sigmoid,hid_out=self.hid_out)
        self.attn = torch.nn.Linear(hidden_size, 1, bias=False)
        self.W1 = torch.nn.Linear(hidden_size, hidden_size)
        self.W2 = torch.nn.Linear(hidden_size, hidden_size)
        self.W3 = torch.nn.Linear(hidden_size, hidden_size)
        self.attnt = torch.nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)

        self.fc_m = nn.Linear(data["micro"].num_nodes, hidden_size)
        self.fc_d = nn.Linear(data["drug"].num_nodes, hidden_size)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)

    def decode(self,z, edge_label_index):

        return (z['m'][edge_label_index[0]] * z['d'][edge_label_index[1]]).sum(dim=-1)

    def init_weight(self,args):

        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()

        layers = [self.hidden_size] + [int(self.hid_out/2),int(self.hid_out/2)]

        for k in range(2):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})


        return weight_dict

    def forward(self, train_data,x,edge_idx,A_hat):

        nodes_embeddings = {}
        embd_tmp = x

        h_trans_embed ,x_dict= self.layers1(train_data)
        drug_gcn1 = h_trans_embed['d']
        mic_gcn1 = h_trans_embed['m']

        ego_embeddings = torch.cat([drug_gcn1, mic_gcn1], 0)
        all_embeddings = []
        for k in range(2):

            side_embeddings = torch.mm(A_hat, ego_embeddings)
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]


        all_embeddings = torch.cat(all_embeddings, 1)
        drug_NGCF = all_embeddings[:self.n_drug, :]
        mic_NGCF = all_embeddings[self.n_drug:, :]


        embd_hetro = self.hetero_transformerEncoder(embd_tmp)
        hetro_gcn1 = torch.relu(self.gcn_hetro1(embd_hetro, edge_idx))
        embd_heter = hetro_gcn1
        drug_z = embd_heter[0:self.n_drug, :]
        mic_z = embd_heter[self.n_drug:, :]


        drug_features = torch.stack([drug_gcn1, drug_NGCF, drug_z], dim=0)
        score = self.attn(drug_features)
        alpha = torch.softmax(score, dim=0)
        drug_out = (alpha * drug_features).sum(dim=0)


        mic_features = torch.stack([mic_gcn1, mic_NGCF, mic_z], dim=0)
        score = self.attn(mic_features)
        alpha = torch.softmax(score, dim=0)
        mic_out = (alpha * mic_features).sum(dim=0)

        nodes_embeddings['d'] = drug_out
        nodes_embeddings['m'] = mic_out

        return nodes_embeddings



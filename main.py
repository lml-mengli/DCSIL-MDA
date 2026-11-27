import csv
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,matthews_corrcoef
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.utils import negative_sampling
from scipy.sparse import coo_matrix

from torch_geometric.data import HeteroData
import torch.nn.functional as F
from model import DCSILGNN
from utils_ import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


micro_dis = pd.read_csv(r'./MDAD/mirco-drug-MDAD.csv', header=None)
micro_dis.columns =['micro_id','drug_id']
def make_edge_index(network):
    source_nodes = micro_dis['micro_id'].values
    target_nodes = micro_dis['drug_id'].values
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]),dtype=torch.long)
    return edge_index

def make_edge_attr(m_sim,d_sim):

    micro_dis = pd.read_csv(r'./MDAD/mirco-drug-MDAD.csv', header=None)
    micro_dis = micro_dis.values.tolist()
    edge_attr = np.zeros((len(micro_dis), len(micro_dis)), dtype=float)
    i=0
    for list in micro_dis:
        micro,dis = list
        for j in range(len(m_sim[micro])):
            edge_attr[i][j] = m_sim[micro][j]
        for p in range(len(d_sim[dis])):
            edge_attr[i][len(m_sim)+p] = d_sim[dis][p]
        i += 1
    return edge_attr


if __name__ == '__main__':

    args = parameters_set()
    set_seed(args.seed)

    association = pd.read_csv("./MDAD/index_association.csv", header=0, index_col=0).values
    m_sim = pd.read_csv("./MDAD/microbe_similarity.txt", header=None, index_col=None, sep='\t').values
    d_sim = pd.read_csv("./MDAD/drug_similarity.txt", header=None, index_col=None ,sep='\t').values

    n_m = len(m_sim)
    n_d = len(d_sim)

    m_sim = np.array(m_sim,dtype=float)
    d_sim = np.array(d_sim,dtype=float)
    data = HeteroData()
    edge_index = make_edge_index(association)
    edge_attr = make_edge_attr(m_sim,d_sim)

    

   
    drug_sim, cir_sim = d_sim, m_sim
    data['micro'].node_id = torch.arange(len(cir_sim))
    data['drug'].node_id = torch.arange(len(drug_sim))
    data['micro'].x = torch.tensor(m_sim,dtype=torch.float32)
    data['drug'].x = torch.tensor(d_sim,dtype=torch.float32)
    data['micro','cause','drug'].edge_index = edge_index
    data['micro','cause','drug'].edge_arrt = edge_attr
    data = ToUndirected()(data)
    print(data)



    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio = 1.0, 
        add_negative_train_samples=False, 
        edge_types=[('micro', 'cause', 'drug')],
        rev_edge_types=[('drug', 'rev_cause', 'micro')],
        )

    train_data, val_data, test_data = transform(data)
    print(train_data)
    print(val_data)
    print(test_data)

    data = data.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    val_data = val_data.to(device)

    np.random.seed(args.seed)


    model = DCSILGNN(meta_paths=args.metapaths, hidden_size=args.hidden_size, num_heads=args.num_heads,
                   n_in_features=n_d + n_m,n_drug=n_d,dropout=args.dropout, args=args, data=data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    train_pos_edges = train_data['micro', 'cause', 'drug'].edge_index

    diag = np.diag(m_sim)
    if np.sum(diag) != 0:
        cir_sim = m_sim - np.diag(diag)
    diag = np.diag(d_sim)
    if np.sum(diag) != 0:
        drug_sim = d_sim - np.diag(diag)
        
    temp_drug_cir = torch.zeros((n_d, n_m), device=device)
    temp_drug_cir[train_pos_edges[1], train_pos_edges[0]] = 1
    temp_drug_cir_np = temp_drug_cir.detach().cpu().numpy()

    het_mat = construct_het_mat(temp_drug_cir_np, cir_sim, drug_sim) 
    adj_mat = construct_adj_mat(temp_drug_cir_np)

    edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
    het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

    norm_adj_mat = normalized_adj(adj_mat).to(device)

    best_val_auc = final_test_auc = best_f1_auc = 0
    for epoch in range(args.num_epochs): 

        model.train()
        optimizer.zero_grad()
        neg_edge_label_index = negative_sampling(edge_index=train_data['micro', 'cause', 'drug'].edge_index,
                                                 num_nodes=(
                                                     [train_data['micro'].num_nodes, train_data['drug'].num_nodes]),
                                                 num_neg_samples=train_data[
                                                     'micro', 'cause', 'drug'].edge_label_index.size(1),
                                                 method='sparse')

        edge_label_index = torch.cat(
            [train_data['micro', 'cause', 'drug'].edge_label_index, neg_edge_label_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data['micro', 'cause', 'drug'].edge_label,
            train_data['micro', 'cause', 'drug'].edge_label.new_zeros(neg_edge_label_index.size(1))
        ], dim=0)

        out = model(train_data, het_mat_device, edge_idx_device, norm_adj_mat)


        train_pos_edges_label=train_data['micro', 'cause', 'drug'].edge_label_index

        out = model.decode(out, edge_label_index).view(-1)

        loss = F.binary_cross_entropy_with_logits(out, edge_label)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(val_data, het_mat_device, edge_idx_device, norm_adj_mat)
            out = model.decode(out, val_data["micro", "cause", "drug"].edge_label_index).view(-1).sigmoid()
            out = out.cpu().detach().numpy()
            pred = (np.array(out) > 0.5).astype(int)
            true = val_data["micro", "cause", "drug"].edge_label.cpu().detach().numpy()
            auroc = roc_auc_score(true, out)
            auprc = average_precision_score(true, out)
            acc = accuracy_score(true, pred)
            f1 = f1_score(true, pred)
            recall = recall_score(true, pred)
            precision = precision_score(true, pred)
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            specificity = tn / (tn + fp + 1e-6)
            mcc = matthews_corrcoef(true, pred)
            metrics = [auroc, auprc, f1, acc, recall, specificity, precision,mcc]

            if auroc > best_val_auc and f1 >= best_f1_auc:
                best_val_auc = auroc
                final_test_auc = auroc
                best_f1_auc = f1
                metrics_max = metrics.copy()
                torch.save(model.state_dict(), "./MDAD/MDAD_GCN_model.pt")


            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val:Auc = {auroc:.4f}, Test:Auc = {auroc:.4f},  Aupr = ',
                  metrics[1], '\t  f1_score = ', metrics[2],
                  '\t  Accuracy = ', metrics[3], '\t  Recall = ', metrics[4], '\t  Specificity = ', metrics[5],
                  '\t  Precision = ', metrics[6], '\t  Mcc = ', metrics[7]) 

    model.load_state_dict(torch.load("./MDAD/MDAD_GCN_model.pt", map_location=device))

    model.eval()
    with torch.no_grad():
        out = model(test_data, het_mat_device, edge_idx_device, norm_adj_mat)
        out = model.decode(out, test_data["micro", "cause", "drug"].edge_label_index).view(-1).sigmoid()
        out = out.cpu().detach().numpy()
        pred = (np.array(out) > 0.5).astype(int)
        true = test_data["micro", "cause", "drug"].edge_label.cpu().detach().numpy()
        auroc = roc_auc_score(true, out)
        auprc = average_precision_score(true, out)
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        specificity = tn / (tn + fp + 1e-6)
        mcc = matthews_corrcoef(true, pred)
        metrics = [auroc, auprc, f1, acc, recall, specificity, precision, mcc]

        print(f'Final Test:{auroc:.4f}')
        print(f'\nIndependent Test Set: Auc = {auroc:.4f},  Aupr = ',
              metrics[1], '\t  f1_score = ', metrics[2],
              '\t  Accuracy = ', metrics[3], '\t  Recall = ', metrics[4], '\t  Specificity = ', metrics[5],
              '\t  Precision = ', metrics[6], '\t  Mcc = ', metrics[7]) 

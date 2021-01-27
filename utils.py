import torch 
import numpy as np 

from sklearn.metrics import roc_auc_score, average_precision_score

def decode(z, ei):
    dot = (z[ei[0]] * z[ei[1]]).sum(dim=1)
    return torch.sigmoid(dot)

def get_score(tp, neg, z):
    # Select a weighted number of negative edges 
    ntp = tp.size(1)
    ntn = neg.size(1)
    
    pscore = decode(z, tp)
    nscore = decode(z, neg)
    score = torch.cat([pscore, nscore]).numpy()

    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return auc, ap

'''
Splits edges into 85:5:10 train val test partition
(Following route of ARGAE paper)
'''
def edge_tvt_split(data):
    ne = data.edge_index.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True 
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True 

    return masks[0], masks[1], masks[2]

'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin 
torch geo method)
'''
def fast_negative_sampling(edge_list, batch_size, oversample=1.25):
    num_nodes = edge_list.max().item() + 1
    
    # For faster membership checking
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

    el1d = el_hash(edge_list).numpy()
    neg = np.array([[],[]])

    while(neg.shape[1] < batch_size):
        maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample)))
        neg_hash = el_hash(maybe_neg)
        
        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
            axis=1
        )

    # May have gotten some extras
    neg = neg[:, :batch_size]
    return torch.tensor(neg).long()

'''
Given a list of src nodes, generate dst nodes that don't exist
'''
def in_order_negative_sampling(edge_list, src):
    # For faster membership checking
    num_nodes = edge_list.max().item()
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes
    el1d = el_hash(edge_list).numpy()

    src = src.numpy()
    dst = np.full((src.shape[0],), -1, dtype=np.long)
    while (dst.min() == -1):
        maybe_neg = np.random.randint(0, num_nodes+1, dst.shape)
        check = src + maybe_neg*num_nodes
        mask = (~np.in1d(check, el1d)) 
        dst[mask] = maybe_neg[mask]

    return torch.tensor(dst).long()

'''
Splits edge_index into multiple ei's based on one-hot 
edge features. Returns ef.size(1) different sets of 
edge indexes, where every edge in ei[n] has ef[:, n] == 1
'''
def edge_feats_to_multi_adj(ei, ef):
    new_eis = []

    for i in range(ef.size(1)):
        new_eis.append(ei[:, ef[:, i] == 1])
    
    return torch.stack(new_eis)
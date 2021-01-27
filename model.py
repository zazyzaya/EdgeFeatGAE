import torch 
import torch.nn as nn

from torch_geometric.nn import GCNConv

class EdgeFeatGAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32, reduce_fn=None):
        super(EdgeFeatGAE, self).__init__()

        self.c1 = GCNConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.sig = nn.Sigmoid()

        self.reduce = reduce_fn if reduce_fn else lambda x : x.sum(dim=2)
        self.embed_dim = embed_dim

    '''
    Runs GCN on a stack of different edge_indices each representing edges
    with different features (maybe in future use multiple GCNs for this?)
    '''
    def forward(self, x, eis, ews=None):
        embeds = []
        if type(ews) == type(None):
            ews = [None] * eis.size(2)

        for i in range(eis.size(2)):
            embeds.append(self.forward_once(x, eis[i], ews[i]))

        embeds = torch.stack(embeds)
        return self.reduce(embeds)


    def forward_once(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        x = self.drop(x)

        return x

    def decode(self, z, ei):
        dot = (z[ei[0]] * z[ei[1]]).sum(dim=1)
        return self.sig(dot)

    def loss_fn(self, z, pos_samples, neg_samples):
        EPS = 1e-6
        pos_loss = -torch.log(self.decode(z, pos_samples)+EPS).mean()
        neg_loss = -torch.log(1-self.decode(z, neg_samples)+EPS).mean()

        return pos_loss + neg_loss
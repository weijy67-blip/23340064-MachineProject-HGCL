# -*- coding: UTF-8 -*-
# @Author  : [Your Name]
# @Email   : [Your Email]

""" HGCL (Hyper-Graph Contrastive Learning)
Reference:
    "Self-supervised Hypergraph Contrastive Learning for Recommendation"
    Wang et al., SIGIR'2022.
CMD example:
    python main.py --model_name HGCL --emb_size 64 --lr 1e-3 --l2 1e-6 --tau 0.2 --lam 0.1 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class HGCL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'tau', 'lam', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--tau', type=float, default=0.2,
                            help='Temperature parameter for contrastive loss.')
        parser.add_argument('--lam', type=float, default=0.1,
                            help='Weight for contrastive loss.')
        parser.add_argument('--hyper_k', type=int, default=3,
                            help='K-nearest neighbors for hypergraph construction.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of hypergraph convolution layers.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.tau = args.tau
        self.lam = args.lam
        self.hyper_k = args.hyper_k
        self.n_layers = args.n_layers

        self._define_params()
        self.apply(self.init_weights)

        # Build hypergraph adjacency matrix
        self.hypergraph_adj = self.build_hypergraph(corpus)

    def _define_params(self):
        # User and item embeddings
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        # Hypergraph convolution layers
        self.hgnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.hgnn_layers.append(nn.Linear(self.emb_size, self.emb_size))

        # Projection heads for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )

    def build_hypergraph(self, corpus):
        """
        Build hypergraph adjacency matrix based on user-item interactions
        and item-item similarity
        """
        user_item = corpus.train_clicked_set  # dict: {user_id: [item_id1, item_id2, ...]}

        # Build user hyperedges (each user connects to their interacted items)
        user_hyperedges = []
        for u in range(self.user_num):
            if u in user_item:
                user_hyperedges.append(list(user_item[u]))

        # Build item hyperedges based on KNN similarity
        item_hyperedges = []

        # First, compute item-item similarity matrix from interactions
        inter_mat = np.zeros((self.user_num, self.item_num))
        for u, items in user_item.items():
            for i in items:
                inter_mat[u, i] = 1

        # Compute item similarity (cosine similarity)
        item_sim = np.dot(inter_mat.T, inter_mat)
        item_norm = np.linalg.norm(inter_mat.T, axis=1, keepdims=True)
        item_norm[item_norm == 0] = 1e-10
        item_sim = item_sim / (item_norm @ item_norm.T)

        # For each item, find k-nearest neighbors
        for i in range(self.item_num):
            sim_scores = item_sim[i]
            # Exclude self
            sim_scores[i] = -1
            # Get top-k similar items
            top_k_indices = np.argsort(sim_scores)[-self.hyper_k:]
            item_hyperedges.append(list(top_k_indices))

        # Combine all hyperedges
        all_hyperedges = user_hyperedges + item_hyperedges
        num_hyperedges = len(all_hyperedges)

        # Create incidence matrix H (|V| x |E|)
        H = sp.lil_matrix((self.user_num + self.item_num, num_hyperedges))

        # Fill user hyperedges
        for e_idx, items in enumerate(user_hyperedges):
            H[self.user_num + np.array(items), e_idx] = 1
            H[self.user_num - 1, e_idx] = 1  # User node

        # Fill item hyperedges
        for e_idx, items in enumerate(item_hyperedges, start=len(user_hyperedges)):
            H[self.user_num + np.array(items), e_idx] = 1

        # Compute hypergraph Laplacian
        D_v = sp.diags(H.sum(axis=1).A1)  # Vertex degree matrix
        D_e = sp.diags(H.sum(axis=0).A1)  # Edge degree matrix

        # Avoid division by zero
        D_v_inv_sqrt = sp.diags(1.0 / np.sqrt(D_v.diagonal() + 1e-10))
        D_e_inv = sp.diags(1.0 / (D_e.diagonal() + 1e-10))

        # Hypergraph Laplacian: Δ = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        theta = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
        laplacian = sp.eye(self.user_num + self.item_num) - theta

        return laplacian

    def hypergraph_convolution(self, embeddings):
        """
        Perform hypergraph convolution
        embeddings: concatenated user and item embeddings [user_num + item_num, emb_size]
        """
        adj = self.hypergraph_adj.toarray() if isinstance(self.hypergraph_adj, sp.spmatrix) else self.hypergraph_adj
        adj = torch.FloatTensor(adj).to(embeddings.device)

        # Multi-layer hypergraph convolution
        h = embeddings
        for layer in self.hgnn_layers:
            h = torch.matmul(adj, h)
            h = layer(h)
            h = F.relu(h)

        return h

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        # Get base embeddings
        u_emb = self.u_embeddings(u_ids)  # [batch_size, emb_size]
        i_emb = self.i_embeddings(i_ids)  # [batch_size, num_neg+1, emb_size]

        # Concatenate all embeddings for hypergraph convolution
        all_u_emb = self.u_embeddings.weight  # [user_num, emb_size]
        all_i_emb = self.i_embeddings.weight  # [item_num, emb_size]
        all_emb = torch.cat([all_u_emb, all_i_emb], dim=0)  # [user_num+item_num, emb_size]

        # Hypergraph convolution
        enhanced_emb = self.hypergraph_convolution(all_emb)
        enhanced_u_emb = enhanced_emb[:self.user_num]
        enhanced_i_emb = enhanced_emb[self.user_num:]

        # Get enhanced embeddings for batch users and items
        enhanced_u = enhanced_u_emb[u_ids]  # [batch_size, emb_size]
        enhanced_i = enhanced_i_emb[i_ids]  # [batch_size, num_neg+1, emb_size]

        # Compute predictions from both views
        base_prediction = (u_emb[:, None, :] * i_emb).sum(dim=-1)  # [batch_size, -1]
        enhanced_prediction = (enhanced_u[:, None, :] * enhanced_i).sum(dim=-1)  # [batch_size, -1]

        # Combine predictions
        prediction = (base_prediction + enhanced_prediction) / 2

        # Project embeddings for contrastive loss
        z_u_base = F.normalize(self.projection(u_emb), dim=-1)
        z_u_enhanced = F.normalize(self.projection(enhanced_u), dim=-1)

        # Get all item embeddings for contrastive loss
        all_i_emb_base = self.i_embeddings.weight
        all_i_emb_enhanced = enhanced_i_emb

        # Sample negative items for contrastive loss
        batch_size = u_ids.shape[0]
        neg_indices = torch.randint(0, self.item_num, (batch_size, 100)).to(u_ids.device)

        return {
            'prediction': prediction,
            'z_u_base': z_u_base,
            'z_u_enhanced': z_u_enhanced,
            'all_i_emb_base': all_i_emb_base,
            'all_i_emb_enhanced': all_i_emb_enhanced,
            'neg_indices': neg_indices,
            'u_emb': u_emb,
            'enhanced_u': enhanced_u
        }

    def calculate_loss(self, feed_dict, out_dict):
        """
        Calculate BPR loss + contrastive loss
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]

        # BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_pred.unsqueeze(1) - neg_pred)).mean()

        # Contrastive loss
        z_u_base = out_dict['z_u_base']
        z_u_enhanced = out_dict['z_u_enhanced']

        # Positive pairs: same user from two views
        pos_sim = torch.sum(z_u_base * z_u_enhanced, dim=-1) / self.tau
        pos_sim = torch.exp(pos_sim)

        # Negative pairs: different users
        batch_size = z_u_base.shape[0]
        all_z_u = torch.cat([z_u_base, z_u_enhanced], dim=0)
        neg_sim = torch.mm(all_z_u, all_z_u.t()) / self.tau
        mask = torch.eye(batch_size * 2, device=z_u_base.device).bool()
        neg_sim = neg_sim.masked_fill(mask, -1e9)
        neg_sim = torch.exp(neg_sim).sum(dim=-1)

        # InfoNCE loss
        cl_loss = -torch.log(pos_sim / neg_sim[:batch_size]).mean()

        # Total loss
        total_loss = bpr_loss + self.lam * cl_loss

        return total_loss, {
            'bpr_loss': bpr_loss.item(),
            'cl_loss': cl_loss.item(),
            'total_loss': total_loss.item()
        }


class HGCLImpression(HGCL):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'

    @staticmethod
    def parse_model_args(parser):
        return HGCL.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

    def forward(self, feed_dict):
        out_dict = super().forward(feed_dict)
        # For impression tasks, we might need additional processing
        return out_dict
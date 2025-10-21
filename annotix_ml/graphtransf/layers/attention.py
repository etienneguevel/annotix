from math import sqrt

import torch
import torch.nn as nn

from annotix_ml.graphtransf.layers.ffn import FfnNodeEdge

class MultiHeadEdgeNode(nn.Module):
    """
    Class implementing the attention classification of the Edge-Node model.
    """
    def __init__(
        self,
        d: int,
        de:int,
        n_heads: int,
    ):
        """
        Args:
        - d: int, hidden dimension of the model 
        - de: int, hidden dimension of the edges
        - n_heads: int, number of heads 
        """
        super().__init__()
        if not (d % n_heads == 0):
            raise ValueError(
                f"Hidden dim : {d} is not a multiple of num heads : {n_heads}"
            )
        self.d = d
        self.de = de
        self.dk = d // n_heads
        self.n_heads = n_heads

        # Put the 3 matrices together to optimize computing
        self.qkv = nn.Linear(d, 3 * d)

        # Initialize the 2 matrices for FiLM
        self.FiLM_E = nn.Linear(de, 2 * d)

        # Initialize the matrices for output projection
        self.Out_N = nn.Linear(d, d)
        self.norm_n = nn.LayerNorm(d)
        self.Out_E = nn.Linear(d, de)
        self.norm_e = nn.LayerNorm(de)
    
    def forward_nested(
            self,
            h: torch.Tensor,
            e: torch.Tensor,
        ):
        """
        Function to get the attention map. attn[i, j] = softmax_j(sum_k(Q_i.K_j.E_ij))
        Args:
        - h: torch.tensor, nested tensor of the nodes representation (bs, *n, d)
        - e: torch.tensor, nested tensor of the edges representation (bs, *n, *n, d)
        TODO : Check if i can do without mask, or if needed for after FiLM
        """
        # Compute the classic attention map
        # h : (bs, *n, d)
        # e : (bs, *n, *n, de)
        # node_mask : (bs, n)

        bs, _, d = h.size()
        bse, *_, de = e.size()
        assert bs == bse, "Wrong batch sizes for nodes and edges."
        assert d == self.d, "Wrong dimension for the nodes embeddings."
        assert de == self.de, "Wrong dimension for the edges embeddings."
        
        # Calculate the Query, Key and Values vectors
        qkv = self.qkv(h).unflatten((-1, (3 * self.dk, self.n_heads))) # (bs, *n, 3 * dk, nh)
        qkv = qkv.permute((0, 3, 1, 2)) # (bs, nh, *n, 3 * dk)
        Q, K, V = qkv.chunk(3, -1) # (bs, nh, *n, dk), (bs, nh, *n, dk), (bs, nh, *n, dk)

        # Change the axis & add a dim for outer product
        Q = Q.unsqueeze(3) # (bs, nh, *n, 1, dk)
        K = K.unsqueeze(2) # (bs, nh, 1, *n, dk)

        # Calculate the edges key vector
        E = self.FiLM_E(e).unflatten((-1, (2 * self.dk, self.n_heads))) # (bs, *n, *n, 2 * dk, nh)
        E1, E2 = E.permute((0, 4, 1, 2, 3)).chunk(2, -1) # (bs, nh, *n, *n, dk), (bs, nh, *n, *n, dk)
        
        # Do the outer product of Q and K
        attn = Q * K # (bs, nh, *n, *n, dk)
        attn /= sqrt(self.dk) # (bs, nh, *n, *n, dk)

        # Add the edges to the attn product
        attn = E1 + (E2 * attn) + attn # (bs, nh, *n, *n, dk)

        # Do the summed softmax of the edges
        node_attn = attn.sum(-1) # (bs, nh, *n, *n)
        node_attn = node_attn.softmax(-1) # (bs, nh, *n, *n)

        # Add to the values
        node_attn = node_attn.unsqueeze(-1) * V 
    
        ...
    
    def forward_normal(
            self,
            h: torch.Tensor,
            e: torch.Tensor,
            mask: torch.Tensor,
            attn_map_mode: bool = False,
        ):
        """
        Function to get the attention map. attn[i, j] = softmax_j(sum_k(Q_i.K_j.E_ij))
        Args:
        - h: torch.tensor, tensor of the nodes representation (bs, n, d)
        - e: torch.tensor, tensor of the edges representation (bs, n, n, d)
        - mask: torch.tensor, binary mask representing the number of nodes
        present in each element of the batch.
        """
        # Compute the classic attention map
        # h : (bs, n, d)
        # e : (bs, n, n, de)
        # mask : (bs, n)

        bs, n, d = h.size()
        bse, *_, de = e.size()
        assert bs == bse, "Wrong batch sizes for nodes and edges."
        assert d == self.d, "Wrong dimension for the nodes embeddings."
        assert de == self.de, (
            "Wrong dimension for the edges embeddings, "
            f"got {de} expected {self.de}."
        )

        # Prepare the node mask
        node_mask = mask.unsqueeze(-1) # (bs, n, 1)
        edge_mask = node_mask.unsqueeze(-1).expand(-1, -1, n, -1) # (bs, n, n, 1)

        # Calculate the Query, Key and Values vectors
        qkv = self.qkv(h).unflatten(-1, (3 * self.dk, self.n_heads)) # (bs, n, 3 * dk, nh)
        qkv = qkv * node_mask.unsqueeze(-1) # Mask the qkv matrix

        qkv = qkv.permute((0, 3, 1, 2)) # (bs, nh, n, 3 * dk)
        Q, K, V = qkv.chunk(3, -1) # (bs, nh, n, dk), (bs, nh, n, dk), (bs, nh, n, dk)

        # Change the axis & add a dim for outer product
        Q = Q.unsqueeze(3) # (bs, nh, n, 1, dk)
        K = K.unsqueeze(2) # (bs, nh, 1, n, dk)

        # Calculate the edges key vector
        E = self.FiLM_E(e).unflatten(-1, (2 * self.dk, self.n_heads)) # (bs, n, n, 2 * dk, nh)
        E = E * edge_mask.unsqueeze(-1) # (bs, n, n, 2 * dk, nh)
        E1, E2 = E.permute((0, 4, 1, 2, 3)).chunk(2, -1) # (bs, nh, n, n, dk), (bs, nh, n, n, dk)
        
        # Do the outer product of Q and K
        attn = Q * K # (bs, nh, n, n, dk)
        attn /= sqrt(self.dk) # (bs, nh, n, n, dk)

        # Add the edges to the attn product
        node_attn = E1 + (E2 * attn) + attn # (bs, nh, n, n, dk)

        # Do the summed softmax of the edges
        node_attn = node_attn.sum(-1) # (bs, nh, n, n)
        node_attn = node_attn.softmax(-1) # (bs, nh, n, n)

        if attn_map_mode:
            return node_attn, node_attn

        # Add to the values
        node_attn = node_attn @ V # (bs, nh, n, dk)
        node_attn = node_attn.transpose(1, 2).flatten(start_dim=2) # (bs, n, d)

        # Stack the edges attn
        edge_attn = attn.permute((0, 2, 3, 1, 4)).flatten(start_dim=3) # (bs, n, n, d)

        # Do the output transformation
        h = self.norm_n(h + self.Out_N(node_attn)) # (bs, n, d)
        e = self.norm_e(e + self.Out_E(edge_attn)) # (bs, n, n, de)

        # Ensure that the masking is still correct
        h = h * node_mask # (bs, n, d)
        e = e * edge_mask # (bs, n, n, de)

        return h, e

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ):
        
        if h.is_nested & e.is_nested:
            # /!\ Not implemented yet
            _ = self.forward_nested(h, e)

        elif h.is_nested | e.is_nested:
            raise TypeError(
                "Only one of the two tensors is nested -> both need to be the same type."
            )
        
        else:
            h, e = self.forward_normal(h, e, mask)
        
        return h, e, mask

class AttentionLayer(nn.Module):
    """
    """
    def __init__(
        self,
        d: int,
        de:int,
        n_heads: int,

    ):
        super().__init__()
        self.d = d
        self.de = de
        self.n_heads = n_heads
        self.attnEdgeNode = MultiHeadEdgeNode(d, de, n_heads)
        self.ffn = FfnNodeEdge(d, de)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ):
        h_attn, e_attn, mask_attn = self.attnEdgeNode(h, e, mask)
        h, e, mask = self.ffn(h_attn, e_attn, mask_attn)
        return h, e, mask

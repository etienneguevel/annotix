from math import sqrt

import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.layers.ffn import FfnNodeEdge
from annotix_ml.graphtransf.layers.mlp import MLP


class MultiHeadEdgeNodeWithY(nn.Module):
    """
    Class implementing the attention classification of the Edge-Node model.
    """

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
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

        # Initialize the matrices for FiLM
        self.FiLM_E = nn.Linear(de, 2 * d)
        self.FilM_yN = nn.Linear(dy, 2 * d)
        self.FilM_yE = nn.Linear(dy, 2 * d)

        # Initialize the matrices for output projection
        self.Out_N = nn.Linear(d, d)
        self.norm_n = nn.LayerNorm(d)

        self.Out_E = nn.Linear(d, de)
        self.norm_e = nn.LayerNorm(de)

        # Make the matrices for y updating
        self.Update_y = nn.Linear(4 * d + 4 * de + dy, dy)
        self.Out_y = MLP(dy, dy, dy)
        self.norm_y = nn.LayerNorm(dy)

    def forward_nested(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
    ):
        """
        Function to get the attention map.
        attn[i, j] = softmax_j(sum_k(Q_i.K_j.E_ij))
        Args:
        - h: torch.tensor, nested tensor of the nodes representation (bs, *n, d)
        - e: torch.tensor, nested tensor of the edges representation (bs, *n, *n, d)
        TODO : Check if i can do without mask, or if needed for after FiLM
        """
        # Compute the classic attention map
        # h : (bs, *n, d)
        # e : (bs, *n, *n, de)

        bs, _, d = h.size()
        bse, *_, de = e.size()
        assert bs == bse, "Wrong batch sizes for nodes and edges."
        assert d == self.d, "Wrong dimension for the nodes embeddings."
        assert de == self.de, "Wrong dimension for the edges embeddings."

        # Calculate the Query, Key and Values vectors
        qkv = self.qkv(h).unflatten(
            (-1, (3 * self.dk, self.n_heads))
        )  # (bs, *n, 3 * dk, nh)
        qkv = qkv.permute((0, 3, 1, 2))  # (bs, nh, *n, 3 * dk)
        Q, K, V = qkv.chunk(
            3, -1
        )  # (bs, nh, *n, dk), (bs, nh, *n, dk), (bs, nh, *n, dk)

        # Change the axis & add a dim for outer product
        Q = Q.unsqueeze(3)  # (bs, nh, *n, 1, dk)
        K = K.unsqueeze(2)  # (bs, nh, 1, *n, dk)

        # Calculate the edges key vector
        E = self.FiLM_E(e).unflatten(
            (-1, (2 * self.dk, self.n_heads))
        )  # (bs, *n, *n, 2 * dk, nh)
        E1, E2 = E.permute((0, 4, 1, 2, 3)).chunk(
            2, -1
        )  # (bs, nh, *n, *n, dk), (bs, nh, *n, *n, dk)

        # Do the outer product of Q and K
        attn = Q * K  # (bs, nh, *n, *n, dk)
        attn /= sqrt(self.dk)  # (bs, nh, *n, *n, dk)

        # Add the edges to the attn product
        attn = E1 + (E2 * attn) + attn  # (bs, nh, *n, *n, dk)

        # Do the summed softmax of the edges
        node_attn = attn.sum(-1)  # (bs, nh, *n, *n)
        node_attn = node_attn.softmax(-1)  # (bs, nh, *n, *n)

        # Add to the values
        node_attn = node_attn.unsqueeze(-1) * V

        return None

    def compute_attn(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
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
        # y: (bs, dy)
        # mask : (bs, n)

        bs, n, d = h.size()
        bse, *_, de = e.size()
        assert bs == bse, "Wrong batch sizes for nodes and edges."
        assert d == self.d, "Wrong dimension for the nodes embeddings."
        assert de == self.de, (
            f"Wrong dimension for the edges embeddings, got {de} expected {self.de}."
        )

        # Calculate the Query, Key and Values vectors
        qkv = self.qkv(h).unflatten(
            -1, (3 * self.dk, self.n_heads)
        )  # (bs, n, 3 * dk, nh)
        qkv = mask_any_tensor(qkv, mask)  # Mask the qkv matrix

        qkv = qkv.permute((0, 3, 1, 2))  # (bs, nh, n, 3 * dk)
        Q, K, V = qkv.chunk(3, -1)  # (bs, nh, n, dk), (bs, nh, n, dk), (bs, nh, n, dk)

        # Change the axis & add a dim for outer product
        Q = Q.unsqueeze(3)  # (bs, nh, n, 1, dk)
        K = K.unsqueeze(2)  # (bs, nh, 1, n, dk)

        # Calculate the edges key vector
        E = self.FiLM_E(e).unflatten(
            -1, (2 * self.dk, self.n_heads)
        )  # (bs, n, n, 2 * dk, nh)
        E = mask_any_tensor(E, mask)  # (bs, n, n, 2 * dk, nh)
        E1, E2 = E.permute((0, 4, 1, 2, 3)).chunk(
            2, -1
        )  # (bs, nh, n, n, dk), (bs, nh, n, n, dk)

        # Do the outer product of Q and K
        # attn[..., i, j, :] = Q[..., i, :] * K[..., j, :]
        attn = Q * K  # (bs, nh, n, n, dk)
        attn /= sqrt(self.dk)  # (bs, nh, n, n, dk)

        # Add the edges to the attn product -> FiLM
        attn = E1 + (E2 * attn) + attn  # (bs, nh, n, n, dk)

        # Do the summed softmax of the edges
        node_attn = attn.sum(-1)  # (bs, nh, n, n)

        # Mask the attention scores before softmax
        mask_attn = mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, n)
        node_attn = node_attn.masked_fill(mask_attn == 0, -1e9)

        node_attn = node_attn.softmax(-1)  # (bs, nh, n, n)

        if attn_map_mode:
            return attn, node_attn

        # Add to the values
        node_attn = node_attn @ V  # (bs, nh, n, dk)
        node_attn = node_attn.transpose(1, 2).flatten(start_dim=2)  # (bs, n, d)

        # Stack the edges attn
        edge_attn = attn.permute((0, 2, 3, 1, 4)).flatten(start_dim=3)  # (bs, n, n, d)

        # FilM the nodes with y
        yN = self.FilM_yN(y)  # (bs, 2*d)
        yN1, yN2 = yN.unsqueeze(1).chunk(2, -1)  # (bs, 1, d), (bs, 1, d)

        node_attn = yN1 + yN2 * node_attn + node_attn  # (bs, n, d)
        node_attn = mask_any_tensor(node_attn, mask)

        # FilM the edges with y
        yE = self.FilM_yE(y)  # (bs, 2*d)
        yE1, yE2 = (
            yE.unsqueeze(1).unsqueeze(1).chunk(2, -1)
        )  # (bs, 1, 1, d), (bs, 1, 1, d)

        edge_attn = yE1 + yE2 * edge_attn + edge_attn  # (bs, n, n, d)
        edge_attn = mask_any_tensor(edge_attn, mask)

        return node_attn, edge_attn

    def update_y(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the features of h
        h_feats = torch.hstack(
            (h.max(1).values, h.min(1).values, h.mean(1), h.std(1))
        )  # (bs, 4 * d)

        # Compute the features of e
        e_feats = torch.hstack(
            (
                e.max(2).values.max(1).values,
                e.min(2).values.min(1).values,
                e.mean((1, 2)),
                e.std((1, 2)),
            )
        )  # (bs, 4 * de)

        # Compute new y
        new_y = self.Update_y(torch.hstack((h_feats, e_feats, y)))  # (bs, dy)

        return new_y

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
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
            # Compute the attention
            node_attn, edge_attn = self.compute_attn(h, e, y, mask)
            new_y = self.update_y(h, e, y)  # (bs, dy)

            # Do the output transformation
            h = self.norm_n(h + self.Out_N(node_attn))  # (bs, n, d)
            e = self.norm_e(e + self.Out_E(edge_attn))  # (bs, n, n, de)
            y = self.norm_y(y + self.Out_y(new_y))  # (bs, dy)

            # Ensure that the masking is still correct
            h = mask_any_tensor(h, mask)  # (bs, n, d)
            e = mask_any_tensor(e, mask)  # (bs, n, n, de)

        return h, e, y, mask


class MultiHeadEdgeNode(nn.Module):
    """
    Class implementing the attention classification of the Edge-Node model,
    without global features y.
    """

    def __init__(
        self,
        d: int,
        de: int,
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

        # Initialize the matrices for FiLM (only for edges)
        self.FiLM_E = nn.Linear(de, 2 * d)

        # Initialize the matrices for output projection
        self.Out_N = nn.Linear(d, d)
        self.norm_n = nn.LayerNorm(d)

        self.Out_E = nn.Linear(d, de)
        self.norm_e = nn.LayerNorm(de)

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
            f"Wrong dimension for the edges embeddings, got {de} expected {self.de}."
        )

        # Calculate the Query, Key and Values vectors
        qkv = self.qkv(h).unflatten(
            -1, (3 * self.dk, self.n_heads)
        )  # (bs, n, 3 * dk, nh)
        qkv = mask_any_tensor(qkv, mask)  # Mask the qkv matrix

        qkv = qkv.permute((0, 3, 1, 2))  # (bs, nh, n, 3 * dk)
        Q, K, V = qkv.chunk(3, -1)  # (bs, nh, n, dk), (bs, nh, n, dk), (bs, nh, n, dk)

        # Change the axis & add a dim for outer product
        Q = Q.unsqueeze(3)  # (bs, nh, n, 1, dk)
        K = K.unsqueeze(2)  # (bs, nh, 1, n, dk)

        # Calculate the edges key vector
        E = self.FiLM_E(e).unflatten(
            -1, (2 * self.dk, self.n_heads)
        )  # (bs, n, n, 2 * dk, nh)
        E = mask_any_tensor(E, mask)  # (bs, n, n, 2 * dk, nh)
        E1, E2 = E.permute((0, 4, 1, 2, 3)).chunk(
            2, -1
        )  # (bs, nh, n, n, dk), (bs, nh, n, n, dk)

        # Do the outer product of Q and K
        # attn[..., i, j, :] = Q[..., i, :] * K[..., j, :]
        attn = Q * K  # (bs, nh, n, n, dk)
        attn /= sqrt(self.dk)  # (bs, nh, n, n, dk)

        # Add the edges to the attn product -> FiLM
        attn = E1 + (E2 * attn) + attn  # (bs, nh, n, n, dk)

        # Do the summed softmax of the edges
        node_attn = attn.sum(-1)  # (bs, nh, n, n)

        # Mask the attention scores before softmax
        mask_attn = mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, n)
        node_attn = node_attn.masked_fill(mask_attn == 0, -1e9)

        node_attn = node_attn.softmax(-1)  # (bs, nh, n, n)

        if attn_map_mode:
            return attn, node_attn

        # Add to the values
        node_attn = node_attn @ V  # (bs, nh, n, dk)
        node_attn = node_attn.transpose(1, 2).flatten(start_dim=2)  # (bs, n, d)

        # Stack the edges attn
        edge_attn = attn.permute((0, 2, 3, 1, 4)).flatten(start_dim=3)  # (bs, n, n, d)

        # Do the output transformation
        h = self.norm_n(h + self.Out_N(node_attn))  # (bs, n, d)
        e = self.norm_e(e + self.Out_E(edge_attn))  # (bs, n, n, de)

        # Ensure that the masking is still correct
        h = mask_any_tensor(h, mask)  # (bs, n, d)
        e = mask_any_tensor(e, mask)  # (bs, n, n, de)

        return h, e

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ):
        if h.is_nested & e.is_nested:
            raise NotImplementedError("Nested tensors not supported yet.")

        elif h.is_nested | e.is_nested:
            raise TypeError(
                "Only one of the two tensors is nested -> both need to be the same type."
            )

        else:
            h, e = self.forward_normal(h, e, mask)

        return h, e, mask


class AttentionLayer(nn.Module):
    """
    Class implementing the total attention layer of the model. It consists of a
    EdgeNode attention layer followed by a ffnNodeEdge layer.
    """

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        n_heads: int,
    ):
        super().__init__()
        self.d = d
        self.de = de
        self.n_heads = n_heads
        self.attnEdgeNode = MultiHeadEdgeNodeWithY(d, de, dy, n_heads)
        self.ffn = FfnNodeEdge(d, de)
        self.mlpy = MLP(dy, 2 * dy, dy)
        self.norm_y = nn.LayerNorm(dy)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Compute the attention, make the residual connection
        h_attn, e_attn, y_attn, mask = self.attnEdgeNode(h, e, y, mask)
        # print(y_attn.shape)
        # print(self.mlpy)

        # Compute the output of the feedforward network, make residual connections
        h, e, _ = self.ffn(h_attn, e_attn, mask)
        y = self.norm_y(y_attn + self.mlpy(y_attn))

        return h, e, y, mask

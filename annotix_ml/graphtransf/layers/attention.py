from math import sqrt

import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.layers.ffn import FfnNodeEdge
from annotix_ml.graphtransf.layers.mlp import MLP


class MultiHeadEdgeNodeWithY(nn.Module):
    """
    Multi-head attention layer for edge and node features with global feature updates.

    This layer implements an attention mechanism that processes node, edge, and global (y) features.
    It uses FiLM (Feature-wise Linear Modulation) to incorporate edge and global information into the
    attention process and node/edge updates.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        dy (int): Hidden dimension for global features.
        n_heads (int): Number of attention heads.
        y_update (bool, optional): Whether to update global features. Defaults to True.
    """

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        n_heads: int,
        y_update: bool = True,
    ):
        """
        Initialize the MultiHeadEdgeNodeWithY layer.

        Args:
            d (int): Hidden dimension of the model.
            de (int): Hidden dimension of the edges.
            dy (int): Hidden dimension of the global features.
            n_heads (int): Number of attention heads.
            y_update (bool, optional): Whether to update global features. Defaults to True.
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
        self.y_update = y_update

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
        Forward pass using nested tensors (experimental).

        Computes the attention map: attn[i, j] = softmax_j(sum_k(Q_i.K_j.E_ij))

        Args:
            h (torch.Tensor): Nested tensor of nodes representation of shape (bs, *n, d).
            e (torch.Tensor): Nested tensor of edges representation of shape (bs, *n, *n, de).

        Returns:
            None: Implementation is incomplete.
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
        r"""
        Compute attention and update node and edge representations using FiLM modulation.

        The attention mechanism is computed as follows:
        1. Node-based query (Q) and key (K) outer product scaled by $\sqrt{d_k}$:
           $A_{ij} = (Q_i \otimes K_j) / \sqrt{d_k}$
        2. Edge-based FiLM modulation of the attention:
           $A'_{ij} = E1_{ij} + E2_{ij} \odot A_{ij} + A_{ij}$
        3. Feature updates for nodes and edges:
           - Node attention weights: $\alpha_{ij} = \text{softmax}_j(\sum_k A'_{ijk})$
           - Updated node representation (before global FiLM): $h_{attn} = \sum_j \alpha_{ij} V_j$
           - Updated edge representation (before global FiLM): $e_{attn, ij} = \text{concat}_k(A'_{ijk})$
        4. Global feature (y) modulation using FiLM:
           - $h_{final} = yN1 + yN2 \odot h_{attn} + h_{attn}$
           - $e_{final} = yE1 + yE2 \odot e_{attn} + e_{attn}$

        Args:
            h (torch.Tensor): Node representation tensor of shape (bs, n, d).
            e (torch.Tensor): Edge representation tensor of shape (bs, n, n, de).
            y (torch.Tensor): Global representation tensor of shape (bs, dy).
            mask (torch.Tensor): Binary mask for nodes of shape (bs, n).
            attn_map_mode (bool, optional): If True, returns the attention weights. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
                - If attn_map_mode is False: (node_attn, edge_attn)
                    - node_attn (torch.Tensor): Updated node representations of shape (bs, n, d).
                    - edge_attn (torch.Tensor): Updated edge representations of shape (bs, n, n, d).
                - If attn_map_mode is True: (attn, node_attn)
                    - attn (torch.Tensor): Raw attention tensor of shape (bs, nh, n, n, dk).
                    - node_attn (torch.Tensor): Softmaxed attention weights of shape (bs, nh, n, n).
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
        """
        Update global feature representation based on node and edge features.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            y (torch.Tensor): Current global features of shape (bs, dy).

        Returns:
            torch.Tensor: New global feature representation of shape (bs, dy).
        """
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
        """
        Forward pass of the MultiHeadEdgeNodeWithY layer.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            y (torch.Tensor): Global features of shape (bs, dy).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Updated (h, e, y, mask).
        """
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
            if self.y_update:
                new_y = self.update_y(h, e, y)  # (bs, dy)
                y = self.norm_y(y + self.Out_y(new_y))  # (bs, dy)

            # Do the output transformation
            h = self.norm_n(h + self.Out_N(node_attn))  # (bs, n, d)
            e = self.norm_e(e + self.Out_E(edge_attn))  # (bs, n, n, de)

            # Ensure that the masking is still correct
            h = mask_any_tensor(h, mask)  # (bs, n, d)
            e = mask_any_tensor(e, mask)  # (bs, n, n, de)

        return h, e, y, mask


class MultiHeadEdgeNode(nn.Module):
    """
    Multi-head attention layer for edge and node features without global updates.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        n_heads (int): Number of attention heads.
    """

    def __init__(
        self,
        d: int,
        de: int,
        n_heads: int,
    ):
        """
        Initialize the MultiHeadEdgeNode layer.

        Args:
            d (int): Hidden dimension of the model.
            de (int): Hidden dimension of the edges.
            n_heads (int): Number of attention heads.
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
        r"""
        Compute attention and update representations without global features.

        The mechanism follows these steps:
        1. Node outer product scaled by $\sqrt{d_k}$: $A_{ij} = (Q_i \otimes K_j) / \sqrt{d_k}$
        2. Edge FiLM modulation: $A'_{ij} = E1_{ij} + E2_{ij} \odot A_{ij} + A_{ij}$
        3. Aggregation:
           - Node update: $h_{attn} = \text{softmax}_j(\sum_k A'_{ijk}) V_j$
           - Edge update: $e_{attn, ij} = \text{concat}_k(A'_{ijk})$

        Args:
            h (torch.Tensor): Node representation tensor of shape (bs, n, d).
            e (torch.Tensor): Edge representation tensor of shape (bs, n, n, de).
            mask (torch.Tensor): Binary mask for nodes of shape (bs, n).
            attn_map_mode (bool, optional): If True, returns the attention weights. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated (h, e) tensors.
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
        """
        Forward pass of the MultiHeadEdgeNode layer.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated (h, e, mask).
        """
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
    Combined attention and feed-forward layer for the Edge-Node model with global updates.
    This layer applies MultiHeadEdgeNodeWithY followed by FfnNodeEdge.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        dy (int): Hidden dimension for global features.
        n_heads (int): Number of attention heads.
        y_update (bool, optional): Whether to update global features. Defaults to True.
    """

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        n_heads: int,
        y_update: bool = True,
    ):
        """
        Initialize the AttentionLayer.

        Args:
            d (int): Hidden dimension of the model.
            de (int): Hidden dimension of the edges.
            dy (int): Hidden dimension of the global features.
            n_heads (int): Number of attention heads.
            y_update (bool, optional): Whether to update global features. Defaults to True.
        """
        super().__init__()
        self.d = d
        self.de = de
        self.n_heads = n_heads
        self.attnEdgeNode = MultiHeadEdgeNodeWithY(d, de, dy, n_heads, y_update)
        self.ffn = FfnNodeEdge(d, de)
        self.mlpy = MLP(dy, 2 * dy, dy)
        self.norm_y = nn.LayerNorm(dy)
        self.y_update = y_update

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Forward pass of the AttentionLayer.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            y (torch.Tensor): Global features of shape (bs, dy).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Updated (h, e, y, mask).
        """
        # Compute the attention, make the residual connection
        h_attn, e_attn, y_attn, mask = self.attnEdgeNode(h, e, y, mask)

        # Compute the output of the feedforward network, make residual connections
        h, e, _ = self.ffn(h_attn, e_attn, mask)
        if self.y_update:
            y = self.norm_y(y_attn + self.mlpy(y_attn))

        return h, e, y, mask


class AttentionLayerWithoutY(nn.Module):
    """
    Combined attention and feed-forward layer for the Edge-Node model without global updates.
    This layer applies MultiHeadEdgeNode followed by FfnNodeEdge.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        n_heads (int): Number of attention heads.
    """

    def __init__(
        self,
        d: int,
        de: int,
        n_heads: int,
    ):
        """
        Initialize the AttentionLayerWithoutY.

        Args:
            d (int): Hidden dimension of the model.
            de (int): Hidden dimension of the edges.
            n_heads (int): Number of attention heads.
        """
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
        """
        Forward pass of the AttentionLayerWithoutY.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated (h, e, mask).
        """
        # Compute the attention, make the residual connection
        h_attn, e_attn, mask = self.attnEdgeNode(h, e, mask)

        # Compute the output of the feedforward network, make residual connections
        h, e, _ = self.ffn(h_attn, e_attn, mask)

        return h, e, mask

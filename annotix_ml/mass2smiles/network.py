import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger

def positional_encoding(max_position, d_model, min_freq=1e-6):
    position = np.arange(max_position)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

class BaseAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        x = self.layernorm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.seq(x)
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = BaseAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.relu(x)
        return x

class TCN(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size, dilations):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TCNBlock(in_channels if i==0 else out_channels, out_channels, kernel_size, dilations[i]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Mass2SmilesModel(nn.Module):
    def __init__(self, units, heads, dropout, dense_dropout, filters, num_layers, embed_dim, output_dim_smiles, output_dim_fg, input_dim=2):
        super().__init__()
        self.mask_value = 0  # Changed from 10 to 0 for direct encoding
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Project input features to embedding dimension
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=embed_dim, num_heads=heads, dff=units, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.tcn = TCN(num_layers=6, in_channels=embed_dim, out_channels=filters, kernel_size=8, dilations=[2 ** i for i in range(6)])
        self.dropout_fg = nn.Dropout(dense_dropout)
        self.dense_fg1 = nn.Linear(filters, 128)
        self.dense_fg2 = nn.Linear(128, output_dim_fg)
        self.dropout_smiles = nn.Dropout(dense_dropout)
        self.dense_smiles1 = nn.Linear(filters, output_dim_smiles)
        self.dense_smiles2 = nn.Linear(output_dim_smiles, output_dim_smiles)

    def forward(self, x):
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Masking (not needed for direct encoding but kept for compatibility)
        x = torch.where(x == self.mask_value, torch.zeros_like(x), x)
        
        for layer in self.encoder_layers:
            x = layer(x)

        # TCN layer
        x = self.tcn(x)

        # Take last time step (or mean over sequence)
        x = x.mean(dim=1)

        # Functional groups output
        fg = self.dropout_fg(x)
        fg = torch.tanh(self.dense_fg1(fg))
        fg = self.dropout_fg(fg)
        fg = torch.sigmoid(self.dense_fg2(fg))

        # Smiles output
        smiles = self.dropout_smiles(x)
        smiles = F.relu(self.dense_smiles1(smiles))
        smiles = self.dropout_smiles(smiles)
        smiles = self.dense_smiles2(smiles)

        return smiles, fg

def trun_n_d(n, d):
    return (  n if not n.find('.') + 1 else n[:n.find('.') + d + 1]  )

def prepro_specs_train(df):
    
    max_len_mz = max([len(mzs) + len(loss) + 1 for _, (mzs, loss) in df[['mzs', 'loss_mzs']].iterrows()])
    max_len_intensities = max([len(intensities) + len(loss) + 1 for _, (intensities, loss) in df[['intensities', 'loss_intensities']].iterrows()])
    assert max_len_mz == max_len_intensities, "Max length of mz and intensities should be the same"
    logger.info(f"Max length of mz/intensities: {max_len_mz}")

    valid=[]
    for _, (precursor_mz, mzs, intensities, loss_mzs, loss_intensities) in df[['precursor_mz', 'mzs', 'intensities', 'loss_mzs', 'loss_intensities']].iterrows():
        mz_list = [round(float(trun_n_d(str(precursor_mz), 2)) * 100)] # add precursor mz
        intes_list = [2.0] # add precursor intensity, arbitrary value > 1 to distinguish from real peaks

        # Into one dictionary and sort by mz
        res = dict(zip(mzs + loss_mzs, intensities + loss_intensities))  # order by mzs
        res = dict(sorted(res.items()))

        for mz, intensities in res.items():
            # m/z
            mz_list.append(round(float(trun_n_d(str(mz), 2)) * 100))

            # Intensities
            intes_list.append(round(intensities, 4))

        if len(mz_list) < max_len_mz:
            mz_list += [0] * (max_len_mz - len(mz_list))
        if len(intes_list) < max_len_mz:
            intes_list += [0] * (max_len_mz - len(intes_list))

        int_mzs = [intes_list, mz_list]

        valid.append(int_mzs) # put intesities at first

    return torch.nn.utils.rnn.pad_sequence([torch.tensor(v, dtype=torch.float32) for v in valid], batch_first=True)

def direct_spectral_encoding(rag_tensor, max_length=None):
    """
    Direct encoding keeping both m/z and intensity information
    Args:
        rag_tensor: tensor with shape [batch, 2, seq_len] where 2 = [intensities, mz_values]
        max_length: maximum sequence length for padding
    Returns:
        tensor with shape [batch, max_seq_len, 2] where 2 = [normalized_mz, normalized_intensity]
    """
    encoded = []
    
    if max_length is None:
        max_length = max([tensor.shape[1] for tensor in rag_tensor])
    
    for sample in rag_tensor:
        intensities = sample[0].numpy()  # First row: intensities
        mz_values = sample[1].numpy()    # Second row: m/z values
        
        # Remove zero-padded values (where both mz and intensity are 0)
        valid_indices = (intensities != 0) | (mz_values != 0)
        valid_intensities = intensities[valid_indices]
        valid_mz = mz_values[valid_indices]
        
        if len(valid_intensities) == 0:
            # Handle empty spectra
            peaks = torch.zeros((max_length, 2))
        else:
            # Normalize values
            normalized_mz = valid_mz / 1000.0  # Scale m/z to reasonable range
            normalized_intensities = valid_intensities / max(valid_intensities) if max(valid_intensities) > 0 else valid_intensities
            
            # Stack m/z and intensity as features
            peaks = torch.stack([torch.tensor(normalized_mz, dtype=torch.float32), 
                               torch.tensor(normalized_intensities, dtype=torch.float32)], dim=1)
            
            # Pad or truncate to max_length
            if len(peaks) < max_length:
                padding = torch.zeros((max_length - len(peaks), 2))
                peaks = torch.cat([peaks, padding], dim=0)
            elif len(peaks) > max_length:
                peaks = peaks[:max_length]
        
        encoded.append(peaks)
    
    return torch.stack(encoded)

class SpectrumEncoder(nn.Module):
    """
    Option 2: Learned embeddings for m/z values with intensity projection
    """
    def __init__(self, embed_dim, num_mz_bins=20000, mz_max=2000):
        super().__init__()
        self.mz_max = mz_max
        self.num_mz_bins = num_mz_bins
        self.embed_dim = embed_dim
        
        # Learn embeddings for discretized m/z values
        self.mz_embedding = nn.Embedding(num_mz_bins, embed_dim//2)
        self.intensity_projection = nn.Linear(1, embed_dim//2)
        
    def forward(self, mz, intensities):
        """
        Args:
            mz: tensor of shape [batch, seq_len] with m/z values
            intensities: tensor of shape [batch, seq_len] with intensity values
        Returns:
            embedded features of shape [batch, seq_len, embed_dim]
        """
        # Discretize m/z into bins (scale and clamp to valid range)
        mz_scaled = (mz / self.mz_max * self.num_mz_bins).clamp(0, self.num_mz_bins - 1).long()
        
        # Get m/z embeddings
        mz_emb = self.mz_embedding(mz_scaled)
        
        # Project intensities
        int_emb = self.intensity_projection(intensities.unsqueeze(-1))
        
        # Concatenate m/z and intensity embeddings
        return torch.cat([mz_emb, int_emb], dim=-1)

def learned_spectral_encoding(rag_tensor, max_length=None):
    """
    Preprocessing for learned embeddings approach
    Args:
        rag_tensor: tensor with shape [batch, 2, seq_len] where 2 = [intensities, mz_values]
        max_length: maximum sequence length for padding
    Returns:
        tuple (mz_tensor, intensity_tensor) both with shape [batch, max_seq_len]
    """
    mz_list = []
    intensity_list = []
    
    if max_length is None:
        max_length = max([tensor.shape[1] for tensor in rag_tensor])
    
    for sample in rag_tensor:
        intensities = sample[0].numpy()  # First row: intensities
        mz_values = sample[1].numpy()    # Second row: m/z values
        
        # Remove zero-padded values (where both mz and intensity are 0)
        valid_indices = (intensities != 0) | (mz_values != 0)
        valid_intensities = intensities[valid_indices]
        valid_mz = mz_values[valid_indices]
        
        # Normalize intensities to [0, 1]
        if len(valid_intensities) > 0 and max(valid_intensities) > 0:
            valid_intensities = valid_intensities / max(valid_intensities)
        
        # Pad or truncate to max_length
        if len(valid_mz) < max_length:
            # Pad with zeros
            padded_mz = np.zeros(max_length)
            padded_intensities = np.zeros(max_length)
            padded_mz[:len(valid_mz)] = valid_mz
            padded_intensities[:len(valid_intensities)] = valid_intensities
        else:
            # Truncate
            padded_mz = valid_mz[:max_length]
            padded_intensities = valid_intensities[:max_length]
        
        mz_list.append(torch.tensor(padded_mz, dtype=torch.float32))
        intensity_list.append(torch.tensor(padded_intensities, dtype=torch.float32))
    
    return torch.stack(mz_list), torch.stack(intensity_list)

class Mass2SmilesModelV2(nn.Module):
    """
    Model using learned embeddings for m/z values (Option 2)
    """
    def __init__(self, units, heads, dropout, dense_dropout, filters, num_layers, embed_dim, 
                 output_dim_smiles, output_dim_fg, num_mz_bins=20000, mz_max=2000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Spectrum encoder with learned embeddings
        self.spectrum_encoder = SpectrumEncoder(embed_dim, num_mz_bins, mz_max)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=embed_dim, num_heads=heads, dff=units, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.tcn = TCN(num_layers=6, in_channels=embed_dim, out_channels=filters, kernel_size=8, 
                       dilations=[2 ** i for i in range(6)])
        
        self.dropout_fg = nn.Dropout(dense_dropout)
        self.dense_fg1 = nn.Linear(filters, 128)
        self.dense_fg2 = nn.Linear(128, output_dim_fg)
        self.dropout_smiles = nn.Dropout(dense_dropout)
        self.dense_smiles1 = nn.Linear(filters, output_dim_smiles)
        self.dense_smiles2 = nn.Linear(output_dim_smiles, output_dim_smiles)

    def forward(self, mz, intensities):
        # Encode spectra using learned embeddings
        x = self.spectrum_encoder(mz, intensities)
        
        # Transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        # TCN layer
        x = self.tcn(x)

        # Take mean over sequence (global pooling)
        x = x.mean(dim=1)

        # Functional groups output
        fg = self.dropout_fg(x)
        fg = torch.tanh(self.dense_fg1(fg))
        fg = self.dropout_fg(fg)
        fg = torch.sigmoid(self.dense_fg2(fg))

        # Smiles output
        smiles = self.dropout_smiles(x)
        smiles = F.relu(self.dense_smiles1(smiles))
        smiles = self.dropout_smiles(smiles)
        smiles = self.dense_smiles2(smiles)

        return smiles, fg

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
    def __init__(self, units, heads, dropout, dense_dropout, filters, num_layers, embed_dim):
        super().__init__()
        self.mask_value = 10
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=embed_dim, num_heads=heads, dff=units, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.tcn = TCN(num_layers=6, in_channels=embed_dim, out_channels=filters, kernel_size=8, dilations=[2 ** i for i in range(6)])
        self.dropout_fg = nn.Dropout(dense_dropout)
        self.dense_fg1 = nn.Linear(filters, 128)
        self.dense_fg2 = nn.Linear(128, 71)
        self.dropout_smiles = nn.Dropout(dense_dropout)
        self.dense_smiles1 = nn.Linear(filters, 512)
        self.dense_smiles2 = nn.Linear(512, 512)

    def forward(self, x):
        # Masking
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

def encoding(rag_tensor, positional_encoding, dimn):
    to_pad=[]
    for sample in rag_tensor:
        all_dim = []
        pos_enc = [positional_encoding[int(i)-1] for i in sample[1].numpy().tolist()]
        for dim in range(dimn):
            dim_n = [i[dim] for i in pos_enc]
            all_dim.append(dim_n)
        to_pad.append(all_dim)

    # pad to maxlen=501 along the sequence dimension
    to_pad = [torch.nn.functional.pad(torch.tensor(i, dtype=torch.float32), (0, 501 - len(i[0]), 0, 0), value=10) \
              if len(i[0]) < 501 else torch.tensor(i, dtype=torch.float32)[:, :501] for i in to_pad]

    to_pad = np.swapaxes(np.stack((to_pad)), 1, -1)

    return torch.tensor(to_pad)

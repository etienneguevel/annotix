import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import pickle
from pathlib import Path

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

# SMILES Tokenization and Dataset Classes
class SMILESTokenizer:
    """Simple SMILES tokenizer"""
    
    def __init__(self):
        # Common SMILES tokens - extend as needed
        self.regex_pattern = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
        self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        
    def tokenize(self, smiles):
        """Tokenize a SMILES string"""
        if pd.isna(smiles) or smiles == '***':  # Handle missing or unknown SMILES
            return ['<UNK>']
        tokens = re.findall(self.regex_pattern, smiles)
        return ['<START>'] + tokens + ['<END>']
    
    def build_vocab(self, smiles_list, min_freq=1):
        """Build vocabulary from list of SMILES strings"""
        token_counts = {}
        
        for smiles in smiles_list:
            if pd.isna(smiles) or smiles == '***':
                continue
            tokens = self.tokenize(smiles)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Create vocab with special tokens first
        vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # Add frequent tokens
        for token, count in token_counts.items():
            if count >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.token_to_idx = vocab
        self.idx_to_token = {idx: token for token, idx in vocab.items()}
        
        return vocab
    
    def encode(self, smiles, max_length=None):
        """Convert SMILES to token indices"""
        tokens = self.tokenize(smiles)
        indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) for token in tokens]
        
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.token_to_idx['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def decode(self, indices):
        """Convert token indices back to SMILES"""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        # Remove special tokens and pad tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<START>', '<END>']]
        return ''.join(tokens)

def parse_peaks_csv(peaks_str):
    """Parse peaks_list from CSV format to arrays"""
    if pd.isna(peaks_str) or not isinstance(peaks_str, str):
        return np.array([]), np.array([])
    
    peaks = []
    for line in peaks_str.split('\\n'):
        if line.strip():
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    peaks.append([mz, intensity])
                except ValueError:
                    continue
    
    if not peaks:
        return np.array([]), np.array([])
    
    peaks = np.array(peaks)
    return peaks[:, 0], peaks[:, 1]  # mz, intensity

class SpectrumSMILESDataset(Dataset):
    """Dataset for spectrum -> SMILES prediction"""
    
    def __init__(self, csv_path, tokenizer, max_spectrum_length=500, max_smiles_length=150, model_type='direct'):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_spectrum_length = max_spectrum_length
        self.max_smiles_length = max_smiles_length
        self.model_type = model_type  # 'direct' or 'learned'
        
        # Filter out invalid entries
        self.data = self.data.dropna(subset=['peaks_list', 'smiles'])
        self.data = self.data[self.data['smiles'] != '***'].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Parse spectrum
        mz_values, intensities = parse_peaks_csv(row['peaks_list'])
        
        if len(mz_values) == 0:
            # Handle empty spectra with dummy data
            mz_values = np.array([0.0])
            intensities = np.array([0.0])
        
        # Normalize and pad/truncate
        if len(mz_values) > self.max_spectrum_length:
            mz_values = mz_values[:self.max_spectrum_length]
            intensities = intensities[:self.max_spectrum_length]
        
        # Normalize intensities
        if np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)
        
        # Pad to max length
        mz_padded = np.zeros(self.max_spectrum_length)
        int_padded = np.zeros(self.max_spectrum_length)
        mz_padded[:len(mz_values)] = mz_values
        int_padded[:len(intensities)] = intensities
        
        # Prepare spectrum input based on model type
        if self.model_type == 'direct':
            # For direct encoding: [batch, seq_len, 2]
            spectrum_input = np.stack([mz_padded / 1000.0, int_padded], axis=1)  # Scale m/z
            spectrum_tensor = torch.tensor(spectrum_input, dtype=torch.float32)
        else:  # learned
            # For learned encoding: separate tensors
            spectrum_tensor = (torch.tensor(mz_padded, dtype=torch.float32), 
                             torch.tensor(int_padded, dtype=torch.float32))
        
        # Encode SMILES
        smiles_encoded = self.tokenizer.encode(row['smiles'], max_length=self.max_smiles_length)
        smiles_tensor = torch.tensor(smiles_encoded, dtype=torch.long)
        
        return spectrum_tensor, smiles_tensor

def custom_collate_fn(batch):
    """Custom collate function for learned embedding model"""
    spectra_mz = []
    spectra_int = []
    smiles = []
    
    for spectrum_tensors, smiles_tensor in batch:
        mz_tensor, int_tensor = spectrum_tensors
        spectra_mz.append(mz_tensor)
        spectra_int.append(int_tensor)
        smiles.append(smiles_tensor)
    
    return (torch.stack(spectra_mz), torch.stack(spectra_int)), torch.stack(smiles)

def prepare_data_loaders(csv_path, batch_size=16, test_size=0.2, max_spectrum_length=500, 
                        max_smiles_length=150, model_type='direct'):
    """
    Prepare data loaders for training
    
    Args:
        csv_path: Path to CSV file with spectrum and SMILES data
        batch_size: Batch size for training
        test_size: Fraction of data to use for validation
        max_spectrum_length: Maximum number of peaks per spectrum
        max_smiles_length: Maximum SMILES token length
        model_type: 'direct' or 'learned'
    
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    
    # Read data and build tokenizer
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=['peaks_list', 'smiles'])
    data = data[data['smiles'] != '***']
    
    # Build SMILES tokenizer
    tokenizer = SMILESTokenizer()
    vocab = tokenizer.build_vocab(data['smiles'].tolist(), min_freq=1)
    logger.info(f'Built SMILES vocabulary with {len(vocab)} tokens')
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    # Save train/val splits temporarily
    train_path = '/tmp/train_data.csv'
    val_path = '/tmp/val_data.csv'
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    
    # Create datasets
    train_dataset = SpectrumSMILESDataset(train_path, tokenizer, max_spectrum_length, 
                                         max_smiles_length, model_type)
    val_dataset = SpectrumSMILESDataset(val_path, tokenizer, max_spectrum_length, 
                                       max_smiles_length, model_type)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=None if model_type == 'direct' else custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=None if model_type == 'direct' else custom_collate_fn)
    
    logger.info(f'Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    
    return train_loader, val_loader, tokenizer

def train_mass2smiles(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, 
                     device='cpu', save_path=None, model_type='direct'):
    """
    Universal training function for both Mass2SmilesModel and Mass2SmilesModelV2
    
    Args:
        model: Either Mass2SmilesModel or Mass2SmilesModelV2 instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save model checkpoints
        model_type: 'direct' for Mass2SmilesModel, 'learned' for Mass2SmilesModelV2
    """
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (spectrum, smiles_target) in enumerate(train_pbar):
            
            if model_type == 'direct':
                spectrum = spectrum.to(device)
            else:  # learned
                spectrum = (spectrum[0].to(device), spectrum[1].to(device))
            
            smiles_target = smiles_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            #TODO fg not used in loss calculation here
            try:
                if model_type == 'direct':
                    smiles_pred, fg_pred = model(spectrum)
                else:  # learned
                    smiles_pred, fg_pred = model(spectrum[0], spectrum[1])
                
                # For sequence prediction, we need to handle the target properly
                #TODO Here we'll use a simplified approach focusing on the next token prediction
                #TODO In practice, you'd want more sophisticated sequence-to-sequence training
                
                # Simple approach: predict first token of SMILES (can be extended)
                target_first_token = smiles_target[:, 1]  # Skip START token, predict first real token
                loss = criterion(smiles_pred, target_first_token)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (spectrum, smiles_target) in enumerate(val_loader):
                try:
                    if model_type == 'direct':
                        spectrum = spectrum.to(device)
                    else:  # learned
                        spectrum = (spectrum[0].to(device), spectrum[1].to(device))
                    
                    smiles_target = smiles_target.to(device)
                    
                    if model_type == 'direct':
                        smiles_pred, fg_pred = model(spectrum)
                    else:  # learned
                        smiles_pred, fg_pred = model(spectrum[0], spectrum[1])
                    
                    target_first_token = smiles_target[:, 1]
                    loss = criterion(smiles_pred, target_first_token)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_losses.append(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint
        if save_path and (epoch + 1) % 5 == 0:
            checkpoint_path = Path(save_path) / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    if save_path:
        final_path = Path(save_path) / 'final_model.pth'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, final_path)
        logger.info(f'Final model saved: {final_path}')
    
    return train_losses, val_losses

# Sequence-to-Sequence Components
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer decoder"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SMILESDecoder(nn.Module):
    """Transformer decoder for SMILES sequence generation"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_length=150):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, memory, tgt_tokens=None, tgt_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for decoder
        
        Args:
            memory: Encoded spectrum features [batch, memory_seq_len, d_model]
            tgt_tokens: Target tokens [batch, tgt_seq_len] (for teacher forcing)
            tgt_mask: Target mask to prevent attention to future tokens
            memory_key_padding_mask: Mask for padded spectrum features
        
        Returns:
            logits: [batch, tgt_seq_len, vocab_size]
        """
        if tgt_tokens is None:
            raise ValueError("tgt_tokens required for forward pass")
        
        # Embed target tokens
        tgt_embedded = self.token_embedding(tgt_tokens) * np.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt_embedded.transpose(0, 1),
            memory=memory.transpose(0, 1),
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        ).transpose(0, 1)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate_sequence(self, memory, tokenizer, max_length=None, temperature=1.0, 
                         memory_key_padding_mask=None):
        """
        Generate SMILES sequence autoregressively
        
        Args:
            memory: Encoded spectrum features [batch, seq_len, d_model]
            tokenizer: SMILES tokenizer with vocab
            max_length: Maximum sequence length
            temperature: Sampling temperature
            memory_key_padding_mask: Mask for memory padding
        
        Returns:
            generated_tokens: [batch, seq_len]
        """
        if max_length is None:
            max_length = self.max_length
        
        batch_size = memory.size(0)
        device = memory.device
        
        # Initialize with START token
        start_token = tokenizer.token_to_idx['<START>']
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            # Create causal mask
            tgt_len = generated.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
            
            # Forward pass
            logits = self.forward(
                memory=memory,
                tgt_tokens=generated,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for END token
            end_token = tokenizer.token_to_idx['<END>']
            newly_finished = (next_token.squeeze(1) == end_token)
            finished = finished | newly_finished
            
            # Stop if all sequences finished
            if finished.all():
                break
        
        return generated

def create_causal_mask(size, device):
    """Create causal mask for transformer decoder"""
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

def create_padding_mask(tokens, pad_token_id=0):
    """Create padding mask for sequences"""
    return (tokens == pad_token_id)

# Enhanced Sequence-to-Sequence Models
class Mass2SmilesSeq2SeqDirect(nn.Module):
    """
    Sequence-to-sequence model with direct spectral encoding (enhanced Option 1)
    """
    
    def __init__(self, units, heads, dropout, dense_dropout, filters, num_layers, embed_dim, 
                 vocab_size, output_dim_fg, input_dim=2, max_smiles_length=150, decoder_layers=6):
        super().__init__()
        self.mask_value = 0
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_smiles_length = max_smiles_length
        
        # Spectrum encoder (same as before)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=embed_dim, num_heads=heads, dff=units, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.tcn = TCN(num_layers=6, in_channels=embed_dim, out_channels=filters, kernel_size=8, 
                       dilations=[2 ** i for i in range(6)])
        
        # Spectrum features to decoder dimension
        self.memory_projection = nn.Linear(filters, embed_dim)
        
        # SMILES sequence decoder
        self.smiles_decoder = SMILESDecoder(
            vocab_size=vocab_size,
            d_model=embed_dim,
            nhead=heads,
            num_layers=decoder_layers,
            max_length=max_smiles_length
        )
        
        # Functional groups branch
        self.dropout_fg = nn.Dropout(dense_dropout)
        self.dense_fg1 = nn.Linear(filters, 128)
        self.dense_fg2 = nn.Linear(128, output_dim_fg)
    
    def encode_spectrum(self, x):
        """Encode spectrum to memory representation"""
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Masking
        x = torch.where(x == self.mask_value, torch.zeros_like(x), x)
        
        # Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # TCN layer
        encoded = self.tcn(x)  # [batch, seq_len, filters]
        
        # Project to decoder dimension
        memory = self.memory_projection(encoded)  # [batch, seq_len, embed_dim]
        
        return encoded, memory
    
    def forward(self, spectrum, smiles_target=None, teacher_forcing_ratio=1.0):
        """
        Forward pass with optional teacher forcing
        
        Args:
            spectrum: Input spectrum [batch, seq_len, input_dim]
            smiles_target: Target SMILES tokens [batch, target_len] (for teacher forcing)
            teacher_forcing_ratio: Probability of using teacher forcing (training only)
        
        Returns:
            smiles_logits: [batch, target_len, vocab_size]
            fg_pred: [batch, output_dim_fg]
        """
        # Encode spectrum
        encoded_spectrum, memory = self.encode_spectrum(spectrum)
        
        # Functional groups prediction (from global representation)
        fg_features = encoded_spectrum.mean(dim=1)  # Global pooling
        fg = self.dropout_fg(fg_features)
        fg = torch.tanh(self.dense_fg1(fg))
        fg = self.dropout_fg(fg)
        fg_pred = torch.sigmoid(self.dense_fg2(fg))
        
        # SMILES sequence generation
        if self.training and smiles_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: use ground truth as input
            # Shift target tokens: input = <START> + tokens[:-1], target = tokens[1:] + <END>
            decoder_input = smiles_target[:, :-1]  # Remove last token
            tgt_mask = create_causal_mask(decoder_input.size(1), decoder_input.device)
            smiles_logits = self.smiles_decoder(
                memory=memory,
                tgt_tokens=decoder_input,
                tgt_mask=tgt_mask
            )
        else:
            # Inference mode or no teacher forcing - would need tokenizer for generation
            # For now, return dummy logits matching expected shape
            if smiles_target is not None:
                batch_size, target_len = smiles_target.shape
                smiles_logits = torch.zeros(batch_size, target_len-1, self.smiles_decoder.vocab_size, 
                                          device=spectrum.device)
            else:
                # This case would require tokenizer for autoregressive generation
                raise ValueError("Need smiles_target or tokenizer for inference")
        
        return smiles_logits, fg_pred
    
    def generate(self, spectrum, tokenizer, max_length=None, temperature=1.0):
        """Generate SMILES sequence autoregressively"""
        self.eval()
        with torch.no_grad():
            # Encode spectrum
            _, memory = self.encode_spectrum(spectrum)
            
            # Generate sequence
            generated_tokens = self.smiles_decoder.generate_sequence(
                memory=memory,
                tokenizer=tokenizer,
                max_length=max_length or self.max_smiles_length,
                temperature=temperature
            )
            
            return generated_tokens

class Mass2SmilesSeq2SeqLearned(nn.Module):
    """
    Sequence-to-sequence model with learned embeddings (enhanced Option 2)
    """
    
    def __init__(self, units, heads, dropout, dense_dropout, filters, num_layers, embed_dim,
                 vocab_size, output_dim_fg, num_mz_bins=20000, mz_max=2000, 
                 max_smiles_length=150, decoder_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_smiles_length = max_smiles_length
        
        # Spectrum encoder with learned embeddings
        self.spectrum_encoder = SpectrumEncoder(embed_dim, num_mz_bins, mz_max)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=embed_dim, num_heads=heads, dff=units, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.tcn = TCN(num_layers=6, in_channels=embed_dim, out_channels=filters, kernel_size=8,
                       dilations=[2 ** i for i in range(6)])
        
        # Spectrum features to decoder dimension
        self.memory_projection = nn.Linear(filters, embed_dim)
        
        # SMILES sequence decoder
        self.smiles_decoder = SMILESDecoder(
            vocab_size=vocab_size,
            d_model=embed_dim,
            nhead=heads,
            num_layers=decoder_layers,
            max_length=max_smiles_length
        )
        
        # Functional groups branch
        self.dropout_fg = nn.Dropout(dense_dropout)
        self.dense_fg1 = nn.Linear(filters, 128)
        self.dense_fg2 = nn.Linear(128, output_dim_fg)
    
    def encode_spectrum(self, mz, intensities):
        """Encode spectrum to memory representation"""
        # Encode spectra using learned embeddings
        x = self.spectrum_encoder(mz, intensities)
        
        # Transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # TCN layer
        encoded = self.tcn(x)  # [batch, seq_len, filters]
        
        # Project to decoder dimension
        memory = self.memory_projection(encoded)  # [batch, seq_len, embed_dim]
        
        return encoded, memory
    
    def forward(self, mz, intensities, smiles_target=None, teacher_forcing_ratio=1.0):
        """
        Forward pass with optional teacher forcing
        
        Args:
            mz: m/z values [batch, seq_len]
            intensities: Intensity values [batch, seq_len]
            smiles_target: Target SMILES tokens [batch, target_len] (for teacher forcing)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            smiles_logits: [batch, target_len, vocab_size]
            fg_pred: [batch, output_dim_fg]
        """
        # Encode spectrum
        encoded_spectrum, memory = self.encode_spectrum(mz, intensities)
        
        # Functional groups prediction
        fg_features = encoded_spectrum.mean(dim=1)  # Global pooling
        fg = self.dropout_fg(fg_features)
        fg = torch.tanh(self.dense_fg1(fg))
        fg = self.dropout_fg(fg)
        fg_pred = torch.sigmoid(self.dense_fg2(fg))
        
        # SMILES sequence generation
        if self.training and smiles_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing
            decoder_input = smiles_target[:, :-1]
            tgt_mask = create_causal_mask(decoder_input.size(1), decoder_input.device)
            smiles_logits = self.smiles_decoder(
                memory=memory,
                tgt_tokens=decoder_input,
                tgt_mask=tgt_mask
            )
        else:
            # Inference mode
            if smiles_target is not None:
                batch_size, target_len = smiles_target.shape
                smiles_logits = torch.zeros(batch_size, target_len-1, self.smiles_decoder.vocab_size,
                                          device=mz.device)
            else:
                raise ValueError("Need smiles_target or tokenizer for inference")
        
        return smiles_logits, fg_pred
    
    def generate(self, mz, intensities, tokenizer, max_length=None, temperature=1.0):
        """Generate SMILES sequence autoregressively"""
        self.eval()
        with torch.no_grad():
            # Encode spectrum
            _, memory = self.encode_spectrum(mz, intensities)
            
            # Generate sequence
            generated_tokens = self.smiles_decoder.generate_sequence(
                memory=memory,
                tokenizer=tokenizer,
                max_length=max_length or self.max_smiles_length,
                temperature=temperature
            )
            
            return generated_tokens

# Enhanced Dataset for Sequence-to-Sequence Training
class SpectrumSMILESSeq2SeqDataset(Dataset):
    """Enhanced dataset for sequence-to-sequence SMILES prediction"""
    
    def __init__(self, csv_path, tokenizer, max_spectrum_length=500, max_smiles_length=150, 
                 model_type='direct', fg_columns=None):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_spectrum_length = max_spectrum_length
        self.max_smiles_length = max_smiles_length
        self.model_type = model_type
        self.fg_columns = fg_columns or []
        
        # Filter out invalid entries
        self.data = self.data.dropna(subset=['peaks_list', 'smiles'])
        self.data = self.data[self.data['smiles'] != '***'].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Parse spectrum
        mz_values, intensities = parse_peaks_csv(row['peaks_list'])
        
        if len(mz_values) == 0:
            mz_values = np.array([0.0])
            intensities = np.array([0.0])
        
        # Normalize and pad/truncate
        if len(mz_values) > self.max_spectrum_length:
            mz_values = mz_values[:self.max_spectrum_length]
            intensities = intensities[:self.max_spectrum_length]
        
        # Normalize intensities
        if np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)
        
        # Pad to max length
        mz_padded = np.zeros(self.max_spectrum_length)
        int_padded = np.zeros(self.max_spectrum_length)
        mz_padded[:len(mz_values)] = mz_values
        int_padded[:len(intensities)] = intensities
        
        # Prepare spectrum input
        if self.model_type == 'direct':
            spectrum_input = np.stack([mz_padded / 1000.0, int_padded], axis=1)
            spectrum_tensor = torch.tensor(spectrum_input, dtype=torch.float32)
        else:  # learned
            spectrum_tensor = (torch.tensor(mz_padded, dtype=torch.float32), 
                             torch.tensor(int_padded, dtype=torch.float32))
        
        # Encode SMILES sequence (full sequence, not just first token)
        smiles_encoded = self.tokenizer.encode(row['smiles'], max_length=self.max_smiles_length)
        smiles_tensor = torch.tensor(smiles_encoded, dtype=torch.long)
        
        # Functional groups (dummy for now - you can extend this)
        if self.fg_columns:
            fg_values = [row[col] if col in row and not pd.isna(row[col]) else 0.0 for col in self.fg_columns]
            fg_tensor = torch.tensor(fg_values, dtype=torch.float32)
        else:
            # Create dummy functional groups
            fg_tensor = torch.zeros(71, dtype=torch.float32)  # 71 is common FG count
        
        return spectrum_tensor, smiles_tensor, fg_tensor

def combined_loss(smiles_logits, smiles_target, fg_pred, fg_target, 
                 smiles_weight=1.0, fg_weight=0.1, ignore_index=0):
    """
    Combined loss for SMILES sequence prediction and functional groups
    
    Args:
        smiles_logits: [batch, seq_len, vocab_size]
        smiles_target: [batch, seq_len] 
        fg_pred: [batch, num_fg]
        fg_target: [batch, num_fg]
        smiles_weight: Weight for SMILES loss
        fg_weight: Weight for functional group loss
        ignore_index: Index to ignore in SMILES loss (padding)
    
    Returns:
        total_loss, smiles_loss, fg_loss
    """
    # SMILES sequence loss (cross-entropy)
    # Target should be shifted: predict tokens[1:] given tokens[:-1]
    smiles_target_shifted = smiles_target[:, 1:]  # Remove <START> token
    
    # Reshape for loss computation
    smiles_logits_flat = smiles_logits.reshape(-1, smiles_logits.size(-1))
    smiles_target_flat = smiles_target_shifted.reshape(-1)
    
    smiles_loss = F.cross_entropy(smiles_logits_flat, smiles_target_flat, ignore_index=ignore_index)
    
    # Functional groups loss (binary cross-entropy)
    fg_loss = F.binary_cross_entropy_with_logits(fg_pred, fg_target)
    
    # Combined loss
    total_loss = smiles_weight * smiles_loss + fg_weight * fg_loss
    
    return total_loss, smiles_loss, fg_loss

def prepare_seq2seq_data_loaders(csv_path, batch_size=16, test_size=0.2, max_spectrum_length=500,
                                max_smiles_length=150, model_type='direct', fg_columns=None):
    """
    Prepare data loaders for sequence-to-sequence training
    
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    # Read data and build tokenizer
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=['peaks_list', 'smiles'])
    data = data[data['smiles'] != '***']
    
    # Build SMILES tokenizer
    tokenizer = SMILESTokenizer()
    vocab = tokenizer.build_vocab(data['smiles'].tolist(), min_freq=1)
    logger.info(f'Built SMILES vocabulary with {len(vocab)} tokens')
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    # Save train/val splits temporarily
    train_path = '/tmp/train_seq2seq_data.csv'
    val_path = '/tmp/val_seq2seq_data.csv'
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    
    # Create datasets
    train_dataset = SpectrumSMILESSeq2SeqDataset(
        train_path, tokenizer, max_spectrum_length, max_smiles_length, model_type, fg_columns
    )
    val_dataset = SpectrumSMILESSeq2SeqDataset(
        val_path, tokenizer, max_spectrum_length, max_smiles_length, model_type, fg_columns
    )
    
    # Custom collate function
    def seq2seq_collate_fn(batch):
        if model_type == 'direct':
            spectra, smiles, fgs = zip(*batch)
            return torch.stack(spectra), torch.stack(smiles), torch.stack(fgs)
        else:  # learned
            spectrum_pairs, smiles, fgs = zip(*batch)
            mz_tensors, int_tensors = zip(*spectrum_pairs)
            return (torch.stack(mz_tensors), torch.stack(int_tensors)), torch.stack(smiles), torch.stack(fgs)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=seq2seq_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=seq2seq_collate_fn)
    
    logger.info(f'Created seq2seq data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    
    return train_loader, val_loader, tokenizer

def train_seq2seq_mass2smiles(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4,
                             device='cpu', save_path=None, model_type='direct', 
                             teacher_forcing_ratio=1.0, smiles_weight=1.0, fg_weight=0.1):
    """
    Training function for sequence-to-sequence mass2smiles models with combined loss
    
    Args:
        model: Mass2SmilesSeq2SeqDirect or Mass2SmilesSeq2SeqLearned instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save checkpoints
        model_type: 'direct' or 'learned'
        teacher_forcing_ratio: Probability of using teacher forcing
        smiles_weight: Weight for SMILES loss in combined loss
        fg_weight: Weight for functional group loss in combined loss
    """
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_smiles_losses = []
    train_fg_losses = []
    val_smiles_losses = []
    val_fg_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_smiles_loss = 0.0
        train_fg_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (spectrum, smiles_target, fg_target) in enumerate(train_pbar):
            
            # Move to device
            if model_type == 'direct':
                spectrum = spectrum.to(device)
            else:  # learned
                spectrum = (spectrum[0].to(device), spectrum[1].to(device))
            
            smiles_target = smiles_target.to(device)
            fg_target = fg_target.to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                if model_type == 'direct':
                    smiles_logits, fg_pred = model(spectrum, smiles_target, teacher_forcing_ratio)
                else:  # learned
                    smiles_logits, fg_pred = model(spectrum[0], spectrum[1], smiles_target, teacher_forcing_ratio)
                
                # Compute combined loss
                total_loss, s_loss, f_loss = combined_loss(
                    smiles_logits, smiles_target, fg_pred, fg_target,
                    smiles_weight=smiles_weight, fg_weight=fg_weight
                )
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_smiles_loss += s_loss.item()
                train_fg_loss += f_loss.item()
                train_batches += 1
                
                train_pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'smiles': f'{s_loss.item():.4f}',
                    'fg': f'{f_loss.item():.4f}'
                })
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Average training losses
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_train_smiles_loss = train_smiles_loss / train_batches if train_batches > 0 else 0
        avg_train_fg_loss = train_fg_loss / train_batches if train_batches > 0 else 0
        
        train_losses.append(avg_train_loss)
        train_smiles_losses.append(avg_train_smiles_loss)
        train_fg_losses.append(avg_train_fg_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_smiles_loss = 0.0
        val_fg_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, (spectrum, smiles_target, fg_target) in enumerate(val_pbar):
                try:
                    # Move to device
                    if model_type == 'direct':
                        spectrum = spectrum.to(device)
                    else:  # learned
                        spectrum = (spectrum[0].to(device), spectrum[1].to(device))
                    
                    smiles_target = smiles_target.to(device)
                    fg_target = fg_target.to(device)
                    
                    # Forward pass (always use teacher forcing for validation)
                    if model_type == 'direct':
                        smiles_logits, fg_pred = model(spectrum, smiles_target, teacher_forcing_ratio=1.0)
                    else:  # learned
                        smiles_logits, fg_pred = model(spectrum[0], spectrum[1], smiles_target, teacher_forcing_ratio=1.0)
                    
                    # Compute combined loss
                    total_loss, s_loss, f_loss = combined_loss(
                        smiles_logits, smiles_target, fg_pred, fg_target,
                        smiles_weight=smiles_weight, fg_weight=fg_weight
                    )
                    
                    val_loss += total_loss.item()
                    val_smiles_loss += s_loss.item()
                    val_fg_loss += f_loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({
                        'total': f'{total_loss.item():.4f}',
                        'smiles': f'{s_loss.item():.4f}',
                        'fg': f'{f_loss.item():.4f}'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Average validation losses
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_smiles_loss = val_smiles_loss / val_batches if val_batches > 0 else 0
        avg_val_fg_loss = val_fg_loss / val_batches if val_batches > 0 else 0
        
        val_losses.append(avg_val_loss)
        val_smiles_losses.append(avg_val_smiles_loss)
        val_fg_losses.append(avg_val_fg_loss)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train - Total: {avg_train_loss:.4f}, SMILES: {avg_train_smiles_loss:.4f}, FG: {avg_train_fg_loss:.4f}')
        logger.info(f'  Val   - Total: {avg_val_loss:.4f}, SMILES: {avg_val_smiles_loss:.4f}, FG: {avg_val_fg_loss:.4f}')
        
        # Save checkpoint
        if save_path and (epoch + 1) % 5 == 0:
            checkpoint_path = Path(save_path) / f'seq2seq_checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_smiles_loss': avg_train_smiles_loss,
                'train_fg_loss': avg_train_fg_loss,
                'val_smiles_loss': avg_val_smiles_loss,
                'val_fg_loss': avg_val_fg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    if save_path:
        final_path = Path(save_path) / 'seq2seq_final_model.pth'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_smiles_losses': train_smiles_losses,
            'train_fg_losses': train_fg_losses,
            'val_smiles_losses': val_smiles_losses,
            'val_fg_losses': val_fg_losses,
        }, final_path)
        logger.info(f'Final model saved: {final_path}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'train_smiles_losses': train_smiles_losses,
        'train_fg_losses': train_fg_losses,
        'val_smiles_losses': val_smiles_losses,
        'val_fg_losses': val_fg_losses
    }

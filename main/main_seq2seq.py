#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import argparse

# Add the annotix_ml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'annotix_ml'))

from mass2smiles.network import (
    Mass2SmilesSeq2SeqDirect, 
    Mass2SmilesSeq2SeqLearned,
    train_seq2seq_mass2smiles,
    SpectrumSMILESSeq2SeqDataset,
    SMILESTokenizer
)
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train Mass2SMILES Seq2Seq Models')
    parser.add_argument('--model', type=str, choices=['direct', 'learned'], default='direct',
                        help='Model type: direct or learned')
    parser.add_argument('--csv_path', type=str, default='main/test_data/test-data.csv',
                        help='Path to CSV file with peaks_list and smiles columns')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--max_length', type=int, default=150,
                        help='Maximum SMILES sequence length')
    parser.add_argument('--test_generation', action='store_true',
                        help='Test sequence generation after training')
    
    args = parser.parse_args()
    
    print(f"Training {args.model} Mass2SMILES Seq2Seq model...")
    print(f"CSV path: {args.csv_path}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Load data
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples from CSV")
    
    # Initialize tokenizer
    tokenizer = SMILESTokenizer()
    
    # Build vocabulary from all SMILES in dataset
    all_smiles = df['smiles'].tolist()
    tokenizer.build_vocab(all_smiles)
    vocab_size = tokenizer.vocab_size
    print(f"Built vocabulary with {vocab_size} tokens")
    
    # Create dataset and dataloader
    dataset = SpectrumSMILESSeq2SeqDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        max_smiles_length=args.max_length,
        model_type=args.model
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    if args.model == 'direct':
        model = Mass2SmilesSeq2SeqDirect(
            units=128,
            heads=args.heads,
            dropout=0.1,
            dense_dropout=0.1,
            filters=64,
            num_layers=6,
            embed_dim=args.embed_dim,
            vocab_size=vocab_size,
            output_dim_fg=200,
            max_smiles_length=args.max_length
        )
    else:  # learned
        model = Mass2SmilesSeq2SeqLearned(
            units=128,
            heads=args.heads,
            dropout=0.1,
            dense_dropout=0.1,
            filters=64,
            num_layers=6,
            embed_dim=args.embed_dim,
            vocab_size=vocab_size,
            output_dim_fg=200,
            max_smiles_length=args.max_length
        )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test model with one batch
    print("\nTesting model with sample batch...")
    sample_batch = next(iter(train_loader))
    
    if args.model == 'direct':
        spectra, target_tokens, target_fg = sample_batch
        print(f"Input spectra shape: {spectra.shape}")
    else:  # learned
        spectrum_input, target_tokens, target_fg = sample_batch
        mz_values, intensities = spectrum_input
        print(f"Input m/z shape: {mz_values.shape}, intensities shape: {intensities.shape}")
    
    print(f"Target tokens shape: {target_tokens.shape}")
    print(f"Target functional groups shape: {target_fg.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if args.model == 'direct':
            smiles_logits, fg_pred = model(spectra, target_tokens[:, :-1])
        else:
            smiles_logits, fg_pred = model(mz_values, intensities, target_tokens[:, :-1])
        
        print(f"SMILES logits shape: {smiles_logits.shape}")
        print(f"FG predictions shape: {fg_pred.shape}")
    
    print("✓ Model forward pass successful!")
    
    # Train the model
    print(f"\nStarting training for {args.epochs} epochs...")
    
    trained_model, train_losses, val_losses = train_seq2seq_mass2smiles(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device='cpu',  # Use CPU for testing
        model_type=args.model
    )
    
    print("Training completed!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    
    # Test sequence generation
    if args.test_generation:
        print("\nTesting sequence generation...")
        trained_model.eval()
        
        # Take first sample from validation set
        val_sample = val_dataset[0]
        
        if args.model == 'direct':
            input_spectra, target_tokens, target_fg = val_sample
            input_spectra = input_spectra.unsqueeze(0)  # Add batch dim
            generated_tokens = trained_model.generate(
                input_spectra, 
                max_length=args.max_length,
                temperature=0.8
            )
        else:
            spectrum_input, target_tokens, target_fg = val_sample
            input_mz, input_intensities = spectrum_input
            input_mz = input_mz.unsqueeze(0)
            input_intensities = input_intensities.unsqueeze(0)
            generated_tokens = trained_model.generate(
                input_mz, 
                input_intensities,
                max_length=args.max_length,
                temperature=0.8
            )
        
        # Decode generated sequence
        generated_smiles = tokenizer.decode(generated_tokens[0])
        target_smiles = tokenizer.decode(target_tokens)
        
        print(f"Target SMILES: {target_smiles}")
        print(f"Generated SMILES: {generated_smiles}")
        
        # Test functional group prediction
        with torch.no_grad():
            if args.model == 'direct':
                _, fg_pred = trained_model(input_spectra, target_tokens[:-1].unsqueeze(0))
            else:
                _, fg_pred = trained_model(input_mz, input_intensities, target_tokens[:-1].unsqueeze(0))
            
            fg_probs = torch.sigmoid(fg_pred[0])  # Convert to probabilities
            predicted_fg = (fg_probs > 0.5).float()
            
            print(f"Target FG: {target_fg.sum().item():.0f} functional groups")
            print(f"Predicted FG: {predicted_fg.sum().item():.0f} functional groups")
    
    print("\n✓ Seq2Seq Mass2SMILES test completed successfully!")

if __name__ == '__main__':
    main()
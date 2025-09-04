"""
Test script for training Mass2SmilesModel with direct spectral encoding (Option 1)
"""

import argparse
import torch
from loguru import logger

from annotix_ml.mass2smiles.network import (
    Mass2SmilesModel, Mass2SmilesModelV2, 
    prepare_data_loaders, train_mass2smiles
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True, help="Path to CSV file with spectrum and SMILES data")
    parser.add_argument("-model_type", choices=['direct', 'learned'], default='direct', 
                       help="Model type: 'direct' for Mass2SmilesModel, 'learned' for Mass2SmilesModelV2")
    parser.add_argument("-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("-device", default='cpu', help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("-save_path", default='/tmp/mass2smiles_models', help="Path to save model checkpoints")
    args = parser.parse_args()

    logger.info(f'Starting training with {args.model_type} encoding approach')
    
    # Prepare data loaders
    logger.info('Preparing data loaders...')
    train_loader, val_loader, tokenizer = prepare_data_loaders(
        csv_path=args.input,
        batch_size=args.batch_size,
        model_type=args.model_type
    )
    
    vocab_size = tokenizer.vocab_size
    logger.info(f'Vocabulary size: {vocab_size}')
    
    # Initialize model
    logger.info(f'Initializing {args.model_type} model...')
    
    if args.model_type == 'direct':
        model = Mass2SmilesModel(
            units=2048,
            heads=8,
            dropout=0.1,
            dense_dropout=0.1,
            filters=256,
            num_layers=5,
            embed_dim=128,
            output_dim_smiles=vocab_size,  # Output size should match vocab size
            output_dim_fg=71,
            input_dim=2
        )
    else:  # learned
        model = Mass2SmilesModelV2(
            units=2048,
            heads=8,
            dropout=0.1,
            dense_dropout=0.1,
            filters=256,
            num_layers=5,
            embed_dim=128,
            output_dim_smiles=vocab_size,  # Output size should match vocab size
            output_dim_fg=71,
            num_mz_bins=20000,
            mz_max=2000
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Train model
    logger.info('Starting training...')
    train_losses, val_losses = train_mass2smiles(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=1e-4,
        device=args.device,
        save_path=args.save_path,
        model_type=args.model_type
    )
    
    logger.info('Training completed!')
    logger.info(f'Final train loss: {train_losses[-1]:.4f}')
    logger.info(f'Final validation loss: {val_losses[-1]:.4f}')
    
    # Test a prediction
    logger.info('Testing prediction on first validation batch...')
    model.eval()
    with torch.no_grad():
        for spectrum, smiles_target in val_loader:
            if args.model_type == 'direct':
                spectrum = spectrum.to(args.device)
                smiles_pred, fg_pred = model(spectrum)
            else:  # learned
                spectrum = (spectrum[0].to(args.device), spectrum[1].to(args.device))
                smiles_pred, fg_pred = model(spectrum[0], spectrum[1])
            
            # Get predicted token for first sample
            pred_token_idx = torch.argmax(smiles_pred[0]).item()
            target_token_idx = smiles_target[0, 1].item()  # First real token (skip START)
            
            pred_token = tokenizer.idx_to_token.get(pred_token_idx, '<UNK>')
            target_token = tokenizer.idx_to_token.get(target_token_idx, '<UNK>')
            
            logger.info(f'Sample prediction - Target: "{target_token}", Predicted: "{pred_token}"')
            
            # Show full SMILES for first sample
            target_smiles_decoded = tokenizer.decode(smiles_target[0].tolist())
            logger.info(f'Full target SMILES: {target_smiles_decoded}')
            
            break
    
    logger.success(f'Training completed successfully! Models saved to {args.save_path}')

if __name__ == "__main__":
    main()
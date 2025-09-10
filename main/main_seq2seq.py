import argparse
from loguru import logger
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import annotix_ml.mass2smiles.network as net

def main():
    parser = argparse.ArgumentParser(description='Train Mass2SMILES Seq2Seq Models')
    parser.add_argument('--model', type=str, choices=['direct', 'learned'], default='direct', help='Model type: direct or learned')
    parser.add_argument('--csv_path', type=str, default='main/test_data/test-data.csv', help='Path to CSV file with peaks_list and smiles columns')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum SMILES sequence length')
    parser.add_argument('--test_generation', action='store_true', help='Test sequence generation after training')
    parser.add_argument("--device", default='cpu', help="Device to use: 'cpu' or 'cuda' or 'mps'")
    args = parser.parse_args()
    
    logger.info(f"Training {args.model} Mass2SMILES Seq2Seq model...")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Load data
    df = pd.read_csv(args.csv_path)
    logger.success(f"Loaded {len(df)} samples from CSV")
    
    # Initialize tokenizer
    tokenizer = net.SMILESTokenizer()
    
    # Build vocabulary from all SMILES in dataset
    all_smiles = df['smiles'].tolist()
    tokenizer.build_vocab(all_smiles)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Built vocabulary with {vocab_size} tokens")
    
    # Create dataset and dataloader
    dataset = net.SpectrumSMILESSeq2SeqDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        max_smiles_length=args.max_length,
        model_type=args.model
    )
    
    # Split into train/val (80/20)
    train_size = int(.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    if args.model == 'direct':
        model = net.Mass2SmilesSeq2SeqDirect(
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

        input_spectra, target_tokens, _ = val_dataset[0]
        print(summary(model, input_data=(input_spectra.unsqueeze(0), target_tokens.unsqueeze(0)), col_names=["input_size", "output_size", "num_params", "trainable"], device="cpu"))

    else:  # learned
        model = net.Mass2SmilesSeq2SeqLearned(
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

        # For model summary, get one batch
        (input_mz, input_intensities), target_tokens, _ = val_dataset[0]
        print(summary(model, input_data=(input_mz.unsqueeze(0), input_intensities.unsqueeze(0), target_tokens.unsqueeze(0)), col_names=["input_size", "output_size", "num_params", "trainable"], device="cpu"))

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    

    # Test model with one batch
    logger.info("\nTesting model with sample batch...")
    sample_batch = next(iter(train_loader))
    
    if args.model == 'direct':
        spectra, target_tokens, target_fg = sample_batch
        logger.info(f"Input spectra shape: {spectra.shape}")
    else:  # learned
        spectrum_input, target_tokens, target_fg = sample_batch
        mz_values, intensities = spectrum_input
        logger.info(f"Input m/z shape: {mz_values.shape}, intensities shape: {intensities.shape}")
    
    logger.info(f"Target tokens shape: {target_tokens.shape}")
    logger.info(f"Target functional groups shape: {target_fg.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if args.model == 'direct':
            smiles_logits, fg_pred = model(spectra, target_tokens[:, :-1])
        else:
            smiles_logits, fg_pred = model(mz_values, intensities, target_tokens[:, :-1])
        
        logger.info(f"SMILES logits shape: {smiles_logits.shape}")
        logger.info(f"FG predictions shape: {fg_pred.shape}")
    
    logger.info("✓ Model forward pass successful!")
    
    # Train the model
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    
    trained_model, train_losses, val_losses = net.train_seq2seq_mass2smiles(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        model_type=args.model
    )
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final val loss: {val_losses[-1]:.4f}")
    
    logger.info("Saving the model in ./model.pt")
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(trained_model, "models/model.pt")


# fig = plt.figure(figsize=(10, 5))
# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Over Epochs")
# fig.savefig("losses.png")



    # Test sequence generation
    if args.test_generation:
        logger.info("\nTesting sequence generation...")
        trained_model.eval()
        
        # Take first sample from validation set
        val_sample = val_dataset[0]
        
        if args.model == 'direct':
            input_spectra, target_tokens, target_fg = val_sample
            input_spectra = input_spectra.unsqueeze(0)  # Add batch dim
            generated_tokens = trained_model.generate(
                input_spectra, 
                tokenizer,
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
                tokenizer,
                max_length=args.max_length,
                temperature=0.8
            )
        
        # Decode generated sequence
        generated_smiles = tokenizer.decode(generated_tokens[0].tolist())
        target_smiles = tokenizer.decode(target_tokens.tolist())
        
        logger.info(f"Target SMILES: {target_smiles}")
        logger.info(f"Generated SMILES: {generated_smiles}")
        
        # Test functional group prediction
        with torch.no_grad():
            if args.model == 'direct':
                _, fg_pred = trained_model(input_spectra, target_tokens[:-1].unsqueeze(0))
            else:
                _, fg_pred = trained_model(input_mz, input_intensities, target_tokens[:-1].unsqueeze(0))
            
            fg_probs = torch.sigmoid(fg_pred[0])  # Convert to probabilities
            predicted_fg = (fg_probs > 0.5).float()
            
            logger.info(f"Target FG: {target_fg.sum().item():.0f} functional groups")
            logger.info(f"Predicted FG: {predicted_fg.sum().item():.0f} functional groups")
    
    logger.success("✓ Seq2Seq Mass2SMILES test completed successfully!")

if __name__ == '__main__':
    main()
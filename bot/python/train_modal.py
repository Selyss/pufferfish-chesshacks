"""
Modal training script for NNUE chess model with A100 GPU.

Run with: modal run train_modal.py
"""

import modal

# Create Modal app
app = modal.App("pufferfish-nnue-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "datasets",
        "python-chess",
        "huggingface-hub",
    )
)

# Create a volume to persist the trained model
volume = modal.Volume.from_name("pufferfish-models", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),  # Single A100 GPU
    timeout=60 * 60 * 4,  # 4 hour timeout
    volumes={"/models": volume},
)
def train_on_modal(
    num_epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_samples: int = None,
):
    """
    Train the NNUE model on Modal with A100 GPU.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import chess
    from datasets import load_dataset
    
    # Configuration constants
    FEATURE_DIM = 768
    ACC_UNITS = 256
    HIDDEN1 = 32
    HIDDEN2 = 32
    
    # Define the model inline
    class NNUEModel(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = FEATURE_DIM
            
            # Accumulator projections
            self.acc_friendly = nn.Linear(input_dim, ACC_UNITS)
            self.acc_enemy = nn.Linear(input_dim, ACC_UNITS)
            
            # Fully connected part
            self.fc1 = nn.Linear(2 * ACC_UNITS, HIDDEN1)
            self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
            self.fc_out = nn.Linear(HIDDEN2, 1)
        
        def forward(self, x):
            acc_f = self.acc_friendly(x)
            acc_e = self.acc_enemy(x)
            
            acc_f = torch.relu(acc_f)
            acc_e = torch.relu(acc_e)
            
            combined = torch.cat([acc_f, acc_e], dim=1)
            
            y = torch.relu(self.fc1(combined))
            y = torch.relu(self.fc2(y))
            y = self.fc_out(y)
            return y
    
    def fen_to_features(fen: str) -> torch.Tensor:
        """Convert FEN to feature vector."""
        board = chess.Board(fen)
        features = torch.zeros(768, dtype=torch.float32)
        
        piece_to_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_offset = piece_to_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_offset += 6
                feature_idx = square * 12 + piece_offset
                features[feature_idx] = 1.0
        
        return features
    
    class ChessDataset(Dataset):
        def __init__(self, split: str = "train", max_samples: int = None):
            super().__init__()
            print(f"Loading dataset split: {split}...")
            dataset = load_dataset("LegendaryAKx3/light-preprocessed", split=split)
            
            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            self.data = dataset
            print(f"Loaded {len(self.data)} samples")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            fen = sample['fen']
            cp_label = sample['cp_label']
            
            features = fen_to_features(fen)
            target = torch.tensor(cp_label, dtype=torch.float32)
            
            return features, target
    
    # Training setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training with batch_size={batch_size}, lr={lr}, epochs={num_epochs}")
    
    model = NNUEModel().to(device)
    
    # Load dataset
    train_dataset = ChessDataset(split="train", max_samples=max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Checkpoint paths
    checkpoint_dir = "/models/checkpoints"
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_latest.pt"
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    else:
        print("No checkpoint found, starting training from scratch")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        grad_norms = []
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device).view(-1)
            
            # Diagnostic on first batch of first epoch
            if epoch == start_epoch and batch_idx == 0:
                print(f"\nData diagnostics:")
                print(f"  Features shape: {features.shape}")
                print(f"  Targets range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
                print(f"  Targets mean: {targets.mean().item():.2f}, std: {targets.std().item():.2f}")
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_grad = sum(grad_norms[-100:]) / min(100, len(grad_norms))
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Avg Grad Norm: {avg_grad:.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Gradient Norm: {avg_grad_norm:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check for dead neurons
        with torch.no_grad():
            sample_features = next(iter(train_loader))[0].to(device)
            acc_f = model.acc_friendly(sample_features)
            acc_e = model.acc_enemy(sample_features)
            dead_acc_f = (acc_f.max(dim=0)[0] == 0).sum().item()
            dead_acc_e = (acc_e.max(dim=0)[0] == 0).sum().item()
            if dead_acc_f > 0 or dead_acc_e > 0:
                print(f"  Warning: {dead_acc_f} dead friendly neurons, {dead_acc_e} dead enemy neurons")
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: epoch {epoch + 1}")
        
        # Save best model separately
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = "/models/nnue_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with loss: {best_loss:.4f}")
        
        # Commit volume after each epoch to persist checkpoints
        volume.commit()
    
    # Save final model to volume
    model_path = "/models/nnue_state_dict.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete! Final model saved to {model_path}")
    print(f"Best loss achieved: {best_loss:.4f}")
    
    # Final commit
    volume.commit()
    
    return {"status": "success", "final_loss": avg_loss, "best_loss": best_loss, "epochs_completed": num_epochs}


@app.local_entrypoint()
def main(
    num_epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_samples: int = None,
):
    """
    Main entry point for training.
    
    Run with:
        modal run train_modal.py
        modal run train_modal.py --num-epochs 20 --batch-size 2048
        modal run train_modal.py --max-samples 100000  # For testing
    """
    print("Starting training on Modal with A100 GPU...")
    result = train_on_modal.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
    )
    print(f"Training complete! Result: {result}")
    print(f"Model saved to Modal volume 'pufferfish-models'")
    print(f"To download: modal volume get pufferfish-models nnue_state_dict.pt")

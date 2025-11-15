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
    num_epochs: int = 5,
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
    
    # Checkpoint paths
    checkpoint_dir = "/models/checkpoints"
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_latest.pt"
    
    # Try to resume from checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch")
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: epoch {epoch + 1}")
        
        # Save best model separately
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = "/models/nnue_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Commit volume after each epoch to persist checkpoints
        volume.commit()
    
    # Save final model to volume
    model_path = "/models/nnue_state_dict.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    # Final commit
    volume.commit()
    
    return {"status": "success", "final_loss": avg_loss, "best_loss": best_loss, "epochs_completed": num_epochs}


@app.local_entrypoint()
def main(
    num_epochs: int = 5,
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
from datasets import load_dataset

from config import FEATURE_DIM
from model import NNUEModel


def fen_to_features(fen: str) -> torch.Tensor:
    """
    Convert a FEN string to a binary feature vector.
    Uses HalfKP-like encoding: for each piece, encode its position relative to king positions.
    
    Feature layout:
    - 64 squares * 10 piece types (WP, WN, WB, WR, WQ, BP, BN, BB, BR, BQ) * 64 king positions
    - Total: 64 * 10 * 64 = 40,960 features per side (white perspective, black perspective)
    - Full feature space: 40,960 * 2 = 81,920
    
    For simplicity, we'll use a simpler encoding:
    - 64 squares * 12 piece types (6 white, 6 black) = 768 features
    """
    board = chess.Board(fen)
    
    # Simple piece-square encoding: 64 squares * 12 piece types
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
            # White pieces: indices 0-5, Black pieces: indices 6-11
            piece_offset = piece_to_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                piece_offset += 6
            
            # Feature index: square * 12 + piece_type
            feature_idx = square * 12 + piece_offset
            features[feature_idx] = 1.0
    
    return features


class ChessDataset(Dataset):
    """
    Chess position dataset from Hugging Face.
    
    Dataset: LegendaryAKx3/light-preprocessed
    Columns: fen (string), depth (int32), knodes (int32), cp_label (int32)
    """

    def __init__(self, split: str = "train", max_samples: int = None):
        super().__init__()
        print(f"Loading dataset split: {split}...")
        
        # Load dataset from Hugging Face
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
        
        # Convert FEN to features
        features = fen_to_features(fen)
        
        # Convert cp_label to float32 tensor to match model dtype
        target = torch.tensor(cp_label, dtype=torch.float32)
        
        return features, target


def train_nnue(
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = None,
    checkpoint_dir: str = "checkpoints",
):
    import os
    
    model = NNUEModel().to(device)
    
    # Load training dataset
    train_dataset = ChessDataset(split="train", max_samples=max_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    
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

    model.train()
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
            best_model_path = os.path.join(checkpoint_dir, "nnue_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with loss: {best_loss:.4f}")

    return model, best_loss


if __name__ == "__main__":
    # Start with a smaller subset for testing, remove max_samples for full training
    model, best_loss = train_nnue(num_epochs=5, batch_size=256, lr=1e-3)
    torch.save(model.state_dict(), "nnue_state_dict.pt")
    print(f"Training complete! Final model saved to nnue_state_dict.pt")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Best model saved in checkpoints/nnue_best.pt")

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

    def __init__(self, data):
        super().__init__()
        self.data = data

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


def create_stratified_split(dataset, val_ratio=0.05, max_samples=None, seed=42):
    """
    Create stratified train/validation split based on binned cp_label values.
    
    Args:
        dataset: HuggingFace dataset
        val_ratio: Fraction of data to use for validation (default 0.05 = 5%)
        max_samples: Limit total samples (applied before split)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Limit samples if requested
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Creating stratified split from {len(dataset)} samples...")
    
    # Get all cp_labels
    cp_labels = np.array(dataset['cp_label'])
    
    # Bin the continuous targets for stratification
    # Use quantile-based binning to ensure balanced bins
    n_bins = 20
    bins = np.percentile(cp_labels, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates
    stratify_labels = np.digitize(cp_labels, bins[:-1])
    
    # Create indices
    indices = np.arange(len(dataset))
    
    # Stratified split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=stratify_labels,
        random_state=seed
    )
    
    # Create subsets
    train_data = dataset.select(train_indices.tolist())
    val_data = dataset.select(val_indices.tolist())
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Print distribution statistics
    train_cp = np.array(train_data['cp_label'])
    val_cp = np.array(val_data['cp_label'])
    print(f"Train cp_label: mean={train_cp.mean():.2f}, std={train_cp.std():.2f}")
    print(f"Val cp_label: mean={val_cp.mean():.2f}, std={val_cp.std():.2f}")
    
    return train_data, val_data


def train_nnue(
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = None,
    checkpoint_dir: str = "checkpoints",
    val_ratio: float = 0.05,
):
    import os
    
    model = NNUEModel().to(device)
    
    # Dataset version identifier
    DATASET_VERSION = "preprocessed-balanced-v1"
    
    # Load and split dataset
    print("Loading dataset from Hugging Face...")
    full_dataset = load_dataset("LegendaryAKx3/heavy-preprocessed", split="train")
    train_data, val_data = create_stratified_split(
        full_dataset, 
        val_ratio=val_ratio, 
        max_samples=max_samples
    )
    
    # Create datasets and loaders
    train_dataset = ChessDataset(train_data)
    val_dataset = ChessDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler - now using validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{DATASET_VERSION}_latest.pt")
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Verify dataset version matches
        checkpoint_dataset_version = checkpoint.get('dataset_version', 'unknown')
        if checkpoint_dataset_version != DATASET_VERSION:
            print(f"Warning: Checkpoint dataset version mismatch!")
            print(f"  Checkpoint version: {checkpoint_dataset_version}")
            print(f"  Current version: {DATASET_VERSION}")
            print("Starting training from scratch with new dataset")
        else:
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device).view(-1)
                
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Average Gradient Norm: {avg_grad_norm:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check for overfitting
        if avg_val_loss > avg_loss * 1.5:
            print(f"  Warning: Possible overfitting detected (val_loss / train_loss = {avg_val_loss / avg_loss:.2f})")
        
        # Check for dead neurons
        with torch.no_grad():
            sample_features = next(iter(train_loader))[0].to(device)
            acc_f = model.acc_friendly(sample_features)
            acc_e = model.acc_enemy(sample_features)
            dead_acc_f = (acc_f.max(dim=0)[0] == 0).sum().item()
            dead_acc_e = (acc_e.max(dim=0)[0] == 0).sum().item()
            if dead_acc_f > 0 or dead_acc_e > 0:
                print(f"  Warning: {dead_acc_f} dead friendly neurons, {dead_acc_e} dead enemy neurons")
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'best_loss': best_loss,
            'dataset_version': DATASET_VERSION,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: epoch {epoch + 1}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "nnue_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with validation loss: {best_loss:.4f}")

    return model, best_loss


if __name__ == "__main__":
    # Start with a smaller subset for testing, remove max_samples for full training
    model, best_loss = train_nnue(num_epochs=5, batch_size=256, lr=1e-3)
    torch.save(model.state_dict(), "nnue_state_dict.pt")
    print(f"Training complete! Final model saved to nnue_state_dict.pt")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Best model saved in checkpoints/nnue_best.pt")

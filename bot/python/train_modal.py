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
        "scikit-learn",
        "numpy",
    )
)

# Create a volume to persist the trained model
volume = modal.Volume.from_name("pufferfish-models", create_if_missing=True)

@app.function(
    image=image,
    timeout=60 * 60 * 3,
    volumes={"/models": volume},
    cpu=8.0,
)
def preprocess_features():
    """
    Preprocess all FEN strings to feature tensors once.
    This eliminates the CPU bottleneck during training.
    
    Run with: modal run train_modal.py::preprocess_features
    """
    import torch
    import chess
    import numpy as np
    from datasets import load_dataset
    import os
    
    def fen_to_features_batch(fens):
        """Batch process FEN strings to features."""
        features_list = []
        for fen in fens:
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
            
            features_list.append(features.numpy())
        
        return np.array(features_list, dtype=np.float32)
    
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("LegendaryAKx3/preprocessed-balanced", split="train")
    
    print(f"Preprocessing {len(dataset):,} FEN strings to features...")
    print("This will take 30-60 minutes but only needs to be done once.")
    
    all_features = []
    all_targets = []
    
    batch_size = 10000
    for i in range(0, len(dataset), batch_size):
        end_idx = min(i + batch_size, len(dataset))
        batch = dataset.select(range(i, end_idx))
        
        fens = [sample['fen'] for sample in batch]
        targets = [sample['cp_label'] for sample in batch]
        
        batch_features = fen_to_features_batch(fens)
        all_features.append(batch_features)
        all_targets.extend(targets)
        
        if (i // batch_size + 1) % 100 == 0:
            print(f"  Processed {end_idx:,}/{len(dataset):,} samples ({100*end_idx/len(dataset):.1f}%)")
    
    print("Concatenating arrays...")
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.array(all_targets, dtype=np.float32)
    
    cache_path = "/models/cached_features_v1.npz"
    print(f"Saving preprocessed features to {cache_path}...")
    
    os.makedirs("/models", exist_ok=True)
    np.savez_compressed(cache_path, features=all_features, targets=all_targets)
    
    volume.commit()
    
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Preprocessing complete!")
    print(f"  Samples: {len(all_features):,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Location: {cache_path}")
    
    return {"samples": len(all_features), "file_size_mb": file_size_mb}


@app.function(
    image=image,
    gpu="A100-40GB",  # Single A100 GPU with 40GB VRAM
    timeout=60 * 60 * 8,  # 8 hour timeout
    volumes={"/models": volume},
)
def train_on_modal(
    num_epochs: int = 10,
    batch_size: int = 8192,  # Increased default for A100
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
        def __init__(self, data):
            super().__init__()
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            fen = sample['fen']
            cp_label = sample['cp_label']
            
            features = fen_to_features(fen)
            target = torch.tensor(cp_label, dtype=torch.float32)
            
            return features, target
    
    def create_stratified_split(dataset, val_ratio=0.05, max_samples=None, seed=42):
        """Create stratified train/validation split based on binned cp_label values."""
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Creating stratified split from {len(dataset)} samples...")
        
        cp_labels = np.array(dataset['cp_label'])
        
        # Bin the continuous targets for stratification
        n_bins = 20
        bins = np.percentile(cp_labels, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        stratify_labels = np.digitize(cp_labels, bins[:-1])
        
        indices = np.arange(len(dataset))
        
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=stratify_labels,
            random_state=seed
        )
        
        train_data = dataset.select(train_indices.tolist())
        val_data = dataset.select(val_indices.tolist())
        
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        train_cp = np.array(train_data['cp_label'])
        val_cp = np.array(val_data['cp_label'])
        print(f"Train cp_label: mean={train_cp.mean():.2f}, std={train_cp.std():.2f}")
        print(f"Val cp_label: mean={val_cp.mean():.2f}, std={val_cp.std():.2f}")
        
        return train_data, val_data
    
    # Training setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training with batch_size={batch_size}, lr={lr}, epochs={num_epochs}")
    
    model = NNUEModel().to(device)
    
    # Dataset version identifier
    DATASET_VERSION = "preprocessed-balanced-v1"
    
    # Load preprocessed features from cache
    print("Loading cached features...")
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset
    
    cache_path = "/models/cached_features_v1.npz"
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached features not found at {cache_path}. "
            "Please run: modal run train_modal.py::preprocess_features"
        )
    
    data = np.load(cache_path)
    features = torch.from_numpy(data['features']).float()
    targets = torch.from_numpy(data['targets']).float()
    
    print(f"Loaded {len(features)} preprocessed samples")
    
    # Apply max_samples limit if specified
    if max_samples is not None and max_samples < len(features):
        indices = torch.randperm(len(features))[:max_samples]
        features = features[indices]
        targets = targets[indices]
        print(f"Subsampled to {max_samples} samples")
    
    # Create stratified split
    # Bin targets for stratification
    n_bins = 100
    target_bins = pd.qcut(targets.numpy(), q=n_bins, labels=False, duplicates='drop')
    
    train_idx, val_idx = train_test_split(
        np.arange(len(features)),
        test_size=0.05,
        stratify=target_bins,
        random_state=42
    )
    
    # Create tensor datasets
    train_dataset = TensorDataset(features[train_idx], targets[train_idx])
    val_dataset = TensorDataset(features[val_idx], targets[val_idx])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Double batch size for validation (no gradients)
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
    )
    
    criterion = nn.MSELoss()
    # Note: fused=True is incompatible with GradScaler, so we use standard Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler - using validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Checkpoint paths with dataset version
    checkpoint_dir = "/models/checkpoints"
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{DATASET_VERSION}_latest.pt"
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    # Use automatic mixed precision (AMP) for faster training
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
    use_amp = True
    
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
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    else:
        print("No checkpoint found, starting training from scratch")
    
    # Enable TF32 for faster matmul on Ampere GPUs (A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking for faster conv operations
    torch.backends.cudnn.benchmark = True
    
    # Keep reference to original model for saving
    original_model = model
    
    # Use torch.compile for JIT optimization (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode='max-autotune')
        print("Model compiled with torch.compile for maximum performance")
    except Exception as e:
        print(f"Could not compile model: {e}")
    
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        grad_norms = []
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1)
            
            # Diagnostic on first batch of first epoch
            if epoch == start_epoch and batch_idx == 0:
                print(f"\nData diagnostics:")
                print(f"  Features shape: {features.shape}")
                print(f"  Targets range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
                print(f"  Targets mean: {targets.mean().item():.2f}, std: {targets.std().item():.2f}")
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use automatic mixed precision
            with autocast(device_type='cuda'):
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
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
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).view(-1)
                
                with autocast(device_type='cuda'):
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
        
        # Check for dead neurons every 5 epochs (expensive operation)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_features = next(iter(train_loader))[0].to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    acc_f = model.acc_friendly(sample_features)
                    acc_e = model.acc_enemy(sample_features)
                dead_acc_f = (acc_f.max(dim=0)[0] == 0).sum().item()
                dead_acc_e = (acc_e.max(dim=0)[0] == 0).sum().item()
                if dead_acc_f > 0 or dead_acc_e > 0:
                    print(f"  Warning: {dead_acc_f} dead friendly neurons, {dead_acc_e} dead enemy neurons")
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save checkpoint every 5 epochs (reduce I/O overhead)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': original_model.state_dict(),  # Use original model, not compiled
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss,
                'dataset_version': DATASET_VERSION,
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: epoch {epoch + 1}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = "/models/nnue_best.pt"
            torch.save(original_model.state_dict(), best_model_path)  # Use original model
            print(f"  New best model saved with validation loss: {best_loss:.4f}")
        
        # Commit volume every 10 epochs (reduce I/O overhead)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            volume.commit()
            print(f"  Volume committed at epoch {epoch + 1}")
    
    # Save final model to volume
    model_path = "/models/nnue_state_dict.pt"
    torch.save(original_model.state_dict(), model_path)  # Use original model
    print(f"\nTraining complete! Final model saved to {model_path}")
    print(f"Best validation loss achieved: {best_loss:.4f}")
    
    # Final commit
    volume.commit()
    
    return {"status": "success", "final_train_loss": avg_loss, "final_val_loss": avg_val_loss, "best_val_loss": best_loss, "epochs_completed": num_epochs}


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

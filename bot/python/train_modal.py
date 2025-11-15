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

# Global functions for multiprocessing (must be at module level for pickling)
def fen_to_features_and_target(fen_and_label):
    """
    Extract 795 features from FEN with proper perspective flipping.
    
    Features (795 total):
    - 768: piece-square occupancies (64 squares × 12 piece types from side-to-move perspective)
    - 1: side-to-move (1.0 if white, 0.0 if black - but we always normalize to 1.0 after flip)
    - 4: castling rights (our kingside, our queenside, enemy kingside, enemy queenside)
    - 8: en-passant file (one-hot encoding, 0-7 or all zeros if none)
    - 6: material balance per piece type (pawns, knights, bishops, rooks, queens, kings)
    - 6: our piece counts (P, N, B, R, Q, K)
    - 1: game phase (0.0=opening, 1.0=endgame based on material)
    
    Returns:
        features: 795 floats
        target: evaluation from side-to-move's perspective
    """
    fen, cp_label = fen_and_label
    import chess
    import numpy as np
    
    try:
        board = chess.Board(fen)
        features = np.zeros(795, dtype=np.float32)
        
        piece_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}  # PAWN through KING
        our_color = board.turn  # True for white, False for black
        
        # === PIECE-SQUARE FEATURES (768) ===
        for square in range(64):
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            is_ours = (piece.color == our_color)
            piece_type_idx = piece_map[piece.piece_type]
            
            # If black to move, flip the board vertically
            sq = square
            if not our_color:
                file = square % 8
                rank = square // 8
                sq = (7 - rank) * 8 + file
            
            # Our pieces: indices 0-5, opponent's pieces: 6-11
            piece_idx = piece_type_idx if is_ours else piece_type_idx + 6
            features[sq * 12 + piece_idx] = 1.0
        
        idx = 768
        
        # === SIDE-TO-MOVE (1) ===
        # After perspective flip, we're always encoding from "our" perspective
        # So this is always 1.0 (we are to move)
        features[idx] = 1.0
        idx += 1
        
        # === CASTLING RIGHTS (4) ===
        # Our kingside, our queenside, enemy kingside, enemy queenside
        features[idx] = 1.0 if board.has_kingside_castling_rights(our_color) else 0.0
        features[idx + 1] = 1.0 if board.has_queenside_castling_rights(our_color) else 0.0
        features[idx + 2] = 1.0 if board.has_kingside_castling_rights(not our_color) else 0.0
        features[idx + 3] = 1.0 if board.has_queenside_castling_rights(not our_color) else 0.0
        idx += 4
        
        # === EN-PASSANT FILE (8) ===
        # One-hot encoding of en-passant file (0-7), or all zeros if none
        if board.ep_square is not None:
            ep_file = board.ep_square % 8
            # If black to move, mirror the file
            if not our_color:
                ep_file = 7 - ep_file
            features[idx + ep_file] = 1.0
        idx += 8
        
        # === MATERIAL BALANCE (6) ===
        # Count pieces: positive if we have more, negative if opponent has more
        piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # P, N, B, R, Q, K
        for piece_type in range(1, 7):
            our_count = len(board.pieces(piece_type, our_color))
            enemy_count = len(board.pieces(piece_type, not our_color))
            features[idx] = float(our_count - enemy_count) / 8.0  # Normalize
            idx += 1
        
        # === OUR PIECE COUNTS (6) ===
        # Normalized counts of our pieces
        for piece_type in range(1, 7):
            count = len(board.pieces(piece_type, our_color))
            features[idx] = float(count) / 8.0  # Normalize (max 8 pawns, etc.)
            idx += 1
        
        # === GAME PHASE (1) ===
        # Estimate game phase based on material (0=opening, 1=endgame)
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type, value in piece_values.items():
                if piece_type != 6:  # Don't count kings
                    total_material += len(board.pieces(piece_type, color)) * value
        
        # Max material at start: 16 pawns + 4 knights + 4 bishops + 4 rooks + 2 queens = 78
        max_material = 78.0
        phase = 1.0 - (total_material / max_material)  # 0 at start, 1 at end
        features[idx] = phase
        
        # Flip target if black to move (cp_label is from white's perspective)
        target = cp_label if our_color else -cp_label
        
        return features, target
    except Exception as e:
        # Return zero vector and zero target for invalid FENs
        return np.zeros(795, dtype=np.float32), 0.0

@app.function(
    image=image,
    timeout=60 * 60 * 4,  # 4 hour timeout for preprocessing
    volumes={"/models": volume},
    cpu=32.0,  # Maximum CPUs for parallel processing
    memory=65536,  # 64GB RAM for large dataset operations
)
def preprocess_features():
    """
    Preprocess all FEN strings to feature tensors once with maximum parallelization.
    This eliminates the CPU bottleneck during training.
    
    Run with: modal run train_modal.py::preprocess_features
    """
    import numpy as np
    from datasets import load_dataset
    import os
    from multiprocessing import Pool, cpu_count
    
    def process_batch_parallel(fen_label_pairs, num_workers):
        """Process a batch of (FEN, label) pairs using all available CPU cores."""
        with Pool(num_workers) as pool:
            results = pool.map(fen_to_features_and_target, fen_label_pairs, chunksize=1000)
        features = np.array([r[0] for r in results], dtype=np.float32)
        targets = np.array([r[1] for r in results], dtype=np.float32)
        return features, targets
    
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("LegendaryAKx3/rebalanced-preprocessed", split="train")
    
    num_workers = cpu_count()
    print(f"Preprocessing {len(dataset):,} FEN strings to features...")
    print(f"Using {num_workers} CPU cores with aggressive parallelization")
    print("Estimated time: 10-20 minutes with maximum optimization")
    
    all_features = []
    all_targets = []
    
    # Much larger batch size for better parallel efficiency
    batch_size = 100000
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    print(f"Processing in {num_batches} batches of {batch_size:,} samples each")
    
    for i in range(0, len(dataset), batch_size):
        end_idx = min(i + batch_size, len(dataset))
        batch = dataset.select(range(i, end_idx))
        
        # Create (fen, label) pairs for parallel processing
        fen_label_pairs = [(sample['fen'], sample['cp_label']) for sample in batch]
        
        # Parallel processing - returns both features and flipped targets
        batch_features, batch_targets = process_batch_parallel(fen_label_pairs, num_workers)
        all_features.append(batch_features)
        all_targets.extend(batch_targets.tolist())
        
        progress = 100 * end_idx / len(dataset)
        print(f"  [{progress:5.1f}%] Processed {end_idx:,}/{len(dataset):,} samples")
    
    print("Concatenating arrays...")
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.array(all_targets, dtype=np.float32)
    
    cache_path = "/models/cached_features_simple_nnue_795.npz"
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
    from torch.utils.data import DataLoader
    
    # Configuration constants
    FEATURE_DIM = 795  # Enhanced feature set
    
    # Architecture: 795 -> 2048 -> 2048 -> 1024 -> 512 -> 256 -> 1
    HIDDEN1 = 2048
    HIDDEN2 = 2048
    HIDDEN3 = 1024
    HIDDEN4 = 512
    HIDDEN5 = 256
    
    DROPOUT_RATE = 0.05
    
    # Define ResidualBlock for reuse
    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout_rate=0.05):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout_rate)
            
        def forward(self, x):
            residual = x
            out = torch.relu(self.fc1(x))
            out = self.dropout(out)
            out = self.fc2(out)
            out = out + residual  # Skip connection
            out = self.norm(out)
            return out
    
    # Define the SimpleNNUE model with deep residual architecture
    class SimpleNNUE(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Stage 1: 795 -> 2048
            self.fc1 = nn.Linear(FEATURE_DIM, HIDDEN1)
            self.norm1 = nn.LayerNorm(HIDDEN1)
            self.dropout1 = nn.Dropout(DROPOUT_RATE)
            self.res1_1 = ResidualBlock(HIDDEN1, DROPOUT_RATE)
            self.res1_2 = ResidualBlock(HIDDEN1, DROPOUT_RATE)
            
            # Stage 2: 2048 -> 2048
            self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
            self.norm2 = nn.LayerNorm(HIDDEN2)
            self.dropout2 = nn.Dropout(DROPOUT_RATE)
            self.res2_1 = ResidualBlock(HIDDEN2, DROPOUT_RATE)
            self.res2_2 = ResidualBlock(HIDDEN2, DROPOUT_RATE)
            
            # Stage 3: 2048 -> 1024
            self.fc3 = nn.Linear(HIDDEN2, HIDDEN3)
            self.norm3 = nn.LayerNorm(HIDDEN3)
            self.dropout3 = nn.Dropout(DROPOUT_RATE)
            self.res3_1 = ResidualBlock(HIDDEN3, DROPOUT_RATE)
            self.res3_2 = ResidualBlock(HIDDEN3, DROPOUT_RATE)
            
            # Stage 4: 1024 -> 512
            self.fc4 = nn.Linear(HIDDEN3, HIDDEN4)
            self.norm4 = nn.LayerNorm(HIDDEN4)
            self.dropout4 = nn.Dropout(DROPOUT_RATE)
            self.res4_1 = ResidualBlock(HIDDEN4, DROPOUT_RATE)
            self.res4_2 = ResidualBlock(HIDDEN4, DROPOUT_RATE)
            
            # Stage 5: 512 -> 256
            self.fc5 = nn.Linear(HIDDEN4, HIDDEN5)
            self.norm5 = nn.LayerNorm(HIDDEN5)
            self.dropout5 = nn.Dropout(DROPOUT_RATE)
            self.res5_1 = ResidualBlock(HIDDEN5, DROPOUT_RATE)
            self.res5_2 = ResidualBlock(HIDDEN5, DROPOUT_RATE)
            
            # Output head: 256 -> 1 (no activation)
            self.fc_out = nn.Linear(HIDDEN5, 1)
        
        def forward(self, x):
            # Stage 1
            x = torch.relu(self.fc1(x))
            x = self.norm1(x)
            x = self.dropout1(x)
            x = self.res1_1(x)
            x = self.res1_2(x)
            
            # Stage 2
            x = torch.relu(self.fc2(x))
            x = self.norm2(x)
            x = self.dropout2(x)
            x = self.res2_1(x)
            x = self.res2_2(x)
            
            # Stage 3
            x = torch.relu(self.fc3(x))
            x = self.norm3(x)
            x = self.dropout3(x)
            x = self.res3_1(x)
            x = self.res3_2(x)
            
            # Stage 4
            x = torch.relu(self.fc4(x))
            x = self.norm4(x)
            x = self.dropout4(x)
            x = self.res4_1(x)
            x = self.res4_2(x)
            
            # Stage 5
            x = torch.relu(self.fc5(x))
            x = self.norm5(x)
            x = self.dropout5(x)
            x = self.res5_1(x)
            x = self.res5_2(x)
            
            # Output (no activation)
            x = self.fc_out(x)
            return x
    
    # Training setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training with batch_size={batch_size}, lr={lr}, epochs={num_epochs}")
    
    model = SimpleNNUE().to(device)
    
    # Dataset version identifier
    DATASET_VERSION = "rebalanced-preprocessed-v2-flipped"
    
    # Load preprocessed features from cache
    print("Loading cached features...")
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset
    
    cache_path = "/models/cached_features_simple_nnue_795.npz"
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached features not found at {cache_path}. "
            "Please run: modal run train_modal.py::preprocess_features"
        )
    
    print("Loading and validating cached features...")
    data = np.load(cache_path)
    
    # Validate cache file structure
    if 'features' not in data or 'targets' not in data:
        raise ValueError(
            f"Cache file {cache_path} is corrupted or from old version. "
            f"Expected 'features' and 'targets' arrays. "
            f"Please rerun preprocessing: modal run train_modal.py::preprocess_features"
        )
    
    features = torch.from_numpy(data['features']).float()
    targets = torch.from_numpy(data['targets']).float()
    
    # Validate dimensions
    if features.shape[1] != 795:
        raise ValueError(
            f"Features have wrong dimension: {features.shape[1]}, expected 795. "
            f"Cache file may be corrupted. Please rerun preprocessing."
        )
    
    print(f"Loaded {len(features)} preprocessed samples")
    print(f"Feature shape: {features.shape}")
    print(f"Target stats: mean={targets.mean():.2f}, std={targets.std():.2f}, "
          f"min={targets.min():.2f}, max={targets.max():.2f}")
    
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
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Double batch size for validation (no gradients)
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=True
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

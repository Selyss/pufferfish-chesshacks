# Modal Training Setup for NNUE Chess Model

This guide explains how to train your NNUE chess model on Modal with an A100 GPU.

## Prerequisites

1. **Install Modal**
   ```bash
   pip install modal
   ```

2. **Create a Modal account**
   - Go to https://modal.com
   - Sign up for an account

3. **Authenticate Modal**
   ```bash
   modal token new
   ```
   This will open a browser window to authenticate.

## Optional: Hugging Face Secret (if dataset is private)

If the dataset requires authentication:

```bash
modal secret create huggingface-secret HF_TOKEN=your_huggingface_token
```

If the dataset is public, you can remove the `secrets` parameter from the function decorator in `train_modal.py`.

## Running Training

### Basic Usage

Train with default parameters (10 epochs, batch size 1024):
```bash
modal run bot/python/train_modal.py
```

### Custom Parameters

Train with custom epochs and batch size:
```bash
modal run bot/python/train_modal.py --num-epochs 20 --batch-size 2048
```

Train on a subset for testing:
```bash
modal run bot/python/train_modal.py --max-samples 100000 --num-epochs 5
```

Full training on entire dataset with custom learning rate:
```bash
modal run bot/python/train_modal.py --num-epochs 15 --batch-size 1024 --lr 0.001
```

## Configuration Options

- `--num-epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 1024)
- `--lr`: Learning rate (default: 1e-3)
- `--max-samples`: Limit dataset size for testing (default: None - full dataset)

## Model Output

The trained model is automatically saved to a Modal volume named `pufferfish-models`.

### Download Trained Model

After training completes:
```bash
modal volume get pufferfish-models nnue_state_dict.pt
```

This downloads the model to your local directory.

### List Files in Volume

```bash
modal volume ls pufferfish-models
```

## Resource Details

- **GPU**: Single NVIDIA A100 (40GB or 80GB depending on availability)
- **Timeout**: 4 hours maximum per run
- **Memory**: Scales automatically with A100
- **Storage**: Persistent volume for model checkpoints

## Cost Estimation

Modal pricing for A100 GPU is approximately:
- ~$4-5 per hour for A100 GPU time

Training time depends on:
- Dataset size
- Number of epochs
- Batch size

## Monitoring

Watch training progress in real-time:
- Modal automatically streams logs to your terminal
- Progress is printed every 100 batches
- Epoch summaries show average loss

## Troubleshooting

### Out of Memory

If you get OOM errors, reduce batch size:
```bash
modal run bot/python/train_modal.py --batch-size 512
```

### Timeout

If training times out, split into multiple runs or increase timeout in `train_modal.py`:
```python
timeout=60 * 60 * 8,  # 8 hours
```

### Dataset Loading Issues

If dataset fails to load, ensure you have internet connectivity in Modal (enabled by default) and check if the dataset requires authentication.

## Advanced: Deploy as a Scheduled Job

To run training on a schedule:

```bash
modal deploy bot/python/train_modal.py
```

Then configure the schedule in Modal's dashboard.

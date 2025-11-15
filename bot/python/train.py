import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import FEATURE_DIM
from model import NNUEModel


class DummyDataset(Dataset):
    """
    Placeholder dataset.

    Replace with:
    - real feature vectors of length FEATURE_DIM
    - real centipawn eval targets
    """

    def __init__(self, n_samples: int = 10000):
        super().__init__()
        self.X = torch.randint(
            0, 2, (n_samples, FEATURE_DIM), dtype=torch.float32
        )
        self.y = torch.randn(n_samples, dtype=torch.float32) * 100.0

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_nnue(
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = NNUEModel().to(device)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train_nnue(num_epochs=5)
    torch.save(model.state_dict(), "nnue_state_dict.pt")
    print("Saved model to nnue_state_dict.pt")

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd


def get_dataloaders(
    processed_path="../../data/processed_features.csv",
    batch_size=32,
    val_size=0.2,
    random_state=42
):
    df = pd.read_csv(processed_path)

    if "label" not in df.columns:
        raise ValueError("Processed features must contain 'label' column")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    obs_dim = X.shape[1]
    n_actions = len(set(y)) if len(set(y)) > 1 else 2

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, obs_dim, n_actions

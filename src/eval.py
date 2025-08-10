import torch
from src.utils import compute_metrics, get_device

def evaluate_agent(agent, data_loader):
    device = get_device()
    agent.set_training(False)

    all_preds, all_labels = [], []
    for state, label in data_loader:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = agent.select_action(state, epsilon=0.0)
        all_preds.append(action)
        all_labels.append(label)

    return compute_metrics(all_labels, all_preds)

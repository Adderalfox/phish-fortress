import torch
import numpy as np
from src.model.ddqn import DDQNAgent
from src.model.replay_buffer import PrioritizedReplayBuffer
from src.rewards import compute_reward
from src.utils import set_seed, get_device, compute_metrics


def train_and_eval(hp, train_loader, val_loader, obs_dim, n_actions, max_steps=5000):
    device = get_device()
    set_seed(42)

    agent = DDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=hp["hidden_size"],
        activation=hp["activation"],
        dueling=True,
        use_noisy=hp["use_noisy"],
        lr=hp["learning_rate"],
        gamma=hp["gamma"],
        tau=hp["tau"],
        optimizer_type=hp["optimizer"],
        loss_type=hp["loss"],
        grad_clip=5.0
    )

    replay = PrioritizedReplayBuffer(
        capacity=50000,
        alpha=0.6,
        beta_start=hp.get("beta_start", 0.4),
        beta_frames=max_steps,
        n_step=1,
        gamma=hp["gamma"]
    )

    epsilon = 1.0
    step_count = 0

    while step_count < max_steps:
        for state, label in train_loader:
            state = torch.tensor(state, dtype=torch.float32)
            label = int(label)

            action = agent.select_action(state, epsilon=epsilon if not hp["use_noisy"] else 0.0)

            reward = compute_reward(action, label)

            done = 1.0
            replay.push(state.numpy(), action, reward, state.numpy(), done)

            epsilon = max(hp["epsilon_min"], epsilon - hp["epsilon_decay"])

            if replay.tree.n_entries > hp["batch_size"]:
                batch = replay.sample(hp["batch_size"])
                loss, td_errors, new_priorities, indices = agent.update(
                    states=batch[0],
                    actions=batch[1],
                    rewards=batch[2],
                    next_states=batch[3],
                    dones=batch[4],
                    gamma_n=batch[5],
                    is_weights=batch[6],
                    indices=batch[7]
                )
                replay.update_priorities(indices, new_priorities)

            step_count += 1
            if step_count >= max_steps:
                break

    all_preds = []
    all_labels = []
    for state, label in val_loader:
        state = torch.tensor(state, dtype=torch.float32)
        action = agent.select_action(state, epsilon=0.0)
        all_preds.append(action)
        all_labels.append(label)

    metrics = compute_metrics(all_labels, all_preds)
    return metrics["f1"]

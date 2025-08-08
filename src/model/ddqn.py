import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.model.net_utils import NoisyLinear, get_activation

class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 2,
        activation: str = "relu",
        dueling: bool = True,
        use_noisy: bool = False,
    ):
        super().__init__()
        assert activation in ("relu", "gelu", "elu")
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dueling = dueling
        self.use_noisy = use_noisy

        act_map = {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU()}
        self.activation = act_map[activation]

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            if use_noisy:
                layers.append(NoisyLinear(in_dim, hidden_size))
            else:
                layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(self.activation)
            in_dim = hidden_size
        self.body = nn.Sequential(*layers)

        if dueling:
            if use_noisy:
                self.value_layer = nn.Sequential(NoisyLinear(hidden_size, hidden_size), self.activation, NoisyLinear(hidden_size, 1))
                self.adv_layer = nn.Sequential(NoisyLinear(hidden_size, hidden_size), self.activation, NoisyLinear(hidden_size, num_actions))
            else:
                self.value_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), self.activation, nn.Linear(hidden_size, 1))
                self.adv_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), self.activation, nn.Linear(hidden_size, num_actions))
        else:
            if use_noisy:
                self.head = nn.Sequential(NoisyLinear(hidden_size, hidden_size), self.activation, NoisyLinear(hidden_size, num_actions))
            else:
                self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), self.activation, nn.Linear(hidden_size, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(x)
        if self.dueling:
            v = self.value_layer(features)
            a = self.adv_layer(features)
            q = v + (a - a.mean(dim=1, keepdim=True))
        else:
            q = self.head(features)
        return q

    def reset_noise(self):
        if not self.use_noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: Optional[str] = None,
        hidden_size: int = 256,
        num_hidden_layers: int = 2,
        activation: str = "relu",
        dueling: bool = True,
        use_noisy: bool = False,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        optimizer_type: str = "adam",
        loss_type: str = "huber",
        grad_clip: Optional[float] = None,
        target_update_freq: Optional[int] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.update_step = 0

        self.online_net = QNetwork(
            input_dim=obs_dim,
            num_actions=n_actions,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            dueling=dueling,
            use_noisy=use_noisy,
        ).to(self.device)

        self.target_net = QNetwork(
            input_dim=obs_dim,
            num_actions=n_actions,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            dueling=dueling,
            use_noisy=use_noisy,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        params = self.online_net.parameters()
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(params, lr=lr)
        elif optimizer_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(params, lr=lr)
        else:
            raise ValueError("Unsupported optimizer type")

        self.loss_type = loss_type
        if loss_type == "mse":
            self.criterion = F.mse_loss
        elif loss_type == "huber":
            self.criterion = F.smooth_l1_loss
        else:
            raise ValueError("Unsupported loss type")

        self.use_noisy = use_noisy
        self._training = True

    def to(self, device: str):
        self.device = torch.device(device)
        self.online_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        self.online_net.eval()
        with torch.no_grad():
            s = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
            if self.use_noisy:
                self.online_net.reset_noise()
                q_vals = self.online_net(s)
                action = int(q_vals.argmax(dim=1).item())
            else:
                if torch.rand(1).item() < epsilon:
                    action = int(torch.randint(0, self.n_actions, (1,)).item())
                else:
                    q_vals = self.online_net(s)
                    action = int(q_vals.argmax(dim=1).item())
        self.online_net.train(self._training)
        return action
    
    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_( (1.0 - self.tau) * target_param.data + self.tau * param.data )

    def hard_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma_n: Optional[torch.Tensor] = None,
        is_weights: Optional[torch.Tensor] = None,
        indices = None,
    ) -> Tuple[float, torch.Tensor, Optional[torch.Tensor]]:
        self.online_net.train()
        self.target_net.eval()

        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).float()

        batch_size = states.size(0)

        if gamma_n is None:
            gamma_n = torch.full((batch_size,), self.gamma, dtype=rewards.dtype, device=self.device)
        else:
            gamma_n = gamma_n.to(self.device)

        if is_weights is None:
            is_weights = torch.ones(batch_size, device=self.device)
        else:
            is_weights = is_weights.to(self.device)

        q_values = self.online_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_noisy:
                self.online_net.reset_noise()
                self.target_net.reset_noise()
            next_q_online = self.online_net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            q_target_next = next_q_target.gather(1, next_actions).squeeze(1)

            y = rewards + (gamma_n * q_target_next * (1.0 - dones))

        td_errors = (y - q_taken).detach()
        if self.loss_type == "huber":
            elementwise_loss = F.smooth_l1_loss(q_taken, y, reduction="none")
        else:
            elementwise_loss = F.mse_loss(q_taken, y, reduction="none")
        weighted_loss = (is_weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_step += 1
        if self.target_update_freq is not None:
            if self.update_step % self.target_update_freq == 0:
                self.hard_update()
        else:
            self.soft_update()

        new_priorities = td_errors.abs().cpu()
        loss_val = weighted_loss.item()

        return loss_val, td_errors.cpu(), new_priorities, indices

    def save(self, path: str):
        ckpt = {
            "online_state": self.online_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_step": self.update_step,
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.online_net.load_state_dict(ckpt["online_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        self.update_step = ckpt.get("update_step", 0)

    def set_training(self, training: bool):
        self._training = training
        self.online_net.train(training)
        self.target_net.train(training)
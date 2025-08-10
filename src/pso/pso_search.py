import numpy as np
from src.pso.pso import PSO
from src.train import train_and_eval
from src.data.loader import get_dataloaders

ACTIVATIONS = {0: "relu", 1: "gelu", 2: "elu"}
OPTIMIZERS = {0: "adam", 1: "rmsprop"}
LOSSES = {0: "huber", 1: "mse"}

def decode_particle(particle_vector):
    return {
        "learning_rate": 10 ** particle_vector[0],
        "batch_size": int([16, 32, 64, 128][int(particle_vector[1])]),
        "activation": ACTIVATIONS[int(particle_vector[2])],
        "gamma": particle_vector[3],
        "epsilon_decay": particle_vector[4],
        "epsilon_min": particle_vector[5],
        "tau": particle_vector[6],
        "hidden_size": int(particle_vector[7]),
        "optimizer": OPTIMIZERS[int(particle_vector[8])],
        "loss": LOSSES[int(particle_vector[9])],
        "use_noisy": bool(int(particle_vector[10])),
    }

def main():
    train_loader, val_loader, obs_dim, n_actions = get_dataloaders()

    bounds = [
        (-5, -2),   # log10 learning_rate
        (0, 3),     # batch_size index
        (0, 2),     # activation index
        (0.90, 0.999),  # gamma
        (1e-5, 1e-2),   # epsilon_decay
        (0.01, 0.2),    # epsilon_min
        (1e-4, 1e-2),   # tau
        (64, 1024),     # hidden_size
        (0, 1),         # optimizer index
        (0, 1),         # loss index
        (0, 1)          # use_noisy
    ]
    is_int_mask = [False, True, True, False, False, False, False, True, True, True, True]

    pso = PSO(
        num_particles=6,
        dim=len(bounds),
        bounds=bounds,
        is_int_mask=is_int_mask,
        w=0.5,
        c1=1.5,
        c2=1.5,
        max_iters=5
    )

    def fitness_fn(vector):
        hp = decode_particle(vector)
        print(f"\nTesting hyperparams: {hp}")
        score = train_and_eval(
            hp,
            train_loader=train_loader,
            val_loader=val_loader,
            obs_dim=obs_dim,
            n_actions=n_actions,
            max_steps=5000
        )
        print(f"Validation F1: {score:.4f}")
        return score

    best_pos, best_score = pso.optimize(fitness_fn)
    print("\n=== Best Hyperparameters Found ===")
    print(decode_particle(best_pos))
    print(f"Best Score: {best_score:.4f}")

if __name__ == "__main__":
    main()

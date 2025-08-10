import random
import numpy as np

class Particle:
    def __init__(self, dim, bounds, is_int_mask):
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_score = -np.inf
        self.is_int_mask = is_int_mask

    def enforce_bounds(self, bounds):
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])
            if self.is_int_mask[i]:
                self.position[i] = int(round(self.position[i]))


class PSO:
    def __init__(self, num_particles, dim, bounds, is_int_mask, w=0.5, c1=1.5, c2=1.5, max_iters=10):
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = bounds
        self.is_int_mask = is_int_mask
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters

        self.swarm = [Particle(dim, bounds, is_int_mask) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = -np.inf

    def optimize(self, fitness_fn):
        for iter in range(self.max_iters):
            print(f"\n=== PSO Iteration {iter+1}/{self.max_iters} ===")
            for i, particle in enumerate(self.swarm):
                score = fitness_fn(particle.position)

                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)

                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(particle.position)

                print(f"Particle {i} score: {score:.4f}")

            for particle in self.swarm:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                particle.position += particle.velocity
                particle.enforce_bounds(self.bounds)

        return self.global_best_position, self.global_best_score

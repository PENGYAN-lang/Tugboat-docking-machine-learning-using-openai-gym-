from unicodedata import name
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class TugboatDockingEnv(gym.Env):
    

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, num_obstacles=3, render_mode=None):
        super().__init__()

        # state space
        # [tug_x, tug_y, ves_x, ves_y, target_x, target_y, v_x, v_y]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # action_space
        # [dx, dy, fx, fy]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # parameter
        self.dt = 0.1
        self.vessel_mass = 10.0
        self.max_steps = 300
        self.safe_radius = 0.05
        self.num_obstacles = num_obstacles

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # init
        self.tug_pos = np.random.uniform(-0.6, 0.6, 2)
        self.vessel_pos = np.random.uniform(-0.6, 0.6, 2)
        self.vessel_vel = np.zeros(2)
        self.target_pos = np.array([0.8, 0.8])

        self.obstacles = []
        for _ in range(self.num_obstacles):
            center = np.random.uniform(-0.8, 0.8, 2)
            radius = np.random.uniform(0.05, 0.15)
            self.obstacles.append((center, radius))

        self.steps = 0
        obs = np.concatenate([self.tug_pos, self.vessel_pos, self.target_pos, self.vessel_vel])
        return obs, {}

    def step(self, action):
        dx, dy, fx, fy = np.clip(action, -1.0, 1.0)

        # mov
        move_vec = np.array([dx, dy]) * 0.05
        self.tug_pos = np.clip(self.tug_pos + move_vec, -1.0, 1.0)

        # push
        force = np.array([fx, fy]) * 0.5
        acc = force / self.vessel_mass
        self.vessel_vel += acc
        self.vessel_pos += self.vessel_vel * self.dt

        # reward
        dist = np.linalg.norm(self.vessel_pos - self.target_pos)
        reward = 0.0
        done = False

        # 1.Success docking
        if dist < 0.05 and np.linalg.norm(self.vessel_vel) < 0.02:
            reward += 10.0
            done = True

        # 2. Collision
        if np.any(np.abs(self.vessel_pos) > 1.0):
            reward -= 10.0
            done = True

        # 3. Energy penalties
        reward -= 0.1 * np.linalg.norm(force)

        # 4. Progress reward
        reward += 0.2 * (1.0 - dist)

        # 5. Time penalty
        reward -= 0.1

    
        for (center, radius) in self.obstacles:
            d_tug = np.linalg.norm(self.tug_pos - center)
            d_vessel = np.linalg.norm(self.vessel_pos - center)

            if d_tug < radius + self.safe_radius or d_vessel < radius + self.safe_radius:
                reward -= 10.0
                done = True
            elif d_tug < radius + 0.2 or d_vessel < radius + 0.2:
                reward -= 0.2 * (0.2 - (min(d_tug, d_vessel) - radius))

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        obs = np.concatenate([self.tug_pos, self.vessel_pos, self.target_pos, self.vessel_vel])
        return obs, reward, done, False, {}

    def render(self):
        # visualize
        import matplotlib.pyplot as plt
        plt.clf()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.scatter(*self.target_pos, c="green", marker="X", s=120, label="Target")
        plt.scatter(*self.vessel_pos, c="blue", s=80, label="Vessel")
        plt.scatter(*self.tug_pos, c="red", s=60, label="Tugboat")

        for (center, radius) in self.obstacles:
            circle = plt.Circle(center, radius, color="gray", alpha=0.5)
            plt.gca().add_artist(circle)

        plt.legend(loc="upper left")
        plt.title("Tugboat Docking with Obstacles")
        plt.pause(0.001)

    def close(self):
        pass


env = TugboatDockingEnv(num_obstacles=3)
obs, _ = env.reset()
env.render()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000)

obs, _ = env.reset()
env.render()
for _ in range(3000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break
    
env.close()
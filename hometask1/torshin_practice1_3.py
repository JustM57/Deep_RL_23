import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


TAXI = "Taxi-v3"
STATE_N = 500
ACTION_N = 6
ITERATIONS_N = 60
POLICY_M = 64
TRAJECTORY_K = 5
QUANTILE = 0.5
LAPLACE_GAMMA = 0
POLICY_GAMMA = 0.25


class DeterministicAgent:
    def __init__(self, policy: np.array):
        self.policy = policy

    def get_action(self, state):
        return int(self.policy[state])


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, laplace_gamma=0.,  policy_gamma=1.):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
        self.laplace_gamma = laplace_gamma
        self.policy_gamma = policy_gamma

    def get_action(self, state):
        action = np.random.choice(
            np.arange(self.action_n), p=self.model[state]
        )
        return int(action)

    def sample_determinist_policy(self):
        return np.array([self.get_action(state) for state in range(self.state_n)])

    def fit(self, elite_trajectories):
        new_model = np.zeros_like(self.model)
        for trajectory in elite_trajectories:
            for state, action in zip(
                    trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if new_model[state].sum() > 0:
                new_model[state] = (new_model[state] + self.laplace_gamma) / (
                        new_model[state].sum() + self.laplace_gamma * self.action_n)
            else:
                new_model[state] = self.model[state].copy()
        self.model = self.policy_gamma * new_model + (1 - self.policy_gamma) * self.model


def get_trajectory(
        env,
        agent,
        max_steps=1000,
        visualize=False
):
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': []
    }
    state = env.reset()
    for _ in range(max_steps):
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        if visualize:
            time.sleep(0.05)
            env.render()
        if done:
            break
    return trajectory


def train_ce_agent(
    iteration_n: int,
    trajectory_n: int,
    env,
    agent: CrossEntropyAgent,
    q_param: float,
    print_mean=False
):
    iteration_rewards = []
    for iteration in range(iteration_n):
        # policy evaluation
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        mean_reward = np.mean(total_rewards)
        iteration_rewards.append(mean_reward)
        if print_mean:
            print(f"iteration: {iteration}, mean total reward: {mean_reward}")
        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for idx, trajectory in enumerate(trajectories):
            if total_rewards[idx] > quantile:
                elite_trajectories.append(trajectory)
        agent.fit(elite_trajectories)
    return iteration_rewards


def train_deterministic_se_agent(
    iteration_n: int,
    policy_m: int,
    trajectory_k: int,
    env,
    agent: CrossEntropyAgent,
    q_param: float,
    print_mean=False
):
    iteration_rewards = []
    for iteration in range(iteration_n):
        # policy evaluation
        trajectories = []
        total_rewards = []
        for m in range(policy_m):
            agent_m = DeterministicAgent(agent.sample_determinist_policy())
            trajectories.append([get_trajectory(env, agent_m) for k in range(trajectory_k)])
            total_rewards.append(np.mean(
                [np.sum(trajectory['rewards']) for trajectory in trajectories[m]]))
        mean_reward = np.mean(total_rewards)
        iteration_rewards.append(mean_reward)
        if print_mean:
            print(f"iteration: {iteration}, mean total reward: {mean_reward}")
        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for idx in range(policy_m):
            if total_rewards[idx] > quantile:
                for trajectory in trajectories[idx]:
                    elite_trajectories.append(trajectory)
        agent.fit(elite_trajectories)
    return iteration_rewards


def experiment(env, laplace_gamma: float, policy_gamma: float,
               policy_m: int, trajectory_k: int, quantile=QUANTILE):
    agent = CrossEntropyAgent(STATE_N, ACTION_N, laplace_gamma, policy_gamma)
    rewards = train_deterministic_se_agent(
        ITERATIONS_N, policy_m, trajectory_k, env, agent, quantile)
    return rewards


def plot_results(rewards: dict, path: str):
    x = np.arange(ITERATIONS_N)
    plt.grid(True)
    for key, value in rewards.items():
        plt.plot(x, value, label=key)
    plt.xlabel("Iterations")
    plt.ylabel("Mean reward")
    plt.title(path.split('.')[0])
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.clf()


def run_experiments():
    env = gym.make(TAXI)
    # Policy M research
    rewards = {
        m: experiment(env, LAPLACE_GAMMA, POLICY_GAMMA, m, TRAJECTORY_K
                      ) for m in tqdm([16, 32, 64, 128])
    }
    plot_results(rewards, "Policy_M.png")
    # Iterations K research
    rewards = {
        k: experiment(env, LAPLACE_GAMMA, POLICY_GAMMA, POLICY_M, k
                      ) for k in tqdm([2, 4, 8, 16, 32])
    }
    plot_results(rewards, "Trajectory_K.png")
    # Quantile research
    rewards = {
        q: experiment(env, LAPLACE_GAMMA, POLICY_GAMMA, POLICY_M, TRAJECTORY_K, q
                      ) for q in tqdm([0.25, 0.5, 0.75, 0.9])
    }
    plot_results(rewards, "Deter_quantile.png")


if __name__ == '__main__':
    run_experiments()

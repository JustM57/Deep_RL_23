import time

import gym
import numpy as np
import matplotlib.pyplot as plt


TAXI = "Taxi-v3"
STATE_N = 500
ACTION_N = 6
ITERATIONS_N = 60
TRAJECTORY_N = 200
QUANTILE = 0.5


class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        action = np.random.choice(
            np.arange(self.action_n), p=self.model[state]
        )
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros_like(self.model)
        for trajectory in elite_trajectories:
            for state, action in zip(
                    trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if new_model[state].sum() > 0:
                new_model[state] /= new_model[state].sum()
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model


def get_trajectory(env, agent, max_steps=1000, visualize=False):
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
        iteration_n: int, trajectory_n: int, env,
        agent: CrossEntropyAgent, q_param: float,
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


def experiment(env, trajectory_n: int, q_quantile: float):
    agent = CrossEntropyAgent(STATE_N, ACTION_N)
    rewards = train_ce_agent(ITERATIONS_N, trajectory_n, env,
                             agent, q_quantile)
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
    trajectory_n = TRAJECTORY_N
    env = gym.make(TAXI)
    # Quantile research
    rewards = {
        quantile: experiment(env, trajectory_n, round(quantile, 1)
                             ) for quantile in np.linspace(0.1, 0.9, 9)
    }
    plot_results(rewards, "quantile.png")
    # number of trajectories per iteration research
    rewards = {
        trajectory_n: experiment(env, trajectory_n, QUANTILE
                                 ) for trajectory_n in [50, 100, 200, 400, 800]
    }
    plot_results(rewards, "n_trajectories.png")


if __name__ == '__main__':
    run_experiments()

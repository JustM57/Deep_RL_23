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
LAPLACE_GAMMA = 0
POLICY_GAMMA = 1


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


def experiment(env, laplace_gamma: float, policy_gamma: float,
               trajectory_n=TRAJECTORY_N, quantile=QUANTILE):
    agent = CrossEntropyAgent(STATE_N, ACTION_N, laplace_gamma, policy_gamma)
    rewards = train_ce_agent(ITERATIONS_N, trajectory_n, env,
                             agent, quantile)
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
    # Laplace gamma
    rewards = {
        gamma: experiment(env, gamma, POLICY_GAMMA
                          ) for gamma in [0., 0.1, 0.25, 0.5, 1., 2., 10.]
    }
    plot_results(rewards, "laplace_smoothing.png")
    # Policy gamma research
    rewards = {
        gamma: experiment(env, LAPLACE_GAMMA, gamma
                          ) for gamma in [0.1, 0.25, 0.5, 0.75, 0.9, 1]
    }
    plot_results(rewards, "policy_smoothing.png")
    rewards = {
        "quantile_0.5": experiment(env, LAPLACE_GAMMA, POLICY_GAMMA),
        "laplace_0.25": experiment(env, 0.25, POLICY_GAMMA),
        "policy_0.25": experiment(env, LAPLACE_GAMMA, 0.25),
        "laplace+policy_0.25": experiment(env, 0.25, 0.25),
        "combined": experiment(env, 0.25, 0.25, trajectory_n=400)
    }
    plot_results(rewards, "features_comparison.png")


if __name__ == '__main__':
    run_experiments()

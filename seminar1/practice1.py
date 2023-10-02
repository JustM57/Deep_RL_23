import time
import math

import gym
import gym_maze
import numpy as np


MAZE = 'maze-sample-10x10-v0'
STATE_N = 100
ACTION_N = 4
ITERATIONS_N = 20
TRAJECTORY_N = 50
QUANTILE = 0.9


class RandomAgent:
    def __init__(self, action_n: int):
        self.action_n = action_n

    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action


class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n),
                                  p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros_like(self.model)
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if new_model[state].sum() > 0:
                new_model[state] /= new_model[state].sum()
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model


def get_state(obs):
    return int(math.sqrt(STATE_N) * obs[0] + obs[1])


def train_ce_agent(
        iteration_n: int, trajectory_n: int, env, agent: CrossEntropyAgent, q_param: float
):
    for iteration in range(iteration_n):
        # policy evaluation
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        mean_reward = np.mean(total_rewards)
        print(f"iteration: {iteration}, mean total reward: {mean_reward}")
        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for idx, trajectory in enumerate(trajectories):
            if total_rewards[idx] > quantile:
                elite_trajectories.append(trajectory)
        agent.fit(elite_trajectories)


def get_trajectory(env, agent, max_steps=1000, visualize=False):
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': []
    }
    obs = env.reset()
    state = get_state(obs)
    for _ in range(max_steps):
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        state = get_state(obs)
        if visualize:
            time.sleep(0.5)
            env.render()
        if done:
            break
    return trajectory


def main():
    env = gym.make(MAZE)
    # agent = RandomAgent(ACTION_N)
    agent = CrossEntropyAgent(STATE_N, ACTION_N)
    train_ce_agent(ITERATIONS_N, TRAJECTORY_N, env, agent, QUANTILE)

    trajectory = get_trajectory(env, agent, 1000, True)
    total_reward = np.sum(trajectory['rewards'])
    print(f"Total reward {total_reward}")


if __name__ == '__main__':
    main()

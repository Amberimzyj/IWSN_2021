# Project: <<selftry_ten_armed>>
# File Created: Friday, 28th August 2020 10:57:50 pm
# Author: <<Yajing Zhang>> (<<amberimzyj@qq.com>>)
# -----
# Last Modified: Saturday, 29th August 2020 1:49:17 am
# Modified By: <<Yajing Zhang>> (<<amberimzyj@qq.com>>>)
# -----
# Copyright 2020 - <<2020>> <<Yajing Zhang>>, <<IWIN>>

import numpy as np

from iwsn.rtsn import RTSN


class RL:

    def __init__(self, data_path: str = 'data/RTSN.csv',
                 k_arm=10, 
                 epsilon=0., 
                 initial=0., 
                 step_size=0.1, 
                 sample_averages=False, 
                 UCB_param=None,
                 gradient=False, 
                 gradient_baseline=False, 
                #  true_reward=0.
                 ):
        self._data = data_path
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        # self.inte_delay_true_reward = self._data['inte_delay']
        self.epsilon = epsilon
        self.initial = initial

        if not os.path.exists(data_path):
            raise FileNotFoundError(f'data path "{data_path}" not exists.')

        self.RTSN_data = np.loadtxt(data_path)  # 读取RTSN相关信息

    def reset(self):
        # real reward for each action

        self.inte_delay_true = np.random.randn(self.k) + self._data['inte_delay']

        # estimation for each action
        self.inte_delay_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.inte_delay_true)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:  # epsilon-贪心算法
            return np.random.choice(self.indices)

        if self.UCB_param is not None:  # UCB算法
            UCB_estimation = self.q_estimation + \
                self.UCB_param * \
                np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:  # 梯度赌博机
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)  # 吉布斯-玻尔兹曼分布
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)  # 贪心算法
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:  # 采样平均更新reward
            # update estimation using sample averages
            self.q_estimation[action] += (reward -
                                          self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward  # 收益基准项即为平均收益
            else:
                baseline = 0
            self.q_estimation += self.step_size * \
                (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * \
                (reward - self.q_estimation[action])
        return reward

    def simulate(runs, time, bandits):
        rewards = np.zeros((len(bandits), runs, time))
        best_action_counts = np.zeros(rewards.shape)
        for i, bandit in enumerate(bandits):
            for r in trange(runs):
                bandit.reset()
                for t in range(time):
                    action = bandit.act() 
                    reward = bandit.step(action)
                    rewards[i, r, t] = reward
                    if action == bandit.best_action:
                        best_action_counts[i, r, t] = 1
        mean_best_action_counts = best_action_counts.mean(axis=1)
        mean_rewards = rewards.mean(axis=1)
        return mean_best_action_counts, mean_rewards

    @ indexedproperty
    def t_5G(self, key: float) -> float:
        return self._data.at[key, 't_5G']

    @ trans_slot.setter
    def t_5G(self, key: int, value: float):
        self._data.at[key, 't_5G'] = value
    
    @ indexedproperty
    def q_t(self, key: float) -> float:
        return self._data.at[key, 'q_t']

    @ trans_slot.setter
    def q_t(self, key: int, value: float):
        self._data.at[key, 'q_t'] = value

    @ indexedproperty
    def t_tsn(self, key: float) -> float:
        return self._data.at[key, 't_tsn']

    @ trans_slot.setter
    def t_tsn(self, key: int, value: float):
        self._data.at[key, 't_tsn'] = value

    @ indexedproperty
    def inte_delay(self, key: float) -> float:
        return self._data.at[key, 'inte_delay']

    @ trans_slot.setter
    def inte_delay(self, key: int, value: float):
        self._data.at[key, 'inte_delay'] = value
    

if __name__ == '__main__':
    q = np.arange(0, 7)
    rb = np.arange(0, 15)
    

    rl = RL()

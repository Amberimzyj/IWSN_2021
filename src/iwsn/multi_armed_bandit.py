# Project: <<selftry_ten_armed>>
# File Created: Friday, 28th August 2020 10:57:50 pm
# Author: <<Yajing Zhang>> (<<amberimzyj@qq.com>>)
# -----
# Last Modified: Saturday, 29th August 2020 1:49:17 am
# Modified By: <<Yajing Zhang>> (<<amberimzyj@qq.com>>>)
# -----
# Copyright 2020 - <<2020>> <<Yajing Zhang>>, <<IWIN>>

import os
from pathlib import Path
import shutil

import numpy as np
from indexedproperty import indexedproperty
from pandas.core.indexes import base
from tqdm import trange
import pandas as pd

from iwsn.rtsn import RTSN
from matplotlib import pyplot as plt


class Bandit:

    def __init__(self, data_path: str = 'data/RTSN_res',
                 k_arm=10,
                 epsilon=0.,
                 initial=0.,
                 step_size=0.1,
                 sample_averages=False,
                 UCB_param=None,
                 gradient=False,
                 gradient_baseline=False,
                 true_reward=0.
                 ):
        self._data = data_path
        # self.rtsn = RTSN()
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

        self.RTSN_data = {}
        for path in os.listdir(data_path):
            self.RTSN_data[int(path[:-4].split('_')[-1])
                           ] = pd.read_csv(os.path.join(data_path, path))

    def gen_true_value(self, k: int, name: str):
        self.rtsn = RTSN()
        self.k = self.rtsn.subslot
        classical_true_value = []
        risk_true_value = []
        for res in range(self.rtsn.subslot):
            # sum = np.sum(self.RTSN_data[res][name])/10000
            mean = -np.mean(self.RTSN_data[res][name])
            classical_true_value.append(mean)
            # 计算risk-sensitive的reward，k是高阶量的系数
            risk_value = -np.exp(-1*k*mean/100)
            risk_true_value.append(risk_value)
        return classical_true_value, risk_true_value

    def gen_inte_delay(self, res: int, q_t: int, k: int, name: str):
        self.rtsn = RTSN(res_subslot_num=res)
        self.k = self.rtsn.subslot * 8
        inte_delay = 0
        for t in range(len(self.RTSN_data[res])):
            t_tsn = self.rtsn.cal_tsn_delay(t, q_t)
            inte_delay += t_tsn
        mean = inte_delay/len(self.RTSN_data[res])
        t_5G = np.mean(self.RTSN_data[res]['t_5G'])
        mean += t_5G
        # mean = inte_delay/len(self.RTSN_data[0])
        return mean

    def reset(self, with_risk=True):
        # real reward for each action

        classical_true_value, risk_true_value = self.gen_true_value(
            2, 'inte_delay')

        if with_risk:
            self.inte_delay_true = np.random.randn(self.k) + risk_true_value
        else:
            self.inte_delay_true = np.random.randn(
                self.k) + classical_true_value

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
            UCB_estimation = self.inte_delay_estimation + \
                self.UCB_param * \
                np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:  # 梯度赌博机
            exp_est = np.exp(self.inte_delay_estimation)
            self.action_prob = exp_est / np.sum(exp_est)  # 吉布斯-玻尔兹曼分布
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.inte_delay_estimation)  # 贪心算法
        return np.random.choice(np.where(self.inte_delay_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.inte_delay_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:  # 采样平均更新reward
            # update estimation using sample averages
            self.inte_delay_estimation[action] += (reward -
                                                   self.inte_delay_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward  # 收益基准项即为平均收益
            else:
                baseline = 0
            self.inte_delay_estimation += self.step_size * \
                (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.inte_delay_estimation[action] += self.step_size * \
                (reward - self.inte_delay_estimation[action])

        inte_delay = self.RTSN_data[action]['inte_delay'].to_numpy().mean()
        return reward, inte_delay


def simulate(runs, time, bandits, with_risk):
    rewards = np.zeros((len(bandits), runs, time))
    inte_delays = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset(with_risk)
            for t in range(time):
                action = bandit.act()
                # print(action)
                reward, inte_delay = bandit.step(action)
                rewards[i, r, t] = reward
                inte_delays[i, r, t] = inte_delay
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards, inte_delays.mean(axis=1)


def figure_2_5(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1,
                          gradient_baseline=True, true_reward=250))
    bandits.append(Bandit(gradient=True, step_size=0.1,
                          gradient_baseline=False, true_reward=250))
    bandits.append(Bandit(gradient=True, step_size=0.4,
                          gradient_baseline=True, true_reward=250))
    bandits.append(Bandit(gradient=True, step_size=0.4,
                          gradient_baseline=False, true_reward=150))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.yticks(np.arange(200, 250))
    plt.legend()
    plt.grid()

    plt.show()

    # plt.savefig('/images/figure_2_5.png')
    # plt.close()


def figure_inte_delay(runs=[20, 100, 500], time=1000):
    bandits = []
    bandits.append(Bandit(gradient=True, k_arm=15, step_size=0.1,
                          gradient_baseline=True, true_reward=0))
    # bandits.append(Bandit(gradient=True, k_arm = 15, step_size=0.4, gradient_baseline=True, true_reward=0))
    # bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=0))
    # bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=-100))
    # bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=250))

    for run in runs:
        base_dir = Path('data/Bandit_data/using') / str(run)
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        for with_risk in [True, False]:
            _, rewards, inte_delays = simulate(
                run, time, bandits, with_risk)
            data = np.zeros((len(bandits), 2))

            # labels = ['alpha = 0.1, with baseline',
            #         'alpha = 0.4, with baseline']
            labels = ['step_size=0.1, with baseline',
                      'step_size=0.4, with baseline',
                      'true_reward=0, without baseline',
                      'true_reward=-100, without baseline']

            # for i in range(len(bandits)):
            #     # data[i][0] =
            #     plt.plot(-rewards[i], label=labels[i])
            # plt.xlabel('Steps')
            # plt.ylabel('% Risk-sensitive delay')
            # # plt.yticks(np.arange(225,240,27))
            # plt.legend()
            # plt.grid(linestyle='--')

            # plt.show()

            if (with_risk):
                np.savetxt(base_dir / 'data_risk.csv', inte_delays)
            else:
                np.savetxt(base_dir / 'data_without_risk.csv', inte_delays)


if __name__ == '__main__':

    figure_inte_delay()

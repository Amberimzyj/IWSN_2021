# Project: <<selftry_ten_armed>>
# File Created: Friday, 28th August 2020 10:57:50 pm
# Author: <<Yajing Zhang>> (<<amberimzyj@qq.com>>)
# -----
# Last Modified: Saturday, 29th August 2020 1:49:17 am
# Modified By: <<Yajing Zhang>> (<<amberimzyj@qq.com>>>)
# -----
# Copyright 2020 - <<2020>> <<Yajing Zhang>>, <<IWIN>>

import os
import numpy as np
import matplotlib.pyplot as plt
import contribution_rl
import pandas as pd
from tqdm import trange



class RTSN:

    def __init__(self, 
                # data_path: str = 'X2_pro.csv',
                #TTI = 0, 
                C = 4, 
                subslot = 15, 
                t_rb = 5, 
                res_subslot_num = 4, 
                signal_ratio = 0.15,
                t_tsn_max = 40,
                t_tsn_min = 8,
                t_ddl = 100,
                D = 60, #假设一个RB传输的数据量是60kbit
                H = 5,
                r_tsn_min = 20,
                t_cyc = 5):
        
        # if not os.path.exists(data_path):
        #     raise FileNotFoundError(f'data path "{data_path}" not exists.')

        # self.X2_pro = pd.read_csv(data_path) #读取卡方概率
        

        #self.TTI = TTI #记录TTI
        self.C = C #信道数
        self.subslot = subslot #子时隙数
        self.t_rb = t_rb #每个rb的duration
        self.res_subslot_num = res_subslot_num #预留子时隙数量
        self.signal_ratio = signal_ratio #信令时延比例
        self.t_tsn_max = t_tsn_max #TSN网络做大时延
        self.t_tsn_min = t_tsn_min #TSN网络最小时延
        self.t_ddl = t_ddl #数据ddl
        self.D = D #一个RB传输的数据量是60kbit
        self.H = H #TSN网络跳数
        self.r_tsn_min = r_tsn_min #TSN网络最小传输速率
        self.t_cyc = t_cyc # TSN网关循环转发时间
        self.res_rb_num = self.C * self.res_subslot_num #预留RB数量
        self.con = contribution_rl.SensorContrib(res_num = res_subslot_num * C)
    
    # def reset(self):
    #      self.TTI = 0

    #      self.t_5G = [] #存放每个TTI的5G时延

    #      self.t_tsn = [] #存放每个TTI的TSN时延

    #      self.inte_delay = [] #存放每个TTI的总时延

    def gen_pro(self, cir_num:int):
        act_times = np.zeros((self.con._length),'int64')
        joint_times = np.zeros((self.con._length,self.con._length),'int64')
        
        for i in range(1,cir_num):
            act_times_temp, joint_times_temp = self.con.circul(1)
            act_times += act_times_temp
            joint_times += joint_times_temp
            pri_pro, cond_pro, pri_pro_re, joint_pro  = self.con.cal_cond_pro(act_times, joint_times, i)
            X2_pro = self.con.cal_X2(act_times, joint_times, i)
        
        return X2_pro

    def data_gen_dynamic_tc(self) -> int:
        """生成动态接入部分的TC流

        Returns:
            int: 返回生成TC的数量
        """
        #生成动态调度部分的TC流，预留部分由contribution_rl生成

        tc = np.random.poisson(lam = 3, size = 100) #产生泊松分布的TC流
        # rb = np.empty((self.C, self.subslot)) #生成C*subslot的二维空矩阵
        tc_num = 0 #记录TC流数量
        for i in range(len(tc)):
            if tc[i] > 3:
                tc_num += 1
        return tc_num


    def reserve_sensor(self, t:int) -> int:
        """预测性预留节点，并记录触发和抢占情况

        Args:
            t (int): 当前TTI

        Returns:
            int： 触发的预分配节点数量
            list：触发的预分配节点index
            int:  存放固定预留RB上预留且触发且被抢占的节点数量
        """
        X2_pro = self.gen_pro(3)
        _ , pre_sensor , trigger_sensor = self.con.cal_pre_accu(t, X2_pro) #预测t时刻触发节点
        # print(pre_sensor)
        # print(trigger_sensor)
        pre_sensor = np.array(pre_sensor)
        pre_sensor = pre_sensor.reshape((self.C, self.res_subslot_num)) #将预测节点存放至预留RB
        # print(pre_sensor)
        trigger_num = 0 #存放固定预留RB上预留且触发的节点数量
        trigger_index = [] #存放固定预留RB上预留且触发的节点标号
        for i in range(self.res_subslot_num):
            # if trigger_sensor.contains(pre_sensor[1][i]):
            if pre_sensor[1][i] in trigger_sensor:
                trigger_num += 1
                trigger_index.append(i)
        _, ns_index, _, =self.data_gen_ns()
        trigger_preempt_num = len(np.intersect1d(trigger_index, ns_index)) #存放固定预留RB上预留且触发且被抢占的节点数量

        return trigger_num, trigger_index, trigger_preempt_num


    def data_gen_ns(self)-> int:
        """生成突发NS流

        Returns:
            int, list, int: NS流数量， NS流index， 预分配区域的NS流数量
        """
        ns = np.random.rand(1, self.subslot) #产生一个长度为subslot的0-1随机数list
        ns_index = [] #建立空list存放NS流的index
        preempt_num = 0 #存放预留RB内产生的NS流数量
        for i in range(self.subslot):
            if ns[0][i] > 0.9:
                ns_index.append(i)
        ns_num = len(ns_index) #存放NS流数量
        for j in range(len(ns_index)):
            if ns_index[j] < self.res_subslot_num:
                preempt_num += 1

        return ns_num, ns_index, preempt_num

    def cal_5G_delay(self, t:int ) -> float:
        """计算5G通信时延

        Args:
            t (int): 当前TTI

        Returns:
            float, float: 5G传输时延， 高优先级RB数量
        """

        tc_num = self.data_gen_dynamic_tc()
        trigger_num, _, trigger_preempt_num = self.reserve_sensor(t)
        ns_num, _, preempt_num = self.data_gen_ns()
        s_ht = (1 - 1/self.C) * self.res_rb_num + tc_num + ns_num - trigger_preempt_num
        dict = {'self.res_rb_num:': self.res_rb_num, 'tc_num:':tc_num,'ns_num:':ns_num, 'self.res_subslot_num:':self.res_subslot_num, 'preempt_num:':preempt_num,'trigger_preempt_num:':trigger_preempt_num,'self.signal_ratio:':self.signal_ratio,'s_ht:':s_ht,'self.t_rb:':self.t_rb }
        # print(dict)
        # S_ht = self.res_num - np.trunc(self.res_subslot_num - (trigger_num + preempt_num - trigger_preempt_num))
        t_5G = np.trunc((self.res_rb_num + (tc_num + ns_num - (self.res_subslot_num - (preempt_num - trigger_preempt_num)))*(1 + self.signal_ratio))/(self.C * s_ht))*self.t_rb

        return t_5G, s_ht

    def cal_tsn_delay(self, t:int, q_t:int) -> int:
        """计算TSN传输时延

        Args:
            t (int): 当前TTI
            q_t (int): 队列级数

        Returns:
            int: TSN传输时延
        """
        _, s_ht = self.cal_5G_delay(t)
        q_duration = (self.t_tsn_max - self.t_tsn_min)/8 #相邻队列之间的传输时间差
        q_delay = self.t_tsn_min + q_t * q_duration #第q_t个队列的时延函数
        t_tsn = self.H * np.trunc(s_ht * self.D * q_delay/self.r_tsn_min/self.t_cyc) * self.t_cyc

        return t_tsn

    def cal_d_q(self, q:int, t_5G:float)-> float:
        """计算不同TSN队列的传输时延

        Args:
            q (int): TSN队列等级
            t_5G (float): 当前周期5G传输时延

        Returns:
            float: 各TSN队列传输时延
        """
        q_duration = (self.t_tsn_max - self.t_tsn_min)/8 #相邻队列之间的传输时间差
        d = self.t_tsn_min + q * q_duration - self.t_ddl + t_5G[1]
        return d


    def cal_q_t(self, t_5G:float) -> int:
        """计算TSN队列级数

        Args:
            t_5G (float): 当前周期5G传输时延
            
        Returns:
            int: 应该注入的TSN队列级数
        """
        # q_duration = (self.t_tsn_max - self.t_tsn_min)/8 #相邻队列之间的传输时间差
        q_t = 0
        while self.cal_d_q(q_t,t_5G) < 0.0:
        # while (self.t_tsn_min + q * q_duration - self.t_ddl + t_5G) < 0.0:
            q_t += 1
        # q_t = np.argmin(self.t_tsn_min + q_t * q_duration - (self.t_ddl - t_5G))

        return q_t


def simulate(runs:int, time:int, rtsns) -> list:
    """运行RTSN函数得到相应参数并存入列表

    Args:
        runs (int): 运行轮数
        time (int): 每轮运行次数
        rtsns ([class]): RTSN参数列表

    Returns:
        list, list, list, list: 5G时延, TSN时延, TSN队列优先级, 总时延
    """
    rtsn = RTSN()
    # delays = np.zeros((len(rtsns), runs, time))
    t_5G = [] #存放5G时延
    q_t = [] #存放队列级数
    t_tsn = [] #存放TSN时延
    inte_delay = [] #存放总时延
    # for i, rtsn in enumerate(rtsns):
    for r in trange(runs):
        # RTSN.reset()
        for t in range(time):
            t_5G.append(rtsn.cal_5G_delay(t))
            q_t.append(rtsn.cal_q_t(t_5G[t]))
            t_tsn.append(rtsn.cal_tsn_delay(t, q_t[t]))
            inte_delay.append((t_5G[t] + t_tsn[t]))
    return t_5G, t_tsn, q_t, inte_delay

def test(runs = 10, time = 10):
    rtsn = RTSN()
    t_5G, t_tsn, q_t, inte_delay = simulate(runs, time, rtsn)
    print('t_5G:')
    print(t_5G)










class RL:
    
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon: #epsilon-贪心算法
            return np.random.choice(self.indices) 

        if self.UCB_param is not None: #UCB算法
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient: #梯度赌博机
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est) #吉布斯-玻尔兹曼分布
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation) #贪心算法
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages: #采样平均更新reward
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:  
                baseline = self.average_reward  #收益基准项即为平均收益
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward
    

if __name__ == '__main__':
    test()



    print('=> Generate done.')

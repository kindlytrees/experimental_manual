import numpy as np
import math

class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma  # 衰减系数
        self.epsilon = 0    # 按一定的概率随机选动作
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.eval = False

    def choose_action(self, state):
        #  智能体的决策函数，需要完成Q表格方法（需要完成）
        if not self.eval:
            self.sample_count += 1

            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon - self.epsilon_end) * \
                        math.exp(-1.0 * self.sample_count / self.epsilon_decay)

            if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
                action = self.predict(state)  # 调用函数获得要在该观察值（或状态）条件下要执行的动作
            else:
                action = np.random.choice(self.action_dim)  # e_greedy概率直接从动作空间中随机选取一个动作
            return action
        else:
            action = self.predict(state)
            return action
        # #

    def predict(self, state):
        # 根据Q表格采样输出的动作值（需要完成）
        #print(state)
        Q_list = self.Q_table[state, :]  # 从Q表中选取状态(或观察值)对应的那一行
        maxQ = np.max(Q_list)  # 获取这一行最大的Q值，可能出现多个相同的最大值

        action_list = np.where(Q_list == maxQ)[0]  # np.where(条件)功能是筛选出满足条件的元素的坐标
        action = np.random.choice(action_list)  # 这里尤其如果最大值出现了多次，随机取一个最大值对应的动作就成

        return action
        # #

    def update(self, state, action, reward, next_state, done):
        # Q表格的更新方法（需要完成）
        """
            on-policy
            obs：交互前的obs, 这里observation和state通用，也就是公式或者伪代码码中的s_t
            action： 本次交互选择的动作， 也就是公式或者伪代码中的a_t
            reward: 本次与环境交互后的奖励,  也就是公式或者伪代码中的r
            next_obs: 本次交互环境返回的下一个状态，也就是s_t+1
            next_action: 根据当前的Q表，针对next_obs会选择的动作，a_t+1
            done: 回合episode是否结束
        """
        predict_Q = self.Q_table[state, action]
        if done:
            target_Q = reward  # 如果到达终止状态， 没有下一个状态了，直接把奖励赋值给target_Q
        else:
            target_Q = reward + self.gamma * np.max(self.Q_table[next_state])  # 这两行代码直接看伪代码或者公式
        self.Q_table[state, action] = predict_Q + self.lr * (target_Q - predict_Q)  # 修正q
        ##

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")

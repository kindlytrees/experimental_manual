import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
import gym
import torch
import torch.nn.functional as F
import os

class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def predict(self, state):  # 用于打印策略
        max_idx = np.argmax(self.Q_table[state])
        return max_idx
    
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")

ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

save_dir = './outputs'
os.makedirs(save_dir, exist_ok=True)

# 创建 CliffWalking 环境
env = gym.make('CliffWalking-v0', render_mode='human')
action_map = {0:0, 1:2, 2:3, 3:1}
def eval(eval_eps, env, agent):
    print('Start to eval !')
    # print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []  # 记录所有episode的reward
    running_rewards = []  # 滑动平均的reward
    for i_ep in range(eval_eps):
        ep_reward = 0  # 记录每个episode的reward
        state, _ = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            ori_action = agent.predict(state)  # 根据算法选择一个动作
            action = action_map[ori_action]
            next_state, reward, done, _, _ = env.step(action)  # 与环境进行一个交互
            env.render()
            print(state, action, next_state, reward)
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if running_rewards:
            running_rewards.append(running_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            running_rewards.append(ep_reward)
        print(f"Episode:{i_ep+1}/{eval_eps}, reward:{ep_reward:.1f}")

    print('Complete evaling！')
    return rewards, running_rewards

eval(5, env, agent)
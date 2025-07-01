# 策略梯度实验代码说明

多次loss.backward()计算的梯度会累积（Gradient Accumulation），self.optimizer.step()更新参数，self.optimizer.zero_grad()会将梯度清零。

```
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数（G为从当前步骤开始到结束状态的折扣奖励和），负数最小对应着正数最大（期望的奖励不断提升），对应policy gradient 1.9和1.11部分的介绍，这里不是梯度，而是对数概率，公式里边还有一个梯度算子
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
```

- 关于损失函数按照上述代码进行实现的说明，策略梯度上升算法，其对应的损失函数得采用深度学习框架里现成的梯度下降算法，因此添加了负号，而损失函数也将策略梯度中的\nabla还原成原始对数函数的计算，上述这个过程完美地利用了深度学习框架的现有工具，通过定义一个巧妙的“代理损失函数”。

$$
\begin{aligned}
\nabla \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} G_t^n \nabla \log \pi_\theta\left(a_t^n \mid s_t^n\right) \\
J(\theta) \approx -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} G_t^n  \log \pi_\theta\left(a_t^n \mid s_t^n\right)
\end{aligned}
$$
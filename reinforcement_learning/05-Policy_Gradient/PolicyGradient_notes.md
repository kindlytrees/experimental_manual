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
            loss = -log_prob * G  # 每一步的损失函数（G为从当前步骤开始到结束状态的折扣奖励和），负数最小对应着正数最大（期望的奖励不断提升），对应policy gradient 1.9和1.11部分的介绍，这里不是梯度，而是对数概率，公式里边还有一个梯度算子，这样和ppo的公式就统一了
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
```
# Actor Critic Notes

- 价值网络的损失函数是当前状态价值网络输出和一阶的回报函数(td_target)的mse

```
td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
```
- 策略梯度的损失函数为REINFORCEMENT策略梯度算法的变种, 其中权重设为时序差分误差。

```
td_delta = td_target - self.critic(states)  # 时序差分误差
actor_loss = torch.mean(-log_probs * td_delta.detach())
```

- 在actor-critic算法中，时序差分误差作为权重的意义解释，是否说明误差表明当前状态价值函数估值差异的大小，越大说明当前低估了当前状态的价值，因此应该在此状态下提升相应的策略梯度？

gemini中的回答和上述观点的一致性：“误差表明当前状态价值函数估值差异的大小，越大说明当前低估了当前状态的价值，因此应该在此状态下提升相应的策略梯度。”
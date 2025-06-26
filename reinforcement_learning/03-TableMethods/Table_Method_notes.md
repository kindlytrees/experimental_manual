# Table Method notes

Sarsa和QLearning都是基于Q表格更新的最优策略求解算法，这里的实验基于cliffwalking仿真环境进行。
- Sarsa是同策略的Q表格学习算法，下一个状态的Q函数Q'(s',a')
- QLearning是一种异策略的Q表格学习算法,下一个状态的Q函数

同策略和异策略和策略迭代和值迭代算法之间是不是直接相对应？即sarsa是基于策略迭代的表格法，QLearning基于Q的状态动作的值函数迭代？

Q-Learning和SARSA是将DP中的VI和PI思想，应用到无法获得完整环境模型的现实场景中的学习算法。
SARSA可以看作是“采样版本的、模型无关的策略迭代”。 它不是像DP那样完整地进行评估和改进，而是在每一步交互中，都同时进行一小步的评估和一小步的改进。
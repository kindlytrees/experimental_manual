# GBDT-XGBoost


## GBDT


## XGBoost

$$
\begin{gathered}
f\left(x_0+\Delta x\right) \approx f\left(x_0\right)+f^{\prime}\left(x_0\right) \Delta x+\frac{f^{\prime \prime}\left(x_0\right)}{2}(\Delta x)^2 \\
O b j=\sum_{i=1}^N\left[L\left[F_{m-1}\left(x_i\right), y_i\right]+\frac{\partial L}{\partial F_{m-1}\left(x_i\right)} f_m\left(x_i\right)+\frac{1}{2} \frac{\partial^2 L}{\partial^2 F_{m-1}\left(x_i\right)} f_m^2\left(x_i\right)\right]+ \sum_{j=1}^m \Omega\left(f_j\right)
\end{gathered}
$$


定义节点 $j$ 上的样本集为 $I(j)=\left\{x_i \mid q\left(x_i\right)=j\right\}$ ，其中 $q\left(x_i\right)$ 为将样本映射到叶节点上的索引函数，叶节点 $j$ 上的回归值为 $w_j=f_m\left(x_i\right), i \in I(j)$ ．

XGBoost中CART树的叶子节点的目标值的求解关乎该叶子节点下面的损失函数相关的一阶导数和二节导数及残差关系式的二次函数最小化的求解过程。

$$
O b j=\sum_{j=1}^T\left[G_j w_j+\frac{1}{2}\left(H_j+\lambda\right) w_j^2\right]+\gamma T
$$
# 核方法

- LMS with kernel trick

$$
\begin{aligned}
\theta= & \theta+\alpha \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right) x^{(i)} \\
\Rightarrow \quad \theta= & \theta+\alpha \sum_{i=1}^n \left(y^{(i)}-\theta^{\top} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right). \\
& \theta=\sum_{i=1}^n \beta_i \phi\left(x^{(i)}\right) \\
\Rightarrow \quad \theta= & \sum_{i=1}^n \beta_i \phi\left(x^{(i)}\right)+\alpha \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right) \\
= & \sum_{i=1}^n\left(\beta_i+\alpha\left(y^{(i)}-\theta^{\top} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right)\right. \\
& \\
& \quad \beta_i=\beta_i+\alpha\left(y^{(i)}-\sum_{j=1}^n \beta_j \phi\left(x^{(j)}\right)^{\top} \phi\left(x^{(i)}\right)\right) \\
& \left.\phi\left(x^{(j)}\right)\right)^{\top} \phi\left(x^{(i)}\right)=\left\langle\phi\left(x^{(j)}\right), \phi\left(x^{(i)}\right)\right\rangle
\end{aligned}
$$

- 线性函数的参数可以看成为训练样本的线性组合的解释

$$
\begin{aligned}
w & =\left(X^{\top} X\right)^{-1} X^{\top} y \\
& =X^{\top}\left(X X^{\top}\right)^{-1} y \\
& =X^{\top} \alpha \\
\left(X^{\top} X\right)^{-1} X^{\top} \cdot X X^{\top} & =\left(X^{\top} X\right)^{-1} \cdot\left(X^{\top} \cdot X\right) \cdot X^{\top}=X^{\top} \\
X^{\top}\left(X X^{\top}\right)^{-1} \cdot X X^{\top} & =X^{\top}\left(X X^{\top}\right)^{-1} \cdot\left(X X^{\top}\right)=X^{\top}
\end{aligned}
$$

## 实验及说明

- 待完成

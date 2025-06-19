# Nenual Network

## 激活函数的种类

- logistic funciton

$$
f(z)=\frac{1}{1+\exp (-z)}
$$

- 双曲正切函数

$$
f(z)=\tanh (z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

- ReLU

$$
f(z)=\max (0, x)
$$


## 神经网络的函数表示

[!Nenual Network](./nn.png)

$$
\begin{gathered}
a_1^{(2)}=f\left(W_{11}^{(1)} x_1+W_{12}^{(1)} x_2+W_{13}^{(1)} x_3+b_1^{(1)}\right) \\
a_2^{(2)}=f\left(W_{21}^{(1)} x_1+W_{22}^{(1)} x_2+W_{23}^{(1)} x_3+b_2^{(1)}\right) \\
a_3^{(2)}=f\left(W_{31}^{(1)} x_1+W_{32}^{(1)} x_2+W_{33}^{(1)} x_3+b_3^{(1)}\right) \\
h_{W, b}(x)=a_1^{(3)}=f\left(W_{11}^{(2)} a_1^{(2)}+W_{12}^{(2)} a_2^{(2)}+W_{13}^{(2)} a_3^{(2)}+b_1^{(2)}\right) \\
z_i^{(2)}=\sum_{j=1}^n W_{i j}^{(1)} x_j+b_i^{(1)} \quad a_i^{(l)}=f\left(z_i^{(l)}\right) \\
z^{(2)}=W^{(1)} x+b^{(1)} \\
a^{(2)}=f\left(z^{(2)}\right) \\
z^{(3)}=W^{(2)} a^{(2)}+b^{(2)} \\
h_{W, b}(x)=a^{(3)}=f\left(z^{(3)}\right)
\end{gathered}
$$

- 单个样本的损失函数

$$
J(W, b ; x, y)=\frac{1}{2}\left\|h_{W, b}(x)-y\right\|^2
$$

- 所有样本及

$$
\begin{aligned}
J(W, b) & =\left[\frac{1}{m} \sum_{i=1}^m J\left(W, b ; x^{(i)}, y^{(i)}\right)\right]+\frac{\lambda}{2} \sum_{l=1}^{n_l-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(W_{j i}^{(l)}\right)^2 \\
& =\left[\frac{1}{m} \sum_{i=1}^m\left(\frac{1}{2}\left\|h_{W, b}\left(x^{(i)}\right)-y^{(i)}\right\|^2\right)\right]+\frac{\lambda}{2} \sum_{l=1}^{n_l-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(W_{j i}^{(l)}\right)^2
\end{aligned}
$$
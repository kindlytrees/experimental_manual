# PCA

## SVD

SVD中的v向量是A^TA的特征向量的时候，其V的正交向量基通过A变换即可变换为另一个不同长度维度的正交向量基U

$$
\begin{aligned}
& \left\|A v_i\right\|^2=\left(A v_i\right)^T *\left(A v_i\right) \\
& \Rightarrow\left\|A v_i\right\|^2=v_i^T A^T A v_i \\
& \Rightarrow\left\|A v_i\right\|^2=\lambda_i v_i^T v_i=\lambda_i \\
& \therefore\left\|A v_i\right\|=\sqrt{\lambda_i}
\end{aligned}
$$


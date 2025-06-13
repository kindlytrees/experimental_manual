import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 生成二维分类数据
X, y = datasets.make_circles(n_samples=100, factor=0.5, noise=0.1)

# 使用SVM模型，选择RBF核
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X, y)

# 创建二维网格数据
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 对网格点进行预测
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print(Z.min(), Z.max())

# 绘制决策边界和数据点
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')  # 决策边界
# contours = plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), linewidths=2, colors='darkred')

# # 为等高线添加数值标签
# plt.clabel(contours, inline=True, fontsize=10)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolors='k')
plt.title('SVM with RBF kernel')
plt.show()

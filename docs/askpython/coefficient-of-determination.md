# 决定系数–Python 中的 R 平方值

> 原文：<https://www.askpython.com/python/coefficient-of-determination>

读者朋友们，你们好！在本文中，我们将关注 Python 中的**决定系数**。所以，让我们开始吧！🙂

* * *

## 什么是决定系数(R 平方值)？

在深入探讨**决定系数**的概念之前，让我们先了解一下通过误差度量来评估一个机器学习模型的必要性。

在数据科学领域，为了解决任何模型，工程师/开发人员在将模型应用于[数据集](https://www.askpython.com/python/examples/standardize-data-in-python)之前，评估模型的[效率是非常必要的。模型的评估基于某些误差度量。决定系数就是这样一种误差度量。](https://www.askpython.com/python/examples/impute-missing-data-values)

决定系数，也就是通常所说的 R 平方值，是一个`regression error metric`,用于评估模型在应用数据值时的准确性和效率。

r 平方值描述了模型的性能。它描述了由数据模型的独立变量预测的响应或目标变量的变化。

因此，简单地说，R 平方值有助于确定模型的混合程度，以及数据集的决定(独立)变量对输出值的解释程度。

**R 平方的取值范围在[0，1]之间。**看看下面的公式！

**R²= 1-SS[RES]/SS[tot]**

这里，

*   SS [res] 表示数据模型的残差的平方和。
*   SS [tot] 代表误差的总和。

**R 平方值越高，模型和结果越好**。

* * *

## 带数字图书馆的 r 广场

现在让我们尝试使用 [Python NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 库来实现 R square。

我们按照以下步骤使用 Numpy 模块获得 R 平方的值:

1.  使用`numpy.corrcoef()`函数计算[相关矩阵](https://www.askpython.com/python/examples/correlation-matrix-in-python)。
2.  对索引为[0，1]的矩阵进行切片，以获取 R 的值，即`Coefficient of Correlation`。
3.  对 R 的值求平方，得到 R 平方的值。

**举例:**

```py
import numpy
actual = [1,2,3,4,5]
predict = [1,2.5,3,4.9,4.9]

corr_matrix = numpy.corrcoef(actual, predict)
corr = corr_matrix[0,1]
R_sq = corr**2

print(R_sq)

```

**输出:**

```py
0.934602946460654

```

* * *

## 带 Python sklearn 库的 R square

现在，让我们尝试使用 sklearn 库计算 R 平方的值。Python sklearn 库为我们提供了一个 r2_score()函数来确定决定系数的值。

**举例:**

```py
from sklearn.metrics import r2_score 
a =[1, 2, 3, 4, 5] 
b =[1, 2.5, 3, 4.9, 5.1] 
R_square = r2_score(a, b) 
print('Coefficient of Determination', R_square) 

```

**输出:**

```py
Coefficient of Determination 0.8929999999999999

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。更多与 Python 相关的帖子，敬请关注。快乐学习！！🙂
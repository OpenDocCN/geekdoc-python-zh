# Python 中损失函数概述

> 原文：<https://www.askpython.com/python/examples/loss-functions>

读者朋友们，你们好！在本文中，我们将详细关注 Python 中的**损失函数。**

所以，让我们开始吧！！🙂

* * *

## Python 损失函数的重要性

Python 在数据科学和机器学习领域一直扮演着重要的角色。当涉及到在数据集上应用模型时，理解它在准确性和错误率方面对数据集的影响是非常重要的。这有助于我们理解模型对因变量的影响。

同样，我们有 Python 提供的损失函数。利用损失函数，我们可以很容易地理解预测数据值和预期/实际数据值之间的差异。有了这些损失函数，我们可以很容易地获取错误率，从而估计基于它的模型的准确性。

* * *

## 4 最常用的 Python 损失函数

了解了 Python 中的损失函数之后，现在我们将看看一些最常用的损失函数，用于误差估计和准确率。

1.  **均方根误差**
2.  **平均绝对误差**
3.  **交叉熵函数**
4.  **均方误差**

* * *

### 1.均方根误差

利用[均方根误差](https://www.askpython.com/python/examples/rmse-root-mean-square-error)，我们计算数据集的预测值和实际值之间的差异。此外，我们计算差异的平方，然后对其应用均值函数。这里，将使用 NumPy 模块和 mean_squared_error()函数，如下所示。使用 mean_squared_error()函数，我们需要将**平方**参数设置为 False，以便它拾取并计算 RMSE。如果设置为 True，它将计算 MSE。

**举例**:

```py
from sklearn.metrics import mean_squared_error
import numpy as np
ac = np.array([1,2,3])
pr = np.array([0.9,1.9,2.1])
print(mean_squared_error(ac, pr, squared = False))

```

**输出**:

```py
0.5259911279353167

```

* * *

### 2.绝对平均误差

平均绝对误差使我们能够获得数据集的预测数据值和实际数据值之间的平均绝对差值。Python 为我们提供了 mean_absolute_error()函数来计算任何数据范围的平均绝对误差。

**举例**:

```py
from sklearn.metrics import mean_absolute_error
import numpy as np
ac = np.array([1,2,3])
pr = np.array([0.9,1.9,2.1])
print(mean_absolute_error(ac, pr))

```

**输出**:

```py
0.3666666666666667

```

* * *

### 3.均方误差

在 RMSE 之后，[均方差](https://www.askpython.com/python/examples/mape-mean-absolute-percentage-error)使我们能够轻松计算实际数据值和预测数据值之间的均方差的平均值。我们可以利用 mean_squared_error()函数来计算所示数据范围的 MSE

**举例**:

```py
from sklearn.metrics import mean_squared_error
import numpy as np
ac = np.array([1,2,3])
pr = np.array([0.9,1.9,2.1])
print(mean_squared_error(ac, pr, squared = True))

```

**输出**:

```py
0.2766666666666666

```

* * *

### 4.交叉熵损失函数

RMSE、MSE 和 MAE 主要用于回归问题。交叉熵损失函数高度用于问题陈述的分类类型。它使我们能够针对分类数据变量定义问题分类类型的错误/丢失率。

Python 的 sklearn 库为我们提供了 log_loss()函数来处理和估计分类/分类数据变量的错误率。

**举例**:

```py
from sklearn.metrics import log_loss
op = log_loss(["Yes", "No", "No", "Yes","Yes","Yes"],[[10, 9], [39, 11], [8, 2], [35, 65], [12, 14], [12,12]])
print(op)

```

**输出**:

```py
0.6931471805599453

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂
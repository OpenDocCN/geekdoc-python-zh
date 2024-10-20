# Python 中的套索回归

> 原文：<https://www.askpython.com/python/examples/lasso-regression>

读者朋友们，你们好！在上一篇文章中，我们详细讨论了 Python 编程中的岭回归。现在，我们将讨论 Python 中的**套索回归**。

所以，让我们开始吧！

* * *

## 一、什么是套索回归？

在数据科学和机器学习领域，我们的主要目标是根据数据值的类型，通过各种算法对现实生活中的问题进行预测。

[线性回归](https://www.askpython.com/python/examples/linear-regression-in-python)就是这样一种算法。使用该算法，我们可以为我们的模型定义最佳拟合线，即了解数据集变量之间的相关性。

它帮助我们找出数据集的因变量和自变量之间的关系，以建立预测的估计模型。

**线性回归的问题**:

*   众所周知，线性回归计算的是模型每个变量的系数。随着数据复杂性的增加，系数的值变成更高的值，这反过来使得模型对提供给它的进一步输入敏感。
*   这反过来又让模型有点不稳定！

**解–套索回归**

所以，我们开始解决这个问题。拉索回归，也称为`L1 regression`就足够了。使用套索回归，我们倾向于用系数的值来惩罚模型。因此，它通过包含模型变量的额外成本来操纵损失函数，而该模型恰好具有大的系数值。

它针对绝对系数值对模型进行惩罚。这样，它让系数的值(对预测变量没有贡献)变为零。除此之外，**它从模型**中移除那些输入特征。

因此，我们可以说，

**Lasso = loss+(λ* L1 _ penalty)**

这里，**λ**是在惩罚值的加权处进行检查的超参数。

* * *

## 套索回归——一种实用的方法

在本例中，我们利用了自行车租赁计数预测数据集。你可以在这里找到数据集[！](https://github.com/Safa1615/BIKE-RENTAL-COUNT/blob/master/day.csv)

最初，我们使用 read_csv()函数将数据集加载到 Python 环境中。除此之外，我们使用 [train_test_split()](https://www.askpython.com/python/examples/split-data-training-and-testing-set) 函数将数据集分割成训练和测试数据。

对于这个例子，我们已经设置了 [MAPE](https://www.askpython.com/python/examples/mape-mean-absolute-percentage-error) 作为误差度量来评估 lasso 回归惩罚模型。

Python 的`sklearn.linear_model library`，为我们提供了`lasso()`函数，在数据集上建立模型。

**举例:**

```py
import os
import pandas

#Changing the current working directory
os.chdir("D:/Ediwsor_Project - Bike_Rental_Count")
BIKE = pandas.read_csv("day.csv")

bike = BIKE.copy()
categorical_col_updated = ['season','yr','mnth','weathersit','holiday']
bike = pandas.get_dummies(bike, columns = categorical_col_updated)
#Separating the depenedent and independent data variables into two dataframes.
from sklearn.model_selection import train_test_split 
X = bike.drop(['cnt'],axis=1) 
Y = bike['cnt']

import numpy as np
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=1.0)
lasso=lasso_model.fit(X_train , Y_train)
lasso_predict = lasso.predict(X_test)
Lasso_MAPE = MAPE(Y_test,lasso_predict)
print("MAPE value: ",Lasso_MAPE)
Accuracy = 100 - Lasso_MAPE
print('Accuracy of Lasso Regression: {:0.2f}%.'.format(Accuracy))

```

**输出:**

```py
MAPE value:  16.55305612241603
Accuracy of Lasso Regression: 83.45%.

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，请随时在下面评论。

建议您尝试使用其他数据集的 Lasso 回归概念，并在评论部分告诉我们您的体验！

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂
# 梯度推进模型——用 Python 实现

> 原文：<https://www.askpython.com/python/examples/gradient-boosting-model-in-python>

读者朋友们，你们好！在本文中，我们将关注 Python 中的**梯度增强模型，以及实现细节。**

所以，让我们开始吧！

* * *

## 第一，什么是梯度提升模型？

在深入探讨梯度提升的概念之前，我们先来了解一下机器学习中的提升概念。

Boosting 技术试图通过以串行方式构建弱模型实例来创建强回归器或分类器。也就是说，前一个实例的**错误分类误差**被馈送到下一个实例，并且它从该误差中学习以提高分类或预测率。

梯度推进算法就是这样一种机器学习模型，它遵循预测推进技术。

在梯度推进算法中，预测器的每个实例从其前一个实例的误差中学习，即它校正由前一个预测器报告或引起的误差，以具有更好的模型和更少的误差率。

每个梯度提升算法的基础学习器或预测器是**分类和回归树**。学习的过程继续进行，直到我们决定构建的所有 N 棵树都已经从模型中学习，并且准备好进行具有更少量的错误分类错误的预测。

梯度推进模型适用于回归和分类变量。

***推荐阅读——[Python XGBoost 教程](https://www.askpython.com/python/examples/gradient-boosting)***

* * *

## 梯度推进模型——一种实用的方法

在本例中，我们利用了自行车租赁计数预测数据集。你可以在这里找到数据集[！](https://github.com/Safa1615/BIKE-RENTAL-COUNT/blob/master/day.csv)

首先，我们使用 [read_csv()](https://www.askpython.com/python-modules/python-csv-module) 函数将数据集加载到 Python 环境中。

为了进一步实现，我们使用来自`sklearn.model selection`库的`train_test_split()`函数将数据集分成训练和测试数据值。

分离数据后，我们进一步使用 [MAPE](https://www.askpython.com/python/examples/mape-mean-absolute-percentage-error) 作为评估算法的误差度量模型。

现在，让我们来关注一下在 Python 中实现梯度推进模型的步骤

*   我们利用 GradientBoostingRegressor()函数对训练数据应用 GBM。
*   在此基础上，我们利用 predict()方法对测试数据进行建模。

**举例:**

```py
import pandas
BIKE = pandas.read_csv("day.csv")

#Separating the depenedent and independent data variables into two dataframes.
from sklearn.model_selection import train_test_split 
X = bike.drop(['cnt'],axis=1) 
Y = bike['cnt']
# Splitting the dataset into 80% training data and 20% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)

import numpy as np
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

from sklearn.ensemble import GradientBoostingRegressor
GR = GradientBoostingRegressor(n_estimators = 200, max_depth = 1, random_state = 1) 
gmodel = GR.fit(X_train, Y_train) 
g_predict = gmodel.predict(X_test)
GB_MAPE = MAPE(Y_test,g_predict)
Accuracy = 100 - GB_MAPE
print("MAPE: ",GB_MAPE)
print('Accuracy of Linear Regression: {:0.2f}%.'.format(Accuracy))

```

**输出:**

结果，我们从数据集上的梯度推进模型获得了 83.10%的准确度。

```py
MAPE:  16.898145257306943
Accuracy of Linear Regression: 83.10%.

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂
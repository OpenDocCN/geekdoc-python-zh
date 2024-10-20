# Python 中的相关回归分析——2 种简单的方法！

> 原文：<https://www.askpython.com/python/examples/correlation-regression-analysis>

读者朋友们，你们好！今天，我们将关注 Python 中的**相关性回归分析**。

所以，让我们开始吧！

* * *

## 首先，变量之间的相关性是什么？

让我们试着在数据科学和机器学习的背景下理解相关性的概念！

在数据科学和机器学习领域，主要步骤是分析和清理数据以供进一步处理。

以[数据预处理](https://www.askpython.com/python/examples/standardize-data-in-python)为借口，了解每个变量/列对其他变量以及响应/目标变量的影响是非常重要的。

这就是相关回归分析登场的时候了！

**相关回归分析是一种检测和分析自变量之间以及与目标值之间关系的技术。**

由此，我们试图分析自变量试图代表目标值添加什么信息或值。

通常，相关性分析适用于回归值，即连续(数字)变量，并通过一个称为[相关性矩阵](https://www.askpython.com/python/examples/correlation-matrix-in-python)的矩阵进行描述。

在相关矩阵中，变量之间的关系是范围 **-1 到+1** 之间的值。

使用相关性分析，我们可以检测冗余变量，即代表目标值相同信息的变量。

如果两个变量高度相关，这就给了我们一个提示，要消除其中一个变量，因为它们描述了相同的信息。

现在让我们来实现相关回归的概念！

* * *

## 使用 Pandas 模块进行相关回归分析

在本例中，我们利用了银行贷款数据集来确定数字列值的相关矩阵。可以在这里 找到数据集 **[！](https://github.com/Safa1615/Dataset--loan/blob/main/bank-loan.csv)**

1.  最初，我们将使用 [pandas.read_csv()](https://www.askpython.com/python-modules/python-csv-module) 函数将数据集加载到环境中。
2.  此外，我们将把数字列分离到不同的 [Python 列表](https://www.askpython.com/python/list/python-list)(变量)中，如下例所示。
3.  现在，我们将对每个数字变量应用`corr() function`,并为该函数的相同输出创建一个相关矩阵。

**举例:**

```py
import os
import pandas as pd
import numpy as np

# Loading the dataset
data = pd.read_csv("loan.csv")
numeric_col = ['age', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']

#Using Correlation analysis to depict the relationship between the numeric/continuous data variables
corr = data.loc[:,numeric_col].corr()
print(corr)

```

**输出:**

**Correlation Regression Analysis Output**

* * *

## 使用 NumPy 模块确定变量之间的相关性

corr()方法并不是唯一可以用于相关回归分析的方法。我们有另一个计算相关性的函数。

[Python NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 为我们提供了`numpy.corrcoef()`函数来计算数值变量之间的相关性。

**语法:**

```py
numpy.corrcoef(col1, col2)

```

因此，它将返回输入回归变量的相关矩阵。

**举例:**

```py
import numpy as np 

x = np.array([2,4,8,6]) 
y = np.array([3,4,1,6]) 

corr_result=np.corrcoef(x, y) 

print(corr_result) 

```

**输出:**

```py
[[ 1\.         -0.24806947]
 [-0.24806947  1\.        ]]

```

* * *

## 结论

到此，我们就结束了这个话题。更多与 Python 相关的此类帖子，敬请关注！！尝试在不同的数据集上实现相关性分析的概念，并在评论部分告诉我们您的体验🙂

在那之前，学习愉快！！🙂
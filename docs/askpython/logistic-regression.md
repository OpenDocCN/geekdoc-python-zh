# 逻辑回归–简单实用的实施

> 原文：<https://www.askpython.com/python/examples/logistic-regression>

读者朋友们，你们好！在本文中，我们将关注 Python 中逻辑回归的实际实现。

在我们用 Python 进行的一系列机器学习中，我们已经了解了各种有监督的 ML 模型，如[线性回归](https://www.askpython.com/python/examples/linear-regression-in-python)、 [K 近邻](https://www.askpython.com/python/examples/knn-in-python)等。今天，我们将集中讨论逻辑回归，并用它来解决现实生活中的问题！激动吗？耶！🙂

让我们开始吧！

* * *

## 一、什么是 Logistic 回归？

在开始逻辑回归之前，让我们了解我们在哪里需要它。

众所周知，监督机器学习模型对连续数据值和分类数据值都有效。其中，[分类数据值](https://www.askpython.com/python/examples/label-encoding)是组成组和类别的数据元素。

因此，当我们以分类数据变量作为因变量时，要做出预测，就需要进行逻辑回归。

**逻辑回归**是一个监督机器学习模型，它以**二元**或**多分类数据变量**为因变量。也就是说，它是一个**分类算法**，它分别分离和分类二进制或多标记值。

例如，如果一个问题希望我们预测结果为“是”或“否”，那么逻辑回归将对相关数据变量进行分类，并计算出数据的结果。

逻辑回归使我们能够使用 **logit 函数**对训练数据进行分类，以符合因变量二元变量的结果。此外，logit 函数仅依赖于**优势值和概率机会**来预测二元响应变量。

现在让我们看看逻辑回归的实现。

* * *

## 实用方法-逻辑回归

在本文中，我们将利用**银行贷款违约者问题**,其中我们预计会预测哪些客户是贷款违约者。

在 这里可以找到数据集 **[。](https://github.com/Safa1615/Dataset--loan/blob/main/bank-loan.csv)**

* * *

### 1.加载数据集

在初始步骤，我们需要使用 [pandas.read_csv()](https://www.askpython.com/python-modules/python-csv-module) 函数将数据集加载到环境中。

```py
import pandas as pd
import numpy as np
data = pd.read_csv("bank-loan.csv") # dataset

```

### 2.数据集的采样

加载数据集后，现在让我们使用 [train_test_split()](https://www.askpython.com/python/examples/split-data-training-and-testing-set) 函数将数据集分为训练数据集和测试数据集。

```py
from sklearn.model_selection import train_test_split 
X = loan.drop(['default'],axis=1) 
Y = loan['default'].astype(str)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)

```

这里，X 是包含除了响应/目标值之外的所有变量的训练数据集，Y 是指仅包含响应变量的测试数据集。

### 3.定义模型的误差度量

现在，在进入模型构建之前，让我们定义误差度量，这将有助于我们以更好的方式分析模型。

这里，我们创建了一个[混淆矩阵](https://www.askpython.com/python/examples/confusion-matrix)，并计算了精确度、召回率、准确度和 F1 分数。

```py
def err_metric(CM): 

    TN = CM.iloc[0,0]
    FN = CM.iloc[1,0]
    TP = CM.iloc[1,1]
    FP = CM.iloc[0,1]
    precision =(TP)/(TP+FP)
    accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
    recall_score  =(TP)/(TP+FN)
    specificity_value =(TN)/(TN + FP)

    False_positive_rate =(FP)/(FP+TN)
    False_negative_rate =(FN)/(FN+TP)

    f1_score =2*(( precision * recall_score)/( precision + recall_score))

    print("Precision value of the model: ",precision)
    print("Accuracy of the model: ",accuracy_model)
    print("Recall value of the model: ",recall_score)
    print("Specificity of the model: ",specificity_value)
    print("False Positive rate of the model: ",False_positive_rate)
    print("False Negative rate of the model: ",False_negative_rate)
    print("f1 score of the model: ",f1_score)

```

### 4.对数据集应用模型

现在终于到了对数据集执行模型构建的时候了。看看下面的代码吧！

```py
logit= LogisticRegression(class_weight='balanced' , random_state=0).fit(X_train,Y_train)
target = logit.predict(X_test)
CM_logit = pd.crosstab(Y_test,target)
err_metric(CM_logit)

```

**说明:**

*   最初，我们在训练数据集上应用了`LogisticRegression()`函数。
*   此外，我们已经使用 [predict()](https://www.askpython.com/python/examples/python-predict-function) 函数输入了上述输出来预测测试数据集的值。
*   最后，我们使用`crosstab()`创建了一个相关矩阵，然后调用误差度量定制函数(之前创建的)来判断结果。

**输出:**

```py
Precision value of the model:  0.30158730158730157
Accuracy of the model:  0.6382978723404256
Recall value of the model:  0.7307692307692307
Specificity of the model:  0.6173913043478261
False Positive rate of the model:  0.3826086956521739
False Negative rate of the model:  0.2692307692307692
f1 score of the model:  0.42696629213483145

```

因此，正如上面所见证的，我们通过我们的模型获得了 **63%** 的准确性。

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。更多关于 Python 和 ML 的文章，请继续关注，

快乐学习！！🙂
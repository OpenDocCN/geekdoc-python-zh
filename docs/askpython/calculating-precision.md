# Python 中的计算精度——分类误差度量

> 原文：<https://www.askpython.com/python/examples/calculating-precision>

读者朋友们，你们好！在本文中，我们将详细关注 Python 中的**计算精度。**

所以，让我们开始吧！！🙂

* * *

* * *

## 精确度——分类误差指标

在深入研究分类错误度量的概念具体来说，精度之前，让我们先了解一下机器学习中的错误度量是什么。

误差指标是一组指标，使我们能够评估模型在准确性方面的效率，并让我们估计最适合我们的问题陈述的模型。

根据机器学习算法的类型，有各种类型的误差度量。

对于回归算法，我们有以下可用于评估的指标-

1.  **[R 方](https://www.askpython.com/python/coefficient-of-determination)**
2.  **[MAPE](https://www.askpython.com/python/examples/mape-mean-absolute-percentage-error)**
3.  **MSE**
4.  **调整后的 R 方**等。

对于分类算法，我们可以利用以下指标-

*   **[混乱矩阵](https://www.askpython.com/python/examples/confusion-matrix)**
*   **精度**
*   **精度**
*   **回忆**等。

精度帮助我们估计被预测为正的和实际上为正的正数据值的百分比。

**精度公式:**

精度=真阳性/(真阳性+假阳性)

**注—**所谓真正值，是指预测为正值且实际为正值的值。而假正值是预测为正值但实际上为负值的值。

精度分值的范围分别为 0.0 到 1.0。

现在，让我们关注 Python 中数据集的精度误差度量的实现。

* * *

## Python 中计算数据集精度的步骤

首先，我们将利用银行贷款数据集进行演示。

你可以在这里找到数据集 **[！](https://github.com/Safa1615/Bike-loan-Dataset/blob/main/bank-loan.csv)**

1.  最初，我们使用 read_csv()函数将数据集加载到 Python 环境中。
2.  使用[缺失值分析](https://www.askpython.com/python/examples/impute-missing-data-values)、[异常值检测](https://www.askpython.com/python/examples/detection-removal-outliers-in-python)技术进行数据分析和清理。
3.  使用 train_test_split()函数将数据集拆分为训练数据和测试数据。
4.  在应用模型之前，我们需要定义用于评估模型的误差度量。我们利用混淆矩阵来获得真阳性和假阳性分数。此外，我们已经应用了上面讨论的公式来获得精度分数。
5.  最后，我们在数据集上应用决策树算法，并用精度分数测试其效率。

你可以在下面找到完整的代码

```py
import pandas as pd
import numpy as np
loan = pd.read_csv("bank-loan.csv") # dataset

from sklearn.model_selection import train_test_split 
X = loan.drop(['default'],axis=1) 
Y = loan['default'].astype(str)

# Error metrics -- Confusion matrix\FPR\FNR\f1 score\
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

#Decision Trees
decision = DecisionTreeClassifier(max_depth= 6,class_weight='balanced' ,random_state =0).fit(X_train,Y_train)
target = decision.predict(X_test)
targetclass_prob = decision.predict_proba(X_test)[:, 1]
confusion_matrix = pd.crosstab(Y_test,target)
err_metric(confusion_matrix)

```

**输出—**

因此，精度分数为 0.25，这意味着总预测正值的 25%实际上是正的。

```py
Precision value of the model:  0.25
Accuracy of the model:  0.6028368794326241

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂
# 基于多种最大似然算法的虹膜数据集分类

> 原文：<https://www.askpython.com/python/examples/iris-dataset-classification>

你好。今天我们将学习一个新的数据集——虹膜数据集。数据集非常有趣，因为它处理花的各种属性，然后根据它们的属性对它们进行分类。

## 1.导入模块

任何项目的第一步都是导入基本模块，包括 [numpy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 、 [pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 和 [matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 。

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```

## 2.加载和准备 Iris 数据集

为了加载数据，我们将从 Kaggle 下载数据集。你可以在这里下载数据集[，但是要确保这个文件和代码文件在同一个目录下。](https://www.kaggle.com/uciml/iris)

我们还将通过对数据进行切片操作来将数据和标签相互分离。

```py
data = pd.read_csv('Iris.csv')
data_points = data.iloc[:, 1:5]
labels = data.iloc[:, 5]

```

## 3.将数据分为测试数据和训练数据

在训练任何一种 ML 模型之前，我们首先需要使用 sklearn 的`train_test_split`函数将数据[分割成测试和训练数据](https://www.askpython.com/python/examples/split-data-training-and-testing-set)。

```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_points,labels,test_size=0.2)

```

## 4.数据的标准化/规范化

在我们进行 ML 建模和数据处理之前，我们需要对数据进行规范化，下面将提到代码。

```py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
Standard_obj = StandardScaler()
Standard_obj.fit(x_train)
x_train_std = Standard_obj.transform(x_train)
x_test_std = Standard_obj.transform(x_test)

```

## 5.应用分类 ML 模型

现在我们的数据已经准备好，可以进入各种 ML 模型，我们将测试和比较各种分类模型的效率

### 5.1 SVM(支持向量机)

我们要测试的第一个模型是 SVM 分类器。下面提到了相同的代码。

```py
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train_std, y_train)
print('Training data accuracy {:.2f}'.format(svm.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(svm.score(x_test_std, y_test)*100))

```

在成功执行时，分类器分别给出了大约 97%和 93%的训练和测试精度，这是相当不错的。

### 5.2 KNN (K 近邻)

[KNN 算法](https://www.askpython.com/python/examples/knn-in-python)是 ML 世界中最基本、最简单、最初级的分类模型之一。直接执行它的代码如下所示。

```py
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
knn.fit(x_train_std,y_train)
print('Training data accuracy {:.2f}'.format(knn.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(knn.score(x_test_std, y_test)*100))

```

在这种情况下，测试精度只有大约 80%,与其他模型相比，这要低一些，但是这是合理的，因为该模型非常基本，并且有一些限制。

### 5.3 决策树

接下来，我们将实现决策树模型，这是一个简单而复杂的 ML 模型。相同的代码如下所示。

```py
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(criterion='gini')
decision_tree.fit(x_train_std, y_train)
print('Training data accuracy {:.2f}'.format(decision_tree.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(decision_tree.score(x_test_std, y_test)*100))

```

这个模型的测试准确率也仍然在 80%左右，因此到目前为止 SVM 给出了最好的结果。

### 5.4 随机森林

随机森林是机器学习中更复杂、更好的决策树。相同的实现如下所示。

```py
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
random_forest.fit(x_train_std, y_train)
print('Training data accuracy {:.2f}'.format(random_forest.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(random_forest.score(x_test_std, y_test)*100))

```

这里的准确率非常高，训练数据是 100%，这太棒了！而测试数据的准确率为 90%，这也是相当不错的。

## 结论

恭喜你！本教程提到了在同一数据集上的许多不同的算法，我们为每个模型获得了不同的结果。希望你喜欢它！继续阅读，了解更多！

感谢您的阅读！
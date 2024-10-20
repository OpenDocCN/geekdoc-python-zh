# 使用 Python 检测帕金森病

> 原文：<https://www.askpython.com/python/examples/detecting-parkinsons-disease>

你好，学习伙伴！今天，我们正在使用 Python 基于一些预先获得的信息构建一个基本的 ML 模型来检测帕金森病。

因此，让我们首先了解帕金森病和我们将用于我们模型的数据集，可以在[这里](https://archive.ics.uci.edu/ml/datasets/parkinsons)找到。我们将在我们的项目中使用`parkinson.data`文件。

**帕金森病**是一种中枢神经系统疾病，会影响身体的运动。到目前为止，这种疾病还没有切实可行的治疗方法。

## 导入所需的库

任何项目的第一步都是将所有必要的模块导入我们的项目。我们需要一些基本模块，如 [numpy](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) 、 [pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 和 [matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 来分别准备、加载和绘制数据。

然后我们还需要一些 sklearn 模型和函数，用于训练和估计精度。最后但同样重要的是，我们将使用`xgboost`库。

XGBoost 库是一个基于决策树的[梯度推进](https://www.askpython.com/python/examples/gradient-boosting-model-in-python)模型，旨在提高系统的速度和精度。

```py
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

```

## 加载数据集

下一步是将我们之前下载的数据加载到与代码文件相同的文件夹中。同样，我们使用 pandas 模块，其代码如下所示。

```py
dataframe=pd.read_csv('parkinsons.csv')
print("The shape of data is: ",dataframe.shape,"\n")
print("FIRST FIVE ROWS OF DATA ARE AS FOLLOWS: \n")
dataframe.head()

```

程序的输出显示了数据集的前五行，该数据集总共由 24 列和 195 个数据点组成。下一步是将标签和数据相互分离。

下面提到了相同的代码。这里的标签列是**状态**列。

```py
data=dataframe.loc[:,dataframe.columns!='status'].values[:,1:]
label=dataframe.loc[:,'status'].values

```

## 标准化数据

下一步是缩放-1 和+1 之间的所有数据点。我们将使用[最小最大缩放器](https://www.askpython.com/python/examples/normalize-data-in-python)来转换特征，并将它们作为参数缩放到给定的范围。`fit_transform`函数有助于拟合数据，然后对其进行转换/标准化。

不需要缩放标签，因为它们已经只有两个值，即 0 和 1。相同的代码如下所示。

```py
Normalizing_object = MinMaxScaler((-1,1))
x_data = Normalizing_object.fit_transform(data)
y_data=label

```

## 训练-测试数据分割

下一步是[根据 80-20 法则将数据分为训练和测试数据](https://www.askpython.com/python/examples/split-data-training-and-testing-set)，其中 80%的数据用于训练，其余 20%用于测试。

我们将使用 sklearn 模块的`train_test_split`函数来实现同样的功能。代码如下所述。

```py
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)

```

## xgb 分类器的初始化和模型的训练

我们的数据现在已经准备好接受训练并适合 xbg 分类器。为此，我们将创建一个分类器对象，然后将训练数据放入分类器。

相同的代码如下所示。

```py
model=XGBClassifier()
model.fit(x_train,y_train)

```

输出显示了分类器的全部训练信息，现在我们准备好对测试数据进行预测，然后获得准确性。

## 获得预测和准确性

下一步也是最后一步是获得测试数据集的预测，并估计我们模型的准确性。完成同样工作的代码如下所示。

```py
predictions=model_obj.predict(x_test)
print(accuracy_score(y_test,predictions)*100)

```

运行代码后，我们知道这个模型已经超过了`97.43%`精度，这很好，对吗？！所以我们走吧！我们建立了自己的帕金森病分类器。

## 结论

在本教程中，我们学习了如何根据各种因素检测个体中帕金森病的存在。

对于这个项目，我们使用 xgb 分类器进行快速准确的检测。模型给了我们超过`97.43%`的准确率，太棒了！

感谢您的阅读！
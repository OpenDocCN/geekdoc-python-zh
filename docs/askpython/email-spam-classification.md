# Python 中的垃圾邮件分类

> 原文：<https://www.askpython.com/python/examples/email-spam-classification>

你好，学习伙伴！在本教程中，我们将讨论如何在数据集的帮助下实现垃圾邮件的分类，该数据集将使用 Python 编程语言中的 scikit-learn 加载。

## 垃圾邮件简介

我们都知道，每天都有数十亿封垃圾邮件发送到用户的电子邮件帐户，其中超过 90%的垃圾邮件是恶意的，会对用户造成重大伤害。

你不觉得垃圾邮件很烦人吗？他们肯定会让我很烦！有时，甚至一些重要的邮件被转移到垃圾邮件中，结果，一些重要的信息由于害怕受到垃圾邮件的伤害而未被阅读。

您知道吗**每 1000 封电子邮件中就有一封包含恶意软件指控**？因此，对我们来说，重要的是学会如何将我们自己的电子邮件分类为安全和不安全。

## 用 Python 实现垃圾邮件分类器

让我们进入使用 Python 实现垃圾邮件分类算法的步骤。这将帮助你理解一个非常基本的垃圾邮件分类器的后端工作。与我下面描述的算法相比，现实世界中使用的算法要先进得多。但是你可以把它作为你旅程的起点。

### 1.导入模块和加载数据

首先，我们将所有必需的模块导入我们的程序。相同的代码如下:

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

```

我们需要一些基本的机器学习模块，如 [numpy](https://www.askpython.com/python-modules/numpy/numpy-bitwise-operations) 、 [pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 和 [matplotlib](https://www.askpython.com/python-modules/matplotlib/python-matplotlib) 。除此之外，我们还需要一些`sklearn`模型和特性。

下一步是在前面导入的 pandas 模块的帮助下加载数据集。我们将使用的数据集是`spam.csv`数据文件，可以在[这里](https://www.kaggle.com/uciml/sms-spam-collection-dataset)找到。

```py
data = pd.read_csv('./spam.csv')

```

我们加载的数据集有 5572 个电子邮件样本以及两个唯一的标签，即`spam`和`ham`。

### 2.培训和测试数据

加载后，我们必须将数据分成[训练和测试数据](https://www.askpython.com/python/examples/split-data-training-and-testing-set)。

将数据分为训练和测试数据包括两个步骤:

1.  将 x 和 y 数据分别分离为电子邮件文本和标签
2.  基于 80:20 规则将 x 和 y 数据分割成四个不同的数据集，即 x_train、y_train、x_test 和 y_test。

将数据分成 x 和 y 数据是在下面的代码中完成的:

```py
x_data=data['EmailText']
y_data=data['Label']

split =(int)(0.8*data.shape[0])
x_train=x_data[:split]
x_test=x_data[split:]
y_train=y_data[:split]
y_test=y_data[split:]

```

### 3.提取重要特征

下一步是从整个数据集中只获取重要的单词/特征。为了实现这一点，我们将利用`CountVectorizer`函数来对训练数据集的单词进行矢量化。

```py
count_vector = CountVectorizer()  
extracted_features = count_vector.fit_transform(x_train)

```

### 4.构建和训练模型

最重要的步骤包括为我们之前创建的数据集构建和训练模型。相同的代码如下:

```py
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}
model = GridSearchCV(svm.SVC(), tuned_parameters)
model.fit(extracted_features,y_train)

print("Model Trained Successfully!")

```

最后一步包括在测试数据集上计算我们的模型的整体准确性。

```py
print("Accuracy of the model is: ",model.score(count_vector.transform(x_test),y_test)*100)

```

我们最终达到了`**98.744%**` 的精确度，这太棒了！！

## 结论

实施电子邮件分类系统是发展该技术并使电子邮件更加安全的下一个重要步骤。

我希望你喜欢这个教程！快乐学习！😇

## 另请参阅:

1.  [Python 中的手写数字识别](https://www.askpython.com/python/examples/handwritten-digit-recognition)
2.  [Python:图像分割](https://www.askpython.com/python/examples/image-segmentation)
3.  [Python 中的拼写检查器](https://www.askpython.com/python/examples/spell-checker-in-python)
4.  [K-最近邻从零开始用 Python](https://www.askpython.com/python/examples/k-nearest-neighbors-from-scratch)
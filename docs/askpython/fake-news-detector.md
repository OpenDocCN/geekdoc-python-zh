# 如何用 Python 创建假新闻检测器？

> 原文：<https://www.askpython.com/python/examples/fake-news-detector>

你好。今天，我们将使用一些常见的机器学习算法，在 Python 中创建一个假新闻检测器。

## 1.导入模块

就像任何其他项目一样，这个项目的第一步也是导入模块。我们正在与 [Numpy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 、[熊猫](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)和 [itertools](https://www.askpython.com/python-modules/python-itertools-module) 合作。相同的代码如下所示。

```py
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

```

## 2.加载数据

现在，让我们[从 csv 文件](https://www.askpython.com/python-modules/python-csv-module)中读取假新闻检测的数据，这里可以找到[。打印前 5 行数据的代码如下所示。](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)

```py
data=pd.read_csv('news.csv')
data.head()

```

确保 CSV 文件与 Python 代码保存在同一个文件夹中。接下来，让我们从刚刚加载的数据中提取标签，并打印前五个标签。

```py
lb=df.label
lb.head()

```

## 3.创建培训和测试数据

在我们将数据传递到最终的模型/分类器之前，我们需要[将数据分割成测试和训练数据](https://www.askpython.com/python/examples/split-data-training-and-testing-set)，这在下面提到的代码中完成。

```py
x_train,x_test,y_train,y_test=train_test_split(data['text'], lb, test_size=0.2, random_state=7)

```

为了分割数据，我们将使用`80-20`规则，其中 80%的数据用于训练，剩下的 20%用于测试数据。

## 4.实现 tfi df-矢量器和 PassiveAggressiveClassifier

使用 Tfidf 矢量器将文本数组转换为`TF-IDF`矩阵。

1.  **TF(词频)**:定义为一个词在文本中出现的次数。
2.  **IDF(逆文档频率)**:衡量一个术语在整个数据中的重要程度。

稍后，我们应用`PassiveAggressiveClassifier`并将数据拟合到训练数据中。分类器**在每次迭代后更新损失**，并对权重向量进行**的微小改变。**

最后，我们对测试数据进行预测，并根据测试数据计算模型的精度。事实证明，我们在测试数据上获得了超过 90% 的**准确率。**

相同的代码如下所示。

```py
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)

print("Accuracy: ",round(score*100,2),"%")

```

## 结论

今天，我们学习了用 Python 在一个有很多新闻数据的数据集上检测假新闻。检测是在 tfidf 矢量器和 PassiveAggressiveClassifier 的帮助下完成的。结果，我们获得了超过 90%的准确率，这是惊人的！

我希望你喜欢假新闻检测器！继续阅读，了解更多！
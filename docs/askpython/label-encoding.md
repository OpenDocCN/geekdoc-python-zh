# Python 中的标签编码——快速指南！

> 原文：<https://www.askpython.com/python/examples/label-encoding>

读者朋友们，你们好！在本文中，我们将关注 Python 中的标签编码。

在我们的上一篇文章中，我们了解了 [One hot Encoding](https://www.askpython.com/python/examples/one-hot-encoding) 的工作和实现，其中标签编码是该过程的初始步骤。

今天，我们将看看数据值分类编码中最基本的步骤之一。

因此，没有任何进一步的拖延，让我们开始吧！

* * *

## Python 中的标签编码是什么？

在深入研究标签编码的概念之前，让我们先了解一下“标签”这个概念对数据集的影响。

标签实际上是代表一组特定实体的数字或字符串。标签有助于模型更好地理解数据集，并使模型能够学习更复杂的结构。

*推荐—[如何标准化机器学习的数据集？](https://www.askpython.com/python/examples/standardize-data-in-python)*

**标签编码器**将分类数据的这些标签转换成数字格式。

例如，如果数据集包含带有标签“男性”和“女性”的变量“性别”，则标签编码器会将这些标签转换为数字格式，结果将是[0，1]。

因此，通过将标签转换成整数格式，机器学习模型可以在操作数据集方面有更好的理解。

* * *

## 标签编码–语法知识！

Python **sklearn 库**为我们提供了一个预定义的函数，对数据集进行标签编码。

**语法:**

```py
from sklearn import preprocessing  
object = preprocessing.LabelEncoder() 

```

这里，我们创建一个 LabelEncoder 类的对象，然后利用该对象对数据应用标签编码。

* * *

### 1.使用 sklearn 进行标签编码

让我们直接进入标签编码的过程。对数据集进行编码的第一步是拥有一个数据集。

因此，我们将在这里创建一个简单的数据集。**示例:数据集的创建**

```py
import pandas as pd 
data = {"Gender":['M','F','F','M','F','F','F'], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}
block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)

```

这里，我们创建了一个[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)‘数据’,然后使用`pandas.DataFrame()` 函数将它转换成一个数据帧。

**输出:**

```py
Original Data frame:

  Gender    NAME
0      M    John
1      F  Camili
2      F  Rheana
3      M  Joseph
4      F  Amanti
5      F   Alexa
6      F    Siri

```

从上面的数据集中，很明显，变量“性别”的标签为“M”和“F”。

此外，现在让我们导入 **LabelEncoder** 类，并将其应用于数据集的“性别”变量。

```py
from sklearn import preprocessing 
label = preprocessing.LabelEncoder() 

block['Gender']= label.fit_transform(block['Gender']) 
print(block['Gender'].unique())

```

我们已经使用`fit_transform() method`将对象指向的标签编码器的功能应用于数据变量。

**输出:**

```py
[1 0]

```

所以，你看，数据已经被转换成[0，1]的整数标签了。

```py
print(block)

```

**输出:**

```py
Gender    NAME
0       1    John
1       0  Camili
2       0  Rheana
3       1  Joseph
4       0  Amanti
5       0   Alexa
6       0    Siri

```

* * *

### **2。使用类别代码的标签编码**

让我们首先检查数据集变量的数据类型。

```py
block.dtypes

```

**数据类型**:

```py
Gender    object
NAME      object
dtype: object

```

现在，将变量“性别”的数据类型转换为**类别**类型。

```py
block['Gender'] = block['Gender'].astype('category')

```

```py
block.dtypes

```

```py
Gender    category
NAME        object
dtype: object

```

现在，让我们使用`pandas.DataFrame.cat.codes`函数将标签转换成整数类型。

```py
block['Gender'] = block['Gender'].cat.codes

```

```py
print(block)

```

如下所示，变量“性别”已被编码为整数值[0，1]。

```py
Gender    NAME
0       1    John
1       0  Camili
2       0  Rheana
3       1  Joseph
4       0  Amanti
5       0   Alexa
6       0    Siri

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

为了更深入地理解这个主题，请尝试在不同的数据集和变量上实现标签编码器的概念。请在评论区告诉我们你的体验！🙂

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂

* * *

## 参考

*   [标签编码器–文档](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
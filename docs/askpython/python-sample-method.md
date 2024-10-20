# 了解 Python sample()函数

> 原文：<https://www.askpython.com/python/built-in-methods/python-sample-method>

读者朋友们，你们好！在本文中，我们将重点关注 **Python sample()函数**及其在数据科学领域的重要性。

所以，让我们开始吧！

* * *

## Python sample()方法是什么？

我们先来了解一下 sample()方法在数据科学这个行业的存在。

在解决与数据预测相关的问题时，我们经常会遇到这样的情况:我们需要在少量数据上[测试算法](https://www.askpython.com/python/examples/split-data-training-and-testing-set)，以估计所应用算法的准确性。

这就是 Python sample()方法发挥作用的时候。

`sample() method`让我们从操作的可用数据中选择一个随机样本。尽管有很多对数据进行采样的技术，sample()方法被认为是同类方法中最简单的一种。

Python 的 sample()方法将所有类型的 iterables 如 [list](https://www.askpython.com/python/list/python-list) 、 [tuple](https://www.askpython.com/python/tuple/python-tuple) 、 [sets](https://www.askpython.com/python/set/python-set) 、 [dataframe](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 等。它通过用户定义数量的数据值从 iterable 中随机选择数据。

现在让我们在下面的部分中理解相同的结构。

* * *

### sample()方法的语法

看看下面的语法！

**语法:**

```py
sample(iterable, sample_amt)

```

我们需要向函数提供我们希望函数从提供的 iterable 或数据结构中随机选取的样本量。

* * *

## 1.带列表的 Python 示例()

在本节中，我们实现了 sample()函数和一个 Python 列表，并使用该函数从数据中随机选择了 4 个样本。

**举例:**

```py
from random import sample 

lst = [10,20,40,30,50,46,89] 
res = sample(lst, 4)
print(res) 

```

**输出:**

```py
[20, 89, 40, 46]

```

* * *

## 2.带集合的 Python 示例()

这里，我们使用字母和数值创建了一个 Python 集合。此外，我们对集合应用了 sample()函数，并随机选择了 4 个值。

**举例:**

```py
from random import sample 

set_sample = {1,2,3,4,5,"s","a","f","a"}
res = sample(set_sample, 4)
print(res) 

```

**输出:**

```py
['f', 1, 5, 2]

```

* * *

## sample()函数的错误和异常

在处理 sample()函数时，我们会遇到一个 **ValueError** 异常。如果我们提供的 sample_amt 值大于 iterable 中数据值的总数，就会调用这个异常。

**举例:**

```py
from random import sample 

set_sample = {1,2,3,4,5,"s","a","f","a"}
res = sample(set_sample, 15)
print(res) 

```

**输出:**

```py
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-9-284724c4b7db> in <module>
      2 
      3 set_sample = {1,2,3,4,5,"s","a","f","a"}
----> 4 res = sample(set_sample, 15)
      5 print(res)

c:\users\hp\appdata\local\programs\python\python36\lib\random.py in sample(self, population, k)
    316         n = len(population)
    317         if not 0 <= k <= n:
--> 318             raise ValueError("Sample larger than population or is negative")
    319         result = [None] * k
    320         setsize = 21        # size of a small set minus size of an empty list

ValueError: Sample larger than population or is negative

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！🙂
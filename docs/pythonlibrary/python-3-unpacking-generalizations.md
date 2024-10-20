# Python 3 -解包一般化

> 原文：<https://www.blog.pythonlibrary.org/2017/02/21/python-3-unpacking-generalizations/>

Python 3.5 在 [PEP 448](https://www.python.org/dev/peps/pep-0448/) 中增加了更多对解包泛化的支持。根据 PEP，它增加了* iterable 解包操作符和** dictionary 解包操作符的*扩展用法，允许在更多位置、任意次数和在其他情况下解包*。这意味着我们现在可以通过任意次数的解包来调用函数。我们来看一个 dict()的例子:

```py

>>> my_dict = {'1':'one', '2':'two'}
>>> dict(**my_dict, w=6)
{'1': 'one', '2': 'two', 'w': 6}
>>> dict(**my_dict, w='three', **{'4':'four'})
{'1': 'one', '2': 'two', 'w': 'three', '4': 'four'}

```

有趣的是，如果键不是字符串，那么解包就不起作用:

```py

>>> my_dict = {1:'one', 2:'two'}
>>> dict(**my_dict)
Traceback (most recent call last):
  File "", line 1, in <module>dict(**my_dict)
TypeError: keyword arguments must be strings
```

**更新:**我的一位读者很快指出这不起作用的原因是因为我试图解包成一个函数调用(即 dict())。如果我只使用 dict 语法进行解包，整数键就能很好地工作。我要说的是:

```py

>>> {**{1: 'one', 2:'two'}, 3:'three'}
{1: 'one', 2: 'two', 3: 'three'}

```

dict 解包的另一个有趣的地方是后面的值总是会覆盖前面的值。PEP 中有一个很好的例子可以证明这一点:

```py

>>> {**{'x': 2}, 'x': 1}
{'x': 1}

```

我觉得这很棒。您可以在 collections 模块中用 ChainMap 做同样的事情，但是这要简单得多。

然而，这种新的解包也适用于元组和列表。让我们尝试将一些不同类型的项目合并到一个列表中:

```py

>>> my_tuple = (11, 12, 45)
>>> my_list = ['something', 'or', 'other']
>>> my_range = range(5)
>>> combo = [*my_tuple, *my_list, *my_range]
>>> combo
[11, 12, 45, 'something', 'or', 'other', 0, 1, 2, 3, 4]

```

在进行这种解包更改之前，您需要做这样的事情:

```py

>>> combo = list(my_tuple) + list(my_list) + list(my_range)
[11, 12, 45, 'something', 'or', 'other', 0, 1, 2, 3, 4]

```

我认为新的语法实际上对于这种情况非常方便。实际上，我在 Python 2 中遇到过一两次这种情况，这种新的增强非常有用。

* * *

### 包扎

在 PEP 448 中有很多其他的例子，读起来很有趣，可以在 Python 的解释器中尝试。我强烈推荐你去看看，并尝试一下这个特性。每当我们最终迁移到 Python 3 时，我希望在我的新代码中开始使用这些特性。
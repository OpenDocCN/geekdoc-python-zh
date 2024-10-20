# Python 201 -有序直接

> 原文：<https://www.blog.pythonlibrary.org/2016/03/24/python-201-ordereddict/>

Python 的 collections 模块有另一个很棒的 dict 子类，叫做 OrderedDict。顾名思义，这个字典在添加键时跟踪它们的顺序。如果您创建一个常规字典，您会注意到它是一个无序的数据集合:

```py

>>> d = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
>>> d
{'apple': 4, 'banana': 3, 'orange': 2, 'pear': 1}

```

每次打印出来，顺序可能都不一样。有时候，你需要按照特定的顺序遍历字典中的键。例如，我有一个用例，我需要对键进行排序，这样我就可以按顺序遍历它们。为此，您可以执行以下操作:

```py

>>> keys = d.keys()
>>> keys
dict_keys(['apple', 'orange', 'banana', 'pear'])
>>> keys = sorted(keys)
['apple', 'banana', 'orange', 'pear']
>>> for key in keys:
...     print (key, d[key])
... 
apple 4
banana 3
orange 2
pear 1

```

让我们使用原始字典创建一个 OrderedDict 的实例，但是在创建过程中，我们将对字典的键进行排序:

```py

>>> from collections import OrderedDict
>>> d = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
>>> new_d = OrderedDict(sorted(d.items()))
>>> new_d
OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
>>> for key in new_d:
...     print (key, new_d[key])
... 
apple 4
banana 3
orange 2
pear 1

```

在这里，我们通过使用 Python 的 **sorted** 内置函数动态排序来创建 OrderedDict。sorted 函数接受字典的条目，这些条目是表示字典的键对的元组列表。它对它们进行排序，然后将它们传递给 OrderedDict，后者将保留它们的顺序。因此，当我们打印键和值时，它们是按照我们期望的顺序排列的。如果你要遍历一个常规的字典(不是一个排序的键列表)，顺序会一直改变。

请注意，如果添加新的键，它们将被添加到 OrderedDict 的末尾，而不是被自动排序。

关于 OrderDicts，需要注意的另一点是，当您比较两个 OrderedDicts 时，它们不仅会测试条目是否相等，还会测试顺序是否正确。正规字典只看字典的内容，不关心它的顺序。

最后，OrderDicts 在 Python 3 中有两个新方法: **popitem** 和 **move_to_end** 。popitem 方法将返回并移除(key，item)对。move_to_end 方法会将现有键移动到 OrderedDict 的任意一端。如果 OrderedDict 的*最后一个*参数设置为 True(这是默认值)，则该项将移动到右端，如果为 False，则该项将移动到开头。

有趣的是，OrderedDicts 使用 Python 的反向内置函数支持反向迭代:

```py

>>> for key in reversed(new_d):
...     print (key, new_d[key])
... 
pear 1
orange 2
banana 3
apple 4

```

非常简洁，尽管你可能不会每天都需要这个功能。

* * *

### 包扎

此时，您应该准备好亲自尝试 OrderedDict 了。它是对您的工具箱的一个有用的补充，我希望您能在您的代码库中找到它的许多用途。

* * *

### 相关阅读

*   订购产品的官方[文件](https://docs.python.org/2/library/collections.html#collections.OrderedDict)
*   本周 Python 模块: [OrderedDict](https://pymotw.com/2/collections/ordereddict.html)
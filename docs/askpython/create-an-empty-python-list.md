# 用 Python 创建一个空列表——两种简单的方法

> 原文：<https://www.askpython.com/python/list/create-an-empty-python-list>

嘿，大家好！在本文中，我们将关注创建空 Python 列表的**不同方法。**

我们先来详细了解一下 [Python](https://www.askpython.com/) List 的工作原理。

* * *

## Python 列表的基础

[Python List](https://www.askpython.com/python/list/python-list) 是一个动态数据结构，将元素存储在其中。列表的美妙之处在于它可以存储不同的元素，即不同数据类型的元素都可以存储在其中。

**举例**:

```py
lst = [1,2,4,'Python','sp']
print(lst)

```

**输出:**

```py
[1, 2, 4, 'Python', 'sp']

```

现在让我们看看创建一个空 Python 列表的方法。

* * *

## 技巧 1:使用方括号

我们可以使用方括号在 Python 中创建一个空列表。众所周知，列表使用方括号来存储它的元素。因此，它也可以应用在这里。

**语法:**

```py
list = []

```

**举例:**

```py
lst = []
print("Empty list: ",lst)

```

在上面的例子中，我们使用方括号创建了一个空列表。

**输出:**

```py
Empty list:  []

```

* * *

## 技巧 2:使用 list()函数

Python list()函数可以用来生成如下所示的空列表-

**语法:**

```py
list = list()

```

如果没有参数传递给它，`list() function`将返回一个空列表。但是，如果数据值作为参数传递给它，list()函数将返回 iterable 中的数据值。

**举例:**

```py
lst = list()
print("Empty list: ",lst)

```

这里，我们使用内置的 list()函数创建了一个空列表。因为没有参数传递给 list()函数，所以它返回一个空列表作为输出。

**输出:**

```py
Empty list:  []

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，请随时在下面评论。如果你想在 Python 中实现一个函数来[检查一个列表是否为空，看看链接的文章。](https://www.askpython.com/python/list/check-if-list-is-empty)

在那之前，学习愉快！！

* * *

## 参考

*   [如何创建空列表— StackOverflow](https://stackoverflow.com/questions/2972212/creating-an-empty-list-in-python)
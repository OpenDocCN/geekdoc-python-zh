# 如何在 Python 中把集合转换成列表？

> 原文：<https://www.askpython.com/python/list/convert-set-to-list-python>

我们可以在 Python 中使用内置的 **list()** 方法将一个[集合](https://www.askpython.com/python/set/python-set)转换成[列表](https://www.askpython.com/python/list/python-list)。让我们看一些使用这个函数的例子。

## 在 Python 中将集合转换为列表的方法

我们将讨论一些简单的方法，这些方法允许快速地将 python 集合对象转换成列表对象。

### 1.使用 list()函数

**list()** 函数将 iterable 作为参数，并将其转换为列表类型的对象。这是一个可供您使用的内置方法。

```py
my_list = list(my_iterable)

```

由于一个集合也是可迭代的，我们可以将它传递给`list()`方法，并得到我们对应的列表。

```py
my_set = set({1, 4, 3, 5})

my_list = list(my_set)

print(my_list)

```

如预期的那样，输出将是包含上述值的列表。

```py
[1, 3, 4, 5]

```

注意，列表的顺序可以是随机的，不一定是有序的。

例如，以下面的片段为例。

```py
s = set()
s.add("A")
s.add("B")

print(list(s))

```

**我的情况下的输出**:

```py
['B', 'A']

```

### 2.使用手动迭代

我们也可以手动将元素添加到列表中，因为集合是可迭代的。除了由您自己编写之外，这个方法与使用 list()方法相比没有任何实际优势。

```py
s = set({1, 2, 3})

a = []

for i in s:
    a.append(i)

print(a)

```

同样，输出是一个列表:

```py
[1, 2, 3]

```

## 将 frozenset 转换为列表

Python **frozenset** 对象类似于集合，但不可变。因此，我们不能修改冷冻集的元素。我们也可以使用 **list()** 来转换这种类型的集合。

```py
f_set = frozenset({1, 3, 2, 5})

a = list(f_set)

print(a)

```

输出

```py
[1, 2, 3, 5]

```

* * *

## 参考

*   JournalDev 关于将集合转换成列表的文章
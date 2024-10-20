# python frozenset()-您需要知道的一切

> 原文：<https://www.askpython.com/python/built-in-methods/python-frozenset>

你好。所以今天我们在这里讨论**Python 的 frozenset()方法。**

所以在我们进入这个方法之前，让我们知道什么是 **frozenset** 。

## 什么是冷冻人？

`frozenset`是一个无序的、无索引的、不可变的元素集合。它提供了一个[集](https://www.askpython.com/python/set/python-set)在 Python 中提供的所有功能，唯一的不同是一个事实，即一个 frozenset 是**不可变的**，即在它被创建之后不能被改变。

因此，简而言之，冻结集合是不可变集合。

## Python frozenset()方法

Python `frozenset()`方法返回一个新的 frozenset 对象，其元素取自传递的`iterable`。如果没有指定`iterable`，则返回一个新的空集。

**注意:**元素必须是 [hashable](https://docs.python.org/3.7/glossary.html#term-hashable) 。

```py
fz = frozenset([iterable])

```

当没有指定任何内容时，`frozenset()`方法将一个空的 frozenset 对象返回给`fz`。

为了更好地理解该方法的工作原理，让我们看一个例子。

```py
# List Initialisation
list1 = [0, 1, 2, 3, 4, 5, 6]

fz = frozenset(list1)
print(fz)

fz = frozenset()  # empty frozenset
print(fz)

print("Type of fz: ", type(fz))

```

**输出:**

```py
frozenset({0, 1, 2, 3, 4, 5, 6})
frozenset()
Type of fz:  <class 'frozenset'>

```

这里，首先我们初始化了一个列表(`list1`)，然后将它作为`iterable`传递给`frozenset()`方法。作为回报，我们得到一个 frozenset 对象( **fz** )和列表中的元素。当没有传递任何东西时， **fz** 现在是一个空的 frozenset 对象。

## 冷冻集初始化

在下面给出的例子中，我们已经使用 Python `frozenset()`方法初始化了一个 frozenset，方法是传递不同的 iterables，如[列表](https://www.askpython.com/python/list/python-list)、[元组](https://www.askpython.com/python/tuple/python-tuple)、[集合](https://www.askpython.com/python/set/python-set)和[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)。

```py
# Initialisation
list1 = [1, 2, 3, 4, 5]

fz = frozenset(list1)  # from list object
print(fz)

fz = frozenset([1, 2, 3, 4, 5])  # from list
print(fz)

fz = frozenset({5: 'a', 4: 'B', 3: 'C', 2: 'D', 1: 'E', 0: '0'})# from dict
print(fz)

fz = frozenset({'Python', 'Java', 'C', 'C++', 'Kotlin', 'R'})# from set
print(fz)

fz = frozenset((17, 55, 26, 90, 75, 34)) # from tuple
print(fz)

```

**输出:**

```py
frozenset({1, 2, 3, 4, 5})
frozenset({1, 2, 3, 4, 5})
frozenset({0, 1, 2, 3, 4, 5})
frozenset({'Java', 'Kotlin', 'Python', 'C', 'R', 'C++'})
frozenset({34, 90, 75, 17, 55, 26})

```

对于每种情况，我们都得到一个 frozenset 对象，其中包含相应的 iterable 元素。但是请仔细注意，在字典的情况下，只考虑键。

## Python frozenset 上的操作

我们可以使用 dir()方法获得与 frozenset 对象相关的所有方法的名称。

```py
fo = frozenset([1, 2, 3, 4, 5])

print(dir(fo))

```

**输出:**

```py
['__and__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__rand__', '__reduce__', '__reduce_ex__', '__repr__', '__ror__', '__rsub__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__xor__', 'copy', 'difference', 'intersection', 'isdisjoint', 'issubset', 'issuperset', 'symmetric_difference', 'union']

```

从上面的输出可以看出，各种函数，如 add()、remove()、update()、pop()等。(用于改变/更新元素，可用于[组](https://www.askpython.com/python/set/python-set))缺失。这也是因为一个冻结的集合是不可变的。

现在让我们看看冻结集合的可用方法，使用它们我们可以执行各种操作。

```py
fs = frozenset([1, 12, 23, 45, 67, 89, 100])

print("Given Frozenset =", fs)

fs_len = len(fs)
print("Length of Frozenset =", fs_len)

print("23 in fs? ", 23 in fs)

print("23 not in fs? ", 23 not in fs)

print("Sets are disjoint? ", fs.isdisjoint(frozenset([10, 5])))

print("Set is Subset? ", fs.issubset(set([1, 2, 3, 4, 12, 23, 45, 67, 89, 100])))

print("fs is superset? ", fs.issuperset(frozenset({1, 100})))

print("Union of sets: ", fs.union(frozenset([-1, -12])))

print("Intersection: ", fs.intersection(set([1, 10, 100])))

print("Difference: ", fs.difference(frozenset([1, 10, 100])))

print("Symmetric difference: ", fs.symmetric_difference(frozenset([1, 10, 100])))

fs_copy = fs.copy()
print("Copy of fs: ", fs_copy)

```

**输出:**

```py
Given Frozenset = frozenset({1, 67, 100, 12, 45, 23, 89})
Length of Frozenset = 7
23 in fs?  True
23 not in fs?  False
Sets are disjoint?  True
Set is Subset?  True
fs is superset?  True
Union of sets:  frozenset({1, 67, 100, 12, 45, -12, 23, 89, -1})
Intersection:  frozenset({1, 100})
Difference:  frozenset({67, 12, 45, 23, 89})
Symmetric difference:  frozenset({67, 10, 12, 45, 23, 89})
Copy of fs:  frozenset({1, 67, 100, 12, 45, 23, 89})

```

这里，

*   **len(s)** :返回 frozenset s 的长度，
*   **s 中的 x**:检查 x 是否存在于 frozenset s 中，
*   **x 不在 s** 中:如果 x 不是 frozenset s 的元素则返回 True，否则返回 False，
*   **isdisjoint(other)** :如果集合中没有与`other`共有的元素，则返回 True。集合是不相交的当且仅当它们的交集是空集。
*   **issubset(other)** :检查 other 是否包含 frozenset 的元素，
*   **issuperset(other)** :检查 frozenset 是否包含`other`的元素，
*   **union(*others)** :返回一个 frozenset，包含所提供的其他集合的并集，
*   **交集(*others)** :返回一个 frozenset，传递 fs 和所有其他元素共有的元素，
*   **difference(*others)** :返回一个新的 frozenset，该 frozenset(fs)中的元素不在其他 frozenset 中，
*   **symmetric _ difference(other)**:返回一个新的 frozenset，其元素在 fs 或 other 中，但不在两者中。

## 总结

今天到此为止。希望你对 Python `frozenset()`方法有一个清晰的理解。

要了解更多信息，我们建议浏览参考资料部分提供的链接。

任何进一步的问题，请在下面随意评论。

## 参考

*   [Python frozenset](https://docs.python.org/3/library/stdtypes.html?highlight=frozenset#frozenset)–文档，
*   [Python 中 tuples 和 frozensets 的区别](https://stackoverflow.com/questions/14422409/difference-between-tuples-and-frozensets-in-python)—堆栈溢出问题，
*   [Set 和 frozenset 在实现上的差异](https://stackoverflow.com/questions/17646007/set-and-frozenset-difference-in-implementation)–堆栈溢出问题。
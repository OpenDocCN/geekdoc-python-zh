# 了解 Python enumerate()方法

> 原文：<https://www.askpython.com/python/built-in-methods/python-enumerate-method>

## 介绍

今天在本教程中，我们将学习 Python enumerate()方法。

## Python enumerate()方法

Python 内置的`enumerate()`方法将传递的序列转换成具有相同元素的元组形式的枚举对象。此外，该函数将**索引**添加到相应的元组元素。

使用 Python `enumerate()`方法的语法是，

```py
enumerate( thing, start)

```

这里，

*   **thing** 是我们需要添加单个元素索引的任意序列，
*   **start(可选)**是索引开始的起始值。如果没有通过，默认值设置为 0。

## 使用 Python enumerate()方法

Python `enumerate()`方法可以将任何可迭代序列转换成添加了索引的枚举对象。这个序列可以是一个[列表](https://www.askpython.com/python/list/python-list)、[字符串](https://www.askpython.com/python/string)，或者是一个[元组](https://www.askpython.com/python/tuple/python-tuple)。但不允许是[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)或[集合](https://www.askpython.com/python/set)，因为它们不是序列。

现在让我们看一些例子来更好地理解。

### 列举名单

当我们将一个[列表](https://www.askpython.com/python/list/python-list)传递给 Python `enumerate()`方法时，它会将基本索引作为第一个值添加到元组元素中。返回的 iterable 属于枚举类型。

要打印这个枚举对象，我们可以使用一个简单的`for`循环遍历它。注意，这次我们可以同时访问元素**的索引**和元素**的值**。

```py
list1 = [ 11, 22, 33, 44]

enumerate_list = enumerate(list1)
print("Enumerated list now looks like: ", list(enumerate_list))

#print the index and corresponding value for enumerated list1
for i, item in enumerate(list1):
    print("Index = ", i,"  :  ","value = ",item)

```

**输出**:

```py
Enumerated list now looks like:  [(0, 11), (1, 22), (2, 33), (3, 44)]
Index =  0   :   value =  11
Index =  1   :   value =  22
Index =  2   :   value =  33
Index =  3   :   value =  44

```

这里，

*   **list1** 是一个列表，里面有一些初始值。我们将它传递给 enumerate()方法，并将返回的对象存储在 **enumerate_list** 变量中，
*   当我们将这个对象类型化到一个列表中并尝试使用`print()`方法打印它时，我们可以清楚地观察到列表中的每个元素现在都被转换成一个添加了索引的元组，
*   我们使用带有两个变量 I 和 item 的 for 循环来遍历枚举对象。这样，我们可以同时访问索引 **(i)** 以及相应的元素 **(item)** 。

因此，输出是合理的。

对于[元组](https://www.askpython.com/python/tuple/python-tuple)来说，`enumerate()`也以同样的方式工作。

### 枚举字符串

类似地，我们也可以使用 Python `enumerate()`方法将一个[字符串](https://www.askpython.com/python/string)转换成一个添加了索引的枚举对象。

让我们看看如何。

```py
string1 = "AskPython"

enumerate_string = enumerate(string1)
print("Enumerated list now looks like: ", list(enumerate_string))

#print the index and corresponding character for enumerated string
for i, item in enumerate(string1):
    print("Index = ", i,"  :  ","character = ",item)

```

**输出**:

```py
Enumerated list now looks like:  [(0, 'A'), (1, 's'), (2, 'k'), (3, 'P'), (4, 'y'), (5, 't'), (6, 'h'), (7, 'o'), (8, 'n')]
Index =  0   :   character =  A
Index =  1   :   character =  s
Index =  2   :   character =  k
Index =  3   :   character =  P
Index =  4   :   character =  y
Index =  5   :   character =  t
Index =  6   :   character =  h
Index =  7   :   character =  o
Index =  8   :   character =  n

```

这里，

*   我们初始化一个字符串 **string1** 并将它的`enumerate()`输出存储在一个变量 **enumerate_string** 中，
*   然后打印列表类型转换**枚举字符串**。正如我们所看到的，它是一个元组列表，包含各个字符元素及其各自的索引，
*   我们再次使用 **for** 循环遍历枚举对象，并打印出带有索引的元素。

### 带开始参数的 Python enumerate()

如前所述，`start`参数是一个可选参数，它确定索引将从由`enumerate()`方法返回的枚举对象的哪个值开始。

让我们看一个例子，其中我们试图将起始索引为 **0** 的列表的索引改为起始索引为 **20** 的列表。

```py
list1 = [ 11, 22, 33, 44]

enumerate_list = enumerate(list1)
print("Enumerated list now looks like: ", list(enumerate_list))

#without start
print("Without Start:")
for i, item in enumerate(list1):
    print("Index = ", i,"  :  ","value = ",item)

#with start = 20
print("With Start:")
for i, item in enumerate(list1, 20):
    print("Index = ", i,"  :  ","value = ",item)

```

**输出**:

```py
Enumerated list now looks like:  [(0, 11), (1, 22), (2, 33), (3, 44)]
Without Start:
Index =  0   :   value =  11
Index =  1   :   value =  22
Index =  2   :   value =  33
Index =  3   :   value =  44
With Start:
Index =  20   :   value =  11
Index =  21   :   value =  22
Index =  22   :   value =  33
Index =  23   :   value =  44

```

这里，从输出可以清楚地看到，使用`start=20`，方法返回的枚举对象的起始索引是 **20** 。而没有 start(默认值 **0** )索引从 0 开始。

## 结论

注意 Python `enumerate()`方法只适用于序列。因此，**字典**或**集合**不能被转换成 enumerate()对象。

因此，在本教程中，我们学习了 **Python** 中的`enumerate()`方法。如有任何问题，欢迎在评论中提问。

## 参考

*   [enumerate()](https://docs.python.org/2.3/whatsnew/section-enumerate.html)–Python 文档，
*   python enumerate()–Journal Dev Post。
# 用 Python 将字典转换成元组列表

> 原文：<https://www.pythonforbeginners.com/dictionary/convert-a-dictionary-to-list-of-tuples-in-python>

我们知道 python 中的字典包含键值对。在本文中，我们将把一个 python 字典转换成一个元组列表，其中每个元组包含一个键值对。

## 使用 for 循环将字典转换为元组列表

我们可以使用 for 循环，通过逐个访问字典的键和值，将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)转换为元组列表。首先，我们将使用键和值创建元组，然后我们将把它们附加到一个列表中，如下所示。

```py
myDict = {1: 2, 3: 4, 5: 6, 7: 8}
print("The Dictionary is:",myDict)
myList = []
for key in myDict:
    value = myDict[key]
    myTuple = (key, value)
    myList.append(myTuple)
print("The list of tuples is:",myList)
```

输出:

```py
The Dictionary is: {1: 2, 3: 4, 5: 6, 7: 8}
The list of tuples is: [(1, 2), (3, 4), (5, 6), (7, 8)]
```

## 使用 items()方法将字典转换为元组列表

当在字典上调用 items()方法时，它返回一个 dict_items 对象。dict_items 对象包含字典的键值对。我们可以使用 list()方法将 dict_items 转换为列表，如下所示。

```py
myDict = {1: 2, 3: 4, 5: 6, 7: 8}
print("The Dictionary is:", myDict)
items = myDict.items()
myList = list(items)
print("The list of tuples is:", myList)
```

输出:

```py
The Dictionary is: {1: 2, 3: 4, 5: 6, 7: 8}
The list of tuples is: [(1, 2), (3, 4), (5, 6), (7, 8)]
```

不使用 list()函数，我们可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 将 dict_items 转换为一个列表，如下所示。

```py
myDict = {1: 2, 3: 4, 5: 6, 7: 8}
print("The Dictionary is:", myDict)
items = myDict.items()
myList = [item for item in items]
print("The list of tuples is:", myList) 
```

输出:

```py
The Dictionary is: {1: 2, 3: 4, 5: 6, 7: 8}
The list of tuples is: [(1, 2), (3, 4), (5, 6), (7, 8)]
```

## 使用 zip()函数将字典转换为元组列表

我们可以使用 zip()函数创建一个包含键值对的元组列表。zip()函数将列表等可迭代对象作为输入，并将它们合并成一个可迭代对象。合并时，zip()函数在每个 iterable 的相同索引处创建元素元组。

例如，如果我们将两个列表[1，2，3]和[4，5，6]传递给 zip()函数，它会将这两个列表合并成[(1，4)，(2，5)，(3，6)]。即来自每个 iterable 的相同索引处的元素被压缩在一起。

**为了使用 zip()函数将字典转换成元组列表，我们将把键列表和值列表作为输入传递给 zip()函数**。

我们可以使用 keys()方法获得密钥列表。当在字典上调用 keys()方法时，它返回一个包含字典键的 dict_keys 对象。我们可以使用 list()函数将 dict_keys 对象转换成一个列表。

类似地，我们可以使用 values()方法获得值列表。在字典上调用 values()方法时，会返回一个包含字典值的 dict_values 对象。我们可以使用 list()函数将 dict_values 对象转换成一个列表。

我们可以使用 keys()和 values()方法以及 zip()函数将字典转换为元组列表，如下所示。

```py
myDict = {1: 2, 3: 4, 5: 6, 7: 8}
print("The Dictionary is:", myDict)
keys = list(myDict.keys())
values = list(myDict.values())
myList = list(zip(keys, values))
print("The list of tuples is:", myList)
```

输出:

```py
The Dictionary is: {1: 2, 3: 4, 5: 6, 7: 8}
The list of tuples is: [(1, 2), (3, 4), (5, 6), (7, 8)]
```

## 结论

在本文中，我们讨论了用 Python 将字典转换成元组列表的不同方法。我们使用 for 循环、list comprehension 和 zip()函数以及 items()、key()和 values()等字典方法将字典转换为元组列表。
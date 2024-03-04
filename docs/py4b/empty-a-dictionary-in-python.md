# 用 Python 清空字典

> 原文：<https://www.pythonforbeginners.com/basics/empty-a-dictionary-in-python>

我们使用一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)来存储键值对。有时，我们可能需要从字典中删除键值对。在本文中，我们将讨论通过在 python 中删除字典中的所有键值对来清空字典的不同方法。

## 使用 pop()方法清空字典

`pop()`方法用于从字典中删除一个键值对。当在字典上调用该方法时，它将键作为第一个输入参数，并将一个可选的默认值作为第二个输入参数。在执行时，它从字典中删除键及其关联值，并返回如下所示的值。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
myDict.pop(1)
print("The output dictionary is:")
print(myDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The output dictionary is:
{2: 4, 3: 9, 4: 16, 5: 25}
```

如果字典中不存在这个键，`pop()`方法返回作为第二个参数传递给它的默认值。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
val = myDict.pop(10, -1)
print("The popped value is:", val)
print("The output dictionary is:")
print(myDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The popped value is: -1
The output dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

如果我们不向`pop()`方法传递任何默认值，并且字典中不存在这个键，那么`pop()`方法就会引发`KeyError`异常，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
val = myDict.pop(10)
print("The output dictionary is:")
print(myDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 4, in <module>
    val = myDict.pop(10)
KeyError: 10
```

您可以通过使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块来避免该异常。

为了使用 pop()方法清空字典，我们将首先使用`keys()`方法提取字典的键。之后，我们将从字典中删除每个键及其关联值，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
key_list = list(myDict.keys())
for key in key_list:
    myDict.pop(key)
print("The output dictionary is:")
print(myDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The output dictionary is:
{}
```

## 使用 del 语句清空字典

代替`pop()`方法，我们可以使用 python 中的`del`语句来清空 python 中的字典。在这种方法中，我们将使用字典的键删除字典中的每个键-值对，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
key_list = list(myDict.keys())
for key in key_list:
    del myDict[key]
print("The output dictionary is:")
print(myDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The output dictionary is:
{} 
```

## 使用 clear()方法清空字典

使用`clear()`方法，我们可以用一条语句清空 python 中的字典。在字典上调用`clear()`方法时，会删除字典中的所有键值对。这给我们留下了一个空字典，正如您在下面的例子中所看到的。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
myDict.clear()
print("The output dictionary is:")
print(myDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The output dictionary is:
{}
```

## 结论

在本文中，我们讨论了 python 中清空字典的三种方法。想了解更多 python 中的字典，可以阅读这篇关于 python 中[字典理解的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)
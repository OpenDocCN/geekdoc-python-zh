# 在 Python 中反转字典

> 原文：<https://www.pythonforbeginners.com/basics/reverse-a-dictionary-in-python>

python 中的字典是存储键值映射的好工具。在本文中，我们将讨论如何在 python 中反转字典。我们将使用不同的例子来做这件事，这样我们可以更好地理解这些方法。

## 如何用 Python 逆向一个字典？

当我们反转一个给定的 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)时，我们改变了键值对中值的顺序。在每个键-值对中，当前键成为值，当前值成为新字典中的键。例如，看看下面的字典。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25} 
```

我们把这本字典翻过来之后，它会是这样的。

```py
reversedDict = {1: 1, 4: 2, 9: 3, 16: 4, 25: 5} 
```

在 python 中反转一个字典，我们可以使用 for 循环或者[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)。让我们逐一讨论这两种方法。

## 使用 For 循环反转 Python 中的字典

要使用 for 循环反转字典，我们将首先创建一个空字典来存储反转的键-值对。之后，我们将遍历输入字典中的每个键值对。在遍历时，我们将使输入字典中的每个值成为输出字典中的一个键，与输入字典中的值相关联的键将成为输出字典中的关联值，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
reversedDict = dict()
for key in myDict:
    val = myDict[key]
    reversedDict[val] = key
print("The reversed dictionary is:")
print(reversedDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The reversed dictionary is:
{1: 1, 4: 2, 9: 3, 16: 4, 25: 5}
```

## 使用 keys()和 values()方法

您还可以使用带有 for 循环的`keys()` 和`values()` 方法来反转字典。在这种方法中，我们将首先创建一个键列表和一个输入字典的值列表。之后，我们将使值列表中的每个元素成为输出字典中的一个键。键列表中的关联元素将成为输出字典中的相应值，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
reversedDict = dict()
key_list = list(myDict.keys())
val_list = list(myDict.values())
n = len(key_list)
for i in range(n):
    key = val_list[i]
    val = key_list[i]
    reversedDict[key] = val
print("The reversed dictionary is:")
print(reversedDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The reversed dictionary is:
{1: 1, 4: 2, 9: 3, 16: 4, 25: 5}
```

## 使用 items()方法

你也可以使用`items()` 方法来反转一个字典。在这种方法中，我们将从输入字典中逐个取出每个条目。之后，我们将如下反转键-值对。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
reversedDict = dict()
for item in myDict.items():
    key = item[1]
    val = item[0]
    reversedDict[key] = val
print("The reversed dictionary is:")
print(reversedDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The reversed dictionary is:
{1: 1, 4: 2, 9: 3, 16: 4, 25: 5}
```

## 使用字典理解

不使用 for 循环，我们可以使用 [dictionary comprehension](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 在一条 python 语句中反转一个字典。这里，我们将通过反转字典的每个键-值对中的键和值来创建输出字典，如下所示。

```py
myDict = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print("The input dictionary is:")
print(myDict)
reversedDict = {val: key for (key, val) in myDict.items()}
print("The reversed dictionary is:")
print(reversedDict)
```

输出:

```py
The input dictionary is:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
The reversed dictionary is:
{1: 1, 4: 2, 9: 3, 16: 4, 25: 5}
```

## 结论

在本文中，我们讨论了用 python 反转字典的四种方法。要了解更多关于 python 中的字典，您可以阅读这篇关于如何将字典转换成元组列表的文章。你可能也会喜欢这篇关于用 python 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)
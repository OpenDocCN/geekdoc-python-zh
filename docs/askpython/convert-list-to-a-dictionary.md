# Python 中如何把列表转换成字典？

> 原文：<https://www.askpython.com/python/list/convert-list-to-a-dictionary>

在本教程中，我们将讨论如何在 Python 中将列表转换成字典。

***也读:[如何在 Python 中把字符串转换成列表？](https://www.askpython.com/python/string/convert-string-to-list-in-python)***

## **Python 中什么是列表？**

Python 中的**列表**是一种线性数据结构，用于存储相同类型或不同类型的值的集合。列表中的每一项都用逗号`(,)`隔开。

Python 列表包含在方括号`[]`中。Python 列表是*可变的*，这意味着在 Python 程序内部创建之后，我们可以更改或修改它的内容。

Python 列表以有序的方式存储元素，我们可以通过索引直接访问列表中的任何元素。这就是我们如何在 Python 中创建一个列表，我们也可以使用`type()`函数来验证它是否是一个列表:

```py
# Defining a Python list

ls = ['AskPython', 'JournalDev', 'LinuxforDevices']

# Printing the results
print(ls)

# Validating the type of 'ls'
print(type(ls))

```

**输出:**

```py
['AskPython', 'JournalDev', 'LinuxforDevices']
<class 'list'>

```

## **Python 中的字典是什么？**

Python 中的[字典是一种特殊类型的数据结构，用于以键值对格式存储数据。Python 字典有两个主要组成部分:键&值，其中键必须是单个实体，但值可以是多值实体，如列表、元组等。](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)

键及其值由冒号(:)分隔，每个项目(键-值对)由逗号(，)分隔，就像 Python 列表一样。一本字典被括在花括号`{}`里。像 Python 列表一样，Python 字典也是可变的。

Python 字典以无序的方式存储元素，我们可以通过它的键直接访问字典中的任何值。这就是我们如何在 Python 中创建一个字典，就像列表一样，我们也可以使用`type()`函数来验证它是否是一个字典:

```py
# Defining a Python dictionary
ds = {'AskPython': "Python", 'JournalDev': "Java, Android, Python, Web", 'LinuxforDevices': "Unix/Linux"}

# Printing the results
print(ds)

# Validating the type of 'ds'
print(type(ds))

```

**输出:**

```py
{'AskPython': 'Python', 'JournalDev': 'Java, Android, Python, Web', 'LinuxforDevices': 'Unix/Linux'}
<class 'dict'>

```

Python 中主要有两种将列表转换成字典的方法，一种是*字典理解*，另一种是使用 *Python zip()方法*。我们将逐一讨论它们。

***亦读: [Python 词典列表](https://www.askpython.com/python/list/list-of-dictionaries)***

## **使用字典理解将列表转换为字典**

在这种将列表转换为字典的方法中，我们遍历给定的 Python 列表，并在花括号`{}`内创建相应的`key:value`。每个列表元素都成为字典中的一个键，我们可以根据自己的用途为每个键生成一个值。让我们看一个使用[字典理解](https://www.askpython.com/python/dictionary/python-dictionary-comprehension)将 Python 列表转换成 Python 字典的例子。以下是演示它的 Python 代码:

```py
# Defining a Python list
ls = ['E', 'J', 'O', 'T', 'Y']

# Using dictionary comprehension to convert the list to a dictionary
ds = {item: ord(item) for item in ls}

# Printing the results
print("Given Python list: ", ls)
print("Generated Python dictionary: ", ds)

# Validating the type of 'ds'
print(type(ds))

```

**输出:**

```py
Given Python list:  ['E', 'J', 'O', 'T', 'Y']
Generated Python dictionary:  {'E': 69, 'J': 74, 'O': 79, 'T': 84, 'Y': 89}
<class 'dict'>

```

在上面的例子中，我们使用 Python `ord()`函数来获取列表中每一项的 [ASCII](https://www.askpython.com/python/built-in-methods/python-ascii-function) 值，列表实际上是一个英文字母表。并将 ord()函数返回的 ASCII 值赋给字典的相应键作为其值。因此，我们为列表中的每一项生成了一个键值对，并将其转换为 Python 字典。

## **使用 Python zip()函数将列表转换成字典**

在这个方法中，我们使用 [Python `iter`](https://www.askpython.com/python/python-iter-function) `()`函数创建一个迭代器，它将遍历给定的 Python 列表。然后我们将使用 [Python `zip()`函数](https://www.askpython.com/python/built-in-methods/python-zip-function)，它实际上将两个项目压缩在一起，并生成相应的`key:value`。

最后一步，我们使用 Python `dict()`函数进行类型转换，该函数根据`zip()`函数返回的压缩值创建并返回一个 Python 字典。

让我们看一个使用 Python `zip()`函数将 Python 列表转换成 Python 字典的例子。以下是演示它的 Python 代码:

```py
# Defining a Python list
ls = ['E', 69, 'J', 74, 'O', 79, 'T', 84, 'Y', 89]

# Creating an iterator for list 'ls'
item = iter(ls)

# Using the Python zip() function to convert the list to a dictionary
ds = dict(zip(item, item))

# Printing the results
print("Given Python list: ", ls)
print("Generated Python dictionary: ", ds)

# Validating the type of 'ds'
print(type(ds))

```

**输出:**

```py
Given Python list:  ['E', 69, 'J', 74, 'O', 79, 'T', 84, 'Y', 89]
Generated Python dictionary:  {'E': 69, 'J': 74, 'O': 79, 'T': 84, 'Y': 89}
<class 'dict'>

```

## **结论**

在本 Python 教程中，我们学习了将 Python 列表转换为 Python 字典的非常有用和方便的技术。我们还学习了各种有用的 Python 函数，如`iter()`、`zip()`、`dict()`和`ord()`。希望您对使用这些 Python 函数进行更有趣的学习感到兴奋。
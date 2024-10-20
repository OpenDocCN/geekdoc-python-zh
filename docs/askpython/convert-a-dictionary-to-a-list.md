# 用 Python 将字典转换成列表的 5 种简单方法

> 原文：<https://www.askpython.com/python/dictionary/convert-a-dictionary-to-a-list>

在本教程中，我们将讨论将 Python 字典转换为 Python 列表的五大方法。所以，让我们开始吧！

* * *

## 将字典转换为列表的 5 种方法

现在让我们学习如何在 Python 中将字典转换成列表。在接下来的几节中，您将看到 Python 中用于将数据类型从 dictionary 更改为 list 的各种方法和函数。我们开始吧！

### 方法 1:使用 dict.items()函数

在这个方法中，我们将使用 dictionary 类的 [dict.items()函数](https://www.askpython.com/python/string/dictionary-to-a-string)将字典转换成列表。`dict.items()`函数用于迭代 Python 字典的**键:值**对。默认情况下，键:值对以 Python **元组**的形式存储在由`dict.items()`函数返回的迭代器中。我们可以将这个返回的迭代器传递给`list()`函数，该函数最终会给我们一个包含给定 Python 字典的键:值对的 Python 元组列表。让我们进入实际的 Python 代码来实现这个方法。

```py
# Method- 1

# Create a Python dictionary
dr = {'AskPython': 'Python', 'LinuxForDevices': 'Linux', 'QuickExcel': 'Excel'}
print('Given Python dictionary:\n')
print(dr)

# Convert the above Python dictionary to a Python list
# Using the dict.items() function
ls = list(dr.items())
print('\nThis is the converted list:\n')
print(ls)

```

**输出:**

```py
Given Python dictionary:

{'AskPython': 'Python', 'LinuxForDevices': 'Linux', 'QuickExcel': 'Excel'}

This is the converted list:

[('AskPython', 'Python'), ('LinuxForDevices', 'Linux'), ('QuickExcel', 'Excel')]

```

### 方法 2:使用 zip()函数

在这个方法中，我们将使用 Python 中的 [zip()函数](https://www.askpython.com/python/built-in-methods/python-zip-function)将字典转换成列表。Python 中的`zip()`函数用于通过压缩值来组合两个迭代器对象。

这里我们将传递两个 Python 列表作为两个迭代器对象，一个是给定 Python 字典的所有键的列表，另一个是给定 Python 字典的所有值的列表。我们将使用 Python dictionary 类的`dict.keys()`和`dict.values()`函数获得这两个列表。默认情况下，键:值对以 Python **元组**的形式存储在由`zip()`函数返回的迭代器中。我们可以将这个返回的迭代器传递给`list()`函数，该函数将最终返回一个 Python 元组列表，其中包含给定 Python 字典的所有键:值对。让我们看看如何通过 Python 代码实现这个方法。

```py
# Method- 2

# Create a Python dictionary
dr = {'Python': '.py', 'C++': '.cpp', 'Csharp': '.cs'}
print('Given Python dictionary:\n')
print(dr)

# Convert the above Python dictionary to a Python list
# Using the zip() function
iter = zip(dr.keys(), dr.values())
ls = list(iter)
print('\nThis is the Converted list:\n')
print(ls)

```

**输出:**

```py
Given Python dictionary:

{'Python': '.py', 'C++': '.cpp', 'Csharp': '.cs'}

This is the Converted list:

[('Python', '.py'), ('C++', '.cpp'), ('Csharp', '.cs')]

```

### 方法 3:使用 map()函数

在这个方法中，我们将使用 Python 中的 [map()函数](https://www.askpython.com/python/built-in-methods/map-method-in-python)将一个字典转换成 Python 中的一个列表。Python 中的`map()`函数用于将任何 Python 函数映射到迭代器对象上。

这里我们将传递 Python `list()`函数作为第一个参数，传递`dict.item()`函数返回的迭代器对象作为第二个参数。然后我们可以将由`map()`函数返回的迭代器传递给`list()`函数，该函数将返回一个 Python 列表，其中包含给定 Python 字典的所有键:值对。让我们编写 Python 代码来实现这个方法。

```py
# Method- 3

# Create a Python dictionary
dr = {'Amazon': 'AWS', 'Microsoft': 'AZURE', 'Google': 'GCP'}
print('Given Python dictionary:\n')
print(dr)

# Convert the above Python dictionary to a Python list
# Using the map() function
iter = map(list, dr.items())
ls = list(iter)
print('\nThis is the Converted list:\n')
print(ls)

```

**输出:**

```py
Given Python dictionary:

{'Amazon': 'AWS', 'Microsoft': 'AZURE', 'Google': 'GCP'}

This is the Converted list:

[['Amazon', 'AWS'], ['Microsoft', 'AZURE'], ['Google', 'GCP']]

```

### 方法 4:使用简单 for 循环迭代

在这个方法中，我们将使用一个简单的循环来将字典转换成列表。我们将首先创建一个空列表，然后使用 for 循环遍历给定 Python 字典的键。最后，我们将为给定 Python 字典的每个 key: value 对创建一个 Python 元组，我们对它进行迭代，并将这个元组添加到我们创建的空列表中。让我们通过 Python 代码来实现这个方法。

```py
# Method- 4

# Create a Python dictionary
dr = {'Sanjay': 'ECE', 'Abhishek': 'EE', 'Sarthak': 'ICE'}
print('Given Python dictionary:\n')
print(dr)

# Create an empty Python list
ls = []

# Convert the above Python dictionary to a Python list
# Using the for loop iteration
for key in dr:
    item = (key, dr[key])
    ls.append(item)
print('\nThis is the Converted list:\n')
print(ls)

```

**输出:**

```py
Given Python dictionary:

{'Sanjay': 'ECE', 'Abhishek': 'EE', 'Sarthak': 'ICE'}

This is the Converted list:

[('Sanjay', 'ECE'), ('Abhishek', 'EE'), ('Sarthak', 'ICE')]

```

### 方法 5:使用列表理解

list comprehension 方法是初始化 Python 列表最广泛使用和最简洁的方法。我们还可以使用 for 循环在同一行中提供列表元素。这里我们将使用 for 循环和`dict.items()`函数获取给定 Python 字典的键:值对。然后，我们将使用获取的给定 Python 字典的键:值对初始化 Python 列表。让我们看看如何使用这个方法将一个给定的 Python 字典转换成一个 Python 列表。

```py
# Method- 4

# Create a Python dictionary
dr = {'A': 65, 'B': 66, 'C': 67, 'D': 68}
print('Given Python dictionary:\n')
print(dr)

# Convert the above Python dictionary to a Python list
# Using the list comprehension
ls = [(key, value) for key, value in dr.items()]
print('\nThis is the Converted list:\n')
print(ls)

```

**输出:**

```py
Given Python dictionary:

{'A': 65, 'B': 66, 'C': 67, 'D': 68}

This is the Converted list:

[('A', 65), ('B', 66), ('C', 67), ('D', 68)]

```

## 总结

在本教程中，我们学习了用 Python 将字典转换成列表的五种方法。希望你已经理解了这些东西，并准备用这些方法进行实验。感谢您阅读这篇文章，请继续关注我们，了解更多令人兴奋的 Python 编程学习内容。
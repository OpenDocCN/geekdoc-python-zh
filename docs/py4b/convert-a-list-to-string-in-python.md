# 在 Python 中将列表转换为字符串

> 原文：<https://www.pythonforbeginners.com/basics/convert-a-list-to-string-in-python>

Python 字符串是最常用的数据类型之一。然而，Python 列表是最常用的数据结构。在本文中，我们将尝试使用 python 中的不同函数将列表转换为字符串。我们将使用 join()之类的字符串方法和 map()和 str()之类的函数将列表转换成字符串。

## 使用字符串串联将列表转换为字符串

在 python 中，将列表转换为字符串的最简单方法是使用 for 循环和[字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)。要将列表转换为字符串，我们可以简单地将列表中的每个元素转换为字符串。然后我们可以把它们连接起来形成一个字符串。

为了执行转换，首先我们将获取一个空字符串。之后，我们将开始将列表中的每个元素转换成字符串后添加到空字符串中。在这个过程中，我们还会在每个元素之间添加空格。这可以如下进行。

```py
myList = ["PFB", 11.2, 11, "Python"]
print("The List is:",myList)
myString = ""
for elem in myList:
    myString = myString + str(elem) + " "

print("Output String is:")
print(myString.rstrip())
```

输出:

```py
 The List is: ['PFB', 11.2, 11, 'Python']
Output String is:
PFB 11.2 11 Python
```

在上面的例子中，我们可以看到在字符串的末尾添加了一个额外的空格。我们可以使用 rstrip()方法从字符串中删除多余的空格。

在 python 中将列表转换为字符串的另一种方法是使用 join()方法。join()方法用于从字符串列表中创建一个字符串。

join()方法在分隔符上调用，分隔符用于分隔字符串中的列表元素。字符串列表作为 join()方法的输入给出，它返回一个由列表元素创建的字符串。

我们将创建一个空字符串，列表中的所有元素都将连接到该字符串。为了创建字符串，我们将逐个获取列表中的每个元素，并将其转换为字符串。然后，我们将使用由列表的先前元素和当前元素组成的字符串创建一个字符串列表。我们将使用 join()方法从当前字符串和先前创建的字符串创建新字符串，方法是将一个空格字符作为分隔符，并对其调用 join()方法。我们将使用 for 循环对列表中的所有元素执行这个过程，直到形成完整的字符串。这可以从下面的例子中看出。

```py
myList = ["PFB", 11.2, 11, "Python"]
print("The List is:", myList)
myString = ""
for elem in myList:
    myString = " ".join([myString, str(elem)])

print("Output String is:")
print(myString.lstrip())
```

输出:

```py
The List is: ['PFB', 11.2, 11, 'Python']
Output String is:
PFB 11.2 11 Python
```

在上面的方法中，在输出字符串的左侧添加了一个额外的空格，必须使用 lstrip()方法将其删除。为了避免这种情况，我们可以使用 map()函数将列表中的每个元素转换为一个字符串，而不是对列表中的每个元素应用 str()函数，这样我们就可以使用 join()方法执行字符串连接，从而使用一条语句获得输出字符串。

map()函数将一个函数和一个 iterable 作为输入参数，对 iterable 对象的每个元素执行该函数，并返回可以转换为列表的输出 map 对象。

为了将列表的元素转换成字符串，我们将把 str()函数和输入列表传递给 map()方法。之后，我们可以使用 join()方法从字符串列表中创建一个字符串，如下所示。

```py
myList = ["PFB", 11.2, 11, "Python"]
print("The List is:", myList)
strList = list(map(str, myList))
myString = " ".join(strList)

print("Output String is:")
print(myString.lstrip())
```

输出:

```py
The List is: ['PFB', 11.2, 11, 'Python']
Output String is:
PFB 11.2 11 Python
```

## 使用列表理解将列表转换为字符串

我们可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 将一个列表转换成一个字符串。为此，我们将使用 list comprehension 和 str()函数将列表中的每个元素转换为字符串，然后使用 join()方法将它们连接起来，如下所示。

```py
myList = ["PFB", 11.2, 11, "Python"]
print("The List is:", myList)
strList = [str(i) for i in myList]
myString = " ".join(strList)

print("Output String is:", myString)
```

输出:

```py
The List is: ['PFB', 11.2, 11, 'Python']
Output String is: PFB 11.2 11 Python
```

或者，我们可以使用 map()函数将列表中的元素转换为字符串。然后，我们将使用 list comprehension 从 map()函数创建的 map 对象创建一个新的列表，并将它作为 join()方法的输入，从列表创建一个字符串，如下所示。

```py
myList = ["PFB", 11.2, 11, "Python"]
print("The List is:", myList)
myString = " ".join([i for i in map(str, myList)])
print("Output String is:", myString)
```

输出:

```py
The List is: ['PFB', 11.2, 11, 'Python']
Output String is: PFB 11.2 11 Python
```

## 结论

在本文中，我们使用了 str()、map()和 join()函数将列表转换为字符串。我们也看到了如何使用列表理解来完成这项任务。要了解如何将字符串转换成列表，请阅读这篇关于 python 中的[字符串拆分](https://www.pythonforbeginners.com/dictionary/python-split)操作的文章。
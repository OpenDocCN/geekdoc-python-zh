# Python 中的元组字符串到元组

> 原文：<https://www.pythonforbeginners.com/basics/tuple-string-to-tuple-in-python>

将数据从一种形式转换成另一种形式是一项单调乏味的任务。在本文中，我们将讨论在 python 中将元组字符串转换为元组的两种方法。

## 如何在 Python 中将元组字符串转换成元组

假设给我们一个字符串形式的元组，如下所示。

```py
myStr = "(1,2,3,4,5)"
```

现在，我们必须从给定的字符串创建元组`(1,2,3,4,5)`。为此，我们将首先删除括号和逗号`“,”` 字符。为此，我们将使用`replace()`方法用空格替换逗号，用空字符串替换括号。在字符串上调用`replace()`方法时，将被替换的字符作为第一个输入参数，新字符作为第二个输入参数。我们将使用 `replace()`方法逐一替换`“(”, “)”`和 `“,”`字符，如下所示。

```py
myStr = "(1,2,3,4,5)"
print("The tuple string is:", myStr)
myStr = myStr.replace("(", "")
myStr = myStr.replace(")", "")
myStr = myStr.replace(",", " ")
print("The output string is:", myStr)
```

输出:

```py
The tuple string is: (1,2,3,4,5)
The output string is: 1 2 3 4 5
```

现在，我们获得了一个包含由空格分隔的数字的字符串。为了从字符串中获取数字，我们现在将使用`split()`方法分割字符串。在字符串上调用`split()` 方法时，该方法将一个字符作为可选的输入参数，并在指定字符处分割字符串后返回一个包含元素的列表。如果我们不给出任何字符作为输入参数，它会在空格处分割字符串。

我们将使用`split()`方法分割字符串，如下所示。

```py
myStr = "(1,2,3,4,5)"
print("The tuple string is:", myStr)
myStr = myStr.replace("(", "")
myStr = myStr.replace(")", "")
myStr = myStr.replace(",", " ")
myList = myStr.split()
print("The output list is:", myList)
```

输出:

```py
The tuple string is: (1,2,3,4,5)
The output list is: ['1', '2', '3', '4', '5']
```

现在，我们已经获得了一个包含所有数字的列表。但是，它们以字符串的形式出现。为了获得一个整数列表，我们将如下使用`map()`函数和`int()`函数。

```py
myStr = "(1,2,3,4,5)"
print("The tuple string is:", myStr)
myStr = myStr.replace("(", "")
myStr = myStr.replace(")", "")
myStr = myStr.replace(",", " ")
myList = myStr.split()
myList = list(map(int, myList))
print("The output list is:", myList)
```

输出:

```py
The tuple string is: (1,2,3,4,5)
The output list is: [1, 2, 3, 4, 5]
```

由于我们已经获得了整数列表，我们将从列表中创建一个元组，如下所示。

```py
myStr = "(1,2,3,4,5)"
print("The tuple string is:", myStr)
myStr = myStr.replace("(", "")
myStr = myStr.replace(")", "")
myStr = myStr.replace(",", " ")
myList = myStr.split()
myList = list(map(int, myList))
myTuple = tuple(myList)
print("The output tuple is:", myTuple)
```

输出:

```py
The tuple string is: (1,2,3,4,5)
The output tuple is: (1, 2, 3, 4, 5)
```

您可以看到，我们已经使用 python 中的 `replace()`方法、 `split()`方法和`int()` 函数将 tuple 字符串转换为 tuple。

## 使用 Python 中的 eval()函数将元组字符串转换为元组

`eval()`函数用于计算表达式。它将一个字符串作为输入参数，遍历该字符串，然后返回输出。我们可以使用如下所示的`eval()`函数直接将 tuple 字符串转换成 tuple。

```py
myStr = "(1,2,3,4,5)"
print("The tuple string is:", myStr)
myTuple = eval(myStr)
print("The output tuple is:", myTuple)
```

输出:

```py
The tuple string is: (1,2,3,4,5)
The output tuple is: (1, 2, 3, 4, 5)
```

## 结论

在本文中，我们讨论了在 python 中将元组字符串转换为元组的两种方法。要了解更多关于字符串的知识，您可以阅读这篇关于 python 中[字符串格式化的文章。你可能也会喜欢这篇关于理解 python 中的 T2 的文章。](https://www.pythonforbeginners.com/basics/strings-formatting)
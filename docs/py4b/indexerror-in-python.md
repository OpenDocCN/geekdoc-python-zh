# Python 中的 IndexError

> 原文：<https://www.pythonforbeginners.com/basics/indexerror-in-python>

列表是 Python 中最常用的数据结构之一。当您的程序在使用列表时遇到错误时，您可能会收到消息“*IndexError:list index out of range*”。在本文中，我们将通过研究 Python 中的 IndexError 来讨论如何避免这种错误。

## Python 中什么是 IndexError？

IndexError 是 python 中的一个异常，当我们试图访问列表中的元素或列表中不存在的索引中的元组时，就会发生这个异常。例如，我们有一个包含 10 个元素的列表，索引从 0 到 9。如果我们试图访问索引 10 或 11 或更多的元素，它将导致程序引发 IndexError，并显示消息"*IndexError:list index out of range*",如下所示。

```py
myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("The list is:", myList)
index = 10
element = myList[index]
print("Element at index {} is {}".format(index,element))
```

输出:

```py
The list is: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 4, in <module>
    element = myList[index]
IndexError: list index out of range
```

类似地，如果我们有一个由 10 个元素组成的元组，元素的索引在 0 到 9 的范围内。如果我们试图访问索引为 10 或更多的元素，程序将导致如下的 IndexError。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
print("The Tuple is:", myTuple)
index = 10
element = myTuple[index]
print("Element at index {} is {}".format(index,element))
```

输出:

```py
The Tuple is: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string12.py", line 4, in <module>
    element = myTuple[index]
IndexError: tuple index out of range
```

这里，我们试图从只有 10 个元素的元组中访问索引为 10 的元素。因此，程序会遇到一个 IndexError 异常。

## Python 中如何避免 IndexError？

我们既可以处理索引错误异常，也可以抢占它。为了抢占，我们可以使用 if-else 块。为了在异常发生后对其进行处理，我们可以使用 Python 中的[异常处理。](https://www.pythonforbeginners.com/error-handling/exception-handling-in-python)

### 使用 If-Else 块抢占 IndexError

在 python 中避免 IndexError 的唯一方法是确保我们不会从超出范围的索引中访问元素。例如，如果一个列表或元组有 10 个元素，这些元素将只出现在索引 0 到 9 处。因此，我们不应该访问索引为 10 或更大的元素。为此，我们可以使用 if-else 语句来检查索引是否在正确的范围内，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
print("The Tuple is:", myTuple)
index = 10
print("Index is:",index)
if index <= 9:
    element = myTuple[index]
    print("Element at index {} is {}".format(index, element))
else:
    print("Index should be smaller.")
```

输出:

```py
The Tuple is: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
Index is: 10
Index should be smaller.
```

### 使用 Try-Except 块处理 IndexError

或者，我们可以使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块来处理程序引发的 IndexError 异常。使用这种方法，我们无法避免 IndexError 异常，但我们可以确保程序不会在引发 IndexError 后过早终止。

```py
myTuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
print("The Tuple is:", myTuple)
index = 10
print("Index is:",index)
try:
    element = myTuple[index]
    print("Element at index {} is {}".format(index, element))
except IndexError:
    print("Index should be smaller.")
```

输出:

```py
The Tuple is: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
Index is: 10
Index should be smaller.
```

## 结论

在本文中，我们讨论了 python 中的 IndexError。我们还讨论了超越这个异常的方法。你可以在这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章中阅读更多关于 python 列表的内容。

要了解更多关于 python 编程的知识，你可以阅读这篇关于[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。您可能也会喜欢这篇关于 [python simplehttpserver](https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver) 的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！
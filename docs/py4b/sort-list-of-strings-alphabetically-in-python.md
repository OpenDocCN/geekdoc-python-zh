# Python 中按字母顺序排列的字符串列表

> 原文：<https://www.pythonforbeginners.com/basics/sort-list-of-strings-alphabetically-in-python>

在 python 中，我们使用字符串来处理文本数据。在编程时，我们可能需要对 python 中的字符串列表进行排序。在本文中，我们将讨论在 python 中按字母顺序对字符串列表进行排序的不同方法。

## 使用 Sort()方法按字母顺序对字符串列表进行排序。

为了对列表中的元素进行排序，我们使用了`sort()`方法。当在列表上调用时，`sort()`方法对现有列表的元素进行排序。例如，我们可以对数字列表进行如下排序。

```py
myList = [1, 23, 12, 345, 34, 45]
print("The list is:")
print(myList)
myList.sort()
print("The reverse sorted List is:")
print(myList)
```

输出:

```py
The list is:
[1, 23, 12, 345, 34, 45]
The reverse sorted List is:
[1, 12, 23, 34, 45, 345]
```

为了按字母顺序对字符串列表进行排序，我们可以使用`sort()`方法，就像我们使用它对数字列表进行排序一样。列表中的字符串将根据它们的第一个字符进行比较。ASCII 值中第一个字符在前的字符串将位于排序列表的第一个位置。例如，“苹果”在“盒子”之前，就像“a”在“b”之前一样。您可以在下面的示例中观察到这一点。

```py
myList = ["apple","box","tickle","python","button"]
print("The list is:")
print(myList)
myList.sort()
print("The sorted List is:")
print(myList)
```

输出:

```py
The list is:
['apple', 'box', 'tickle', 'python', 'button']
The sorted List is:
['apple', 'box', 'button', 'python', 'tickle']
```

如果两个字符串具有相同的第一个字符，将使用第二个字符来比较这两个字符串。类似地，如果两个字符串的第二个字符相同，它们将根据第三个字符在排序列表中排序，依此类推。您可以在下面的示例中观察到这一点。

```py
myList = ["apple", "aaple", "aaaple", "tickle", "python", "button"]
print("The list is:")
print(myList)
myList.sort()
print("The sorted List is:")
print(myList)
```

输出:

```py
The list is:
['apple', 'aaple', 'aaaple', 'tickle', 'python', 'button']
The sorted List is:
['aaaple', 'aaple', 'apple', 'button', 'python', 'tickle']
```

您也可以使用参数“`reverse`”按相反的字母顺序对字符串列表排序，如下所示。

```py
myList = ["apple", "aaple", "aaaple", "tickle", "python", "button"]
print("The list is:")
print(myList)
myList.sort(reverse=True)
print("The reverse sorted List is:")
print(myList)
```

输出:

```py
The list is:
['apple', 'aaple', 'aaaple', 'tickle', 'python', 'button']
The reverse sorted List is:
['tickle', 'python', 'button', 'apple', 'aaple', 'aaaple']
```

## 使用 sorted()函数按字母顺序对字符串列表排序

如果不想修改已有的列表，可以使用`sorted()`函数对 python 中的字符串列表进行排序。`sorted()`函数的工作方式类似于`sort()`方法。唯一的区别是它返回一个新的排序列表，而不是修改原来的列表。

为了按字母顺序对字符串列表进行排序，我们将把字符串列表传递给 sorted()函数，它将返回如下排序列表。

```py
myList = ["apple", "aaple", "aaaple", "tickle", "python", "button"]
print("The list is:")
print(myList)
newList = sorted(myList)
print("The sorted List is:")
print(newList)
```

输出:

```py
The list is:
['apple', 'aaple', 'aaaple', 'tickle', 'python', 'button']
The sorted List is:
['aaaple', 'aaple', 'apple', 'button', 'python', 'tickle']
```

这里，对字符串进行排序的机制类似于我们在上一节中讨论的机制。

您也可以使用“`reverse`”参数按字母顺序对列表进行反向排序，如下所示。

```py
myList = ["apple", "aaple", "aaaple", "tickle", "python", "button"]
print("The list is:")
print(myList)
newList = sorted(myList, reverse=True)
print("The reverse sorted List is:")
print(newList)
```

输出:

```py
The list is:
['apple', 'aaple', 'aaaple', 'tickle', 'python', 'button']
The reverse sorted List is:
['tickle', 'python', 'button', 'apple', 'aaple', 'aaaple']
```

## 结论

在本文中，我们讨论了如何使用`sort()`方法和`sorted()`函数在 python 中按字母顺序对字符串列表进行排序。我们还讨论了在对字符串列表进行排序时，`sort()`方法和`sorted()`函数是如何操作的。要了解更多关于字符串的内容，你可以阅读这篇关于[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。你可能也会喜欢[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)上的这篇文章。
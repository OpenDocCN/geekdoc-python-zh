# 在 Python 中将字符串转换为列表

> 原文：<https://www.pythonforbeginners.com/basics/convert-string-to-list-in-python>

字符串和列表是最常用的 python 对象。有时，在操作字符串时，我们需要将字符串转换成列表。在本文中，我们将讨论在 Python 中将字符串转换成列表的不同方法。

## 使用 Python 中的 List()函数将字符串转换为列表

`list()`函数用于从现有的 iterable 对象创建一个列表。`list()`函数将一个 iterable 对象作为它的输入参数。执行后，它返回一个包含 iterable 对象所有元素的列表。

为了在 Python 中将字符串转换成列表，我们将把输入字符串传递给 `list()`函数。执行后，它将返回一个包含输入字符串字符的列表。您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
output_list = list(myStr)
print("The input string is:", myStr)
print("The output list is:", output_list)
```

输出:

```py
The input string is: pythonforbeginners
The output list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

在上面的例子中，我们已经将字符串`pythonforbeginners`传递给了`list()`函数。执行后，它返回字符串中的字符列表。

## Python 中使用 append()方法列出的字符串

方法用于将一个元素添加到一个列表中。当在列表上调用时，它将一个元素作为其输入参数，并将该元素追加到列表中。

要使用`append()`方法将字符串转换成列表，我们将使用以下步骤。

*   首先，我们将创建一个名为`myList`的空列表来存储输出列表。
*   之后，我们将使用 for 循环遍历输入字符串的字符。
*   在迭代过程中，我们将使用`append()`方法将当前字符添加到`myList`中。

执行 for 循环后，我们将在`myList`中得到输出列表。您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
myList = []
for character in myStr:
    myList.append(character)
print("The input string is:", myStr)
print("The output list is:", myList)
```

输出:

```py
The input string is: pythonforbeginners
The output list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

在这里，您可以观察到我们使用 for 循环和`append()`方法从输入字符串中获得了一个列表。

## 使用 Python 中的 extend()方法列出的字符串

方法用来同时向一个列表中添加几个元素。当在列表上调用时，`extend()` 方法将 iterable 对象作为其输入参数。执行后，它将 iterable 对象的所有元素追加到列表中。

为了使用`extend()`方法将字符串转换成列表，我们将首先创建一个名为`myList`的空列表。之后，我们将调用`myList`上的`extend()` 方法，并将输入字符串作为`extend()`方法的输入参数。

在执行了`extend()`方法之后，我们将在`myList`中得到结果输出。您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
myList = []
myList.extend(myStr)
print("The input string is:", myStr)
print("The output list is:", myList)
```

输出:

```py
The input string is: pythonforbeginners
The output list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

在上面的例子中，我们使用 extend()方法获得了字符串的字符列表。

## 使用 Python 中的列表理解将字符串转换为列表

[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)用于从现有的可迭代对象创建一个新列表。理解列表的语法如下。

```py
newList=[expression for element in iterable]
```

为了在 python 中将字符串转换成列表，我们将使用输入字符串来代替`iterable`。在`expression`和`element`的位置，我们将使用字符串的字符。

执行上述语句后，我们将在`newList`中得到输出列表。

您可以在下面的示例中观察到这一点。

```py
myStr = "pythonforbeginners"
myList = [character for character in myStr]
print("The input string is:", myStr)
print("The output list is:", myList)
```

输出:

```py
The input string is: pythonforbeginners
The output list is: ['p', 'y', 't', 'h', 'o', 'n', 'f', 'o', 'r', 'b', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

## 结论

在本文中，我们讨论了在 Python 中将字符串转换为列表的四种方法。在所有这些方法中，您可以使用`list()`函数从输入字符串创建一个新的列表。

如果您想将字符串中的字符添加到现有的列表中，您可以使用带有`append()` 方法或`extend()`方法的方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中的[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。你可能也会喜欢这篇关于用 Python 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)

请继续关注更多内容丰富的文章。

快乐学习！
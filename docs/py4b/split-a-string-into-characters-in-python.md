# 在 Python 中将字符串拆分成字符

> 原文：<https://www.pythonforbeginners.com/basics/split-a-string-into-characters-in-python>

python 中使用字符串来处理文本数据。在本文中，我们将讨论在 Python 中将字符串拆分成字符的不同方法。

## 我们可以使用 Split()方法将一个字符串拆分成多个字符吗？

在 python 中，我们通常在一个字符串上使用`split()`方法将它拆分成子字符串。在字符串上调用`split()`方法时，它将一个字符作为分隔符作为输入参数。执行后，它在分隔符出现的所有情况下拆分字符串，并返回子字符串列表。例如，考虑下面的例子。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=myStr.split()
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: ['Python', 'For', 'Beginners']
```

有人会说我们可以用一个空字符串来把一个字符串分割成字符。让我们试试。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=myStr.split("")
print("The output is:",output)
```

输出:

```py
ValueError: empty separator
```

在这个例子中，我们传递了一个空字符串作为分隔符，将字符串分割成字符。但是，程序会遇到 ValueError 异常，指出您使用了空分隔符。

因此，我们不能使用`split()`方法将 python 字符串分割成字符。让我们讨论一下其他的方法。

## 使用 Python 中的 for 循环将字符串分割成字符

python 中的字符串是可迭代的对象。因此，我们可以使用 for 循环逐个访问字符串的字符。

为了使用 for 循环分割字符串，我们将首先定义一个空列表来包含输出字符。然后，我们将使用一个 [python for 循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)遍历字符串的字符。迭代时，我们将使用`append()`方法将字符串的每个字符添加到列表中。当在列表上调用`append()`方法时，它将一个字符作为输入参数，并将其附加到列表的末尾。

在执行 for 循环后，我们将获得列表中字符串的所有字符。您可以在下面的示例中观察到这一点。

```py
output=[]
myStr="Python For Beginners"
print("The input string is:",myStr)
for character in myStr:
    output.append(character)
print("The output is:",output) 
```

输出:

```py
The input string is: Python For Beginners
The output is: ['P', 'y', 't', 'h', 'o', 'n', ' ', 'F', 'o', 'r', ' ', 'B', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

在上面的例子中，我们将字符串`"Python For Beginners"` 分成了一系列字符。

## 使用列表理解将字符串转换为字符列表

[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)用于从 python 中现有的 iterable 对象的元素创建一个列表。这是使用 for 循环和`append()`方法的更好的替代方法，因为我们可以用一条 python 语句将任何可迭代对象转换成一个列表。

您可以使用列表理解将一个字符串拆分成一个字符列表，如下所示。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=[character for character in myStr]
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: ['P', 'y', 't', 'h', 'o', 'n', ' ', 'F', 'o', 'r', ' ', 'B', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

在这个例子中，我们使用了列表理解而不是 for 循环。因此，我们能够用更少的代码行获得相同的结果。

## 使用 Python 中的 List()函数将字符串转换为字符列表

`list()` 构造函数用于从任何可迭代对象创建列表，比如 python 中的字符串、集合或元组。它将 iterable 对象作为其输入参数，并返回一个包含 iterable 对象元素的列表。

为了将字符串分解成字符，我们将把输入字符串传递给`list()`函数。执行后，它将返回一个包含输入字符串所有字符的列表。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=list(myStr)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: ['P', 'y', 't', 'h', 'o', 'n', ' ', 'F', 'o', 'r', ' ', 'B', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's']
```

## 使用 tuple()函数

元组是列表的不可变版本。因此，您也可以将字符串转换为元组字符。为此，您可以使用`tuple()`功能。`tuple()`函数将 iterable 对象作为其输入参数，并返回一个包含 iterable 对象元素的元组。

为了分割一个字符串，我们将把输入字符串传递给 `tuple()`函数。执行后，它将返回一个包含输入字符串所有字符的元组。您可以在下面的示例中观察到这一点。

```py
myStr="Python For Beginners"
print("The input string is:",myStr)
output=tuple(myStr)
print("The output is:",output)
```

输出:

```py
The input string is: Python For Beginners
The output is: ('P', 'y', 't', 'h', 'o', 'n', ' ', 'F', 'o', 'r', ' ', 'B', 'e', 'g', 'i', 'n', 'n', 'e', 'r', 's')
```

## 结论

在本文中，我们讨论了在 python 中将字符串拆分成字符的不同方法。

要了解更多关于 python 编程的知识，你可以阅读这篇关于[字符串操作](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)的文章。您可能也会喜欢这篇关于 [python simplehttpserver](https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver) 的文章。

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！
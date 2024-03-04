# 如何在 Python 中拆分字符串变量

> 原文：<https://www.pythonforbeginners.com/basics/how-to-split-string-variables-in-python>

在很多情况下，我们可能需要拆分字符串变量。在 python 中拆分字符串变量，我们可以使用带有不同参数的`split()`方法、`rsplit()`方法和`splitlines()`方法来完成不同的任务。在本文中，我们将探讨在 python 中拆分字符串变量的几种方法。

## 在空格处拆分字符串

要在每个空格处分割字符串变量，我们可以使用不带参数的`split()`函数。

`split()`函数的语法是`split(separator, maxsplit)`，其中`separator`指定了字符串应该被拆分的字符。`maxsplit`指定字符串被拆分的次数。默认情况下，`separator`的值为`whitespace`，这意味着如果没有指定参数，字符串将在`whitespaces`处被拆分。参数`maxsplit`的默认值为-1，这表示字符串应该在所有`separator`出现时被拆分。

当我们在没有任何参数的情况下对任何字符串调用`split()`方法时，它将在每个有`whitespace`的地方执行 [python 字符串拆分](https://www.pythonforbeginners.com/dictionary/python-split)操作，并返回一个子字符串列表。

例子

```py
myString="Python For Beginners"
print("The string is:")
print(myString)
myList=myString.split()
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python For Beginners
Output List is:
['Python', 'For', 'Beginners']
```

## 在任意特定字符处拆分字符串

要在 python 中以任何特定字符分割字符串变量，我们可以提供该字符作为分隔符参数。

例如，如果我们想在下划线`_`出现的地方拆分字符串`Python_For_Beginners`，我们将如下执行。

```py
myString="Python_For_Beginners"
print("The string is:")
print(myString)
myList=myString.split("_")
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python_For_Beginners
Output List is:
['Python', 'For', 'Beginners']
```

在输出中可以看到，输出列表包含输入字符串的子字符串，这些子字符串在下划线字符处被拆分。

## 从左到右拆分字符串一定次数

如果我们想在指定字符的特定位置分割输入字符串，我们可以通过在`split()`方法中指定分割次数作为`maxsplit`参数来实现。这样，`split()`方法将把字符串从左到右拆分指定的次数。

下面的程序只在`whitespace`出现的地方分割字符串`"Python For Beginners"`，如程序中的 [python 注释](https://www.pythonforbeginners.com/comments/comments-in-python)所述。

```py
#The string will be split at only one place where whitespace occurs.
myString="Python For Beginners"
print("The string is:")
print(myString)
myList=myString.split(maxsplit=1)
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python For Beginners
Output List is:
['Python', 'For Beginners']
```

在输出中可以看到，输入字符串仅在第一个`whitespace`处被拆分，输出列表仅包含两个子字符串。

## 在某个字符处从左到右将字符串变量拆分一定次数

如果我们想在指定字符的特定位置拆分输入字符串，我们可以通过指定字符串必须拆分的字符作为第一个参数，指定要拆分的数量作为`maxsplit`参数和 in `split()`方法来实现。

例如，如果我们想从左到右只在一个有下划线`_`的地方分割字符串`Python_For_Beginners`，我们可以这样做。

```py
myString="Python_For_Beginners"
print("The string is:")
print(myString)
myList=myString.split("_",maxsplit=1)
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python_For_Beginners
Output List is:
['Python', 'For_Beginners']
```

## 在空格处从右向左拆分字符串变量一定次数

假设我们想从右到左拆分一个字符串一定的次数。为此我们可以使用`rsplit()`方法。

`rsplit()`函数的语法是`rsplit(separator, maxsplit)`，其中`separator`指定了字符串应该被拆分的字符。`maxsplit`指定字符串被拆分的次数。默认情况下，`separator`的值为`whitespace`，这意味着如果没有指定参数，字符串将在`whitespaces`处被拆分。参数`maxsplit`的默认值为-1，这表示字符串应该在所有`separator`出现时被拆分。

为了从右到左在空格处分割字符串一定的次数，我们可以在`maxsplit`参数处指定字符串必须被分割的次数。

```py
myString="Python For Beginners"
print("The string is:")
print(myString)
myList=myString.rsplit(maxsplit=1)
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python or Beginners
Output List is:
['Python For', 'Beginners']
```

在上面的输出中，我们可以看到字符串是从右向左拆分的，这与从左向右拆分字符串的`split()`方法不同。

## 在特定字符处从右向左拆分字符串变量一定次数

要在特定字符处从右到左拆分字符串一定次数，我们可以指定字符串必须拆分的字符作为 rsplit()方法的第一个参数。

例如，如果我们想从右到左只在一个有下划线`_`的地方分割字符串`Python_For_Beginners`，我们可以这样做。

```py
myString="Python_For_Beginners"
print("The string is:")
print(myString)
myList=myString.rsplit("_",maxsplit=1)
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python_For_Beginners
Output List is:
['Python_For', 'Beginners']
```

## 在换行符或换行符处拆分字符串变量

要在换行符处拆分字符串变量，我们可以使用 python 中的`splitlines()`方法。当在任何字符串上调用时，它返回原始字符串的子字符串列表，原始字符串在换行符处被拆分以创建子字符串。

示例:

```py
myString="Python is a good language.\n I love PythonForBeginners"
print("The string is:")
print(myString)
myList=myString.splitlines()
print("Output List is:")
print(myList)
```

输出:

```py
The string is:
Python is a good language.
 I love PythonForBeginners
Output List is:
['Python is a good language.', ' I love PythonForBeginners']
```

在输出中可以观察到，原始字符串包含两行文本，输出列表包含这两行作为其元素。

## 结论

在本文中，我们看到了使用`split()`、`rsplit()`和`splitlines()`方法在 python 中拆分字符串变量的各种方法。敬请关注更多文章。
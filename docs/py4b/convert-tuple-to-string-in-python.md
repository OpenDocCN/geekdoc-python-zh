# 在 Python 中将元组转换为字符串

> 原文：<https://www.pythonforbeginners.com/strings/convert-tuple-to-string-in-python>

在 python 中，一个元组可以包含不同的元素。另一方面，字符串只是一系列字符。在本文中，我们将讨论在 python 中给定一组字符时，如何将元组转换为字符串。

## 如何将字符元组转换成字符串？

假设我们有下面的字符组。

```py
myTuple = ("P", "y", "t", "h", "o", "n")
```

现在，我们必须将这个元组转换成字符串`“Python”`。

为了将 tuple 转换成 string，我们将首先创建一个名为`output_string`的空字符串。之后，我们将使用 for 循环遍历元组。遍历时，我们将使用[字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)操作将元组中的每个字符添加到`output_string`中。在执行 for 循环后，我们将在变量`output_string`中获得所需的字符串。您可以在下面的示例中观察到这一点。

```py
myTuple = ("P", "y", "t", "h", "o", "n")
output_string = ""
for character in myTuple:
    output_string += character
print("The input tuple is:", myTuple)
print("The output string is:", output_string)
```

输出:

```py
The input tuple is: ('P', 'y', 't', 'h', 'o', 'n')
The output string is: Python
```

不使用 for 循环，我们可以使用`join()`方法将字符元组转换成字符串。在字符串上调用`join()`方法时，该方法将包含字符串的 iterable 对象作为输入参数。执行后，它返回包含新字符的修改后的字符串以及原始字符串中的字符。

为了将一组字符转换成一个字符串，我们将首先创建一个名为`output_string`的空字符串。之后，我们将调用`output_string`上的 `join()` 方法，并将元组作为输入参数。执行后， `join()`方法将返回所需的字符串，如下所示。

```py
myTuple = ("P", "y", "t", "h", "o", "n")
output_string = ""
output_string = output_string.join(myTuple)
print("The input tuple is:", myTuple)
print("The output string is:", output_string)
```

输出:

```py
The input tuple is: ('P', 'y', 't', 'h', 'o', 'n')
The output string is: Python
```

## 将数字元组转换为字符串

如果你试图用上面讨论的任何一种方法将一组数字转换成一个字符串，程序将会抛出`TypeError`异常。您可以在下面的示例中观察到这一点。

```py
myTuple = (1, 2, 3, 4, 5, 6)
output_string = ""
for character in myTuple:
    output_string += character
print("The input tuple is:", myTuple)
print("The output string is:", output_string)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 4, in <module>
    output_string += character
TypeError: can only concatenate str (not "int") to str
```

类似地，当我们使用第二种方法时，程序会遇到如下所示的`TypeError`异常。

```py
myTuple = (1, 2, 3, 4, 5, 6)
output_string = ""
output_string = output_string.join(myTuple)
print("The input tuple is:", myTuple)
print("The output string is:", output_string)
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/string1.py", line 3, in <module>
    output_string = output_string.join(myTuple)
TypeError: sequence item 0: expected str instance, int found 
```

为了避免错误，我们只需在将元组添加到`output_string`之前将它的每个元素转换成一个字符串。在第一种方法中，我们将首先使用`str()`函数将元组的每个元素转换成一个字符串。之后，我们将执行连接操作。这样，我们可以将一组数字转换成一个字符串。

```py
myTuple = (1, 2, 3, 4, 5, 6)
output_string = ""
for element in myTuple:
    character = str(element)
    output_string += character
print("The input tuple is:", myTuple)
print("The output string is:", output_string)
```

输出:

```py
The input tuple is: (1, 2, 3, 4, 5, 6)
The output string is: 123456
```

对于使用`join()`方法的方法，我们将首先把数字元组转换成字符串元组。为此，我们将使用`map()`函数和`str()`函数。

`map()`函数将一个函数作为第一个参数，将一个 iterable 对象作为第二个输入参数。执行后，它返回一个 map 对象，其中函数应用于 iterable 对象的每个元素。我们将把`str()`函数和数字元组作为输入参数传递给`map()`函数。之后，我们将使用`tuple()`构造函数转换地图对象。因此，我们将得到一个字符串元组，如下所示。

```py
myTuple = (1, 2, 3, 4, 5, 6)
newTuple = tuple(map(str, myTuple))
print("The input tuple is:", myTuple)
print("The output tuple is:", newTuple)
```

输出:

```py
The input tuple is: (1, 2, 3, 4, 5, 6)
The output tuple is: ('1', '2', '3', '4', '5', '6')
```

从数字元组得到字符串元组后，我们可以直接得到字符串如下。

```py
myTuple = (1, 2, 3, 4, 5, 6)
newTuple = tuple(map(str, myTuple))
output_string = ''.join(newTuple)
print("The input tuple is:", myTuple)
print("The output string is:", output_string)
```

输出:

```py
The input tuple is: (1, 2, 3, 4, 5, 6)
The output string is: 123456
```

## 结论

在本文中，我们讨论了如何在 python 中将元组转换为字符串。要了解 python 中字符串的更多信息，可以阅读这篇关于 python 中[字符串格式化的文章。你可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/basics/strings-formatting)[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。
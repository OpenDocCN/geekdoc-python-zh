# Python 中不带换行符的打印

> 原文：<https://www.pythonforbeginners.com/basics/print-without-newline-in-python>

在编程时，我们使用`print()`函数将一个字符串或一个值打印到标准输出。通常，每个`print()`函数在新的一行打印一个值。然而，我们可以改变函数的这种行为。在本文中，我们将看到如何在 python 中不使用换行符进行打印。

## Python 中如何不换行打印？

`print()`函数接受各种输入参数以及要打印的值。通常，当我们不向`print()`函数传递任何额外的参数时，它会在每次调用该函数时在一个新行上打印值。您可以在下面的示例中观察到这一点。

```py
print("This is the first line.")
print("This is the second line.")
print("This is the third line.")
print("This is the fourth line.")
```

输出:

```py
This is the first line.
This is the second line.
This is the third line.
This is the fourth line.
```

我们可以通过使用参数“`end`”来改变`print()` 函数的这种行为。参数“`end`”用于指定字符。`print()`函数在打印传递给它的值后进行打印。参数“`end`”的默认值为“`\n`”，即换行符。这就是为什么`print()`函数在打印值后打印一个换行符。因此，下一个`print()`语句总是在新的一行打印值。

您可以传递任何字符作为参数“`end`”的输入参数。`print()`函数在执行时，总是在打印实际值后打印作为输入参数传递的字符。例如，你可以通过句号“.”字符给参数“`end`”作为输入参数。这将使`print()`功能在打印实际输入值后打印一个句点。您可以在下面的示例中观察到这一点。

```py
print("This is the first line.", end=".")
print("This is the second line.", end=".")
print("This is the third line.", end=".")
print("This is the fourth line.", end=".")
```

输出:

```py
This is the first line..This is the second line..This is the third line..This is the fourth line..
```

要在每个 print 语句执行后打印一个逗号，可以将“，”作为输入参数传递给`print()`函数，如下所示。

```py
print("This is the first line.", end=",")
print("This is the second line.", end=",")
print("This is the third line.", end=",")
print("This is the fourth line.", end=",")
```

输出:

```py
This is the first line.,This is the second line.,This is the third line.,This is the fourth line.,
```

同样，您可以传递连字符、&、*、$、@、！、圆括号、方括号或任何其他字符添加到参数"`end`"中。您甚至可以将字母和数字作为输入参数传递给`print()` 函数。在打印实际值之后，`print()`函数将简单地打印传递给参数“`end`的字符，而不打印到换行符。

## 结论

在本文中，我们讨论了如何在 python 中不使用换行符进行打印。通过使用 python 中的[字符串格式，还可以在打印时修改字符串的外观。要了解更多关于 python 中字符串的知识，你可以阅读这篇关于](https://www.pythonforbeginners.com/basics/strings-formatting)[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。
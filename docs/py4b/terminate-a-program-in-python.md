# 用 Python 终止程序

> 原文：<https://www.pythonforbeginners.com/basics/terminate-a-program-in-python>

在 python 中编写程序时，可能需要在满足某个条件后多次结束程序。在本文中，我们将讨论在 python 中终止程序的不同方法。

## 使用 quit()函数终止程序

`quit()`函数是一个内置函数，可以用来终止 python 中的程序。当执行`quit()`函数时，它会引发`SystemExit`异常。这导致了程序的终止。您可以在下面的示例中观察到这一点。

```py
number = 10
if number >= 10:
    print("The number is:", number)
    quit()
    print("This line will not be printed")
```

输出:

```py
The number is: 10
```

`quit()`功能是在现场模块中定义的，只有在安装了模块的情况下才起作用。因此，我建议您不要在实际应用程序中使用该函数。然而，在编写更小的程序时，您可以使用`quit()`函数。

## 使用 exit()函数终止程序

站点模块中也定义了`exit()`功能。它的工作方式类似于`quit()`函数，并在执行时终止程序，如下例所示。

```py
number = 10
if number >= 10:
    print("The number is:", number)
    exit()
    print("This line will not be printed")
```

输出:

```py
The number is: 10
```

同样，我建议您不要在生产环境中使用这个函数，因为它依赖于模块。

## 使用 sys.exit()函数终止程序

`sys`模块包含在核心 python 安装中。您可以按如下方式导入`sys`模块。

```py
import sys
```

要使用 sys 模块终止程序，我们将使用 sys.exit()函数。sys.exit()函数在执行时将一个字符串消息作为可选的输入参数，并在终止程序之前打印该消息，如下所示。

```py
import sys
number = 10
if number >= 10:
    print("The number is:", number)
    sys.exit()
    print("This line will not be printed")
```

输出:

```py
The number is: 10
```

但是，传递除 0 以外的任何输入参数都意味着程序异常终止。

## 使用 SystemExit 异常

以上所有函数都会引发`SystemExit`异常来终止程序。因此，我们也可以显式地调用`SystemExit`程序来终止 python 中的程序，如下所示。

```py
import sys
number = 10
if number >= 10:
    print("The number is:", number)
    raise SystemExit
    print("This line will not be printed")
```

输出:

```py
The number is: 10
```

## 结论

在本文中，我们看到了在 python 中终止程序的不同方法。终止程序的最好方法是使用 StackOverflow 上许多人讨论过的`sys.exit()`函数。所以，我建议你用这个函数来终止一个程序。要了解更多关于 python 编程的知识，你可以在[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)上阅读这篇文章。您可能也会喜欢这篇关于 python 中[字符串连接的文章。](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)
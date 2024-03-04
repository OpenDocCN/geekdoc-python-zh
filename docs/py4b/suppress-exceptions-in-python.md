# 在 Python 中禁止异常

> 原文：<https://www.pythonforbeginners.com/basics/suppress-exceptions-in-python>

在 python 中，我们通常使用 try-except 块来处理 python 中的异常。如果我们不想处理异常怎么办？如果我们只想忽略例外呢？在本文中，我们将讨论如何在 python 中抑制异常

## Python 中的异常处理

当程序中出现异常时，程序的执行会突然中断。例如，如果我们试图将一个数除以 0，将会发生异常，程序将给出如下异常回溯，如下所示。

```py
num1 = 10
num2 = 0
print("The first number is:", num1)
print("The second number is:", num2)
output = num1 / num2
print("The output is:", output)
```

输出:

```py
The first number is: 10
The second number is: 0
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/webscraping.py", line 5, in <module>
    output = num1 / num2
ZeroDivisionError: division by zero
```

当一个程序突然停止时，该程序所做的所有工作都将丢失。为了确保程序在保存工作后执行，我们通常使用如下的 [python try-except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块来处理异常。

```py
try:
    num1 = 10
    num2 = 0
    print("The first number is:", num1)
    print("The second number is:", num2)
    output = num1 / num2
    print("The output is:", output)
except:
    print("Exception occurred.")
```

输出:

```py
The first number is: 10
The second number is: 0
Exception occurred.
```

这里，业务逻辑写在 try 块中，处理异常的代码写在 except 块中。现在，让我们看看如何使用 try-except 块来抑制异常。

## 在 Python 中使用 Try-Except 抑制异常

抑制异常意味着我们不会显式地处理异常，并且它不应该导致程序终止。使用 python 中的 try-except 块和 pass 语句，我们可以抑制 python 中的异常。pass 语句用作空语句。执行时，它什么也不做。

为了抑制异常，我们可以在 except 块中使用 pass，而不是异常处理代码。这样，异常也将得到处理，如果发生异常，也不会做额外的工作。您可以使用带有 try-except 块的 pass 语句来抑制 python 中的异常，如下所示。

```py
try:
    num1 = 10
    num2 = 0
    print("The first number is:", num1)
    print("The second number is:", num2)
    output = num1 / num2
    print("The output is:", output)
except:
    pass
```

输出:

```py
The first number is: 10
The second number is: 0
```

## 使用 Python 中的 contextlib 模块抑制异常

不使用 try-except 块和 pass 语句，我们可以使用`contextlib`模块来抑制 python 中的异常。在这种方法中，我们将使用 which 语句和`suppress()`函数创建一个上下文，将异常作为输入参数提供给`suppress()`方法。

每当上下文中出现异常时，python 解释器会自动抑制它。您可以在下面的示例中观察到这一点。

```py
import contextlib

with contextlib.suppress(Exception):
    num1 = 10
    num2 = 0
    print("The first number is:", num1)
    print("The second number is:", num2)
    output = num1 / num2
    print("The output is:", output)
```

输出:

```py
The first number is: 10
The second number is: 0
```

## 结论

在本文中，我们讨论了在 python 中抑制异常的两种方法。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。
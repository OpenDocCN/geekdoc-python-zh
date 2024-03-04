# Python 中数字的绝对值

> 原文：<https://www.pythonforbeginners.com/basics/absolute-value-of-a-number-in-python>

在 python 中处理数字时，我们需要比较两个数字的大小，而不考虑它们的符号。例如，量值-10 大于量值 1。但是当比较-10 和 1 时，由于 1 的正号，它被声明为更大的数字。为了比较这两个数字的大小，首先我们需要找到这两个数字的绝对值。然后我们可以通过比较绝对值来比较数字的大小。在本文中，我们将用 python 实现程序来寻找不同的[数字数据类型](https://www.pythonforbeginners.com/basics/numeric-types-python)的绝对值。

## Python 中如何计算一个数的绝对值？

我们可以使用 abs()函数在 python 中计算任意数字的绝对值。abs()函数将一个数字作为唯一的参数，并返回该数字的绝对值。

输入参数可以是浮点数、整数或复数。我们还可以传递一个二进制数、一个八进制数或一个十六进制数作为 abs()函数的输入。

我们可以从下一节的例子中理解 abs()函数的工作原理。

## Python 中使用 abs()函数的示例

我们可以使用 abs()函数找到一个整数的绝对值，如下所示。

```py
myNum=10
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
myNum=-10
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
```

输出:

```py
Absolute value of 10 is 10.
Absolute value of -10 is 10.
```

我们可以使用 abs()函数找到浮点数的绝对值，如下所示。

```py
myNum=10.5
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
myNum=-10.5
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
```

输出:

```py
Absolute value of 10.5 is 10.5.
Absolute value of -10.5 is 10.5.
```

如果我们想知道一个复数的大小，我们可以使用 abs()函数。abs()函数接受一个复数作为输入，并返回该复数的大小，如下所示。

```py
myNum=3+5j
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
```

输出:

```py
Absolute value of (3+5j) is 5.830951894845301.
```

如果一个数以二进制、八进制或十六进制表示，我们也可以使用 abs()函数来确定十进制数的绝对值，如下所示。

```py
#hexadecimal number
myNum=0x123
absoluteVal=abs(myNum)
print("Absolute value {} is {}.".format(myNum,absoluteVal))
#binary number
myNum=0b1001
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
#octal number
myNum=0o123
absoluteVal=abs(myNum)
print("Absolute value of {} is {}.".format(myNum,absoluteVal))
```

输出:

```py
Absolute value 291 is 291.
Absolute value of 9 is 9.
Absolute value of 83 is 83.
```

## 绝对值计算过程中的错误处理

Python 是一种具有动态类型的编程语言。这意味着 python 解释器在运行时确定变量的数据类型。因此，当我们将一个变量作为输入传递给 abs()函数时，它可能没有正确数据类型的值来产生正确的输出。例如，当我们将一个字符串作为输入参数传递给 abs()函数时，该函数将引发一个 TypeError 异常，如下所示。

```py
myNum="PFB"
absoluteVal=abs(myNum)
print("Absolute value {} is {}.".format(myNum,absoluteVal))
```

输出:

```py
Traceback (most recent call last):
  File "/home/aditya1117/PycharmProjects/pythonProject/main.py", line 2, in <module>
    absoluteVal=abs(myNum)
TypeError: bad operand type for abs(): 'str'
```

我们知道，当程序的任何部分出现异常时，程序会立即终止。这将导致写入文件的数据或程序中执行的任何重要计算丢失。我们可以通过使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 块进行异常处理来避免这种损失。每当我们将一个类型不正确的变量作为输入参数传递给函数时，abs()函数都会引发一个异常。如果需要，我们可以在 except 块中处理异常，保存必要的数据并关闭文件。

## 结论

在本文中，我们学习了如何使用 python 中的 abs()函数来计算数字的绝对值。我们还研究了如何在十进制中求二进制、八进制和十六进制中数的绝对值。最后，我们看到了如何使用 try except 块处理 abs()函数生成的任何异常。请继续关注更多内容丰富的文章。
# 在 Python 中检查难看的数字

> 原文：<https://www.pythonforbeginners.com/basics/check-for-ugly-number-in-python>

你可能听说过自然数、质数、偶数和奇数。但是，你有没有想过什么是丑陋的数字？在这篇文章中，我们将讨论什么是丑陋的数字。我们还将编写一个程序来检查 python 中的难看数字。

## 什么是难看的数字？

如果一个数只有 2，3 和 5 个质因数，那么这个数就是难看的数。换句话说，如果一个数可以通过 2、3 或 5 的乘方得到，那么这个数就是难看的数。

例如，18 可以由 2¹x 3²得到。因此，这是一个丑陋的数字。同样，90 也是一个难看的数字，因为它可以被得到为 2¹x 3²x5¹。相反，126 不是一个难看的数字，因为它可以通过 2¹x 3²x7¹得到。这里，126 的质因数之一是 7。因此，这不是一个难看的数字。

现在，让我们制定一个算法来检查 Python 中的难看数字。

## Python 中检查难看数字的算法

为了检查一个数是不是丑数，我们会对它进行重复除法。首先，我们将把给定的数反复除以 2。如果得到的数不能被 2 除尽，我们将检查这个数是否是 1。如果是 1，我们会说一个数是难看的数。否则，我们将对结果执行 3 的重复除法。一旦结果被 3 整除，我们将检查它是否是 1。如果是，我们会说给定的数是一个难看的数。否则，我们将对结果执行 5 的重复除法。一旦结式被 5 整除，我们将检查结式是否为 1。如果是的话，我们会说一个数字是一个丑陋的数字。否则，我们会说这个数字不是一个丑陋的数字。

现在让我们用 Python 实现这个想法

## Python 程序检查难看的数字

这里，我们实现了一个函数 is_ugly()，使用 [while 循环](https://www.pythonforbeginners.com/loops/python-while-loop)进行重复除法，使用 if-else 语句进行条件检查。该函数将一个整数作为输入参数，如果输入的数字是一个难看的数字，则返回 True。否则，它返回 False。

```py
def is_ugly(N):
    while N % 2 == 0:
        N = N // 2
    if N == 1:
        return True
    while N % 3 == 0:
        N = N // 3
    if N == 1:
        return True
    while N % 5 == 0:
        N = N // 5
    if N == 1:
        return True
    return False

input_number = 18
output = is_ugly(input_number)
print("{} is an ugly number:{}".format(input_number, output))
input_number = 126
output = is_ugly(input_number)
print("{} is an ugly number:{}".format(input_number, output)) 
```

输出:

```py
18 is an ugly number:True
126 is an ugly number:False
```

## 结论

在本文中，我们讨论并实现了一个程序来检查一个给定的数字是否是一个丑陋的数字。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。您可能还会喜欢这篇关于 python 中的[复数的文章。](https://www.pythonforbeginners.com/data-types/complex-numbers-in-python)
# 在 Python 中计算整数的位数

> 原文：<https://www.pythonforbeginners.com/basics/count-digits-of-an-integer-in-python>

在 python 中，整数数据类型用于表示正整数和负整数。在本文中，我们将讨论一个用 python 计算整数位数的程序。

## Python 中如何计算一个整数的位数？

为了计算一个数字的位数，我们将使用一种将数字除以 10 的方法。当我们将一个整数除以 10 时，得到的数减少了一位。

例如，如果我们将 1234 除以 10，结果将是 123。这里，1234 有 4 个数字，而 123 只有 3 个数字。类似地，当我们将 123 除以 10 时，它将被缩减为只有两位数的数，以此类推。最后，该数字将变为 0。

你可以观察到，在 1234 变成 0 之前，我们只能用 10 除它 4 次。换句话说，如果一个整数有 n 位，我们只能将这个整数除以 10n 次，直到它变成 0。

## Python 中计算整数位数的程序

如上所述，我们将使用以下方法在 python 中计算数字的位数。

*   首先，我们将声明一个值计数，并将其初始化为 0。
*   然后，我们将[使用 while 循环](https://www.pythonforbeginners.com/loops/python-while-loop)将给定的数反复除以 10。
*   在 while 循环中，每当我们将数字除以 10 时，count 就会增加 1。
*   一旦数字变为 0，我们将退出 while 循环。
*   在执行 while 循环之后，我们将获得 count 变量中整数的位数。

我们可以用 python 实现上述方法来计算数字的位数，如下所示。

```py
number = 12345
print("The given number is:", number)
count = 0
while number > 0:
    number = number // 10
    count = count + 1
print("The number of digits is:", count)
```

输出:

```py
The given number is: 12345
The number of digits is: 5
```

## 结论

在本文中，我们讨论了一种在 python 中计算整数位数的方法。想了解更多 python 中的数字，可以阅读这篇关于 python 中的[十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数](https://www.pythonforbeginners.com/basics/decimal-module-in-python)的文章。
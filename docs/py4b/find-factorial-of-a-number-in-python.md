# 在 Python 中求一个数的阶乘

> 原文：<https://www.pythonforbeginners.com/basics/find-factorial-of-a-number-in-python>

在数学中，我们可以对任何给定的数进行不同的运算。一个这样的操作是寻找一个数的阶乘。在本文中，我们将讨论什么是阶乘，以及如何在 python 中找到一个数的阶乘。

## 什么是数的阶乘？

一个数 N 的阶乘定义为从 1 到 N 的所有数的乘积，换句话说，要求给定数 N 的阶乘，我们只需将 1 到 N 的所有数相乘。

例如，5 的阶乘将被计算为`1x2x3x4x5`，即 120。同样，6 的阶乘将是`1x2x3x4x5x6`即 720。

0 的阶乘已被定义为 1。另一方面，没有为负整数定义阶乘。

知道了这么多，让我们设计一个算法来寻找一个给定数字的阶乘。

## 在 Python 中求一个数的阶乘

要找到任意给定数字 N 的阶乘，我们只需将 1 到 N 的所有数字相乘，在此之前，我们将首先检查给定数字是否为负数。如果是，我们将通知用户没有为给定的数字定义阶乘。如果输入数是 0，我们就说阶乘是 1。

在正整数的情况下，我们可以定义一个变量“myFact”，并使用 python 中的 for 循环找到从 1 到 N 的所有数字的乘积，如下所示。

```py
def factorial(N):
    if N < 0:
        return -1
    if N == 0:
        return 1
    myFact = 1
    for i in range(1, N + 1):
        myFact = myFact * i
    return myFact

input_number = 0
output = factorial(input_number)
if output == -1:
    print("Factorial not defined for negative number {}.".format(input_number))
else:
    print("Factorial of {} is {}.".format(input_number, output))
input_number = 6
output = factorial(input_number)
if output == -1:
    print("Factorial not defined for negative number {}.".format(input_number))
else:
    print("Factorial of {} is {}.".format(input_number, output))
input_number = -10
output = factorial(input_number)
if output == -1:
    print("Factorial not defined for negative number {}.".format(input_number))
else:
    print("Factorial of {} is {}.".format(input_number, output)) 
```

输出:

```py
Factorial of 0 is 1.
Factorial of 6 is 720.
Factorial not defined for negative number -10. 
```

或者，我们可以使用 while 循环来寻找阶乘。这里，我们将使用一个计数变量，从 1 开始，一直到 N，在 while 循环的每次迭代中递增 1。在每次迭代中，我们将把计数乘以先前数的乘积。因此，我们可以使用 python 中的 while 循环来计算数字的阶乘，如下所示。

```py
def factorial(N):
    if N < 0:
        return -1
    if N == 0:
        return 1
    myFact = 1
    count = 1
    while count <= N:
        myFact = myFact * count
        count = count + 1
    return myFact

input_number = 0
output = factorial(input_number)
if output == -1:
    print("Factorial not defined for negative number {}.".format(input_number))
else:
    print("Factorial of {} is {}.".format(input_number, output))
input_number = 6
output = factorial(input_number)
if output == -1:
    print("Factorial not defined for negative number {}.".format(input_number))
else:
    print("Factorial of {} is {}.".format(input_number, output))
input_number = -10
output = factorial(input_number)
if output == -1:
    print("Factorial not defined for negative number {}.".format(input_number))
else:
    print("Factorial of {} is {}.".format(input_number, output)) 
```

输出:

```py
Factorial of 0 is 1.
Factorial of 6 is 720.
Factorial not defined for negative number -10.
```

## 结论

在本文中，我们讨论了一个数的阶乘。我们还使用 python 中的 for 循环和 while 循环实现了寻找数字阶乘的程序。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。您可能还会喜欢这篇关于 python 中的[复数的文章。](https://www.pythonforbeginners.com/data-types/complex-numbers-in-python)
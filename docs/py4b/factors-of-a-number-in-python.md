# Python 中数字的因子

> 原文：<https://www.pythonforbeginners.com/basics/factors-of-a-number-in-python>

您可能听说过 Python 中的倍数和因数。如果你正在读这个博客，我可以肯定地告诉你，你正在寻找写一个程序，寻找一个数字的因素。在这篇文章中，我们将讨论并实现一个用 python 寻找数字因子的程序。

## 什么是数的因数？

如果一个数 N 能整除 M，则称它是另一个数 M 的因数。换句话说，如果给我们两个数 N 和 M，M 除以 N 后没有余数，那么 N 称为 M 的因数。你还可以很容易地发现，一个数的任何因数总是小于或等于该数本身。

例如，5 是 20 的因数，因为 20 除以 5 得到的输出是 4，没有余数。

## 如何在 Python 中求一个数的因子？

为了求一个数 M 的因子，我们可以用 1 到 M 的数来除 M，在除 M 的时候，如果一个数 N 没有余数，我们就说 N 是 M 的因子，为此，我们可以使用 python 中的 For 循环如下。

```py
factors = set()
M = 120  # number whose factors we need to find
for N in range(1, M + 1):
    if M % N == 0:  # remainder is zero
        factors.add(N)
print("Factors of {} are {}".format(M, factors)) 
```

输出:

```py
Factors of 120 are {1, 2, 3, 4, 5, 6, 8, 40, 10, 12, 120, 15, 20, 24, 60, 30}
```

在上面的示例中，我们声明了一个名为 factors 的集合来存储数字 M 的因子。如果任何数字在除以 M 时余数为 0，我们将该数字添加到该集合中。在 for 循环执行之后，我们得到了数字 m 的所有因子的集合。

我们知道，一个数 M 大于 M/2 的唯一因子是 M 本身。因此，我们可以跳过将 M 除以大于 M/2 的数，从而更有效地找到 M 的因子，如下所示。

```py
factors = set()
M = 120  # number whose factors we need to find
factors.add(M)  # a number is a factor of itself
for N in range(1, M // 2 + 1):
    if M % N == 0:  # remainder is zero
        factors.add(N)
print("Factors of {} are {}".format(M, factors)) 
```

输出:

```py
Factors of 120 are {1, 2, 3, 4, 5, 6, 8, 40, 10, 12, 15, 20, 120, 24, 60, 30} 
```

我们知道一个数的因子成对出现。例如，数量为 M 的因子可以出现在(1，M)，(2，M/2)，(3，M/3)，(4，M/4)对中，直到(M1/2，M^(1/2) )。因此，我们将检查 M^(1/2) 之前的因子，而不是使用 for 循环来检查 M/2 之前的因子。每当我们找到一个因子时，我们也会将它的配对存储在包含所有因子的集合中。这样，我们可以更高效地在 python 中找到一个数的因子。

```py
factors = set()
M = 120  # number whose factors we need to find
factors.add(M)  # a number is a factor of itself
for N in range(1, M):
    if N * N > M:
        break
    if M % N == 0:  # remainder is zero
        factors.add(N)
        factors.add(M // N)
print("Factors of {} are {}".format(M, factors)) 
```

输出:

```py
Factors of 120 are {1, 2, 3, 4, 5, 6, 40, 8, 10, 12, 15, 20, 120, 24, 60, 30}
```

## 结论

在本文中，我们讨论了三个在 python 中寻找数字因子的程序。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数的文章。
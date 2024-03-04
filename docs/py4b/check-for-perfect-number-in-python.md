# 在 Python 中检查完全数

> 原文：<https://www.pythonforbeginners.com/basics/check-for-perfect-number-in-python>

我们根据他们的专业来命名数字。一个这样的数字是完美的数字。在本文中，我们将讨论完全数的性质。我们还将在 python 中实现一个检查完全数的程序。

## 什么是完美的数字？

如果一个数等于除该数本身之外的所有因子之和，则称该数为完全数。如果我们考虑和中的数本身，我们可以说一个完全数的所有因子之和是给定数的两倍。

例如，考虑数字 6。它有四个因素，即 1、2、3 和 6。因为我们排除了这个数字本身，所以其他因素(即 1、2 和 3)的总和是 6。因此，6 是一个完美的数字。

或者，6 的所有因子的和是 1+2+3+6，即 12，它是该数本身的两倍。因此，6 是一个完美的数字。

让我们来看另一个数字 10。10 的因数是 1，2，5 和 10。10 的所有因子之和等于 18，它不是给定数字的两倍。因此，10 不是一个完美的数字。

## 在 Python 中检查完全数

要检查一个完美的数字，我们首先要找到它的因子。之后，我们将检查所有因素的总和是否是给定数字的两倍。

求给定数 N 的因子，我们将该数除以从 1 到 N 的所有数，完全除给定数的数将被声明为 N 的因子，我们将这些因子存储在一个列表中，如下所示。

```py
dedef calculate_factors(N):
    factors = []
    for i in range(1, N + 1):
        if N % i == 0:
            factors.append(i)
    return factors

input_number = 10
output = calculate_factors(input_number)
print("factors of {} are {}".format(input_number, output)) 
```

输出:

```py
factors of 10 are [1, 2, 5, 10]
```

找到给定数字的因子后，我们将使用 sum()函数找到因子的和。找到总和后，我们将检查总和是否是给定数字的两倍。如果是，我们就说给定的数是完全数。否则不会。

我们可以实现这个逻辑来检查 python 中的完全数，如下所示。

```py
def calculate_factors(N):
    factors = []
    for i in range(1, N + 1):
        if N % i == 0:
            factors.append(i)
    return factors

def check_perfect_number(N):
    factors = calculate_factors(N)
    sumOfFactors = sum(factors)
    if sumOfFactors // 2 == N:
        return True
    else:
        return False

input_number = 10
output = check_perfect_number(input_number)
print("{} is a perfect number: {}".format(input_number, output))
input_number = 6
output = check_perfect_number(input_number)
print("{} is a perfect number: {}".format(input_number, output))
```

输出:

```py
10 is a perfect number: False
6 is a perfect number: True
```

## 结论

在这篇文章中，我们讨论了什么是完美的数字。我们还实现了一个程序来检查一个给定的数是否是一个完全数。想了解更多 python 中的数字，可以阅读这篇关于 python 中的[十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数](https://www.pythonforbeginners.com/basics/decimal-module-in-python)的文章。
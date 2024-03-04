# 在 Python 中检查 Moran 数

> 原文：<https://www.pythonforbeginners.com/basics/check-for-moran-number-in-python>

数字有魔力。人们根据他们的专业给不同的数字命名。在本文中，我们将讨论什么是莫兰数。我们还将用 python 实现一个程序来检查一个数是否是 Moran 数。

## 什么是莫兰数？

莫兰数是这样一个数，当它除以它的数字之和时，结果是一个质数。换句话说，如果我们取一个莫兰数，计算它的位数之和，用这个数除以位数之和，结果将是一个质数。莫兰数是哈沙德数的子集。

例如，42 是一个莫兰数。如果计算它的位数之和，就是 6。42 除以 6 的结果是 7，这是一个质数。因此，42 是一个莫兰数。

另一方面，如果我们取 20，它的位数之和是 2。20 除以 2 的结果是 10，这不是质数。因此，20 不是一个莫兰数。

## 在 Python 中检查 Moran 数

要在 Python 中检查 Moran 数，我们必须执行以下操作。

1.  计算给定数字的位数之和。
2.  用数字的和除这个数。
3.  检查除法的结果是否是质数。

让我们先讨论如何求数的位数之和。

要计算给定数字的位数之和，我们将该数字除以 10，直到该数字变为 0。在每次除法运算中，我们将得到最右边的数字作为余数。我们将使用余数来计算数字的总和，方法是将每个除法运算中的余数相加，如下所示。

```py
def calculate_sum_of_digits(N):
    sumOfDigits = 0
    while N > 0:
        digit = N % 10
        sumOfDigits = sumOfDigits + digit
        N = N // 10
    return sumOfDigits
```

求出位数之和后，就可以用给定的数除以位数之和来求结果。现在，我们必须检查这个数是否是质数。为此，我们将结果除以从 2 到结果平方根的所有数字。如果结果能被这个范围内的任何一个数整除，这个数就不是质数。下面给出的 isPrime()函数执行这个操作。它接受一个数字作为输入参数，如果给定的数字是质数，则返回 True。否则，它返回 False。

```py
def isPrime(N):
    count = 2
    while count ** 2 <= N:
        if N % count == 0:
            return False
        count = count + 1
    return True
```

## Python 中检查 Moran 数的程序

在定义了计算数字总和和检查素数的函数之后，我们可以用 Python 编写一个程序来检查 Moran 数，如下所示。

```py
def calculate_sum_of_digits(N):
    sumOfDigits = 0
    while N > 0:
        digit = N % 10
        sumOfDigits = sumOfDigits + digit
        N = N // 10
    return sumOfDigits

def isPrime(N):
    count = 2
    while count ** 2 <= N:
        if N % count == 0:
            return False
        count = count + 1
    return True

def check_for_moran_number(N):
    sumOfDigits = calculate_sum_of_digits(N)
    if N % sumOfDigits == 0:
        result = N // sumOfDigits
        return isPrime(result)
    else:
        return False

input_number = 42
output = check_for_moran_number(input_number)
print("{} is a Moran Number:{}".format(input_number, output))
input_number = 20
output = check_for_moran_number(input_number)
print("{} is a Moran Number:{}".format(input_number, output)) 
```

输出:

```py
42 is a Moran Number:True
20 is a Moran Number:False
```

## 结论

在本文中，我们讨论了什么是莫兰数。我们还讨论了检查 Moran 数的步骤，并用 Python 实现了它。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。您可能还会喜欢这篇关于 python 中的[复数的文章。](https://www.pythonforbeginners.com/data-types/complex-numbers-in-python)
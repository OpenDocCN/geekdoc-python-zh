# 在 Python 中求一个数的质因数

> 原文：<https://www.pythonforbeginners.com/basics/find-prime-factors-of-a-number-in-python>

你可能听说过数字的因数。任何给定数的因数都是那些能把给定数整除而不留余数的数。在本文中，我们将讨论一种在 python 中寻找一个数的质因数的算法。

## 一个数的质因数是什么？

质数是那些只有两个因子的数，1 和数本身。此外，我们可以将任何给定的数表示为质数的乘积。在这里，我们可以用来表示任何一个给定数字的所有那些质数都叫做这个给定数字的质因数。

例如，我们可以将 20 表示为素数 2 的乘积，将 5 表示为 2x2x5。因此，2 和 5 是 20 的质因数。同样，我们可以用质数的组合来表示任何数。

## 求一个数的质因数的算法

要找出一个数的质因数，我们只需用质数来除给定的数。但是，我们如何知道给定数字的一个因子是否是质数呢？为了消除这个问题，我们将使用重复除法。让我们以数字 1260 为例，这样我们就可以找到它的质因数。

首先，我们将给定的数重复除以 2，并存储结果数，直到余数变为 1。这样，我们将确保任何偶数都不会除以合成数。现在，我们可以跳过所有偶数，同时找到给定数字的因子。例如，我们将 1260 除以 2 两次，得到的结果是 315。在这里，你可以看到结果 315 是一个奇数。因此，没有偶数可以是它的因子。

除以 2 后，我们将重复地将结果除以 3，并存储结果，直到该数不能被 3 除。这样，我们将确保 3 的任何倍数都不会将结果除尽。在这里，我们将 315 除以 3 两次，得到 35 作为结果。可以看到没有 3 的倍数能除以 35。

同样，我们将使用 5、7、9、11 和其他奇数执行重复除法，直到结果变为 1。用这种方法，我们可以找到任何给定数的质因数。这里，1260 的质因数是 2、3、5 和 7。

## Python 程序寻找一个数的质因数

现在，我们将实现上面讨论的算法来寻找任何给定数字的质因数。这里，我们将创建一个集合来存储质因数，并如下实现上述算法。

```py
def calculate_prime_factors(N):
    prime_factors = set()
    if N % 2 == 0:
        prime_factors.add(2)
    while N % 2 == 0:
        N = N // 2
        if N == 1:
            return prime_factors
    for factor in range(3, N + 1, 2):
        if N % factor == 0:
            prime_factors.add(factor)
            while N % factor == 0:
                N = N // factor
                if N == 1:
                    return prime_factors

input_number = 20
output = calculate_prime_factors(input_number)
print("Prime factors of {} are {}".format(input_number, output))
input_number = 1260
output = calculate_prime_factors(input_number)
print("Prime factors of {} are {}".format(input_number, output)) 
```

输出:

```py
Prime factors of 20 are {2, 5}
Prime factors of 1260 are {2, 3, 5, 7} 
```

## 结论

在本文中，我们讨论并实现了一个用 python 来寻找给定数字的质因数的算法。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。您可能还会喜欢这篇关于 python 中的[复数的文章。](https://www.pythonforbeginners.com/data-types/complex-numbers-in-python)
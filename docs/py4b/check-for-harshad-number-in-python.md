# 检查 Python 中的 Harshad 数

> 原文：<https://www.pythonforbeginners.com/basics/check-for-harshad-number-in-python>

数字有很多特殊性。根据专业，他们被赋予独特的名字。一个这样的特殊号码是哈沙德号码或尼文号码。在本文中，我们将讨论一个程序来检查一个给定的数是否是一个 Harshad 数。

## 什么是 Harshad 数？

如果一个数能被它的数位之和整除，那么这个数就是哈沙德数或尼文数。换句话说，如果给我们一个能被它自己的数字之和整除的数，这个数就叫做 Harshad 数。

例如，让我们考虑 4320 这个数字。它的数字之和是 9，可以通过将数字 4、3、2 和 0 相加得到。我们可以看到 4320 能被 9 整除。因此，4320 是一个苛刻的数字。另一方面，4321 中的位数之和是 10。在这里，4321 不能被 10 整除。因此，这不是一个苛刻的数字。

## 在 Python 中检查 Harshad 数的程序

为了检查一个数是否是 Harshad 数，我们将首先计算给定数的位数之和。之后，我们将数字除以位数之和，看它是否能被完全整除。如果这个数能被数位之和整除，我们就说它是一个 Harshad 数。否则不会。

为了编写完整的程序，我们将首先编写一个函数来计算给定数字的位数之和。为此，我们将继续用 10 除这个数，直到这个数变成 0。每次我们将数字除以 10，我们得到最右边的数字作为余数。我们可以用这个余数，通过把所有的余数相加，直到数字变成 0，来求数字的和。

python 中用于计算数字位数总和的以下函数接受一个数字，并返回该数字的位数总和。

```py
def calculate_sum_of_digits(N):
    sumOfDigits = 0
    while N > 0:
        digit = N % 10
        sumOfDigits = sumOfDigits + digit
        N = N // 10
    return sumOfDigits

input_number = 4320
output = calculate_sum_of_digits(input_number)
print("Sum of digits of {} is {}.".format(input_number, output)) 
```

输出:

```py
Sum of digits of 4320 is 9. 
```

求完位数和后，我们将数除以位数和。如果除法的余数是零，我们就说这个数是哈沙德数或尼文数。否则，我们将打印出该号码不是 Harshad 号码。

```py
def calculate_sum_of_digits(N):
    sumOfDigits = 0
    while N > 0:
        digit = N % 10
        sumOfDigits = sumOfDigits + digit
        N = N // 10
    return sumOfDigits

def check_for_harshad_number(N):
    sumOfDigits = calculate_sum_of_digits(N)
    if N % sumOfDigits == 0:
        return True
    else:
        return False

input_number = 4320
output = check_for_harshad_number(input_number)
print("{} is a Harshad Number:{}".format(input_number, output))
input_number = 4321
output = check_for_harshad_number(input_number)
print("{} is a Harshad Number:{}".format(input_number, output)) 
```

输出:

```py
4320 is a Harshad Number:True
4321 is a Harshad Number:False
```

## 结论

在本文中，我们讨论了什么是哈沙德数或尼文数。我们还用 python 实现了一个程序来检查一个给定的数是否是 Harshad 数。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数的文章。
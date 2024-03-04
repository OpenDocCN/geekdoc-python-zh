# 在 Python 中检查裸体数字

> 原文：<https://www.pythonforbeginners.com/basics/check-for-nude-number-in-python>

你可能已经读过质数、合数、偶数、奇数等等。但是，你有没有想过什么是裸体数字？在本文中，我们将讨论什么是裸号。我们还将讨论一个 python 程序来检查一个数字是否是裸数字。

## 什么是裸号？

一个裸数是一个能被它的所有非零数字整除的数，也就是说，如果一个给定数中的所有数字都是这个数的因数，那么这个给定数就叫做裸数。

例如，24 可以被 2 和 4 整除。因此，24 是一个裸数字。相反，26 能被 2 整除，但不能被 6 整除。所以，这不是一个裸体数字。

现在，让我们尝试开发一种算法来检查一个裸体数字。

## 在 Python 中检查裸体数字

为了检查一个给定的数字是否是裸数字，我们首先必须提取它的所有数字。之后，我们需要用所有的数字除这个数。如果一个数不能被它的任何一位整除，我们就说给定的数不是裸数。否则，该号码将被宣布为裸号。

为了从给定的数字中提取数字，我们将首先创建一个集合来存储所有的数字。之后，我们将开始把给定的数除以 10。在每次除法运算中，数字最右边的数字将被提取为余数，并存储在集合中。我们将继续用 10 除这个数，并将余数存储在集合中，直到这个数变成 0。

```py
def create_digits(N):
    digits = set()
    while N > 0:
        digit = N % 10
        N = N // 10
        digits.add(digit)
    return digits

input_number = 12345
digits = create_digits(input_number)
print("Digits in {} are {}.".format(input_number,digits)) 
```

输出:

```py
Digits in 12345 are {1, 2, 3, 4, 5}.
```

提取所有数字后，我们将给定的数除以每个数字。如果任何数字不是给定数字的因子，我们将声明该数字不是裸数。永远记住，在除法运算中，我们只能考虑非零数字。否则，程序会出错。

检查裸体数字的整个 Python 程序如下。

```py
def create_digits(N):
    digits = set()
    while N > 0:
        digit = N % 10
        N = N // 10
        digits.add(digit)
    return digits

def is_nude(N):
    digits = create_digits(N)
    for digit in digits:
        if digit != 0 and N % digit != 0:
            return False
    return True

input_number = 12345
digits = create_digits(input_number)
output = is_nude(input_number)
print("Digits in {} are {}.".format(input_number, digits))
print("{} is a nude number:{}".format(input_number, output))
input_number = 24
output = is_nude(input_number)
digits = create_digits(input_number)
print("Digits in {} are {}.".format(input_number, digits))
print("{} is a nude number:{}".format(input_number, output)) 
```

输出:

```py
Digits in 12345 are {1, 2, 3, 4, 5}.
12345 is a nude number:False
Digits in 24 are {2, 4}.
24 is a nude number:True
```

## 结论

在这篇文章中，我们讨论了什么是裸号。我们还讨论了检查给定数字是否为裸数字的算法，并用 python 实现了该算法。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。您可能还会喜欢这篇关于 python 中的[复数的文章。](https://www.pythonforbeginners.com/data-types/complex-numbers-in-python)
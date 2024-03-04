# 用 Python 求一个整数的位数之和

> 原文：<https://www.pythonforbeginners.com/basics/find-the-sum-of-digits-of-an-integer-in-python>

当你写程序的时候，整数几乎无处不在。在本文中，我们将讨论在 python 中寻找给定整数的数字总和的方法。

## 如何求一个整数的位数之和？

求一个数 N 的位数之和，我们可以提取一个数的位数，然后一个一个的相加。为此，我们将从右到左提取数字。为了提取数字，我们将数字除以 10，并在每次除法运算中存储余数。我们将继续用 10 除这个数，直到它变成 0。这样我们就可以一个一个的提取数字了。

例如，如果给我们一个数字 12345，我们将如下计算数字的和。

*   首先，我们将把变量 sumOfDigits 初始化为零。
*   之后，我们将 12345 除以 10。
*   除法之后，我们将在 sumOfDigits 中添加 reminder 5，并将数字更新为 1234。
*   然后，我们将 1234 除以 10。
*   除法之后，我们将在 sumOfDigits 中添加 reminder 4，并将数字更新为 123。
*   同样，我们将 123 除以 10。
*   除法之后，我们将在 sumOfDigits 中添加 reminder 3，并将数字更新为 12。
*   同样，我们将 12 除以 10。
*   除法之后，我们将在 sumOfDigits 中添加 reminder 2，并将数字更新为 1。
*   现在，我们将 1 除以 10。
*   除法之后，我们会将 reminder 1 加到 sumOfDigits 上，并将数字更新为 0。
*   当数字变为 0 时，我们将在变量 sum of digits 中得到数字 12345 的数字总和，即 15。

## 用 Python 实现

正如我们在上面看到的，要在 python 中找到一个整数的位数之和，我们只需要将这个数除以 10，直到它变成 0。与此同时，我们必须将每个除法中的余数相加，以获得位数之和。我们可以如下实现程序来执行这个操作。

```py
def calculate_sum_of_digits(N):
    sumOfDigits = 0
    while N > 0:
        digit = N % 10
        sumOfDigits = sumOfDigits + digit
        N = N // 10
    return sumOfDigits

input_number = 12345
output = calculate_sum_of_digits(input_number)
print("Sum of digits of {} is {}.".format(input_number, output))
input_number = 126
output = calculate_sum_of_digits(input_number)
print("Sum of digits of {} is {}.".format(input_number, output)) 
```

输出:

```py
Sum of digits of 12345 is 15.
Sum of digits of 126 is 9.
```

## 结论

在本文中，我们讨论并实现了一个用 Python 计算整数位数总和的程序。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数的文章。
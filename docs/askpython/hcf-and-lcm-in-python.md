# Python 中的 HCF 和 LCM–使用 Python 计算 HCF 和 LCM

> 原文：<https://www.askpython.com/python/examples/hcf-and-lcm-in-python>

嘿程序员朋友！今天在本教程中，我们将学习如何使用 python 编程语言计算最高公因数(HCF)和最低公倍数(LCM)。

让我们先了解一下我们所说的两个数的 HCF 和 LCM 是什么意思，如果你现在还不熟悉这些术语的话。

***也读作:[Python 中的计算精度—分类误差度量](https://www.askpython.com/python/examples/calculating-precision)***

* * *

## 什么是最高公因数？

两个数的最大公因数定义为两个数的最大公因数。例如，让我们考虑两个数字 12 和 18。

提到的这两个数字的公因数是 2、3 和 6。三者中最高的是 6。在这种情况下，HCF 是 6。

* * *

## 什么是最低公共乘数？

两个数的最小/最低公倍数叫做两个数的最低公倍数。例如，让我们再次考虑 12 和 18 这两个数字。

这两个数字的乘数可以是 36、72、108 等等。但是我们需要最低的公共乘数，所以 12 和 18 的 LCM 是 36。

* * *

## 用 Python 计算 HCF 和 LCM

让我们开始用 Python 代码实现 HCF 和 LCM。

### 1.求两个数的 HCF

```py
a = int(input("Enter the first number: "))
b = int(input("Enter the second number: "))

HCF = 1

for i in range(2,a+1):
    if(a%i==0 and b%i==0):
        HCF = i

print("First Number is: ",a)
print("Second Number is: ",b)
print("HCF of the numbers is: ",HCF)

```

让我们传递两个数字作为输入，看看我们的结果是什么。

```py
First Number is:  12
Second Number is:  18
HCF of the numbers is:  6

```

### 2.求两个数的 LCM

在我们计算了这两个数字的 HCF 之后，找到 LCM 并不是一件困难的事情。LCM 简单地等于数的乘积除以数的 HCF。

```py
a = int(input("Enter the first number: "))
b = int(input("Enter the second number: "))

HCF = 1

for i in range(2,a+1):
    if(a%i==0 and b%i==0):
        HCF = i

print("First Number is: ",a)
print("Second Number is: ",b)

LCM = int((a*b)/(HCF))
print("LCM of the two numbers is: ",LCM)

```

让我们传递这两个数字，看看结果是什么。

```py
First Number is:  12
Second Number is:  18
LCM of the two numbers is:  36

```

* * *

## 结论

我希望你现在清楚了两个数的 HCF 和 LCM 的计算。我想你也已经了解了 python 编程语言中的相同实现。

感谢您的阅读！快乐学习！😇

* * *
# Python 中的算术级数——完全指南

> 原文：<https://www.askpython.com/python/examples/arithmetic-progression-in-python>

嘿伙计们！在本教程中，我们将了解什么是算术级数，以及如何在 Python 编程语言中实现它。

* * *

## 等差数列入门

算术级数是一个项级数，其中下一项是通过将前一项加上一个公共差而生成的。

A.P .系列是一个数列，其中任意两个连续数字之间的差总是相同的。这种区别被称为普通差异。

等差数列的数学计算如下:

***应付帐款系列之和:Sn = n/2(2a+(n–1)d)
应付帐款系列的 Tn 项:Tn = a+(n–1)d***

* * *

## Python 中算术级数的代码实现

让我们使用 Python 来研究算术级数的实现。我们将举两个相同的例子来帮助你更好地理解这个概念。

### 1.打印算术级数的前 n 项

要实现 n AP 条款，需要几个步骤。步骤如下:

**第一步**–输入 a(第一项)、d(第一步)和 n(项数)
**第二步**–从 1 到 n+1 循环，在每次迭代中计算第 n 项，并继续打印这些项。

```py
# 1\. Take input of 'a','d' and 'n'
a = int(input("Enter the value of a: "))
d = int(input("Enter the value of d: "))
n = int(input("Enter the value of n: "))

# 2\. Loop for n terms
for i in range(1,n+1):
    t_n = a + (i-1)*d
    print(t_n)

```

* * *

### 2.获得算术级数中前 n 项的和

计算前 n 个 AP 项的和需要很多步骤。步骤如下:

**步骤 1**–输入 a(第一项)、d(步骤)和 n(项数)
**步骤 2**–使用上面提到的公式计算前“n”项的总和。

```py
# 1\. Take input of 'a','d' and 'n'
a = int(input("Enter the value of a: "))
d = int(input("Enter the value of d: "))
n = int(input("Enter the value of n: "))

S_n = (n/2)*(2*a + (n-1)*d)
print("Sum of first n terms: ", S_n)

```

```py
Enter the value of a: 1
Enter the value of d: 2
Enter the value of n: 5
Sum of first n terms:  25.0

```

* * *

## 结论

恭喜你！您刚刚学习了如何用 Python 实现算术级数。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 中的记忆化——简介](https://www.askpython.com/python/examples/memoization-in-python)
2.  [Python 中的字谜简介](https://www.askpython.com/python/examples/anagrams-in-python)
3.  [Python Wonderwords 模块——简介](https://www.askpython.com/python-modules/wonderwords-module)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *
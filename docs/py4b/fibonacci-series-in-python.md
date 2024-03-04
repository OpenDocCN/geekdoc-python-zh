# Python 中的斐波那契数列

> 原文：<https://www.pythonforbeginners.com/basics/fibonacci-series-in-python>

你可能听说过不同类型的数列。在本文中，我们将讨论这样一个数列，名为斐波那契数列。斐波那契数列在金融市场中有很多重要意义，因为它给出了一个黄金比例，有助于确定任何金融市场的走向。我们还将实现一个用 python 寻找斐波那契数列的程序。

## 什么是斐波那契数列？

斐波那契数列是以意大利数学家斐波那契的名字命名的。在斐波纳契数列中，位置 N 处的任何数字都被定义为位置(N-1)和(N-2)处的数字之和。

斐波纳契数列的第一项和第二项被定义为 0 和 1。

数学上，斐波纳契数列 F 可以定义如下。

```py
F₁=0
F₂=1
F[N]=F[N-1]+F[N-2]
```

使用上面的公式，我们可以找到斐波纳契数列中任意位置的数字。举个例子，

```py
F3=F2+F1 
  =1+0 
  =1 
F4=F3+F2 
  =1+1 
  =2
```

我们可以使用上面的公式找到斐波纳契数列中任意位置的数字。

## Python 中如何确定斐波那契数列？

要确定 python 中的斐波那契数列，我们可以简单地使用上面使用的方法。我们可以从第一项和第二项开始，使用 for 循环或 python 中的 [while 循环来寻找斐波那契数列中的其他项。](https://www.pythonforbeginners.com/loops/python-while-loop)

例如，为了找到斐波那契数列中第 N 个位置的数字，我们将执行 while 循环 N-2 次来计算从第 3 个位置到第 N 个位置的项。为了存储这些术语，我们将使用一个 [python 列表](https://www.pythonforbeginners.com/basics/python-lists-cheat-sheet)。

```py
fibonacciList = [0, 1]
# finding 10 terms of the series starting from 3rd term
N = 10
term = 3
while term < N + 1:
    value = fibonacciList[term - 2] + fibonacciList[term - 3]
    fibonacciList.append(value)
    term = term + 1
print("10 terms of the fibonacci series are:")
print(fibonacciList)
```

输出:

```py
10 terms of the fibonacci series are:
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

除了使用 while 循环，我们还可以使用 for 循环来确定 Python 中的斐波那契数列，如下所示。

```py
fibonacciList = [0, 1]
# finding 10 terms of the series starting from 3rd term
N = 10
for term in range(3, N + 1):
    value = fibonacciList[term - 2] + fibonacciList[term - 3]
    fibonacciList.append(value)
print("10 terms of the fibonacci series are:")
print(fibonacciList)
```

输出:

```py
10 terms of the fibonacci series are:
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

## 在 Python 中使用递归确定斐波那契数列

你可能知道，如果我们能把问题分解成更小的子问题，我们可以用递归来解决问题。当我们使用斐波纳契数列中的前一项来定义它时，我们可以很容易地创建一个递归解决方案，使用 Python 中的递归来确定斐波纳契数列中任何位置的项。

在递归解决方案中，我们将定义一个函数 Fibonacci()，它将数字 N 作为输入，并返回 Fibonacci 序列中第 N 个位置的项。

对于 N=1，函数返回 0，而对于 N=2，函数返回 1。对于 N 的任何其他值，Fibonacci(N)返回 Fibonacci(N-1)和 Fibonacci(N-2)的和。我们可以用 python 实现这个解决方案，如下所示。

```py
def fibonacci(N):
    if N == 1:
        return 0
    if N == 2:
        return 1
    return fibonacci(N - 1) + fibonacci(N - 2)

print("10th term of the fibonacci series is:")
print(fibonacci(10))
```

输出:

```py
10th term of the fibonacci series is:
34
```

## 结论

在本文中，我们讨论了斐波那契数列。我们还讨论了在 python 中确定斐波那契数列的方法。要阅读更多关于 python 中数字的内容，你可以阅读这篇关于 python 中十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数的文章。
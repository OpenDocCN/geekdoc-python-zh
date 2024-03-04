# Python 中的递归

> 原文：<https://www.pythonforbeginners.com/basics/recursion-in-python>

你可能学过 python 中的[函数。在 Python 中编程时，您可能还使用了 for 循环和 while 循环来重复执行任务。在本文中，我们将讨论 Python 中的递归和递归函数。](https://www.pythonforbeginners.com/basics/python-functions-cheat-sheet)

## 什么是递归？

递归是一个数学概念，在这个概念中，我们根据事物本身来定义它。

例如，我们可以将前十个自然数的和定义为前九个自然数加上第十个自然数的和。

同样，我们可以把前九个自然数之和定义为前八个自然数加上第九个自然数之和。

在这里，你可以看到我们把前十个自然数的问题分解成更小的问题，比如求前 9 个数的和，然后求前 8 个数的和，等等。这样，我们将到达一个位置，我们必须找到第一个自然数的和，即 1 本身。在此之后，我们将只执行原子加法，而不用担心我们正在寻找的自然数的总数。

## Python 中什么时候使用递归？

如上所述，每当我们可以将一个问题分解成一个相似但更小的问题时，我们就可以使用递归。我们使用递归时最常见的问题是:

1.  [二叉树](https://www.pythonforbeginners.com/data-structures/binary-search-tree-in-python)遍历问题
2.  求一个数的阶乘
3.  汉诺塔问题
4.  寻找斐波那契数列

## Python 中如何使用递归？

在编程中，如果一个函数调用它自己，我们说它是一个递归函数，也就是说它是基于递归的概念工作的。您可以在 python 中使用递归来实现任何问题的解决方案，这些问题可以简化为一个类似但更小的问题。

例如，让我们试着找出前 10 个自然数的和。为此，让我们定义一个 sumOfNumbers()函数，它接收一个输入数字 N，并返回从 1 到 N 的数字之和。

*   为了计算前 10 个自然数的和，即 sumOfNumbers(10)，我们将找到前 9 个自然数的和，即 sumOfNumbers(9)，并将它加上 10。
*   类似地，为了找到前 9 个自然数的和，即 sumOfNumbers(9)，我们将找到前 8 个自然数的和，即 sumOfNumbers(8)，并将在其上加上 9。
*   同样，为了求出前 8 个自然数的和，即 sumOfNumbers(8)，我们将求出前 7 个自然数的和，即 sumOfNumbers(7)，并将它加上 8。
*   之后，要找到前 7 个自然数的和，即 sumOfNumbers(7)，我们将找到前 6 个自然数的和，即 sumOfNumbers(6)，并将它加上 7。
*   这样，我们将达到必须计算第一个自然数的和，即 sumOfNumbers(1)的位置。在这里，我们可以简单地返回 1。这也称为递归的基本情况，因为问题不能进一步简化成更小的问题。

我们可以如下实现上述算法。

```py
def sumOfNumbers(N):
    if N == 1:
        return N
    else:
        return N + sumOfNumbers(N - 1)

input_number = 10
output = sumOfNumbers(input_number)
print("Sum of first {} natural numbers is {}".format(input_number, output))
input_number = 20
output = sumOfNumbers(input_number)
print("Sum of first {} natural numbers is {}".format(input_number, output)) 
```

输出:

```py
Sum of first 10 natural numbers is 55
Sum of first 20 natural numbers is 210
```

使用递归时，我们必须指定基本情况。否则，程序将继续不断地执行，并运行到 RecursionError。这是因为在 Python 中，一个函数可以进行的最大递归调用次数被限制在 1000 次。如果任何函数进行的递归调用超过 1000 次，就会出现 RecursionError 异常。

## 结论

在本文中，我们讨论了 python 中的递归。我们还实现了一个程序，用 Python 计算前 10 个自然数的总和。要了解关于函数的更多信息，您可以阅读这篇关于 python 中的[闭包的文章。](https://www.pythonforbeginners.com/basics/closures-in-python)
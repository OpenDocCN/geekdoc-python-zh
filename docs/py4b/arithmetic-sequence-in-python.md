# Python 中的算术序列

> 原文：<https://www.pythonforbeginners.com/basics/arithmetic-sequence-in-python>

你可能在数学课上听说过算术数列和几何数列。在这篇文章中，我们将讨论算术序列。我们还将在 Python 中实现对算术序列执行不同操作的程序。

## 什么是等差数列？

一个算术序列是一个数列，其中任何两个连续的数字有一个固定的差异。这种差异也称为等差数列中各项之间的共同差异。

例如，3，5，7，9，11，13，…是一个连续项之间的普通差为 2 的算术序列。

## 算术数列中的第 n 项

如果给我们第一项 A ₁ 和公差带 D，我们可以把第二项写成`A₁+D`，第三项写成 `A₁+2D`，第四项写成`A₁+3D`，以此类推。第 N 项将被写成`A₁+(N-1)D`为了在 python 中找到一个等差序列的第 N 项，我们可以简单地用一个 for 循环将第一项 A ₁ 加上公差(N-1)倍，如下所示。

```py
commonDifference = 2
print("Common Difference in the arithmetic sequence is:", commonDifference)
firstTerm = 3
print("First term in the arithmetic sequence is:", firstTerm)
# calculating 100th term
N = 100
nthTerm = firstTerm
for i in range(1, N):
    nthTerm = nthTerm + commonDifference
print("100th term in the arithmetic sequence is:", nthTerm) 
```

输出:

```py
Common Difference in the arithmetic sequence is: 2
First term in the arithmetic sequence is: 3
100th term in the arithmetic sequence is: 201
```

或者，我们可以使用以下公式直接计算第 n 项。

```py
commonDifference = 2
print("Common Difference in the arithmetic sequence is:", commonDifference)
firstTerm = 3
print("First term in the arithmetic sequence is:", firstTerm)
# calculating 100th term
N = 100
nthTerm = firstTerm + (N - 1) * commonDifference
print("100th term in the arithmetic sequence is:", nthTerm)
```

输出:

```py
Common Difference in the arithmetic sequence is: 2
First term in the arithmetic sequence is: 3
100th term in the arithmetic sequence is: 201
```

## Python 中算术序列中 N 项的和

要计算算术表达式中 N 项的总和，我们可以使用 for 循环简单地将每一项相加。在 for 循环中，我们将首先使用上面讨论的公式找到每一项。之后，我们将每个项相加，计算 N 项之和，如下所示。

```py
commonDifference = 2
print("Common Difference in the arithmetic sequence is:", commonDifference)
firstTerm = 3
print("First term in the arithmetic sequence is:", firstTerm)
# calculating sum of 50 terms
N = 50
sumOfTerms = 0
for i in range(1, N + 1):
    ithTerm = firstTerm + (i - 1) * commonDifference
    sumOfTerms = sumOfTerms + ithTerm
print("Sum of 50 terms in the arithmetic sequence is:", sumOfTerms)
```

输出:

```py
Common Difference in the arithmetic sequence is: 2
First term in the arithmetic sequence is: 3
Sum of 50 terms in the arithmetic sequence is: 2600
```

或者，我们也可以导出一个数学表达式来计算等差数列的 N 项之和。

我们知道 N 个数的和将等于 N *(所有项的平均值)。在这里，我们可以很容易地找到所有项的平均值。

对于第一项为 A₁ ，第 N 项为 A[N] 的等差数列，所有项的平均值定义为`(A₁+A[N])/2`。由于程序中会给出一个 ₁ 和公差带 D，我们可以求出`A[N]= A₁+ (N-1)*D`。

因此，算术序列中所有数字的平均值将变成 `(2A₁+ (N-1)*D)/2`。

随后，等差数列的 N 项之和将变为`N*((2A₁+ (N-1)*D)/2)`。

我们可以使用 python 中的这个公式来计算算术方程中 N 项的和，如下所示。

```py
commonDifference = 2
print("Common Difference in the arithmetic sequence is:", commonDifference)
firstTerm = 3
print("First term in the arithmetic sequence is:", firstTerm)
# calculating sum of 50 terms
N = 50
sumOfTerms = (N * (2 * firstTerm + (N - 1) * commonDifference)) // 2
print("Sum of 50 terms in the arithmetic sequence is:", sumOfTerms)
```

输出:

```py
Common Difference in the arithmetic sequence is: 2
First term in the arithmetic sequence is: 3
Sum of 50 terms in the arithmetic sequence is: 2600
```

## 结论

在这篇文章中，我们讨论了算术序列的基础和公式。我们还执行了不同的操作，比如在 python 中查找算术序列的第 N 项和 N 项之和。要了解更多关于 python 中的数字，你可以阅读这篇关于 python 中的十进制数字的文章。你可能也会喜欢这篇关于 python 中的复数的文章。
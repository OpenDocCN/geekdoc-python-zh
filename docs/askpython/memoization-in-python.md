# Python 中的记忆化——简介

> 原文：<https://www.askpython.com/python/examples/memoization-in-python>

在本教程中，我们将讨论一种非常流行的优化技术 Python 中的记忆化——主要用于加速计算机程序。所以，让我们开始吧！

* * *

## Python 中的记忆化是什么？

在计算机编程的世界里，**Python 中的 Memoization 或 memo ization**是一种特殊的优化技术，主要用于加速我们的计算机程序。通过将昂贵的(就运行时间而言)函数调用的结果存储到存储器中，并在需要任何存储或缓存的值时使用它，它有效地减少了计算机程序的运行时间。

它确保特定的函数或方法不需要对同一组输入运行多次，因为结果已经作为缓存/存储的数据可用。

类似于**缓存**。它包括根据函数的输入参数缓存函数的返回值。

在 Python 中，我们可以使用 **[函数和基于类的装饰器](https://www.askpython.com/python/examples/decorators-in-python)** 在程序中实现记忆化技术。在我们的整个讨论中，我们将使用一个[递归 Python 程序](https://www.askpython.com/python/python-recursion-function)来计算第 n 个^(斐波纳契数，因为对于更大的输入，这个程序将变得非常非常慢，因为对相同输入值的函数调用的数量随着输入大小的增加而增加。)

## Python 中使用基于函数的装饰器的记忆化

对于那些想了解这种优化技术实际上是如何工作的人来说，这是在 Python 中实现记忆化技术的最好也是最复杂的方法。在这种实现记忆化技术的方法中，我们在 Python 中定义了自己的装饰函数来缓存/存储函数调用的返回值。让我们看看如何编写 Python 代码来实现这一点。

```py
# Memoization using function-based decorators

def memoize(f):
    cache = {}
    def foo(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]
    return foo

@memoize
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Driver code
fibonacci(20)

```

**输出:**

```py
6765

```

## 使用基于类的装饰器的记忆化

对于想了解这种优化技术实际上是如何工作的初学者来说，这是在 Python 中实现记忆化技术的第二好也是最复杂的方法。在这种实现记忆化技术的方法中，我们在 Python 中定义了自己的 decorator 类来缓存/存储函数调用的返回值。让我们编写 Python 代码来实现这一点。

```py
# Memoization using class-based decorators

class classMemoize:
    def __init__(self, f):
        self.f = f
        self.cache = {}
    def __call__(self, *x):
        if x not in self.cache:
            self.cache[x] = self.f(*x)
        return self.cache[x]

@classMemoize
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Driver code
fibonacci(50)

```

**输出**

```py
12586269025

```

## 使用内置装饰函数的记忆化

这是在 Python 中实现记忆技术的最简单和最容易的方法之一。

在这种实现记忆化技术的方法中，我们不定义自己的装饰函数或类，而是利用内置函数如`lru_cache()`和`cache()`来缓存/存储函数调用的中间结果。

这些`lru_cache()`和`cache()`函数是在`funtools`库中定义的，它是一个标准的 Python 库，并且是在正常的 Python 安装中附带的。

`lru_cache(maxsize=None, typed=False)`函数通过其参数如`maxsize`和`typed`提供一些定制功能。参数`maxsize`通过将其值设置为**无**或整数值**来决定是否启用 LRU 功能。并且参数`typed`决定不同类型的数据是否被分开缓存。**

整数值将限制 Python 程序执行期间维护的缓存大小，而 None 值将禁用 LRU 功能，然后缓存可以无限制增长。

从 Python 3.9 版本开始，`cache()`函数可用，它相当于`funtools`库中的`lru_cache(maxsize=None)`函数。

```py
# Memoization using built-in function

import functools

@functools.cache
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Driver code
fibonacci(100)

```

**输出:**

```py
354224848179261915075

```

## 结论

在本教程中，我们学习了如何使用函数和基于类的装饰器来使用 Python 中的记忆化技术。我希望你已经很好地理解了上面讨论的内容，并准备在你的 Python 程序中使用/实现这种记忆技术来提高它的速度。
# 使用 Python randint()生成随机整数

> 原文：<https://www.askpython.com/python-modules/python-randint-method>

在本文中，我们将看看如何使用 Python **randint()** 方法生成随机整数。

这个方法在 Python 的 random 模块中，我们将使用它来生成**伪随机数**，所以我们需要导入它来加载这个方法。现在就来看看这个方法吧！

* * *

## Python randint()的语法

Python randint()方法返回两个界限`lower`和`upper`之间的随机整数(包括两个界限)。所以这个随机数也可能是两个极限之一。

我们可以如下调用这个函数:

```py
random_integer = random.randint(lower, upper)

```

这里，**下限**是随机数的下限，**上限**是随机数的上限。

我们必须保证`lower`和`upper`是整数，并且`lower` < = `upper`。否则，将引发一个`ValueError`异常。

现在让我们来看看如何使用这个功能。

* * *

## 使用 Python randint()方法

我们需要导入`random`模块。之后，我们可以使用语法调用函数。

```py
import random

beg = 10
end = 100

# Generates a random integer between (10, 100)
random_integer = random.randint(beg, end)

print(f"A random integer between {beg} and {end} is: {random_integer}")

```

**可能的输出**

```py
A random integer between 10 and 100 is: 59

```

事实上，我们可以看到这个数字确实位于范围(10，100)之间。

如果我们想要重复这个伪随机生成，让我们使用一个[循环](https://www.askpython.com/python/python-loops-in-python)来实现。

```py
import random

beg = 0
end = 100

rand_list = [random.randint(beg, end) for _ in range(10)]

print(rand_list) 

```

**可能的输出**

```py
[61, 16, 39, 86, 25, 11, 16, 89, 99, 70]

```

我们可以看到这些数字在 0 到 100 的范围内。并且伪随机条件指示没有两个连续的**数字重复。**

**注**:前面我提到过，`beg`和`end`都必须是整数，其中`beg` < = `end`。否则，我们将得到一个`ValueError`异常。

* * *

## 结论

在本文中，我们学习了如何使用 Python 中的`randint()`方法来生成随机整数。

* * *

## 参考

*   [Python 官方文档](https://docs.python.org/3/library/random.html)关于随机模块
*   关于 randint()方法的 JournalDev 文章

* * *
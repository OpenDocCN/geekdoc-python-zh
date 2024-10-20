# Python 中的回调函数——完整概述

> 原文：<https://www.askpython.com/python/built-in-methods/callback-functions-in-python>

回调是 Python 以及 Javascript、C 等其他语言中的一个通用概念。我们知道 Python 是面向对象的语言，函数是 Python 中的一级对象。这意味着，在 Python 中，我们可以将一个函数返回的值赋给一个变量，并从另一个函数返回一个函数。

在本教程中，我们将学习 Python 编程语言中回调的概念，也将看到一些与之相关的例子。所以让我们开始吧。

## Python 中的回调函数

当一个函数被另一个函数调用时，这就是所谓的回调。回调函数是作为参数传递给另一个函数的函数。有两种方法可以做到:

1.  将一个函数作为参数传递给另一个函数
2.  在另一个函数中调用一个函数

以上两点可以帮助你判断一个给定的函数是不是回调函数。作为回调函数，上述两个条件不一定都为真。即使满足上述两个条件中的一个，该函数也称为回调函数。

回调可以使用**内置函数**和**用户自定义函数**来实现。在本教程中，我们将看到这两种实现的例子。

## 使用内置函数回调

我们都熟悉 Python 中的`sorted`函数。让我们先快速回顾一下。

语法:

```py
sorted(iterable, key)

```

| **参数** | **意为** |
| iterable(强制) | 要排序的数据结构(列表、字典等)。) |
| key(关键字，可选) | 分类标准。它可以被分配任何你想使用的功能。 |

sorted function parameters

当调用`sorted`函数时，iterable 将首先被传递给`key`函数，它将在 iterable 上实现。那么新的/改变的 iterable 将被排序。

这里，我们是**调用** `sorted` **函数内部的** `key` **函数。**所以，这里满足第二个条件。因此，这里的`key`函数是一个**回调函数**。

让我们看一个例子。假设我们要对字符串列表进行排序。

```py
#list
a = ["lmn", "AbCd", "khJH", "ert", "SuNnY"]

#sorted based on the ASCII values
print(sorted(a))

#first convert all letters to lowercase & then sort based on ASCII values
print(sorted(a, key=str.lower))

```

**输出:**

```py
['AbCd', 'SuNnY', 'ert', 'khJH', 'lmn']
['AbCd', 'ert', 'khJH', 'lmn', 'SuNnY']

```

如果您还记得的话，A 的 ASCII 值是 65，B 是 66，e 是 101 等等。ASCII 值表请参考此[链接](https://www.cs.cmu.edu/~pattis/15-1XX/common/handouts/ascii.html)。
所以，在第一种情况下，列表只根据 [ASCII 值](https://www.askpython.com/python/built-in-methods/python-ascii-function)进行排序。

如果我们观察列表，我们可以看到列表中的字符串是大写字母和小写字母的混合。比方说，我们现在想只根据每个字母的小写形式对列表进行排序。为此，我们在函数`key=str.lower`中引入了一个新参数。

这里，`str.lower`首先在 iterable 列表上实现。也就是说，列表中的字符串被转换成小写。所以现在，这个列表实际上是`'lmn', 'abcd', 'khjh', 'ert', 'sunny'`，然后这个列表被传递给`sorted`函数。

注意，全小写列表与原始列表不同，但在原始列表中，这里的' lmn '表示' lmn '，' abcd '表示' AbCd '，' khjh '表示' khJH '，' ert '表示' ert '，而' sunny '表示' SuNnY '。新列表的排序版本是`'abcd', 'ert', 'khjh', 'lmn, 'sunny'`，因此输出如上所示。

这就是 Python 内置函数中回调的工作方式。

## 用户定义函数中的回调

现在让我们学习一下用户定义函数中的回调。让“调用”一个函数的函数名为`caller`，被调用的函数名为`called`。这里，`caller`功能可以说是主功能，而`called`功能是从功能，因为主功能总是可以调用从功能，但反之则不行。

假设给我们一个由两个整数组成的元组，我们需要计算这两个整数的乘积。

```py
def called(n):
    return n[0]*n[1]

def caller(func, n):
    return func(n)

#tuple
num = (8, 5)

ans = caller(called, num)

print("Multiplication =", ans)

```

**输出:**

```py
Multiplication = 40

```

在上面的代码中，我们定义了两个函数:`caller`和`called`。`called`函数是作为参数传递给`caller`函数的**。因此，第一个条件在这里得到满足，所以这也是一个回调函数，`called`就是这里的回调函数。**

当程序执行时，首先创建元组，然后调用`caller`函数。现在，控制转到`caller`函数的定义。它有两个参数:一个函数(`func`)和一个 iterable ( `n`)，在本例中是一个元组。在`caller`函数中，希望将作为参数传递的函数应用于 iterable，并返回结果(func(n))。

所以现在，控制权在`called`函数定义处。它将一个元组作为参数，并返回其成员的乘积。这个值然后被返回给`caller`函数，然后`caller`函数返回答案，即 8×5 = 40。

## 摘要

因此，我们已经了解了 Python 中的回调，以及如何使用内置函数和用户自定义函数来实现回调。
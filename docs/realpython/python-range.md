# Python range()函数(指南)

> 原文：<https://realpython.com/python-range/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解:[**Python range()函数**](/courses/python-range-function/)

当你需要执行一个动作特定的次数时，Python 内置的 **`range`** 函数非常方便。作为一个有经验的 python 爱好者，你很可能以前用过它。但是它有什么用呢？

**学完本指南后，您将:**

*   理解 Python `range`函数的工作原理
*   了解 Python 2 和 Python 3 中的实现有何不同
*   我看到了许多实际操作的例子
*   准备好解决它的一些限制

让我们开始吧！

**免费奖励:** ，它向您展示 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## Python 的`range()`函数的历史

虽然 Python 2 中的`range()`和 Python 3 中的`range()`可能共用一个名字，但它们是完全不同的动物。事实上，Python 3 中的`range()`只是 Python 2 中一个名为`xrange`的函数的重命名版本。

最初，`range()`和`xrange()`都产生可以用 for 循环迭代的[数字](https://realpython.com/python-numbers/)，但是前者一次产生这些数字的列表，而后者产生延迟的数字，这意味着数字在需要时一次返回一个。

庞大的列表会占用内存，所以用`xrange()`代替`range()`、name 等等也就不足为奇了。你可以在 [PEP 3100](https://www.python.org/dev/peps/pep-3100) 中了解更多关于这个决定和`xrange()` vs `range()`的背景。

**注:** PEP 代表 Python 增强提案。pep 是可以涵盖广泛主题的文档，包括提议的新特性、风格、治理和理念。

有很多这样的人。 [PEP 1](https://www.python.org/dev/peps/pep-0001/) 解释了它们是如何工作的，这是一个很好的起点。

在本文的其余部分，您将使用 Python 3 中的函数。

开始了。

[*Remove ads*](/account/join/)

## 让我们循环

在我们深入了解`range()`如何工作之前，我们需要了解一下循环是如何工作的。循环是一个关键的计算机科学概念。如果你想成为一名优秀的程序员，掌握循环是你需要采取的第一步。

下面是 Python 中 for 循环的一个例子:

```py
captains = ['Janeway', 'Picard', 'Sisko']

for captain in captains:
    print(captain)
```

输出如下所示:

```py
Janeway
Picard
Sisko
```

如您所见，for 循环使您能够执行特定的代码块，无论您想执行多少次。在这种情况下，我们循环遍历一个船长列表，并打印出他们每个人的名字。

尽管《星际迷航》很棒，但你可能想做的不仅仅是浏览船长名单。有时，您只想执行一段代码特定的次数。循环可以帮你做到这一点！

用能被 3 整除的数字尝试下面的代码:

```py
numbers_divisible_by_three = [3, 6, 9, 12, 15]

for num in numbers_divisible_by_three:
    quotient = num / 3
    print(f"{num} divided by 3 is {int(quotient)}.")
```

该循环的输出将如下所示:

```py
3 divided by 3 is 1.
6 divided by 3 is 2.
9 divided by 3 is 3.
12 divided by 3 is 4.
15 divided by 3 is 5.
```

这就是我们想要的输出，所以这个循环很好地完成了工作，但是还有一种方法可以通过使用`range()`得到相同的结果。

**注意:**最后一个代码示例有一些字符串格式。要了解这个主题的更多信息，您可以查看 [Python 字符串格式最佳实践](https://realpython.com/python-string-formatting/)和 [Python 3 的 f-Strings:一个改进的字符串格式语法(指南)](https://realpython.com/python-f-strings/)。

现在你对循环已经比较熟悉了，让我们看看如何使用`range()`来简化你的生活。

### Python `range()`基础知识

那么 Python 的`range`函数是如何工作的呢？简单来说，`range()`允许你在给定范围内生成一系列数字。根据传递给函数的参数数量，您可以决定一系列数字的开始和结束位置，以及一个数字和下一个数字之间的差异有多大。

下面先睹为快`range()`的行动:

```py
for i in range(3, 16, 3):
    quotient = i / 3
    print(f"{i} divided by 3 is {int(quotient)}.")
```

在这个 for 循环中，您可以简单地创建一系列可以被`3`整除的数字，因此您不必自己提供每一个数字。

**注意:**虽然这个例子展示了`range()`的正确用法，但是在 for 循环中过于频繁地使用`range()`通常是不可取的。

例如，下面对`range()`的使用通常被认为不是 Pythonic 式的:

```py
captains = ['Janeway', 'Picard', 'Sisko']

for i in range(len(captains)):
    print(captains[i])
```

`range()`非常适合创建数字的可迭代项，但是当你需要迭代可以用 [`in`操作符](https://docs.python.org/3/reference/expressions.html#in)循环的数据时，它不是最佳选择。

如果你想知道更多，查看[如何让你的 Python 循环更 Python 化](https://dbader.org/blog/pythonic-loops)。

有三种方法可以调用`range()`:

1.  `range(stop)`采用一个参数。
2.  `range(start, stop)`需要两个参数。
3.  `range(start, stop, step)`需要三个参数。

#### `range(stop)`

当您用一个参数调用`range()`时，您将得到一系列从`0`开始的数字，包括所有整数，但不包括您作为`stop`提供的数字。

实际情况是这样的:

```py
for i in range(3):
    print(i)
```

循环的输出将如下所示:

```py
0
1
2
```

这表明:我们有从`0`到不包括`3`的所有整数，你提供的数字是`stop`。

#### `range(start, stop)`

当你用两个参数调用`range()`时，你不仅要决定数列在哪里结束，还要决定它从哪里开始，所以你不必总是从`0`开始。您可以使用`range()`生成一系列数字，从 *A* 到 *B* 使用一个`range(A, B)`。让我们看看如何生成从`1`开始的范围。

尝试用两个参数调用`range()`:

```py
for i in range(1, 8):
    print(i)
```

您的输出将如下所示:

```py
1
2
3
4
5
6
7
```

到目前为止，一切顺利:您拥有从`1`(您提供的作为`start`的数字)到不包括`8`(您提供的作为`stop`的数字)的所有整数。

但是如果您再添加一个参数，那么您将能够再现您在使用名为`numbers_divisible_by_three`的列表时得到的输出。

#### `range(start, stop, step)`

当你用三个参数调用`range()`时，你不仅可以选择数字序列的开始和结束位置，还可以选择一个数字和下一个数字之间的差异有多大。如果你不提供一个`step`，那么`range()`将自动表现为`step`就是`1`。

**注意:** `step`可以是正数，也可以是负数，但不能是`0`:

>>>

```py
>>> range(1, 4, 0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: range() arg 3 must not be zero
```

如果你试图使用`0`作为你的步骤，那么你会得到一个错误。

现在你知道了如何使用`step`，你终于可以重温我们之前看到的用`3`除的循环了。

自己尝试一下:

```py
for i in range(3, 16, 3):
    quotient = i / 3
    print(f"{i} divided by 3 is {int(quotient)}.")
```

您的输出将与您在本文前面看到的 for 循环的输出完全一样，当时您使用了名为`numbers_divisible_by_three`的列表:

```py
3 divided by 3 is 1.
6 divided by 3 is 2.
9 divided by 3 is 3.
12 divided by 3 is 4.
15 divided by 3 is 5.
```

正如您在这个例子中看到的，您可以使用`step`参数增加到一个更大的数字。这叫做递增。

[*Remove ads*](/account/join/)

### 用`range()` 递增

如果你想递增，那么你需要`step`是一个正数。要了解这在实践中意味着什么，请键入以下代码:

```py
for i in range(3, 100, 25):
    print(i)
```

如果您的`step`是`25`，那么您的循环的输出将如下所示:

```py
3
28
53
78
```

你得到了一系列的数字，每个数字都比前一个数字大了`25`，即你提供的`step`。

现在，您已经看到了如何在一个范围内前进，是时候看看如何后退了。

### 用`range()`和递减

如果你的`step`是正的，那么你移动通过一系列增加的数字，并且正在增加。如果你的`step`是负的，那么你会经历一系列递减的数字，并且是递减的。这可以让你倒着看这些数字。

在下面的例子中，你的`step`是`-2`。这意味着每循环你将减少`2`:

```py
for i in range(10, -6, -2):
    print(i)
```

递减循环的输出如下所示:

```py
10
8
6
4
2
0
-2
-4
```

你得到了一系列的数字，每一个都比前一个数字小了`2`，即你提供的`step`的[绝对值](https://realpython.com/python-absolute-value)。

创建递减范围的最有效方法是使用`range(start, stop, step)`。但是 Python 确实内置了 [`reversed`函数](https://realpython.com/python-reverse-list/)。如果您将`range()`包装在`reversed()`中，那么[可以以相反的顺序打印](https://realpython.com/python-print/)整数。

试试这个:

```py
for i in reversed(range(5)):
    print(i)
```

你会得到这个:

```py
4
3
2
1
0
```

`range()`可以遍历一个递减的数字序列，而`reversed()`通常用于以相反的顺序遍历一个序列。

**注:** `reversed()`也适用于弦乐。你可以在[如何在 Python](https://dbader.org/blog/python-reverse-string) 中反转一个字符串中了解更多关于`reversed()`带字符串的功能。

[*Remove ads*](/account/join/)

### Python 的`range()`函数的高级用法示例

现在你已经知道了如何使用`range()`的基本知识，是时候深入一点了。

`range()`主要用于两个目的:

1.  执行特定次数的 for 循环体
2.  创建比使用[列表或元组](https://realpython.com/python-lists-tuples/)更有效的整数可迭代表

第一种用法可能是最常见的，你可以证明 [itertools](https://realpython.com/python-itertools/) 提供了一种比`range()`更有效的构造可迭代对象的方法。

在使用 range 时，还有几点需要记住。

`range()`是 Python 中的一种类型:

>>>

```py
>>> type(range(3))
<class 'range'>
```

您可以通过索引访问`range()`中的项目，就像使用列表一样:

>>>

```py
>>> range(3)[1]
1
>>> range(3)[2]
2
```

您甚至可以在`range()`上使用切片符号，但是 REPL 的输出乍一看可能有点奇怪:

>>>

```py
>>> range(6)[2:5]
range(2, 5)
```

虽然这个输出看起来很奇怪，但是切分一个`range()`只是返回另一个`range()`。

您可以通过索引访问一个`range()`的元素并切片一个`range()`的事实凸显了一个重要的事实:`range()`是懒惰的，不像列表，但是[不是迭代器](http://treyhunner.com/2018/02/python-range-is-not-an-iterator/)。

## 浮动和`range()`

你可能已经注意到，到目前为止，我们处理的所有数字都是整数，也称为整数。那是因为`range()`只能接受整数作为参数。

### 关于浮动的一句话

在 Python 中，如果一个数不是整数，那么它就是浮点数。整数和浮点数之间有一些区别。

一个整数(`int`数据类型):

*   是一个整数
*   不包括小数点
*   可以是正的、负的或`0`

浮点数(`float`数据类型):

*   可以是包含小数点的任何数字
*   可以是正面的，也可以是负面的

尝试用浮点数调用`range()`,看看会发生什么:

```py
for i in range(3.3):
    print(i)
```

您应该会看到以下错误消息:

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'float' object cannot be interpreted as an integer
```

如果您需要找到一种允许您使用 floats 的变通方法，那么您可以使用 NumPy。

[*Remove ads*](/account/join/)

### 将`range()`与 NumPy 一起使用

[NumPy](http://www.numpy.org/) 是第三方 Python 库。如果您打算使用 NumPy，您的第一步是检查您是否安装了它。

以下是如何在您的 REPL 中实现这一点:

>>>

```py
>>> import numpy
```

如果你得到了一个`ModuleNotFoundError`，那么你需要安装它。为此，请在命令行中输入`pip install numpy`。

安装好之后，输入以下内容:

```py
import numpy as np

np.arange(0.3, 1.6, 0.3)
```

它将返回以下内容:

```py
array([0.3, 0.6, 0.9, 1.2, 1.5])
```

如果要在单独的行上打印每个数字，可以执行以下操作:

```py
import numpy as np

for i in np.arange(0.3, 1.6, 0.3):
    print(i)
```

这是输出:

```py
0.3
0.6
0.8999999999999999
1.2
1.5
```

`0.8999999999999999`从何而来？

计算机很难将十进制浮点数保存为二进制浮点数。这导致了各种意想不到的数字表现。

**注意:**要了解更多关于为什么会有代表小数的问题，你可以查看[这篇文章](https://realpython.com/python37-new-features/#timing-precision)和 [Python 文档](https://docs.python.org/3/tutorial/floatingpoint.html)。

您可能还想看看[十进制库](https://docs.python.org/3/library/decimal.html)，它在性能和可读性方面有所下降，但允许您精确地表示十进制数。

另一个选择是使用`round()`，你可以在[如何在 Python](https://realpython.com/python-rounding/) 中舍入数字中读到更多。请记住，`round()`有自己的怪癖，可能会产生一些令人惊讶的结果！

这些浮点错误对您来说是否是一个问题取决于您正在解决的问题。误差大约在小数点后第 16 位，这在大多数情况下是不重要的。它们是如此之小，除非你正在计算卫星轨道或其他东西，否则你不需要担心它。

或者，您也可以使用 [`np.linspace()`](https://realpython.com/np-linspace-numpy/) 。它本质上做同样的事情，但是使用不同的参数。使用`np.linspace()`，您可以指定`start`和`end`(包含两端)以及数组的长度(而不是`step`)。

例如，`np.linspace(1, 4, 20)`给出 20 个等间距的数字:`1.0, ..., 4.0`。另一方面，`np.linspace(0, 0.5, 51)`给出了`0.00, 0.01, 0.02, 0.03, ..., 0.49, 0.50`。

**注:**要了解更多信息，你可以阅读 [Look Ma，No For-Loops:Array Programming With NumPy](https://realpython.com/numpy-array-programming/)和这个方便的 [NumPy 参考](https://docs.scipy.org/doc/numpy/reference/)。

[*Remove ads*](/account/join/)

## 前进并循环

您现在了解了如何使用`range()`并解决其局限性。您也知道这个重要的功能在 Python 2 和 Python 3 之间是如何发展的。

下一次当你需要执行一个动作特定的次数时，你就可以全心投入了！

快乐的蟒蛇！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解:[**Python range()函数**](/courses/python-range-function/)*******
# Python 的 range()函数解释

> 原文：<https://www.pythoncentral.io/pythons-range-function-explained/>

## Python 的 range()函数是什么？

作为一名有经验的 Python 开发人员，甚至是初学者，您可能听说过 Python `range()`函数。但是它有什么用呢？简而言之，它生成一个数字列表，通常用于通过`for`循环进行迭代。有许多使用案例。当你想执行一个动作 X 次时，你通常会想使用这个方法，在这种情况下，你可能关心也可能不关心索引。其他时候，您可能希望迭代一个列表(或另一个 iterable 对象)，同时能够使用索引。

Python 2.x 和 3.x 中的`range()`函数工作方式略有不同，但是概念是相同的。不过，我们稍后会谈到这一点。

## Python 的 range()参数

`range()`功能有如下两组参数:

`range(stop)`

*   `stop`:要生成的整数个数(整数)，从零开始。`range(3) == [0, 1, 2]`如。

`range([start], stop[, step])`

*   `start`:序列的起始编号。
*   `stop`:生成不超过本数的数。
*   `step`:序列中各数字之间的差异。

请注意:

*   所有参数必须是整数。
*   所有参数都可以是正的或负的。
*   `range()`(以及一般的 Python)是基于 0 索引的，这意味着列表索引从 0 开始，而不是从 1 开始。访问列表第一个元素的语法是`mylist[0]`。因此由`range()`产生的最后一个整数直到但不包括`stop`。例如`range(0, 5)`生成从 0 到 5 的整数，但不包括 5。

## Python 的 range()函数示例

#### 简单用法

```py

>>> # One parameter

>>> for i in range(5):

...     print(i)

...

0

1

2

3

4

>>> # Two parameters

>>> for i in range(3, 6):

...     print(i)

...

3

4

5

>>> # Three parameters

>>> for i in range(4, 10, 2):

...     print(i)

...

4

6

8

>>> # Going backwards

>>> for i in range(0, -10, -2):

...     print(i)

...

0

-2

-4

-6

-8

```

#### 迭代列表

```py

>>> my_list = ['one', 'two', 'three', 'four', 'five']

>>> my_list_len = len(my_list)

>>> for i in range(0, my_list_len):

...     print(my_list[i])

...

one

two

three

four

five

```

#### 墙上有 99 瓶啤酒...

用下面的代码:
【python】
for I in range(99，0，-1):
if i == 1:
print('墙上 1 瓶啤酒，1 瓶啤酒！')
print('所以把它拿下来，传来传去，不要再在墙上挂啤酒了！')
elif i == 2:
print('墙上再来 2 瓶啤酒，再来 2 瓶啤酒！'打印('拿一瓶下来，传一传，墙上还有一瓶啤酒！')
else:
打印(' {0}瓶啤酒在墙上，{0}瓶啤酒！'。format(i))
print('所以把它拿下来，传来传去，{0}多瓶啤酒挂在墙上！'。格式(i - 1))

我们得到以下输出:

```py

99 bottles of beer on the wall, 99 bottles of beer!

So take one down, pass it around, 98 more bottles of beer on the wall!

98 bottles of beer on the wall, 98 bottles of beer!

So take one down, pass it around, 97 more bottles of beer on the wall!

97 bottles of beer on the wall, 97 bottles of beer!

So take one down, pass it around, 96 more bottles of beer on the wall!

...

3 bottles of beer on the wall, 3 bottles of beer!

So take one down, pass it around, 2 more bottles of beer on the wall!

2 more bottles of beer on the wall, 2 more bottles of beer!

So take one down, pass it around, 1 more bottle of beer on the wall!

1 bottle of beer on the wall, 1 bottle of beer!

So take it down, pass it around, no more bottles of beer on the wall!

```

太棒了。终于可以看到 Python 的真正威力了:)。如果你有点困惑，请参考维基百科文章。

## Python 的 range()与 xrange()函数

你可能听说过一个叫做`xrange()`的函数。这是 Python 2.x 中的一个函数，但是在 Python 3.x 中它被重命名为`range()`,而最初的`range()`函数在 Python 3.x 中被弃用。那么有什么不同呢？嗯，在 Python 2.x 中`range()`产生了一个列表，`xrange()`返回了一个迭代器——一个序列对象。我们可以在下面的例子中看到这一点:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
>>> range(1)
range(0, 1)
>>> type(range(1))
 <class>
```

*   [Python 2.x](#)

```py
>>> range(1)
[0]
>>> type(range(1))
 <type>
```

所以在 Python 3.x 中，`range()`函数有了自己的`type`。基本上，如果你想在一个`for`循环中使用`range()`，那么你就可以开始了。然而，你不能把它纯粹当作一个`list`物体来使用。例如，您不能对`range`类型进行切片。

当你使用迭代器时，`for`语句的每个循环都会动态产生下一个数字。而最初的`range()`函数在`for`循环开始执行之前就立即产生了所有的数字。最初的`range()`函数的问题是它在产生大量数字时使用了大量的内存。然而，数字越少，速度越快。注意，在 Python 3.x 中，您仍然可以通过将生成器返回给`list()`函数来生成一个列表。如下所示:

*   [Python 3.x](#custom-tab-1-python-3-x)

*   [Python 3.x](#)

```py
>>> list_of_ints = list(range(3))
>>> list_of_ints
[0, 1, 2]
```

## 在 Python 的 range()函数中使用浮点数

不幸的是，`range()`函数不支持`float`类型。然而，不要太早沮丧！我们可以很容易地用函数来实现它。有几种方法可以做到这一点，但这里有一个。

```py

>>> # Note: All arguments are required.

>>> # We're not fancy enough to implement all.

>>> def frange(start, stop, step):

...     i = start

...     while i < stop:
...         yield i
...         i += step
... 
>>> for i in frange(0.5, 1.0, 0.1):

...         print(i)

...

0.5

0.6

0.7

0.8

0.9

1.0

```

太棒了。
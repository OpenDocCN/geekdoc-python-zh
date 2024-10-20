# Python 生成器和 Yield 关键字

> 原文：<https://www.pythoncentral.io/python-generators-and-yield-keyword/>

乍看之下，`yield`语句用于定义生成器，代替函数的`return`向其调用者提供结果，而不破坏局部变量。与每次调用都以新的变量集开始的函数不同，生成器会从停止的地方继续执行。

## 关于 Python 生成器

因为`yield`关键字只用于生成器，所以首先回忆一下生成器的概念是有意义的。

生成器的思想是按需(即时)逐个计算一系列结果。在最简单的情况下，一个生成器可以被用作一个`list`，其中每个元素都被延迟计算。让我们比较一下做同样事情的列表和生成器——返回 2 的幂:

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
>>> # First, we define a list
>>> the_list = [2**x for x in range(5)]
>>>
>>> # Type check: yes, it's a list
>>> type(the_list)
<class 'list'>
>>>
>>> # Iterate over items and print them
>>> for element in the_list:
... print(element)
...
1
2
4
8
16
>>>
>>> # How about the length?
>>> len(the_list)
5
>>>
>>> # Ok, now a generator.
>>> # As easy as list comprehensions, but with '()' instead of '[]':
>>> the_generator = (x+x for x in range(3))
>>>
>>> # Type check: yes, it's a generator
>>> type(the_generator)
<class 'generator'>
>>>
>>> # Iterate over items and print them
>>> for element in the_generator:
... print(element)
...
0
2
4
>>>
>>> # Everything looks the same, but the length...
>>> len(the_generator)
Traceback (most recent call last):
File "", line 1, in
TypeError: object of type 'generator' has no len()
[/python]

*   [Python 2.x](#)

[python]
>>> # First, we define a list
>>> the_list = [2**x for x in range(5)]
>>>
>>> # Type check: yes, it's a list
>>> type(the_list)
<type 'list'>
>>>
>>> # Iterate over items and print them
>>> for element in the_list:
... print(element)
...
1
2
4
8
16
>>>
>>> # How about the length?
>>> len(the_list)
5
>>>
>>> # Ok, now a generator.
>>> # As easy as list comprehensions, but with '()' instead of '[]':
>>> the_generator = (x+x for x in range(3))
>>>
>>> # Type check: yes, it's a generator
>>> type(the_generator)
<type 'generator'>
>>>
>>> # Iterate over items and print them
>>> for element in the_generator:
... print(element)
...
0
2
4
>>>
>>> # Everything looks the same, but the length...
>>> len(the_generator)
Traceback (most recent call last):
File "", line 1, in
TypeError: object of type 'generator' has no len()
[/python]

遍历列表和生成器看起来完全一样。然而，尽管生成器是可迭代的，但它不是集合，因此没有长度。集合(列表、元组、集合等)将所有值保存在内存中，我们可以在需要时访问它们。生成器动态地计算这些值，然后忘记它们，所以它对自己的结果集没有任何了解。

生成器对于内存密集型任务特别有用，在这种情况下，不需要同时访问内存密集型列表中的所有元素。在从不需要完整结果的情况下，逐个计算一系列值也很有用，可以向调用者提供中间结果，直到满足某些要求，进一步的处理停止。

# 使用 Python“yield”关键字

一个很好的例子是搜索任务，通常不需要等待找到所有结果。执行文件系统搜索时，用户会更乐意即时收到结果，而不是等待搜索引擎遍历每个文件，然后才返回结果。有没有人真的会在所有谷歌搜索结果中导航到最后一页？

由于不能使用列表理解来创建搜索功能，我们将使用带有`yield`语句/关键字的函数来定义一个生成器。`yield`指令应该放在一个地方，在那里生成器向调用者返回一个中间结果，并休眠直到下一次调用发生。让我们定义一个生成器，在一个巨大的文本文件中逐行搜索一些关键字。

```py

def search(keyword, filename):

print('generator started')

f = open(filename, 'r')

# Looping through the file line by line

for line in f:

if keyword in line:

# If keyword found, return it

yield line

f.close()

```

现在，假设我的“directory.txt”文件包含一个巨大的姓名和电话号码列表，让我们查找名字中带有“Python”的人:

```py

>>> the_generator = search('Python', 'directory.txt')

>>> # Nothing happened

```

当我们调用搜索函数时，它的主体代码不会运行。生成器函数将只返回生成器对象，充当构造函数:

*   [Python 3.x](#custom-tab-1-python-3-x)
*   [Python 2.x](#custom-tab-1-python-2-x)

*   [Python 3.x](#)

[python]
>>> type(search)
<class 'function'>
>>> type(the_generator)
<class 'generator'>
[/python]

*   [Python 2.x](#)

[python]
>>> type(search)
<type 'function'>
>>> type(the_generator)
<type 'generator'>
[/python]

这有点棘手，因为通常情况下，`def search(keyword, filename):`下面的所有东西都要在调用它之后执行，但在生成器的情况下不是这样。事实上，甚至有一个很长的讨论，建议使用“gen”，或其他关键字来定义一个生成器。但是，圭多决定坚持用“def”，就这样。你可以在 [PEP-255](https://www.python.org/dev/peps/pep-0255/ "PEP-255") 上看到动机。

为了让新创建的生成器计算一些东西，我们需要通过迭代器协议访问它，即调用它的`next`方法:

*   [Python 3.x](#custom-tab-2-python-3-x)
*   [Python 2.x](#custom-tab-2-python-2-x)
*   [Python 3.x](#custom-tab-2-python-3-x)
*   [Python 2.x](#custom-tab-2-python-2-x)
*   [Python 3.x](#custom-tab-2-python-3-x)
*   [Python 2.x](#custom-tab-2-python-2-x)
*   [Python 3.x](#custom-tab-2-python-3-x)
*   [Python 2.x](#custom-tab-2-python-2-x)

*   [Python 3.x](#)

[python]
>>> print(next(the_generator))
generator started
Anton Pythonio 111-222-333
[/python]

*   [Python 2.x](#)

[python]
>>> print(the_generator.next())
generator started
Anton Pythonio 111-222-333
[/python]

调试字符串被打印出来，我们没有查看整个文件就获得了第一个搜索结果。现在让我们请求下一场比赛:

*   [Python 3.x](#)

[python]
>>> print(next(the_generator))
generator started
Fritz Pythonmann 128-256-512
[/python]

*   [Python 2.x](#)

[python]
>>> print(the_generator.next())
generator started
Fritz Pythonmann 128-256-512
[/python]

生成器在最后一个`yield`关键字/语句上继续，并遍历循环，直到再次遇到`yield`关键字/语句。然而，弗里茨仍然不是合适的人选。接下来，请:

*   [Python 3.x](#)

[python]
>>> print(next(the_generator))
generator started
Guido Pythonista 123-456-789
[/python]

*   [Python 2.x](#)

[python]
>>> print(the_generator.next())
generator started
Guido Pythonista 123-456-789
[/python]

最后，我们找到了他。现在你可以打电话给他，用 Python 对伟大的`generators`说声“谢谢”!

## 更多生成器细节和示例

正如您可能注意到的，该函数第一次运行时，将从头开始，直到到达`yield`关键字/语句，将第一个结果返回给调用者。然后，每个其他调用将从离开的地方恢复生成器代码。如果生成器函数不再命中`yield`关键字/语句，它将引发一个`StopIteration`异常(就像所有 iterable 对象在耗尽/完成时一样)。

为了在后续调用中运行`yield`，生成器可以包含一个循环或多个`yield`语句:

```py

def hold_client(name):

yield 'Hello, %s! You will be connected soon' % name

yield 'Dear %s, could you please wait a bit.' % name

yield 'Sorry %s, we will play a nice music for you!' % name

yield '%s, your call is extremely important to us!' % name

```

使用生成器作为传送带通常更有意义，链接功能可以有效地处理一些序列。一个很好的例子是缓冲:以大块获取数据，以小块进行处理:

```py

def buffered_read():

while True:

buffer = fetch_big_chunk()

for small_chunk in buffer:

yield small_chunk

```

这种方法允许处理功能从任何缓冲问题中抽象出来。它可以使用负责缓冲的生成器一个接一个地获取值。

使用生成器的思想，即使是简单的任务也会更加高效。在 Python 2 中。x 是 Python 中一个常见的`range()`函数，通常被`xrange()`代替，后者是`yields`的值，而不是一次创建整个列表:

*   [Python 3.x](#)

[python]
>>> # "range" returns a list
>>> type(range(0, 3))
<class 'list'>
>>> # xrange does not exist in Python 3.x
[/python]

*   [Python 2.x](#)

[python]
>>> # "range" returns a list
>>> type(range(0, 3))
<type 'list'>
>>> # xrange returns a generator-like object "xrange"
>>> type(xrange(0, 3))
<type 'xrange'>
>>>
>>> # It can be used in loops just like range
>>> for i in xrange(0, 3):
... print(i)
...
0
1
2
[/python]

最后，一个生成器的“经典”例子:计算 Fibonacci 数的前 N 个给定数:

```py

def fibonacci(n):

curr = 1

prev = 0

counter = 0

while counter < n:

yield curr

prev, curr = curr, prev + curr

counter += 1

```

数字一直计算到计数器达到'`n`'为止。这个例子非常流行，因为斐波那契数列是无限的，很难放入内存。

到目前为止，已经描述了 Python 生成器最实用的方面。要了解更多详细信息和有趣的讨论，请看一下 [Python 增强提案 255](https://www.python.org/dev/peps/pep-0255/ "Python Enhancement Proposal 255") ，其中详细讨论了该语言的特性。

快乐的蟒蛇！
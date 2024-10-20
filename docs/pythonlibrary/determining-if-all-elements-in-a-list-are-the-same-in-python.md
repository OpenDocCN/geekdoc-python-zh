# 在 Python 中确定列表中的所有元素是否都相同

> 原文：<https://www.blog.pythonlibrary.org/2018/05/09/determining-if-all-elements-in-a-list-are-the-same-in-python/>

编者按:本周我们有一篇来自亚历克斯的客座博文，他是[check io](https://checkio.org/)T3 的首席执行官

在生活中，我们总是有选择，不管我们是否知道。编码也是一样。我们可以用许多不同的方法来完成一项特定的任务。我们可能没有考虑过这些方式，或者对它们一无所知，但是它们确实存在。成为一名程序员不仅仅是了解语言和编写代码的过程。通常，成为一名程序员意味着成为你自己最有创造力的版本，考虑你以前从未考虑过的事情。因此，我想介绍一下我自己。嗨！我叫 Alex，是 CheckiO 的首席执行官，我已经处理这个项目的创意方面有一段时间了。

我们的用户有不同的编码知识水平和经验，这就是为什么我经常看到标准的和更明显的任务解决方法。但我不时会遇到这样独特而不寻常的解决方案，让我再次学习这门语言的新的微妙之处。

在这篇文章中，我想回顾一下一个非常简单的任务的一些解决方案，在我看来，这些方案是最有趣的。这个任务要求你写一个函数来决定是否所有的数组元素都有相同的值。

**1。**首先想到的解决方案之一是比较输入元素列表的长度和第一个元素进入列表的次数。如果这些值相等，则列表包含相同的元素。还需要检查列表是否为空，因为在这种情况下也需要返回 True。

```py

def all_the_same(elements):
   if len(elements) < 1:
       return True
   return len(elements) == elements.count(elements[0])

```

或者更简短的版本:

```py

def all_the_same(elements):
   return len(elements) < 1 or len(elements) ==
elements.count(elements[0])

```

**2。**在这个解决方案中，使用了一个有用的 Python 特性——只使用比较操作符来比较列表的能力——**= =**(不像其他一些编程语言那样简单)。让我们看看这是如何工作的:

```py

>>>[1, 1, 1] == [1, 1, 1]
True
>>> [1, 1, 0] == [0, 1, 1]
False

```

这种语言还有另一个很棒的特性——它提供了将列表乘以一个数字的能力，在这个操作的结果中，我们将得到一个列表，其中所有的元素都被复制了指定的次数。让我给你看一些例子:

```py

>>> [1] * 3
[1, 1, 1]
>>> [1] * 5
[1, 1, 1, 1, 1]
>>> [1] * 0
[]
>>> [1, 2] * 3
[1, 2, 1, 2, 1, 2]

```

因此，您可以提出一个简单的解决方案——如果您将一个包含一个元素的数组(即输入数组的第一个元素)乘以该输入数组的长度，那么在结果中您应该再次返回该输入数组，如果该数组的所有元素确实相同的话。

```py

def all_the_same(elements):
    if not elements:
        return True
    return [elements[0]] * len(elements) == elements

```

这里，解决方案也可以简化为:

```py

def all_the_same(elements):
    return not elements or [elements[0]] * len(elements) == elements

```

**3。**在这个解决方案中使用了标准的 [set()函数](https://docs.python.org/3/library/functions.html?#func-set)。这个函数将一个对象转换成一个集合，根据定义，这个集合中的所有元素必须是唯一的。看起来是这样的:

```py

>>> elements = [1, 2, 5, 1, 5, 3, 2]
>>> set(elements)
{1, 2, 3, 5}

```

如果结果集包含 1 个或 0 个元素，那么输入列表包含所有相同的元素或者为空。解决方案可能是这样的:

```py

def all_the_same(elements):
    return len(set(elements)) in (0, 1)

```

或者像这样:

```py

def all_the_same(elements):
    return len(set(elements)) <= 1

```

这种方法可以与 [NumPy](http://www.numpy.org/) 模块一起使用，该模块有一个 [unique()函数](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.unique.html)，其工作方式如下:

```py

>>> from numpy import unique
>>> a = [1, 2, 1, 2, 3, 1, 1, 1, 1]
>>> unique(a)
[1 2 3]

```

正如您所看到的，它的工作与 set()函数非常相似，唯一的区别是在这种情况下对象类型不变——list 仍然是 list。此函数的解决方案如下所示:

```py

from numpy import unique

def all_the_same(elements):
    return len(unique(elements)) <= 1

```

**4。**下面是一个非常有创意的解决方案的例子，其中使用的 Python 的标准 [all()函数](https://docs.python.org/3/library/functions.html#all)就是这个任务名称上的玩法。如果传输列表的所有元素都为真，函数 all()将返回真。例如:

```py

>>> all([1, 2, 0, True])
False
#(0 isn't true)
>>> all([1, 2, None, True])
False
#(None isn't true)
>>> all([1, 2, False, True])
False
>>> all([1, 2, 0.1, True])
True

```

首先，变量 first 被赋予列表中第一个元素的值，其余的是除第一个元素之外的所有其他元素的列表。然后，根据剩余列表的下一个元素是否等于输入列表的第一个元素，将 True 或 False 值添加到 _same 元组。之后，如果 _same 元组只包含“真”元素，则 all()函数将返回 True，如果元组中至少有一个“假”元素，则返回 False。

```py

def all_the_same(elements):
    try:
        first, *rest = elements    
    except ValueError:
        return True
    the_same = (x == first for x in rest)
    return all(the_same)

```

只有当数组为空时，才会引发 ValueError 异常。但是我们可以进行一个更熟悉的测试:

```py

def all_the_same(elements):
    if not elements:
        return True
    first, *rest = elements
    the_same = (x == first for x in rest)
    return all(the_same)

```

**5。**下一个解决方案与上一个非常相似。只有一个小的修改——输入列表的第一个元素和其余的元素被一个迭代器分隔开。 [iter()函数](https://docs.python.org/3/library/functions.html#iter)从传递的列表中创建一个迭代器， [next()函数](https://docs.python.org/3/library/functions.html#next)从中获取下一个元素(即第一个元素——在第一次调用时)。如果打印 el 和 first 中列出的元素，您将看到以下内容:

```py

>>> el = iter([1, 2, 3])
>>> first = next(el, None)
>>> print(first)
1
>>> for i in el:
>>>     print(i)
2
3

```

除此之外，这个解决方案与前一个类似，只是我们不需要检查列表是否为空。

```py

def all_the_same(elements):
    el = iter(elements)
    first = next(el, None)
    return all(element == first for element in el)

```

**6。解决这一任务的创造性方法之一是重新排列元素。我们改变元素的位置，并检查列表是否因此而改变。它告诉我们列表中的所有元素都是相同的。以下是这种方法的几个例子:**

```py

def all_the_same(elements):
    return elements[1:] == elements[:-1]

```

或者

```py

def all_the_same(elements):
    return elements == elements[1:] + elements[:1]

```

还必须承认，数组的比较可以使用 [zip()函数](https://docs.python.org/3/library/functions.html#zip)逐个元素地进行。让我们考虑以下解决方案。

**7。**[zip()函数](https://docs.python.org/3/library/functions.html#zip)将一个对象的第 I 个元素与其余对象的第 I 个元素组合，直到最短的对象结束。

```py

>>> x = [1, 2, 3]
>>> y = [10, 11]
>>> list(zip(x, y))
[(1, 10), (2, 11)]

```

如您所见，尽管 x 由三个元素组成，但只使用了两个元素，因为最短的对象(在本例中为 y)只包含两个元素。

下面的解决方案是这样工作的:首先，创建第二个列表(elements [1:])，它等于输入列表，但是没有第一个元素。然后依次比较这两个列表中的元素，作为每次比较的结果，我们得到 True 或 False。之后，all()函数返回这个 True 和 False 集合的处理结果。

```py

def all_the_same(elements):
    return all(x == y for x, y in zip(elements, elements[1:]))

```

假设我们的输入列表是 elements = [2，2，2，3]。然后使用 zip()，我们将完整列表([2，2，2，3])和没有第一个元素的列表([2，2，3])组合如下:[(2，2)，(2，2)，(2，3)]。元素之间的比较将集合[True，True，False]传递给 all()函数，结果我们得到 False，这是正确的答案，因为输入列表中的所有元素并不相同。

**8。**下面这个使用了 [groupby()迭代器](https://docs.python.org/3.6/library/itertools.html#itertools.groupby)的解决方案非常有趣。groupby()迭代器的工作方式是这样的:它将第 I 个元素与第(i-1)个元素进行比较，如果元素相等，则继续移动，如果不相等，则在摘要列表中保留第(i-1)个元素，并继续与下一个元素进行比较。实际上，它看起来像这样:

```py

>>> from itertools import groupby
>>> elements = [1, 1, 1, 2, 1, 1, 1, 2]
>>> for key, group in groupby(elements):
>>>     print(key)
1
2
1
2

```

如您所见，只有那些与下一个位置的元素不同的元素保留了下来(元素[0]、元素[1]、元素[4]和元素[5]被排除在外)。

在这个解决方案中，函数在 groupby()迭代器的帮助下，每当输入列表的下一项与前一项不同时，就在列表中加 1。因此，如果输入列表包含 0 个元素或所有元素都相等，则 sum(sum(1 for _ in group by(elements)))将是 0 或 1，这在任何情况下都小于 2，如解决方案中所指定的。

```py

from itertools import groupby

def all_the_same(elements):
    return sum(1 for _ in groupby(elements)) < 2

```

**9。**另一个创造性的解决方案，其中一个标准的 Python 模块——[集合](https://docs.python.org/3/library/collections.html)——已经使用。[计数器](https://docs.python.org/3/library/collections.html#collections.Counter)创建一个字典，其中存储了输入列表中每个元素的数量信息。让我们看看它是如何工作的:

```py

>>> from collections import Counter
>>> a = [1, 1, 1, 2, 2, 3]
>>> Counter(a)
Counter({1: 3, 2: 2, 3: 1})

```

因此，如果这个字典的长度为 2 或更多，那么在输入列表中至少有 2 个不同的元素，并且它们并不都是相同的。

```py

def all_the_same(elements):
    from collections import Counter
    return not len(list(Counter(elements))) > 1

```

10。这个解决方案建立在与 7 号解决方案相同的逻辑上，但是使用的函数是 [eq()](https://docs.python.org/3/library/operator.html#operator.eq) 和 [starmap()](https://docs.python.org/3/library/itertools.html#itertools.starmap) 。让我们弄清楚它们是如何工作的:

```py

>>> from operator import eq
>>> eq(1, 2)
False

```

基本上，eq()函数的作用与“==”相同——比较两个对象，如果相等则返回 True，否则返回 False(eq 代表等价)。但是，请注意，函数是一个对象，例如，它可以作为一个参数传递给另一个函数，这在进一步描述的解决方案中已经完成。

函数创建一个迭代器，将另一个函数应用于对象列表。当对象已经分组为元组时使用。例如:

```py

>>> import math
>>> from itertools import starmap
>>> list(starmap(math.pow, [(1, 2), (3, 4)]))
[1.0, 81.0]

```

正如您所看到的，math.pow()函数由于 starmap()函数而被指定了两次——应用于两组对象(1**2 = 1.0，3**4 = 81.0)。

更简单地说，本例中的 starmap()函数可以表示为一个循环:

```py

import math

elements = [(1, 2), (3, 4)]
result = []
for i in elements:
    result.append(math.pow(i[0], i[1]))

```

使用上述函数的解决方案如下所示:

```py

from operator import eq
from itertools import starmap

def all_the_same(elements):
    return all(starmap(eq, zip(elements, elements[1:])))

```

### 结论

所以，在这里我们讨论了一些创造性的解决方案，它们都与一个最简单的编码难题有关。我甚至无法开始描述我们的用户由于处理其他有趣和更复杂的挑战而分享的独特方法的数量。我希望你喜欢读这篇文章，就像我喜欢写它一样。我期待你的反馈。请告诉我。这对你有用吗？你会如何解决这个任务？
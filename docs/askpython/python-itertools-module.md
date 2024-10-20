# Python Itertools 模块使用指南

> 原文：<https://www.askpython.com/python-modules/python-itertools-module>

在本文中，我们将看看如何使用 Python itertools 模块。

如果您想创建适合各种任务的不同类型的迭代器，这个模块非常有用。

如果你能学会这个模块的一些方法，这将是你工具箱中非常有用的一个补充！让我们从现在开始，通过一些有用的方法。

* * *

## python ITER tools–有用的方法

在这一节，我们将看看一些有用的生成迭代器的方法。

要使用这个模块，我们必须首先导入它。这个在标准库中已经有了，所以是预装的！

```py
import itertools

```

### 使用 Python itertools.chain()将 iterables 链接在一起

Python `itertools.chain()`方法从多个可迭代对象中生成一个迭代器。

这只是将所有的 iterables 链接到一个序列中，并向该组合序列返回一个迭代器。

此方法的语法如下

```py
iterator = itertools.chain(*sequence)

```

让我们看一个简单的例子，来理解这一点。

```py
import itertools

list1 = ['hello', 'from', 'AskPython']
list2 = [10, 20, 30, 40, 50]
dict1 = {'site': 'AskPython', 'url': 'https://askpython.com'}

# We can combine lists and dicts (iterables) into a single chain
for item in itertools.chain(list1, list2, dict1):
    print(item)

```

这里，我们直接使用迭代器，通过使用`for item in ...`遍历它

**输出**

```py
hello
from
AskPython
10
20
30
40
50
site
url

```

在这里，虽然我们正确地获得了列表的内容，但是没有显示字典值。

为了解决这个问题，我们可以使用`dict.items()`来获得一组`(key, value)`对。

```py
import itertools

list1 = ['hello', 'from', 'AskPython']
list2 = [10, 20, 30, 40, 50]
dict1 = {'site': 'AskPython', 'url': 'https://askpython.com'}

# We can combine lists and dicts (iterables) into a single chain
for item in itertools.chain(list1, list2, dict1.items()):
    print(item)

```

**输出**

```py
hello
from
AskPython
10
20
30
40
50
('site', 'AskPython')
('url', 'https://askpython.com')

```

事实上，我们现在也打印了值，使用`dict1.items()`作为 iterable！

### 使用 Python itertools.count()生成基于计数器的序列

我们可以使用函数 Python `itertools.count()`来制作对应于一个计数的迭代器。

```py
iterator = itertools.count(start=0, step=1)

```

这里，这是一个迭代器，从 0 开始无限计数。

这使得计数持续增加`step=1`。我们也可以将其设置为十进制/负数。

例如，如果你想证明你有一个无限循环，你可以运行下面的代码片段，但它是**而不是**推荐的。

只要确保你能理解`itertools.count()`是无限计数的。

```py
for num in itertools.count(start=0, step=1):
    # Infinite loop!
    print(num)

```

现在，虽然您可能不会立即发现这个函数的用途，但是您可以将它与其他函数结合使用，例如 [zip 方法](https://www.askpython.com/python/built-in-methods/python-zip-function)来构造序列。

考虑下面的例子:

```py
import itertools
numbers = [100, 200, 300, 400]

data = list(zip(itertools.count(0, 10), numbers))

print(data)

```

在这里，您现在可以看到迭代器的威力了！由于迭代器只在需要时产生输出，我们可以用另一个有限的可迭代对象来`zip()`它，比如一个列表！

现在，它被用来为列表中的条目构建索引，您可以使用输出来验证这一点！

```py
[(0, 100), (10, 200), (20, 300), (30, 400)]

```

现在，如果您想使用 Python `itertools.count()`获得迭代器序列的子集，那么您也可以使用`itertools.islice()`只构建迭代器的一部分。

```py
import itertools

for num in itertools.islice(itertools.count(start=0, step=10), 4):
    print(num)

for num in itertools.islice(itertools.count(), 0, 50, 10):
    print(num)

```

输出

```py
0
10
20
30
0
10
20
30
40

```

正如你所观察到的，两个序列是相同的。这表明您可以有多种方法来生成序列！

根据要解决的问题，使用你认为合适的方法！

## 使用 itertools.repeat()重复一个值

假设您想要重复一个特定的值，您可以使用`itertools.repeat(value)`为重复的值构造一个迭代器。

例如，如果你想构造一个形式为`(i, 5)`的序列，其中 I 的范围是从 0 到 10，你可以使用这个函数！

```py
import itertools

data = list(zip(range(10), itertools.repeat(5)))
print(data)

```

**输出**

```py
[(0, 5),
 (1, 5),
 (2, 5),
 (3, 5),
 (4, 5),
 (5, 5),
 (6, 5),
 (7, 5),
 (8, 5),
 (9, 5)]

```

事实上，我们能够轻松地制作这个序列！

该函数有用的另一个例子是，如果您试图使用 Python 中的 [map()构建正方形。](https://www.askpython.com/python/built-in-methods/map-method-in-python)

```py
squares = list(map(pow, range(10), itertools.repeat(2)))
print(squares)

```

**输出**

```py
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

```

看看我们用`map()`构建它有多容易？

### 使用 itertools.tee()克隆序列

还有一个有用的函数叫做`tee()`，它克隆一个序列，并产生两个序列。

```py
cloned1, cloned2 = itertools.tee(original_sequence)

```

这基于 Linux tee 命令，该命令克隆其输出。

这里，当您使用`tee()`克隆一个序列时，您不能再次使用同一个迭代器。因此，使用该功能时必须非常小心！

```py
import itertools

single_iterator = itertools.islice(itertools.count(), 3)
cloned1, cloned2 = itertools.tee(single_iterator)

for num in cloned1:
    print('cloned1: {}'.format(num))
for num in cloned2:
    print('cloned2: {}'.format(num))

```

**输出**

```py
cloned1: 0
cloned1: 1
cloned1: 2
cloned2: 0
cloned2: 1
cloned2: 2

```

事实上，我们可以看到两个克隆序列，具有相同的输出！

### 使用 itertools.cycle()遍历序列

函数提供了一个我们可以无限循环的迭代器！

如果您希望在应用程序中不断切换状态，这将非常有用。

考虑灯泡的两种状态:“开”和“关”。

你可以构造一个迭代器，每当开关被按下时，迭代器就在两种状态之间循环。

```py
import itertools

# Initially, bulb is switched off, so off is the first element in the list
bulb_states = itertools.cycle(["off", "on"])

for _ in range(5):
    # Use next(iterator) to get the current state
    curr_state = next(bulb_states)
    print(f"Bulb state currently {curr_state}")

```

**输出**

```py
Bulb state currently off
Bulb state currently on
Bulb state currently off
Bulb state currently on
Bulb state currently off

```

事实上，正如你所看到的，灯泡的状态一直在“开”和“关”这两个值之间循环！

### 使用 takewhile()和 dropwhile()筛选项目

我们可以使用 Python `itertools.takewhile()`函数来过滤序列项，只要条件是`True`。如果条件变为`False`，则停止过滤。

```py
iterator = itertools.takewhile(condition, *sequence)

```

这里有一个简单的例子，它过滤数字，只要数字是正数。

```py
import itertools

sequence = itertools.takewhile(lambda x: x > 0, [1, 2, 3, -1, 10])

for item in sequence:
    print(item)

```

**输出**

```py
1
2
3

```

这里，序列在 3 之后停止，因为下一个元素是-1。

类似地，`itertools.dropwhile()`只要条件为`False`就过滤元素，并返回第一个非 false 值之后的所有元素。

```py
import itertools

data = itertools.dropwhile(lambda x: x < 5, [3, 12, 7, 1, -5])
for item in data:
    print(item)

```

**输出**

```py
12
7
1
-5

```

### 使用组合()构造组合

我们还可以使用 Python `itertools.combinations()`构建组合序列。

```py
iterator = itertools.combinations(*sequence, r)

```

这里有一个简单的例子:

```py
import itertools
words = ['hello', 'from', 'AskPython', 'how']
results = itertools.combinations(words, 2)
for item in results:
    print(item)

```

**输出**

```py
('hello', 'from')
('hello', 'AskPython')
('hello', 'how')
('from', 'AskPython')
('from', 'how')
('AskPython', 'how')

```

如果你想在组合中有连续元素的重复，你可以使用`combinations_with_replacement()`。

```py
results = itertools.combinations_with_replacement(words, 3)

for item in results:
    print(item)

```

**输出**

```py
('hello', 'hello', 'hello')
('hello', 'hello', 'from')
('hello', 'hello', 'AskPython')
('hello', 'hello', 'how')
('hello', 'from', 'from')
('hello', 'from', 'AskPython')
('hello', 'from', 'how')
('hello', 'AskPython', 'AskPython')
('hello', 'AskPython', 'how')
('hello', 'how', 'how')
('from', 'from', 'from')
('from', 'from', 'AskPython')
('from', 'from', 'how')
('from', 'AskPython', 'AskPython')
('from', 'AskPython', 'how')
('from', 'how', 'how')
('AskPython', 'AskPython', 'AskPython')
('AskPython', 'AskPython', 'how')
('AskPython', 'how', 'how')
('how', 'how', 'how')

```

类似地，您可以使用`permutations()`和`permutations_with_replacement()`列出排列。

这就结束了这个模块的一些重要功能。更多功能可以咨询[官方文档](https://docs.python.org/3/library/itertools.html)。

* * *

## 结论

在本文中，我们查看了 Python `itertools`模块中的各种函数。根据您的问题，您可以使用多种方法中的一种来快速构建序列！

## 参考

*   [Itertools 模块](https://docs.python.org/3/library/itertools.html)文档
*   关于 itertools 模块的 JournalDev 文章

* * *
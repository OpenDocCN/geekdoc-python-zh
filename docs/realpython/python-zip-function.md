# 使用 Python zip()函数进行并行迭代

> 原文：<https://realpython.com/python-zip-function/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python 的 zip()函数**](/courses/python-zip-function/) 并行迭代

Python 的`zip()`函数创建了一个迭代器，它将聚合两个或更多可迭代对象的元素。您可以使用结果迭代器快速一致地解决常见的编程问题，比如创建[字典](https://realpython.com/courses/dictionaries-python/)。在本教程中，您将发现 Python `zip()`函数背后的逻辑，以及如何使用它来解决现实世界中的问题。

**本教程结束时，您将学会:**

*   **`zip()`** 如何在 Python 3 和 Python 2 中工作
*   如何使用 Python `zip()`函数进行**并行迭代**
*   如何使用`zip()`动态地**创建字典**

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 了解 Python `zip()`函数

`zip()`在[内置命名空间](https://docs.python.org/3/library/builtins.html)中可用。如果您使用`dir()`来检查`__builtins__`，那么您会在列表的末尾看到`zip()`:

>>>

```py
>>> dir(__builtins__)
['ArithmeticError', 'AssertionError', 'AttributeError', ..., 'zip']
```

您可以看到`'zip'`是可用对象列表中的最后一个条目。

根据[官方文档](https://docs.python.org/3/library/functions.html#zip)，Python 的`zip()`函数表现如下:

> 返回元组的迭代器，其中第 *i* 个元组包含来自每个参数序列或可迭代对象的第 *i* 个元素。当最短的输入 iterable 用尽时，迭代器停止。使用一个可迭代的参数，它返回一个 1 元组的迭代器。如果没有参数，它将返回一个空迭代器。([来源](https://docs.python.org/3/library/functions.html#zip))

在本教程的剩余部分，您将解开这个定义。在研究代码示例时，您会看到 Python zip 操作的工作方式就像包或牛仔裤上的物理拉链一样。拉链两侧的互锁齿对被拉到一起以闭合开口。事实上，这个直观的类比对于理解`zip()`来说是完美的，因为这个功能是以物理拉链命名的！

[*Remove ads*](/account/join/)

## 在 Python 中使用`zip()`

Python 的`zip()`函数定义为`zip(*iterables)`。该函数将 [iterables](https://docs.python.org/3/glossary.html#term-iterable) 作为参数，并返回一个**迭代器**。这个迭代器生成一系列元组，其中包含来自每个 iterable 的元素。`zip()`可以接受任何类型的 iterable，比如[文件](https://realpython.com/read-write-files-python/)、[列表、元组](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/python-dicts/)、[集合](https://realpython.com/python-sets/)等等。

### 传递`n`个参数

如果将`zip()`与`n`参数一起使用，那么函数将返回一个迭代器，生成长度为`n`的元组。要了解这一点，请看下面的代码块:

>>>

```py
>>> numbers = [1, 2, 3]
>>> letters = ['a', 'b', 'c']
>>> zipped = zip(numbers, letters)
>>> zipped  # Holds an iterator object
<zip object at 0x7fa4831153c8>
>>> type(zipped)
<class 'zip'>
>>> list(zipped)
[(1, 'a'), (2, 'b'), (3, 'c')]
```

这里，您使用`zip(numbers, letters)`创建一个迭代器，该迭代器产生形式为`(x, y)`的元组。在这种情况下，`x`值取自`numbers`，而`y`值取自`letters`。注意 Python `zip()`函数是如何返回迭代器的。要检索最终的列表对象，需要使用`list()`来消耗迭代器。

如果你正在处理像列表、元组或[字符串](https://realpython.com/python-strings/)这样的序列，那么你的 iterables 肯定会从左到右被求值。这意味着元组的结果列表将采用`[(numbers[0], letters[0]), (numbers[1], letters[1]),..., (numbers[n], letters[n])]`的形式。然而，对于其他类型的可重复项(如[集合](https://realpython.com/python-sets/)，您可能会看到一些奇怪的结果:

>>>

```py
>>> s1 = {2, 3, 1}
>>> s2 = {'b', 'a', 'c'}
>>> list(zip(s1, s2))
[(1, 'a'), (2, 'c'), (3, 'b')]
```

在这个例子中，`s1`和`s2`是`set`对象，它们的元素没有任何特定的顺序。这意味着`zip()`返回的元组将包含随机配对的元素。如果你打算将 Python `zip()`函数用于像集合这样的无序可重复项，那么这一点需要记住。

### 不传递参数

您也可以不带任何参数调用`zip()`。在这种情况下，您将简单地得到一个空迭代器:

>>>

```py
>>> zipped = zip()
>>> zipped
<zip object at 0x7f196294a488>
>>> list(zipped)
[]
```

在这里，您调用没有参数的`zip()`，所以您的`zipped` [变量](https://realpython.com/python-variables/)持有一个空迭代器。如果您使用`list()`来使用迭代器，那么您也会看到一个空列表。

您也可以尝试强制空迭代器直接产生一个元素。在这种情况下，你会得到一个`StopIteration` [异常](https://realpython.com/python-exceptions/):

>>>

```py
>>> zipped = zip()
>>> next(zipped)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

当您在`zipped`上调用 [`next()`](https://docs.python.org/3/library/functions.html#next) 时，Python 会尝试检索下一项。然而，由于`zipped`持有一个空迭代器，所以没有东西可以取出，所以 Python 引发了一个`StopIteration`异常。

### 传递一个参数

Python 的`zip()`函数也可以只接受一个参数。结果将是一个迭代器，产生一系列 1 项元组:

>>>

```py
>>> a = [1, 2, 3]
>>> zipped = zip(a)
>>> list(zipped)
[(1,), (2,), (3,)]
```

这可能不是那么有用，但它仍然有效。也许你能找到一些`zip()`这种行为的用例！

正如您所看到的，您可以调用 Python `zip()`函数，使用任意多的输入可重复项。结果元组的长度将始终等于作为参数传递的 iterables 的数量。下面是一个包含三个可迭代项的示例:

>>>

```py
>>> integers = [1, 2, 3]
>>> letters = ['a', 'b', 'c']
>>> floats = [4.0, 5.0, 6.0]
>>> zipped = zip(integers, letters, floats)  # Three input iterables
>>> list(zipped)
[(1, 'a', 4.0), (2, 'b', 5.0), (3, 'c', 6.0)]
```

这里，您用三个 iterables 调用 Python `zip()`函数，所以得到的元组每个都有三个元素。

[*Remove ads*](/account/join/)

### 传递长度不等的参数

当你使用 Python `zip()`函数时，注意你的 iterables 的长度是很重要的。作为参数传入的 iterables 可能长度不同。

在这些情况下，`zip()`输出的元素数量将等于最短的*的长度。任何更长的 iterables 中的剩余元素将被`zip()`完全忽略，正如你在这里看到的:*

>>>

```py
>>> list(zip(range(5), range(100)))
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
```

由于`5`是第一个(也是最短的) [`range()`](https://realpython.com/python-range/) 对象的长度，`zip()`输出一个五元组列表。第二个`range()`对象仍有 95 个不匹配的元素。这些都被`zip()`忽略了，因为没有更多来自第一个`range()`对象的元素来完成配对。

如果尾随或不匹配的值对你很重要，那么你可以用 [`itertools.zip_longest()`](https://docs.python.org/3/library/itertools.html#itertools.zip_longest) 代替`zip()`。使用这个函数，丢失的值将被替换为传递给`fillvalue`参数的值(默认为 [`None`](https://realpython.com/null-in-python/) )。迭代将继续，直到最长的可迭代次数用完:

>>>

```py
>>> from itertools import zip_longest
>>> numbers = [1, 2, 3]
>>> letters = ['a', 'b', 'c']
>>> longest = range(5)
>>> zipped = zip_longest(numbers, letters, longest, fillvalue='?')
>>> list(zipped)
[(1, 'a', 0), (2, 'b', 1), (3, 'c', 2), ('?', '?', 3), ('?', '?', 4)]
```

在这里，您使用`itertools.zip_longest()`生成五个元组，其中包含来自`letters`、`numbers`和`longest`的元素。只有当`longest`耗尽时，迭代才会停止。`numbers`和`letters`中缺失的元素用问号`?`填充，这是你用`fillvalue`指定的。

自从 [Python 3.10](https://realpython.com/python310-new-features/) ，`zip()`有了一个新的可选关键字参数叫做 [`strict`](https://docs.python.org/3/library/functions.html#zip) ，它是通过 [PEP 618 引入的——给 zip](https://www.python.org/dev/peps/pep-0618/) 添加可选的长度检查。这个参数的主要目标是提供一种安全的方式来处理长度不等的可重复项。

`strict`的缺省值是`False`，这确保了`zip()`保持向后兼容，并且具有与它在旧 Python 3 版本中的行为相匹配的缺省行为:

>>>

```py
>>> # Python >= 3.10

>>> list(zip(range(5), range(100)))
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
```

在 Python >= 3.10 中，调用`zip()`而不将默认值更改为`strict`仍然会给出一个五元组列表，忽略第二个`range()`对象中不匹配的元素。

或者，如果您将`strict`设置为`True`，那么`zip()`将检查您作为参数提供的输入可重复项是否具有相同的长度，如果不相同，将引发 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) :

>>>

```py
>>> # Python >= 3.10

>>> list(zip(range(5), range(100), strict=True))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: zip() argument 2 is longer than argument 1
```

当您需要确保函数只接受等长的可重复项时,`zip()`的这个新特性非常有用。将`strict`设置为`True`会使期望等长可重复项的代码更加安全，确保对调用者代码的错误更改不会导致数据无声地丢失。

## 比较 Python 3 中的`zip()`和 2 中的

Python 的`zip()`函数在该语言的两个版本中工作方式不同。在 Python 2 中，`zip()`返回元组的`list`。产生的`list`被截断为最短输入 iterable 的长度。如果你调用`zip()`而没有参数，那么你得到一个空的`list`作为回报:

>>>

```py
>>> # Python 2
>>> zipped = zip(range(3), 'ABCD')
>>> zipped  # Hold a list object
[(0, 'A'), (1, 'B'), (2, 'C')]
>>> type(zipped)
<type 'list'>
>>> zipped = zip()  # Create an empty list
>>> zipped
[]
```

在这种情况下，您对 Python `zip()`函数的调用返回在值`C`处截断的元组列表。当你调用没有参数的`zip()`时，你得到一个空的`list`。

然而，在 Python 3 中，`zip()`返回一个**迭代器**。该对象按需生成元组，并且只能被遍历一次。一旦最短的输入 iterable 用尽，迭代以一个`StopIteration`异常结束。如果没有给`zip()`提供参数，那么函数返回一个空迭代器:

>>>

```py
>>> # Python 3
>>> zipped = zip(range(3), 'ABCD')
>>> zipped  # Hold an iterator
<zip object at 0x7f456ccacbc8>
>>> type(zipped)
<class 'zip'>
>>> list(zipped)
[(0, 'A'), (1, 'B'), (2, 'C')]
>>> zipped = zip()  # Create an empty iterator
>>> zipped
<zip object at 0x7f456cc93ac8>
>>> next(zipped)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    next(zipped)
StopIteration
```

这里，您对`zip()`的调用返回一个迭代器。第一次迭代在`C`被截断，第二次迭代导致`StopIteration`异常。在 Python 3 中，您还可以通过将返回的迭代器封装在对`list()`的调用中来模拟`zip()`的 Python 2 行为。这将遍历迭代器并返回一个元组列表。

如果您经常使用 Python 2，那么请注意将`zip()`与长输入 iterables 一起使用会无意中消耗大量内存。在这些情况下，考虑使用`itertools.izip(*iterables)`来代替。这个函数创建了一个迭代器，它聚集了每个可迭代对象的元素。它产生了与 Python 3 中的`zip()`相同的效果:

>>>

```py
>>> # Python 2
>>> from itertools import izip
>>> zipped = izip(range(3), 'ABCD')
>>> zipped
<itertools.izip object at 0x7f3614b3fdd0>
>>> list(zipped)
[(0, 'A'), (1, 'B'), (2, 'C')]
```

在这个例子中，您调用`itertools.izip()`来创建一个迭代器。当您用`list()`消费返回的迭代器时，您会得到一个元组列表，就像您在 Python 3 中使用`zip()`一样。当最短的输入 iterable 用尽时，迭代停止。

如果您真的需要编写在 Python 2 和 Python 3 中行为相同的代码，那么您可以使用如下技巧:

```py
try:
    from itertools import izip as zip
except ImportError:
    pass
```

在这里，如果在`itertools`中`izip()`可用，那么您将知道您在 Python 2 中，并且`izip()`将使用别名`zip`被导入。否则，你的程序会抛出一个`ImportError`，你就知道你在 Python 3 中了。(这里的 [`pass`语句](https://realpython.com/python-pass/)只是一个占位符。)

有了这个技巧，您可以在整个代码中安全地使用 Python `zip()`函数。运行时，您的程序将自动选择并使用正确的版本。

到目前为止，您已经了解了 Python 的`zip()`函数是如何工作的，并了解了它的一些最重要的特性。现在是时候卷起袖子开始编写真实世界的例子了！

[*Remove ads*](/account/join/)

## 在多个可迭代对象上循环

在多个可迭代对象上循环是 Python 的`zip()`函数最常见的用例之一。如果您需要遍历多个列表、元组或任何其他序列，那么您很可能会求助于`zip()`。本节将向您展示如何使用`zip()`来同时迭代多个可迭代对象。

### 并行遍历列表

Python 的`zip()`函数允许你在两个或更多的可迭代对象上并行迭代。由于`zip()`生成元组，您可以在 [`for`循环](https://realpython.com/courses/python-for-loop/)的头中解包这些元组:

>>>

```py
>>> letters = ['a', 'b', 'c']
>>> numbers = [0, 1, 2]
>>> for l, n in zip(letters, numbers):
...     print(f'Letter: {l}')
...     print(f'Number: {n}')
...
Letter: a
Number: 0
Letter: b
Number: 1
Letter: c
Number: 2
```

在这里，您遍历由`zip()`返回的一系列元组，并将元素解包到`l`和`n`。当你组合`zip()`、`for`循环、[元组解包](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)时，你可以得到一个有用的[python 式](https://realpython.com/courses/idiomatic-python-101/)习语，用于一次遍历两个或更多的 iterables。

您也可以在一个`for`循环中遍历两个以上的 iterables。考虑下面的例子，它有三个输入项:

>>>

```py
>>> letters = ['a', 'b', 'c']
>>> numbers = [0, 1, 2]
>>> operators = ['*', '/', '+']
>>> for l, n, o in zip(letters, numbers, operators):
...     print(f'Letter: {l}')
...     print(f'Number: {n}')
...     print(f'Operator: {o}')
...
Letter: a
Number: 0
Operator: *
Letter: b
Number: 1
Operator: /
Letter: c
Number: 2
Operator: +
```

在这个例子中，您使用带有三个 iterables 的`zip()`来创建并返回一个迭代器，该迭代器生成 3 项元组。这使您可以一次遍历所有三个可迭代对象。对于 Python 的`zip()`函数，可以使用的 iterables 的数量没有限制。

**注意:**如果你想更深入地研究 Python `for`循环，请查看[Python“for”循环(确定迭代)](https://realpython.com/python-for-loop/)。

### 并行遍历字典

在 Python 3.6 及更高版本中，字典是[有序集合](https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-compactdict)，这意味着它们保持其元素被引入的相同顺序。如果您利用了这个特性，那么您可以使用 Python `zip()`函数以一种安全和一致的方式遍历多个字典:

>>>

```py
>>> dict_one = {'name': 'John', 'last_name': 'Doe', 'job': 'Python Consultant'}
>>> dict_two = {'name': 'Jane', 'last_name': 'Doe', 'job': 'Community Manager'}
>>> for (k1, v1), (k2, v2) in zip(dict_one.items(), dict_two.items()):
...     print(k1, '->', v1)
...     print(k2, '->', v2)
...
name -> John
name -> Jane
last_name -> Doe
last_name -> Doe
job -> Python Consultant
job -> Community Manager
```

这里，您并行迭代`dict_one`和`dict_two`。在这种情况下，`zip()`用两个字典中的条目生成元组。然后，您可以解包每个元组并同时访问两个字典的条目。

**注意:**如果你想更深入地研究字典迭代，请查看[如何在 Python 中迭代字典](https://realpython.com/iterate-through-dictionary-python/)。

注意，在上面的例子中，从左到右的求值顺序是有保证的。还可以使用 Python 的`zip()`函数并行遍历集合。然而，你需要考虑到，与 Python 3.6 中的字典不同，集合*不会*保持它们的元素有序。如果你忘记了这个细节，你的程序的最终结果可能并不完全是你想要的或期望的。

### 解压缩序列

在新 Pythonistas 的论坛中经常出现一个问题:“如果有一个`zip()`函数，那么为什么没有一个`unzip()`函数做相反的事情？”

Python 中之所以没有`unzip()`函数，是因为`zip()`的反义词是……嗯，`zip()`。你还记得 Python `zip()`函数就像一个真正的拉链一样工作吗？到目前为止，示例已经向您展示了 Python 如何压缩关闭的内容。那么，如何解压 Python 对象呢？

假设您有一个元组列表，并希望将每个元组的元素分成独立的序列。为此，您可以将`zip()`与[解包操作符`*`](https://realpython.com/python-kwargs-and-args/#unpacking-with-the-asterisk-operators) 一起使用，如下所示:

>>>

```py
>>> pairs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
>>> numbers, letters = zip(*pairs)
>>> numbers
(1, 2, 3, 4)
>>> letters
('a', 'b', 'c', 'd')
```

这里，您有一个包含某种混合数据的元组`list`。然后，使用解包操作符`*`解压数据，创建两个不同的列表(`numbers`和`letters`)。

[*Remove ads*](/account/join/)

### 并行排序

[排序](https://realpython.com/sorting-algorithms-python/)是编程中常见的操作。假设你想合并两个列表，同时对它们进行排序。为此，您可以将`zip()`与 [`.sort()`](https://realpython.com/python-sort/) 一起使用，如下所示:

>>>

```py
>>> letters = ['b', 'a', 'd', 'c']
>>> numbers = [2, 4, 3, 1]
>>> data1 = list(zip(letters, numbers))
>>> data1
[('b', 2), ('a', 4), ('d', 3), ('c', 1)]
>>> data1.sort()  # Sort by letters
>>> data1
[('a', 4), ('b', 2), ('c', 1), ('d', 3)]
>>> data2 = list(zip(numbers, letters))
>>> data2
[(2, 'b'), (4, 'a'), (3, 'd'), (1, 'c')]
>>> data2.sort()  # Sort by numbers
>>> data2
[(1, 'c'), (2, 'b'), (3, 'd'), (4, 'a')]
```

在这个例子中，首先用`zip()`合并两个列表，并对它们进行排序。请注意`data1`是如何按照`letters`排序的，而`data2`是如何按照`numbers`排序的。

您也可以同时使用`sorted()`和`zip()`来获得类似的结果:

>>>

```py
>>> letters = ['b', 'a', 'd', 'c']
>>> numbers = [2, 4, 3, 1]
>>> data = sorted(zip(letters, numbers))  # Sort by letters
>>> data
[('a', 4), ('b', 2), ('c', 1), ('d', 3)]
```

在这种情况下，`sorted()`遍历由`zip()`生成的迭代器，并通过`letters`对条目进行排序，这一切都是一气呵成的。这种方法会快一点，因为你只需要两个函数调用:`zip()`和`sorted()`。

使用`sorted()`，你还可以编写一段更通用的代码。这将允许你排序任何种类的序列，而不仅仅是列表。

### 成对计算

可以使用 Python `zip()`函数进行一些快速计算。假设您在电子表格中有以下数据:

| 元素/月份 | 一月 | 二月 | 三月 |
| --- | --- | --- | --- |
| 销售总额 | Fifty-two thousand | Fifty-one thousand | Forty-eight thousand |
| 生产成本 | Forty-six thousand eight hundred | Forty-five thousand nine hundred | Forty-three thousand two hundred |

你将使用这些数据来计算你的月利润。`zip()`可以为您提供一种快速的计算方式:

>>>

```py
>>> total_sales = [52000.00, 51000.00, 48000.00]
>>> prod_cost = [46800.00, 45900.00, 43200.00]
>>> for sales, costs in zip(total_sales, prod_cost):
...     profit = sales - costs
...     print(f'Total profit: {profit}')
...
Total profit: 5200.0
Total profit: 5100.0
Total profit: 4800.0
```

在这里，您通过从`sales`中减去`costs`来计算每个月的利润。Python 的`zip()`函数结合正确的数据对进行计算。您可以推广这个逻辑，用`zip()`返回的对进行任何复杂的计算。

### 构建字典

Python 的[字典](https://realpython.com/python-dicts/)是一种非常有用的数据结构。有时，您可能需要从两个不同但密切相关的序列中构建一个字典。实现这一点的一个方便方法是同时使用`dict()`和`zip()`。例如，假设您从表单或数据库中检索一个人的数据。现在，您拥有以下数据列表:

>>>

```py
>>> fields = ['name', 'last_name', 'age', 'job']
>>> values = ['John', 'Doe', '45', 'Python Developer']
```

有了这些数据，您需要创建一个字典来进行进一步的处理。在这种情况下，您可以将`dict()`与`zip()`一起使用，如下所示:

>>>

```py
>>> a_dict = dict(zip(fields, values))
>>> a_dict
{'name': 'John', 'last_name': 'Doe', 'age': '45', 'job': 'Python Developer'}
```

在这里，您创建了一个结合了两个列表的字典。`zip(fields, values)`返回一个生成 2 项元组的迭代器。如果您在这个迭代器上调用`dict()`，那么您将构建您需要的字典。`fields`的元素成为字典的键，`values`的元素代表字典中的值。

您也可以通过组合`zip()`和`dict.update()`来更新现有的字典。假设约翰换了工作，你需要更新字典。您可以执行如下操作:

>>>

```py
>>> new_job = ['Python Consultant']
>>> field = ['job']
>>> a_dict.update(zip(field, new_job))
>>> a_dict
{'name': 'John', 'last_name': 'Doe', 'age': '45', 'job': 'Python Consultant'}
```

这里，`dict.update()`用您使用 Python 的`zip()`函数创建的键值元组更新字典。使用这种技术，您可以很容易地覆盖`job`的值。

[*Remove ads*](/account/join/)

## 结论

在本教程中，你已经学会了如何使用 Python 的`zip()`函数。`zip()`可以接收多个 iterables 作为输入。它返回一个迭代器，该迭代器可以从每个参数生成带有成对元素的元组。当您需要在一个循环中处理多个可迭代对象并同时对它们的项执行一些操作时，结果迭代器会非常有用。

现在您可以:

*   **在 Python 3 和 Python 2 中都使用`zip()`函数**
*   **循环遍历多个 iterables** 并对它们的项目并行执行不同的操作
*   **通过将两个输入的可重复项压缩在一起，动态创建和更新字典**

您还编写了一些例子，可以作为使用 Python 的`zip()`函数实现自己的解决方案的起点。当您深入探索`zip()`时，请随意修改这些示例！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 Python 的 zip()函数**](/courses/python-zip-function/) 并行迭代*******
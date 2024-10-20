# Python 中的计数器

> 原文：<https://www.askpython.com/python/counter-in-python>

计数器是`dict`的子类，是**集合**模块的一部分。它用于计算可折叠物体的数量。

它是一个无序的集合，其中的元素存储为字典键，它们的计数就是值。

计数器对象格式:`{element1: count1, element2: count2}`

元素从一个*可迭代的*开始计数，或者从另一个映射(或计数器)开始初始化

使用`Counter()`调用完成`Counter`对象的初始化。

我们还可以将 iterable 传递到调用中，并获得相应的映射对象。

```py
>>> from collections import Counter
>>> # Empty Counter object
>>> c = Counter()
>>> c
Counter()

>>> # Tuple of values
>>> d = Counter(a=1, b=2, c=1)
>>> d
Counter({'b': 2, 'a': 1, 'c': 1})

>>> # Pass a list, which is an iterable
>>> e = Counter(['a', 'b', 'a', 'c'])
>>> e
Counter({'a': 2, 'b': 1, 'c': 1})

>>> # Pass a string, also an iterable
>>> f = Counter('red-dish')
>>> f
Counter({'d': 2, 'r': 1, 'e': 1, '-': 1, 'i': 1, 's': 1, 'h': 1})

>>> # Pass a Dictionary
>>> g = Counter({'a': 10, 'b': 11})
>>> g
Counter({'b': 11, 'a': 10})

```

请注意，当显示计数器对象时，键-值对按递减计数的顺序显示。

计数器对象有一个字典接口，除了它们为丢失的项目返回一个零计数，而不是引发一个`[KeyError](https://docs.python.org/2/library/exceptions.html#exceptions.KeyError)`。

* * *

## 计数器方法

### 1.获取单个元素的计数

单个元素计数的访问方式与字典相同，这意味着`counter_object[key]`给出了`key`的计数。

```py
>>> c = Counter(a=1, b=2, c=1)
>>> c
Counter({'b': 2, 'a': 1, 'c': 1})
>>> c['b']
2
>>> c['d'] # Does not give KeyError, unlike a Dictionary
0

```

### 2.设置元素的计数

要设置一个元素的计数，使用`counter_object[key] = value`。如果`key`不存在，它将和新的计数一起被添加到计数器字典中。

```py
>>> c = Counter(a=1, b=2, c=1)
>>> c
Counter({'b': 2, 'a': 1, 'c': 1})
>>> c['d'] = 4
>>> c
Counter({'d': 4, 'b': 2, 'a': 1, 'c': 1})

```

### 3.从计数器中取出元件

要从计数器对象中删除一个键，使用`del counter_object[key]`。

```py
>>> del c['d']
>>> c
Counter({'b': 2, 'a': 1, 'c': 1})

```

### 4.元素()

这个方法返回一个元素的迭代器，元素值的重复次数与它们的计数一样多。该方法忽略所有计数小于 1 的*元素。*

```py
>>> c
Counter({'b': 2, 'a': 1, 'c': 1})
>>> c['d'] = -1
>>> c
>>> c.elements()
<itertools.chain object at 0x102e2a208>
>>> type(c.elements())
<class 'itertools.chain'>
>>> for i in c.elements():
...     print(i)
...
a
b
b
c
>>> list(c.elements())
['a', 'b', 'b', 'c']

>>> c['d'] = -1
>>> c
Counter({'b': 2, 'a': 1, 'c': 1, 'd': -1})
>>> # Ignores d since count[d] < 1
>>> list(c.elements())
['a', 'b', 'b', 'c']

```

### 5.最常见(n)

这将返回一个列表，列出了 n 个最常见的元素及其数量，从最常见到最少。如果 *n* 被省略或者`None`，`*most_common()*`返回*计数器中所有的*元素。计数相等的元素是任意排序的。

```py
>>> c
Counter({'b': 2, 'a': 1, 'c': 1, 'd': -1})
>>> c.most_common()
[('b', 2), ('a', 1), ('c', 1), ('d', -1)]
>>> c.most_common(2)
[('b', 2), ('a', 1)]

```

### 6.减法(可迭代/映射)

这将在减去两个 iterable/mappings 的内容后返回一个 mapping/iterable。元素不会被替换，只会减去它们的计数。

```py
>>> a = Counter('redblue')
>>> a
Counter({'e': 2, 'r': 1, 'd': 1, 'b': 1, 'l': 1, 'u': 1})
>>> b = Counter('blueorange')
>>> b
Counter({'e': 2, 'b': 1, 'l': 1, 'u': 1, 'o': 1, 'r': 1, 'a': 1, 'n': 1, 'g': 1})

>>> # Subtracts b from a and updates a accordingly
>>> a.subtract(b)
>>> a
Counter({'d': 1, 'r': 0, 'e': 0, 'b': 0, 'l': 0, 'u': 0, 'o': -1, 'a': -1, 'n': -1, 'g': -1})

```

### 7.更新(可迭代/映射)

这和`subtract()`类似，只是计数相加而不是相减。

```py
>>> a = Counter('12321')
>>> b = Counter('23432')
>>> a
Counter({'1': 2, '2': 2, '3': 1})
>>> b
Counter({'2': 2, '3': 2, '4': 1})

>>> # Add counts into a
>>> a.update(b)
>>> a
Counter({'2': 4, '3': 3, '1': 2, '4': 1})

```

## 其他 Counter()方法

*   `***counter.clear()***`用于重置计数器中所有元素的计数
*   `***counter.values()***`返回一个*字典值*对象，用于其他方法，如`sum()`获得所有元素的总数。
*   `***list(counter)***`用于列出所有独特的元素
*   `***set(counter)***`将计数器转换成集合
*   `***counter.items()***`返回计数器中`(key, value)`对的列表。
*   `***counter += Counter()***`删除所有计数为零或负数的元素

* * *

## 计数器上的算术运算

我们可以在计数器上使用基本的算术运算，如加、减、并和交。

```py
>>> c = Counter(a=3, b=1)
>>> d = Counter(a=1, b=2)
>>> c + d
Counter({'a': 4, 'b': 3})

>>> # Subtract c, d while keeping only positive counts
>>> c - d
Counter({'a': 2})

>>> # Intersection of c, d (Min(c, d))
>>> c & d
Counter({'a': 1, 'b': 1})

>>> # Union of c, d (Max (c, d))
>>> c | d
Counter({'a': 3, 'b': 2})

```

* * *

## 结论

我们学习了 Counter 类，它为我们提供了每个元素到其计数的映射对象。我们还学习了集合的一些方法。计数器为我们提供了，用于操作计数器对象。

## 参考

Python 集合文档:[https://docs.python.org/2/library/collections.html](https://docs.python.org/2/library/collections.html)

关于 Python 的计数器的 JournalDev 文章:https://www . journal dev . com/20806/Python-Counter-Python-collections-Counter

* * *
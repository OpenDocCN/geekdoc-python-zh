# 从 Python 中的列表/元组/数据结构中随机选择一项

> 原文：<https://www.pythoncentral.io/select-random-item-list-tuple-data-structure-python/>

需要*随机*动作的最常见的任务之一是从一个组中选择一个项目，可以是字符串、unicode 或缓冲区中的一个字符，bytearray 中的一个字节，或者是列表、元组、集合或 xrange 中的一个项目。想要一个以上项目的样本也很常见。

## 随机选择一个项目时不要这样做

完成这些任务的一种简单方法包括如下内容:要选择单个项目，您可以使用来自`random`模块的`randrange`(或`randint`),该模块根据其参数指定的范围生成一个伪随机整数:
【python】
导入随机

items = ['here '，' are '，' some '，' strings '，
' which '，' we '，' will '，' select '，' one']

rand _ item = items[random . rand range(len(items))]

一种同样简单的选择多项的方法可能是使用`random.randrange`在列表理解中生成索引，比如:
【python】
rand _ items =[items[random . rand range(len(items))]
for item in range(4)]
[/Python]
这些都可以工作，但是如果您已经编写 Python 有一段时间了，您应该会想到，有一种内置的方法可以更简洁、更易读地完成这项工作。

## 请在选择项目时执行此操作

从 Python 序列类型(即`str`、`unicode`、`list`、`tuple`、`bytearray`、`buffer`、`xrange`中选择一项的 Python 方法是使用`random.choice`。例如，我们单项选择的最后一行是:

```py

rand_item = random.choice(items)

```

简单多了，不是吗？有一个同样简单的方法从序列中选择 n 个项目:

```py

rand_items = random.sample(items, n)

```

## 从`set`中随机选择

`sets`是*不可转位*，意味着`set([1, 2, 3])[0]`产生错误。因此`random.choice`不支持`sets`，而`random.sample`支持。

例如:

```py

>>> from random import choice, sample

>>>

>>> # INVALID: set([1, 2, 3])[0]

>>> choice(set([1, 2, 3, 4, 5]))

Traceback (most recent call last):

  File "", line 1, in <module>

  File "<python-dist>/random.py", line 275, in choice

    return seq[int(self.random() * len(seq))]  # raises IndexError if seq is empty

TypeError: 'set' object does not support indexing

```

有几种方法可以解决这个问题，其中两种方法是首先将`set`转换成`list`，然后使用支持`sets`的`random.sample`。

示例:

```py

>>> from random import choice, sample

>>>

>>> # Convert the set to a list

>>> choice(list(set([1, 2, 3])))

1

>>>

>>> # random.sample(), selecting 1 random element

>>> sample(set([1, 2, 3]), 1)

[1]

>>> sample(set([1, 2, 3]), 1)[0]

3

```

重复项目
如果序列包含重复值，则每个值都是独立的候选值。为了避免重复，一种方法是将`list`转换成`set`，然后再转换回`list`。例如:

```py

>>> my_list = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

>>> my_set = set(my_list)

>>> my_list = list(my_set) # No duplicates

>>> my_list

[1, 2, 3, 4, 5]

>>> my_elem = random.choice(my_list)

>>> my_elem

2

>>> another_elem = random.choice(list(set([1, 1, 1])))

```


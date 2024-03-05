# 如何在 Python 中迭代字典

> 原文：<https://realpython.com/iterate-through-dictionary-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 字典迭代:进阶提示&招数**](/courses/python-dictionary-iteration/)

字典是 Python 中最重要和最有用的数据结构之一。它们可以帮助您解决各种各样的编程问题。本教程将带您深入了解如何在 Python 中迭代字典。

本教程结束时，你会知道:

*   什么是字典，以及它们的一些主要特性和实现细节
*   如何使用 Python 语言提供的基本工具来遍历该语言中的字典
*   通过在 Python 中遍历字典，可以执行什么样的实际任务
*   如何使用一些更高级的技术和策略来遍历 Python 中的字典

有关词典的更多信息，您可以查阅以下资源:

*   [Python 中的字典](https://realpython.com/python-dicts)
*   [以 Python 3 中的 Itertools 为例](https://realpython.com/python-itertools)
*   该文档为 [`map()`](https://docs.python.org/3/library/functions.html#map) 和 [`filter()`](https://docs.python.org/3/library/functions.html#filter)

准备好了吗？我们走吧！

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

***参加测验:****通过我们的交互式“Python 字典迭代”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-dictionary-iteration/)

## 字典上的几个词

字典是 Python 的基石。语言本身是建立在字典的基础上的。模块、类、对象、`globals()`、`locals()`:这些都是字典。从一开始，字典就是 Python 的核心。

[Python 的官方文档](https://docs.python.org/3/index.html)对字典的定义如下:

> 一个关联数组，其中任意键被映射到值。这些键可以是具有`__hash__()`和`__eq__()`方法的任何对象。([来源](https://docs.python.org/3/glossary.html#term-dictionary)

有几点需要记住:

1.  字典将键映射到值，并将它们存储在数组或集合中。
2.  这些键必须是[哈希](https://docs.python.org/3/glossary.html#term-hashable)类型，这意味着它们必须有一个在键的生命周期中不会改变的哈希值。

字典经常用于解决各种编程问题，因此作为 Python 开发人员，字典是您的基本工具。

与 **[序列](https://docs.python.org/3/glossary.html#term-sequence)** 不同，后者是支持使用整数索引的元素访问的 [iterables](https://docs.python.org/3/glossary.html#term-iterable) ，字典是通过键索引的。

字典中的键很像一个 [`set`](https://realpython.com/python-sets/) ，它是可散列的和唯一的对象的集合。因为对象需要是可散列的，[可变的](https://docs.python.org/3/glossary.html#term-mutable)对象不能用作字典键。

另一方面，值可以是任何 Python 类型，不管它们是否是可散列的。实际上，价值观没有任何限制。

在 Python 3.6 和更高版本中，字典的键和值按照它们被创建的顺序被迭代。然而，这种行为在不同的 Python 版本中可能会有所不同，这取决于字典的插入和删除历史。

在 Python 2.7 中，字典是无序的结构。字典条目的顺序是**打乱的**。这意味着项目的顺序是确定的和可重复的。让我们看一个例子:

>>>

```py
>>> # Python 2.7
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> a_dict
{'color': 'blue', 'pet': 'dog', 'fruit': 'apple'}
>>> a_dict
{'color': 'blue', 'pet': 'dog', 'fruit': 'apple'}
```

如果您离开解释器，稍后打开一个新的交互会话，您将获得相同的项目顺序:

>>>

```py
>>> # Python 2.7\. New interactive session
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> a_dict
{'color': 'blue', 'pet': 'dog', 'fruit': 'apple'}
>>> a_dict
{'color': 'blue', 'pet': 'dog', 'fruit': 'apple'}
```

仔细观察这两个输出会发现，两种情况下的结果顺序完全相同。这就是为什么你可以说排序是确定性的。

在 Python 3.5 中，字典仍然是无序的，但这一次，**随机化了**数据结构。这意味着每次你重新运行字典，你会得到一个不同的条目顺序。让我们来看看:

>>>

```py
>>> # Python 3.5
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> a_dict
{'color': 'blue', 'pet': 'dog', 'fruit': 'apple'}
>>> a_dict
{'color': 'blue', 'pet': 'dog', 'fruit': 'apple'}
```

如果您进入一个新的交互式会话，您将得到以下内容:

>>>

```py
>>> # Python 3.5\. New interactive session
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> a_dict
{'fruit': 'apple', 'pet': 'dog', 'color': 'blue'}
>>> a_dict
{'fruit': 'apple', 'pet': 'dog', 'color': 'blue'}
```

这一次，您可以看到两个输出中的项目顺序不同。这就是为什么你可以说它们是随机化的数据结构。

在 Python 3.6 及更高版本中，[字典是**有序的**数据结构](https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-compactdict)，这意味着它们保持其元素被引入的相同顺序，正如您在这里看到的:

>>>

```py
>>> # Python 3.6 and beyond
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> a_dict
{'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> a_dict
{'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
```

这是 Python 字典中相对较新的特性，而且非常有用。但是如果您正在编写应该在不同 Python 版本中运行的代码，那么您一定不能依赖这个特性，因为它会产生错误的行为。

字典的另一个重要特性是它们是可变的数据结构，这意味着您可以添加、删除和更新它们的条目。值得注意的是，这也意味着它们不能用作其他字典的键，因为它们不是可散列对象。

**注意:**您在本节中学习的所有内容都与核心 Python 实现相关， [CPython](https://www.python.org/about/) 。

其他 Python 实现，如 [PyPy](https://realpython.com/pypy-faster-python/) 、 [IronPython](http://ironpython.net/) 或 [Jython](http://www.jython.org/index.html) ，可以展示不同的字典行为和特性，这超出了本文的范围。

[*Remove ads*](/account/join/)

## 如何用 Python 迭代字典:基础知识

字典是 Python 中一种有用且广泛使用的数据结构。作为一名 Python 程序员，您经常会遇到需要遍历 Python 中的字典，同时对其键值对执行一些操作的情况。

当谈到用 Python 迭代字典时，这种语言为您提供了一些很棒的工具，我们将在本文中介绍。

### 直接遍历关键字

Python 的字典是[映射对象](https://docs.python.org/3/glossary.html#term-mapping)。这意味着它们继承了一些**特殊方法**，Python 在内部使用这些方法来执行一些操作。这些方法使用在方法名的开头和结尾添加双下划线的命名约定来命名。

为了可视化任何 Python 对象的方法和属性，您可以使用`dir()`，这是一个内置的函数。如果您使用一个空字典作为参数运行`dir()`，那么您将能够看到字典实现的所有方法和属性:

>>>

```py
>>> dir({})
['__class__', '__contains__', '__delattr__', ... , '__iter__', ...]
```

如果仔细看看前面的输出，您会看到`'__iter__'`。这是一个当容器需要迭代器时调用的方法，它应该返回一个新的[迭代器对象](https://docs.python.org/3/glossary.html#term-iterator)，可以遍历容器中的所有对象。

**注意:**为了节省空间，前面代码的输出被缩写为(`...`)。

对于映射(比如字典)，`.__iter__()`应该遍历这些键。这意味着如果你将一个字典直接放入一个 [`for`循环](https://realpython.com/python-for-loop/)，Python 将自动调用该字典上的`.__iter__()`，你将得到一个覆盖其键的迭代器:

>>>

```py
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> for key in a_dict:
...     print(key)
...
color
fruit
pet
```

Python 足够聪明，知道`a_dict`是一个字典，它实现了`.__iter__()`。在这个例子中，Python 自动调用了`.__iter__()`，这允许您迭代`a_dict`的键。

这是 Python 中遍历字典最简单的方法。只要直接放入一个`for`循环，就大功告成了！

如果您使用这种方法和一个小技巧，那么您可以处理任何字典的键和值。诀窍在于使用索引操作符`[]`和字典及其键来访问值:

>>>

```py
>>> for key in a_dict:
...     print(key, '->', a_dict[key])
...
color -> blue
fruit -> apple
pet -> dog
```

前面的代码允许您同时访问`a_dict`的键(`key`)和值(`a_dict[key]`)。这样，您可以对键和值进行任何操作。

### 迭代通过`.items()`和

当您使用字典时，您可能希望同时使用键和值。在 Python 中迭代字典的最有用的方法之一是使用`.items()`，这是一种返回字典条目的新[视图](https://docs.python.org/3/library/stdtypes.html#dict-views)的方法:

>>>

```py
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> d_items = a_dict.items()
>>> d_items  # Here d_items is a view of items
dict_items([('color', 'blue'), ('fruit', 'apple'), ('pet', 'dog')])
```

像`d_items`这样的字典视图提供了字典条目的动态视图，这意味着当字典发生变化时，视图会反映这些变化。

视图可以被迭代以产生它们各自的数据，因此您可以通过使用由`.items()`返回的视图对象来迭代 Python 中的字典:

>>>

```py
>>> for item in a_dict.items():
...     print(item)
...
('color', 'blue')
('fruit', 'apple')
('pet', 'dog')
```

由`.items()`返回的视图对象一次产生一个键-值对，并允许你在 Python 中遍历一个字典，但是通过这种方式你可以同时访问键和值。

如果您仔细观察由`.items()`生成的单个项目，您会注意到它们实际上是`tuple`对象。让我们来看看:

>>>

```py
>>> for item in a_dict.items():
...     print(item)
...     print(type(item))
...
('color', 'blue')
<class 'tuple'>
('fruit', 'apple')
<class 'tuple'>
('pet', 'dog')
<class 'tuple'>
```

一旦知道了这一点，就可以使用 [`tuple`解包](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)来遍历正在使用的字典的键和值。为了实现这一点，您只需要将每个条目的元素分解成两个不同的[变量](https://realpython.com/python-variables/)来表示键和值:

>>>

```py
>>> for key, value in a_dict.items():
...     print(key, '->', value)
...
color -> blue
fruit -> apple
pet -> dog
```

这里，`for`循环头中的变量`key`和`value`进行解包。每次循环运行时，`key`将存储键，`value`将存储被处理的项的值。这样，您将对字典的条目有更多的控制，并且您将能够以更具可读性和 Pythonic 性的方式分别处理键和值。

**注意:**注意到`.values()`和`.keys()`像`.items()`一样返回视图对象，你将在接下来的两节中看到。

[*Remove ads*](/account/join/)

### 迭代通过`.keys()`和

如果您只需要使用字典的键，那么您可以使用`.keys()`，这是一个返回包含字典键的新视图对象的方法:

>>>

```py
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> keys = a_dict.keys()
>>> keys
dict_keys(['color', 'fruit', 'pet'])
```

这里由`.keys()`返回的对象提供了一个关于`a_dict`键的动态视图。这个视图可以用来遍历`a_dict`的键。

要使用`.keys()`遍历 Python 中的字典，只需在`for`循环的头中调用`.keys()`:

>>>

```py
>>> for key in a_dict.keys():
...     print(key)
...
color
fruit
pet
```

当你在`a_dict`上调用`.keys()`时，你得到一个键的视图。Python 知道视图对象是可迭代的，所以它开始循环，你可以处理`a_dict`的键。

另一方面，使用您之前见过的相同技巧(索引操作符`[]`)，您可以访问字典的值:

>>>

```py
>>> for key in a_dict.keys():
...     print(key, '->', a_dict[key])
...
color -> blue
fruit -> apple
pet -> dog
```

这样你就可以同时访问`a_dict`的键(`key`)和值(`a_dict[key]`)，并且你可以对它们执行任何操作。

### 迭代通过`.values()`和

在 Python 中，只使用值来遍历字典也很常见。一种方法是使用`.values()`，它返回一个包含字典值的视图:

>>>

```py
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> values = a_dict.values()
>>> values
dict_values(['blue', 'apple', 'dog'])
```

在前面的代码中，`values`保存了对包含`a_dict`值的视图对象的引用。

与任何视图对象一样，`.values()`返回的对象也可以被迭代。在这种情况下，`.values()`产生了`a_dict`的值:

>>>

```py
>>> for value in a_dict.values():
...     print(value)
...
blue
apple
dog
```

使用`.values()`，您将只能访问`a_dict`的值，而不用处理密钥。

值得注意的是，它们还支持[成员测试(`in` )](https://docs.python.org/3/library/stdtypes.html#typesseq-common) ，如果您想知道某个特定元素是否在字典中，这是一个重要的特性:

>>>

```py
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> 'pet' in a_dict.keys()
True
>>> 'apple' in a_dict.values()
True
>>> 'onion' in a_dict.values()
False
```

如果键(或值或项)存在于正在测试的字典中，使用`in`的成员测试返回`True`，否则返回`False`。如果您只想知道某个键(或值或项)是否存在于字典中，成员测试允许您不遍历 Python 中的字典。

[*Remove ads*](/account/join/)

## 修改值和键

在 Python 中遍历字典时，需要修改值和键是很常见的。要完成这项任务，你需要考虑几个要点。

例如，您可以随时修改这些值，但是您需要使用原始的字典和映射您想要修改的值的键:

>>>

```py
>>> prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> for k, v in prices.items():
...     prices[k] = round(v * 0.9, 2)  # Apply a 10% discount
...
>>> prices
{'apple': 0.36, 'orange': 0.32, 'banana': 0.23}
```

在前面的代码示例中，为了修改`prices`的值并应用 10%的折扣，您使用了表达式`prices[k] = round(v * 0.9, 2)`。

那么，如果可以访问它的键(`k`)和值(`v`)，为什么还要使用原来的字典呢？你应该能够直接修改它们吗？

真正的问题是`k`和`v`的变化没有反映在原始字典中。也就是说，如果您直接在循环中修改其中的任何一个(`k`或`v`)，那么实际发生的情况是您将丢失对相关字典组件的引用，而不会改变字典中的任何内容。

另一方面，可以通过将由`.keys()`返回的视图转换成一个`list`对象来添加或删除字典中的键:

>>>

```py
>>> prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> for key in list(prices.keys()):  # Use a list instead of a view
...     if key == 'orange':
...         del prices[key]  # Delete a key from prices
...
>>> prices
{'apple': 0.4, 'banana': 0.25}
```

这种方法可能会对性能产生一些影响，主要与内存消耗有关。例如，您将在系统内存中拥有一个全新的`list`,而不是按需生成元素的视图对象。然而，这可能是在 Python 中迭代字典时修改键的一种安全方式。

最后，如果您试图通过直接使用`.keys()`从`prices`中删除一个键，那么 Python 将引发一个`RuntimeError`，告诉您字典的大小在迭代过程中已经改变:

>>>

```py
>>> # Python 3\. dict.keys() returns a view object, not a list
>>> prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> for key in prices.keys():
...     if key == 'orange':
...         del prices[key]
...
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    for key in prices.keys():
RuntimeError: dictionary changed size during iteration
```

这是因为`.keys()`返回一个 dictionary-view 对象，该对象根据需要一次生成一个键，如果您删除一个条目(`del prices[key]`，那么 Python 会产生一个`RuntimeError`，因为您在迭代过程中修改了字典。

**注意:Python 2 中的**，`.items()`，`.keys()`，`.values()`返回`list`对象。但是`.iteritems()`、`iterkeys()`和`.itervalues()`返回迭代器。因此，如果您使用 Python 2，那么您可以通过直接使用`.keys()`来修改字典的键。

另一方面，如果您在 Python 2 代码中使用了`iterkeys()`，并试图修改字典的键，那么您将得到一个`RuntimeError`。

## 真实世界的例子

到目前为止，您已经看到了 Python 中遍历字典的更基本的方法。现在是时候看看如何在迭代过程中对字典的条目执行一些操作了。让我们看一些真实世界的例子。

**注意:**在本文的后面，您将看到通过使用其他 Python 工具来解决这些完全相同的问题的另一种方法。

### 将键转换为值，反之亦然

假设您有一个字典，出于某种原因需要将键转换为值，反之亦然。在这种情况下，您可以使用一个`for`循环来遍历字典，并通过使用键作为值来构建新字典，反之亦然:

>>>

```py
>>> a_dict = {'one': 1, 'two': 2, 'thee': 3, 'four': 4}
>>> new_dict = {}
>>> for key, value in a_dict.items():
...     new_dict[value] = key
...
>>> new_dict
{1: 'one', 2: 'two', 3: 'thee', 4: 'four'}
```

表达式`new_dict[value] = key`通过将键转换成值并将值用作键，为您完成了所有工作。为了让这段代码正常工作，存储在原始值中的数据必须是可哈希的数据类型。

[*Remove ads*](/account/join/)

### 过滤项目

有时您会遇到这样的情况:您有一个字典，但您想创建一个新字典来只存储满足给定条件的数据。您可以使用`for`循环中的 [`if`语句](https://realpython.com/python-conditional-statements/)来实现这一点，如下所示:

>>>

```py
>>> a_dict = {'one': 1, 'two': 2, 'thee': 3, 'four': 4}
>>> new_dict = {}  # Create a new empty dictionary
>>> for key, value in a_dict.items():
...     # If value satisfies the condition, then store it in new_dict
...     if value <= 2:
...         new_dict[key] = value
...
>>> new_dict
{'one': 1, 'two': 2}
```

在本例中，您已经过滤掉了值大于`2`的项目。现在`new_dict`只包含满足条件`value <= 2`的项目。这是解决这类问题的一种可能的方法。稍后，您将看到获得相同结果的更 Pythonic 化、可读性更强的方法。

### 做一些计算

在 Python 中遍历字典时，还经常需要做一些计算。假设您已经将公司的销售数据存储在一个字典中，现在您想知道一年的总收入。

为了解决这个问题，你可以定义一个初始值为零的变量。然后，您可以在该变量中累加字典中的每个值:

>>>

```py
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> total_income = 0.00
>>> for value in incomes.values():
...     total_income += value  # Accumulate the values in total_income
...
>>> total_income
14100.0
```

在这里，您已经遍历了`incomes`，并按照您想要的那样在`total_income`中依次累积了它的值。`total_income += value`这个表达式有魔力，在循环结束时，你会得到一年的总收入。注意`total_income += value`相当于`total_income = total_income + value`。

## 使用理解

一个**字典理解**是一种处理集合中所有或部分元素并返回一个字典作为结果的紧凑方式。与[列表理解](https://realpython.com/list-comprehension-python/)相反，它们需要两个表达式，用冒号分隔，后跟`for`和`if`(可选)子句。当运行字典理解时，产生的键-值对按照它们产生的顺序插入到新字典中。

例如，假设您有两个数据列表，您需要根据它们创建一个新的字典。在这种情况下，您可以使用 Python 的`zip(*iterables)`成对地遍历两个列表的元素:

>>>

```py
>>> objects = ['blue', 'apple', 'dog']
>>> categories = ['color', 'fruit', 'pet']
>>> a_dict = {key: value for key, value in zip(categories, objects)}
>>> a_dict
{'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
```

这里， [`zip()`](https://realpython.com/python-zip-function/) 接收两个可迭代对象(`categories`和`objects`)作为参数，并创建一个迭代器，从每个可迭代对象中聚合元素。由`zip()`生成的`tuple`对象然后被解包到`key`和`value`中，最终用于创建新的字典。

字典理解开辟了新的可能性，并为您提供了一个在 Python 中迭代字典的好工具。

### 将键转换为值，反之亦然:重温

如果您再看一下将键转换为值的问题，或者相反，您会发现您可以通过使用字典理解来编写一个更 Pythonic 化、更有效的解决方案:

>>>

```py
>>> a_dict = {'one': 1, 'two': 2, 'thee': 3, 'four': 4}
>>> new_dict = {value: key for key, value in a_dict.items()}
>>> new_dict
{1: 'one', 2: 'two', 3: 'thee', 4: 'four'}
```

有了对字典的理解，您就创建了一个全新的字典，其中键取代了值，反之亦然。这种新方法使您能够编写更可读、简洁、高效和 Pythonic 化的代码。

这段代码工作的条件与您之前看到的相同:值必须是可散列的对象。否则，您将无法将它们用作`new_dict`的按键。

[*Remove ads*](/account/join/)

### 过滤项目:重新访问

要用一个理解过滤字典中的条目，只需要添加一个`if`子句，定义想要满足的条件。在前面过滤字典的例子中，条件是`if v <= 2`。将这个`if`子句添加到字典理解的末尾，您将过滤掉值大于`2`的条目。让我们来看看:

>>>

```py
>>> a_dict = {'one': 1, 'two': 2, 'thee': 3, 'four': 4}
>>> new_dict = {k: v for k, v in a_dict.items() if v <= 2}
>>> new_dict
{'one': 1, 'two': 2}
```

现在`new_dict`只包含满足您的条件的项目。与之前的解决方案相比，这个解决方案更加 Pythonic 化和高效。

### 做一些计算:重温

还记得公司销售的例子吗？如果您使用列表理解来遍历字典的值，那么您将获得更紧凑、更快速、更 Pythonic 化的代码:

>>>

```py
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> total_income = sum([value for value in incomes.values()])
>>> total_income
14100.0
```

list comprehension 创建了一个包含`incomes`的值的`list`对象，然后您使用`sum()`对所有的值求和，并将结果存储在`total_income`中。

如果您正在使用一个非常大的字典，并且内存使用对您来说是一个问题，那么您可以使用一个[生成器表达式](https://docs.python.org/3/glossary.html#term-generator-expression)来代替列表理解。**生成器表达式**是一个返回迭代器的表达式。这看起来像一个列表理解，但你需要用括号来定义它，而不是括号:

>>>

```py
>>> total_income = sum(value for value in incomes.values())
>>> total_income
14100.0
```

如果您将方括号改为一对圆括号(这里是`sum()`的圆括号)，您将把列表理解变成一个生成器表达式，并且您的代码将是内存高效的，因为生成器表达式按需生成元素。您不必创建整个列表并将其存储在内存中，而是一次只需要存储一个元素。

**注:**如果你对生成器表达式完全陌生，可以看看[Python 生成器简介](https://realpython.com/introduction-to-python-generators/)和 [Python 生成器 101](https://realpython.com/courses/python-generators/) 来更好的理解题目。

最后，有一种更简单的方法来解决这个问题，只需直接使用`incomes.values()`作为`sum()`的参数:

>>>

```py
>>> total_income = sum(incomes.values())
>>> total_income
14100.0
```

`sum()`接收 iterable 作为参数，并返回其元素的总和。在这里，`incomes.values()`扮演传递给`sum()`的 iterable 的角色。结果就是你想要的总收入。

### 移除特定项目

现在，假设您有一个字典，并且需要创建一个删除了所选键的新字典。还记得关键视图对象如何像[集合](https://realpython.com/python-sets/)吗？嗯，这些相似之处不仅仅是可散列的和独特的对象的集合。键视图对象也支持常见的`set`操作。让我们看看如何利用这一点来删除字典中的特定条目:

>>>

```py
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> non_citric = {k: incomes[k] for k in incomes.keys() - {'orange'}}
>>> non_citric
{'apple': 5600.0, 'banana': 5000.0}
```

这段代码可以工作，因为键视图对象支持像并集、交集和差集这样的`set`操作。当你在字典理解里面写`incomes.keys() - {'orange'}`的时候，你真的是在做一个`set`差运算。如果您需要用字典的键执行任何`set`操作，那么您可以直接使用 key-view 对象，而不必先将其转换为`set`。这是键视图对象的一个鲜为人知的特性，在某些情况下非常有用。

### 整理字典

通常有必要对集合中的元素进行排序。从 Python 3.6 开始，字典就是有序的数据结构，所以如果你使用 Python 3.6(以及更高版本)，你将能够通过使用 [`sorted()`](https://realpython.com/python-sort/) 并借助字典理解对任何字典的条目进行排序:

>>>

```py
>>> # Python 3.6, and beyond
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> sorted_income = {k: incomes[k] for k in sorted(incomes)}
>>> sorted_income
{'apple': 5600.0, 'banana': 5000.0, 'orange': 3500.0}
```

这段代码允许你创建一个新的字典，它的键按排序顺序排列。这是可能的，因为`sorted(incomes)`返回一个排序的键列表，您可以用它来生成新的字典`sorted_dict`。

有关如何微调排序的更多信息，请查看[对 Python 字典进行排序:值、键等等](https://realpython.com/sort-python-dictionary/)。

[*Remove ads*](/account/join/)

## 按排序顺序迭代

有时，您可能需要在 Python 中遍历一个字典，但希望按排序顺序进行。这可以通过使用`sorted()`来实现。当您调用`sorted(iterable)`时，您会得到一个`list`，其中包含按照排序顺序排列的`iterable`的元素。

让我们看看，当您需要按照排序的顺序来遍历一个字典时，如何使用`sorted()`来遍历 Python 中的字典。

### 按键排序

如果您需要在 Python 中遍历一个字典，并希望它按键排序，那么您可以将您的字典用作`sorted()`的参数。这将返回一个包含排序后的键的`list`,您将能够遍历它们:

>>>

```py
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> for key in sorted(incomes):
...     print(key, '->', incomes[key])
...
apple -> 5600.0
banana -> 5000.0
orange -> 3500.0
```

在这个例子中，您使用`for`循环头中的`sorted(incomes)`按关键字对字典进行了排序(按字母顺序)。请注意，您也可以使用`sorted(incomes.keys())`来获得相同的结果。在这两种情况下，您都将得到一个`list`，其中包含您的字典中按排序顺序排列的键。

**注意:**排序顺序将取决于您用于键或值的[数据类型](https://realpython.com/python-data-types/)以及 Python 用于排序这些数据类型的内部规则。

### 按值排序

您可能还需要遍历 Python 中的字典，其中的条目按值排序。你也可以使用`sorted()`，但是使用第二个参数`key`。

`key`关键字参数指定了一个参数的函数，用于从正在处理的每个元素中提取一个比较键。

要按值对字典中的条目进行排序，您可以编写一个函数来返回每个条目的值，并将该函数用作`sorted()`的`key`参数:

>>>

```py
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> def by_value(item):
...     return item[1]
...
>>> for k, v in sorted(incomes.items(), key=by_value):
...     print(k, '->', v)
...
('orange', '->', 3500.0)
('banana', '->', 5000.0)
('apple', '->', 5600.0)
```

在本例中，您定义了`by_value()`，并使用它按值对`incomes`的项目进行排序。然后使用`sorted()`按照排序顺序遍历字典。key 函数(`by_value()`)告诉`sorted()`按照每个项目的第二个元素，也就是按照值(`item[1]`)对`incomes.items()`进行排序。

您可能还想按排序顺序遍历字典中的值，而不用担心键。在这种情况下，您可以如下使用`.values()`:

>>>

```py
>>> for value in sorted(incomes.values()):
...     print(value)
...
3500.0
5000.0
5600.0
```

`sorted(incomes.values())`根据您的需要，按排序顺序返回字典中的值。如果你使用`incomes.values()`，这些键是不可访问的，但是有时候你并不真的需要这些键，只需要这些值，这是一种快速访问它们的方法。

### 反转

如果需要对字典进行逆序排序，可以将`reverse=True`作为参数添加到`sorted()`中。关键字参数`reverse`应该取一个[布尔值](https://realpython.com/python-boolean/)。如果设置为`True`，那么元素以相反的顺序排序:

>>>

```py
>>> incomes = {'apple': 5600.00, 'orange': 3500.00, 'banana': 5000.00}
>>> for key in sorted(incomes, reverse=True):
...     print(key, '->', incomes[key])
...
orange -> 3500.0
banana -> 5000.0
apple -> 5600.0
```

在这里，您通过在`for`循环的头中使用`sorted(incomes, reverse=True)`以相反的顺序遍历了`incomes`的键。

最后，需要注意的是`sorted()`并没有真正修改底层字典的顺序。实际发生情况是，`sorted()`创建一个独立的列表，其中的元素按顺序排列，所以`incomes`保持不变:

>>>

```py
>>> incomes
{'apple': 5600.0, 'orange': 3500.0, 'banana': 5000.0}
```

这段代码告诉你`incomes`没有改变。`sorted()`没修改`incomes`。它只是从`incomes`的键中创建了一个新的排序列表。

[*Remove ads*](/account/join/)

## 破坏性地迭代`.popitem` ()

有时，您需要在 Python 中遍历一个字典，并按顺序删除它的条目。要完成这个任务，您可以使用`.popitem()`，它将从字典中移除并返回一个任意的键值对。另一方面，当你在一个空字典上调用`.popitem()`时，它会引出一个 [`KeyError`](https://realpython.com/python-keyerror/) 。

如果你真的需要在 Python 中破坏性地遍历一个字典，那么`.popitem()`会很有用。这里有一个例子:

```py
 1# File: dict_popitem.py
 2
 3a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
 4
 5while True:
 6    try:
 7        print(f'Dictionary length: {len(a_dict)}')
 8        item = a_dict.popitem()
 9        # Do something with item here...
10        print(f'{item} removed')
11    except KeyError:
12        print('The dictionary has no item now...')
13        break
```

这里，您使用了一个 [`while`循环](https://realpython.com/python-while-loop/)而不是一个`for`循环。这样做的原因是，如果您假装以这种方式修改字典，也就是说，如果您正在删除或添加条目，那么在 Python 中迭代字典是不安全的。

在`while`循环中，您定义了一个`try...except`块来捕捉当`a_dict`变空时由`.popitems()`引发的`KeyError`。在`try...except`块中，您处理字典，在每次迭代中删除一个条目。变量`item`保存了对连续项目的引用，并允许您对它们进行一些操作。

**注意:**在前面的代码示例中，您使用 Python 的 f-strings 进行字符串格式化。如果你想更深入地研究 f 字符串，那么你可以看看 [Python 3 的 f 字符串:一个改进的字符串格式化语法(指南)](https://realpython.com/python-f-strings/)。

如果您从命令行运行这个脚本，那么您将得到以下结果:

```py
$ python3 dict_popitem.py
Dictionary length: 3
('pet', 'dog') removed
Dictionary length: 2
('fruit', 'apple') removed
Dictionary length: 1
('color', 'blue') removed
Dictionary length: 0
The dictionary has no item now...
```

这里`.popitem()`依次移除了`a_dict`的物品。当字典变空时循环中断，`.popitem()`引发了一个`KeyError`异常。

## 使用 Python 的一些内置函数

Python 提供了一些内置函数，在处理集合(如字典)时可能会很有用。这些函数是一种迭代工具，为您提供了另一种在 Python 中遍历字典的方法。让我们看看其中的一些。

### `map()`

Python 的 [`map()`](https://realpython.com/python-map-function/) 被定义为`map(function, iterable, ...)`，并返回一个迭代器，将`function`应用于`iterable`的每一项，按需产生结果。因此，`map()`可以被看作是一个迭代工具，你可以用它来遍历 Python 中的字典。

假设您有一个包含一堆产品价格的字典，您需要对它们应用折扣。在这种情况下，您可以[定义一个管理折扣的函数](https://realpython.com/defining-your-own-python-function/)，然后将它用作`map()`的第一个参数。第二个参数可以是`prices.items()`:

>>>

```py
>>> prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> def discount(current_price):
...     return (current_price[0], round(current_price[1] * 0.95, 2))
...
>>> new_prices = dict(map(discount, prices.items()))
>>> new_prices
{'apple': 0.38, 'orange': 0.33, 'banana': 0.24}
```

这里，`map()`遍历字典(`prices.items()`)的条目，通过使用`discount()`对每种水果应用 5%的折扣。在这种情况下，需要使用`dict()`从`map()`返回的迭代器中生成`new_prices`字典。

注意，`discount()`返回一个`(key, value)`形式的`tuple`，其中`current_price[0]`代表键，`round(current_price[1] * 0.95, 2)`代表新值。

### `filter()`

`filter()`是另一个内置函数，可以用来遍历 Python 中的一个字典，过滤掉其中的一些条目。这个函数被定义为`filter(function, iterable)`，并从`iterable`的元素中返回一个迭代器，其中`function`返回`True`。

假设你想了解价格低于`0.40`的产品。您需要定义一个函数来确定价格是否满足该条件，并将其作为第一个参数传递给`filter()`。第二个论点可以是`prices.keys()`:

>>>

```py
>>> prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> def has_low_price(price):
...     return prices[price] < 0.4
...
>>> low_price = list(filter(has_low_price, prices.keys()))
>>> low_price
['orange', 'banana']
```

在这里，您用`filter()`迭代了`prices`的键。然后`filter()`将`has_low_price()`应用到`prices`的每一个键上。最后，你需要使用`list()`来生成低价产品的列表，因为`filter()`返回一个迭代器，你真的需要一个`list`对象。

[*Remove ads*](/account/join/)

## 使用`collections.ChainMap`

[`collections`](https://docs.python.org/3/library/collections.html) 是 Python 标准库中一个有用的模块，提供了专门的容器数据类型。其中一种数据类型是`ChainMap`，它是一个类似字典的类，用于创建多个映射的单一视图(类似字典)。使用`ChainMap`，你可以将多个字典组合在一起，创建一个单一的、可更新的视图。

现在，假设您有两个(或更多)字典，您需要将它们作为一个字典一起迭代。为了实现这一点，您可以创建一个`ChainMap`对象并用您的字典初始化它:

>>>

```py
>>> from collections import ChainMap
>>> fruit_prices = {'apple': 0.40, 'orange': 0.35}
>>> vegetable_prices = {'pepper': 0.20, 'onion': 0.55}
>>> chained_dict = ChainMap(fruit_prices, vegetable_prices)
>>> chained_dict  # A ChainMap object
ChainMap({'apple': 0.4, 'orange': 0.35}, {'pepper': 0.2, 'onion': 0.55})
>>> for key in chained_dict:
...     print(key, '->', chained_dict[key])
...
pepper -> 0.2
orange -> 0.35
onion -> 0.55
apple -> 0.4
```

从`collections`导入`ChainMap`后，您需要用您想要链接的字典创建一个`ChainMap`对象，然后您可以像使用常规字典一样自由地遍历结果对象。

`ChainMap`对象也像标准字典一样实现了`.keys()`、`values()`和`.items()`，因此您可以使用这些方法来遍历由`ChainMap`生成的类似字典的对象，就像您使用常规字典一样:

>>>

```py
>>> for key, value in chained_dict.items():
...     print(key, '->', value)
...
apple -> 0.4
pepper -> 0.2
orange -> 0.35
onion -> 0.55
```

在这种情况下，您已经在一个`ChainMap`对象上调用了`.items()`。`ChainMap`对象的行为就好像它是一个普通的字典，`.items()`返回了一个字典视图对象，可以像往常一样被迭代。

## 使用`itertools`

Python 的`itertools`是一个模块，它提供了一些有用的工具来执行迭代任务。让我们看看如何使用它们中的一些来遍历 Python 中的字典。

### 用`cycle()` 循环迭代

假设您想在 Python 中遍历一个字典，但是您需要在一个循环中重复遍历它。要完成这个任务，您可以使用`itertools.cycle(iterable)`，它使一个迭代器从`iterable`返回元素并保存每个元素的副本。当`iterable`用尽时，`cycle()`从保存的副本中返回元素。这是以循环的方式进行的，所以由你来决定是否停止循环。

在下面的例子中，您将连续三次遍历字典中的条目:

>>>

```py
>>> from itertools import cycle
>>> prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> times = 3  # Define how many times you need to iterate through prices
>>> total_items = times * len(prices)
>>> for item in cycle(prices.items()):
...     if not total_items:
...         break
...     total_items -= 1
...     print(item)
...
('apple', 0.4)
('orange', 0.35)
('banana', 0.25)
('apple', 0.4)
('orange', 0.35)
('banana', 0.25)
('apple', 0.4)
('orange', 0.35)
('banana', 0.25)
```

前面的代码允许您迭代给定次数的`prices`(在本例中为`3`)。这个循环可以是你需要的那样长，但是你有责任停止它。当`total_items`倒数到零时,`if`状态打破循环。

### 用`chain()` 链式迭代

`itertools`还提供了`chain(*iterables)`，它获取一些`iterables`作为参数，并生成一个迭代器，从第一个可迭代对象开始产生元素，直到用完为止，然后遍历下一个可迭代对象，依此类推，直到用完所有元素为止。

这允许您在一个链中遍历多个字典，就像您对`collections.ChainMap`所做的那样:

>>>

```py
>>> from itertools import chain
>>> fruit_prices = {'apple': 0.40, 'orange': 0.35, 'banana': 0.25}
>>> vegetable_prices = {'pepper': 0.20, 'onion': 0.55, 'tomato': 0.42}
>>> for item in chain(fruit_prices.items(), vegetable_prices.items()):
...     print(item)
...
('apple', 0.4)
('orange', 0.35)
('banana', 0.25)
('pepper', 0.2)
('onion', 0.55)
('tomato', 0.42)
```

在上面的代码中，`chain()`返回了一个 iterable，它组合了来自`fruit_prices`和`vegetable_prices`的条目。

也可以使用`.keys()`或`.values()`，这取决于您的需要，条件是同构:如果您使用`.keys()`作为`chain()`的参数，那么您需要使用`.keys()`作为其余参数。

[*Remove ads*](/account/join/)

## 使用字典解包运算符(`**` )

Python 3.5 带来了一个有趣的新特性。 [PEP 448 -额外的解包归纳](https://www.python.org/dev/peps/pep-0448)可以让你在 Python 中遍历多个字典时更加轻松。让我们用一个简短的例子来看看这是如何工作的。

假设您有两个(或更多)字典，您需要一起遍历它们，而不使用`collections.ChainMap`或`itertools.chain()`，正如您在前面的章节中看到的。在这种情况下，您可以使用字典解包操作符(`**`)将两个字典合并成一个新字典，然后遍历它:

>>>

```py
>>> fruit_prices = {'apple': 0.40, 'orange': 0.35}
>>> vegetable_prices = {'pepper': 0.20, 'onion': 0.55}
>>> # How to use the unpacking operator **
>>> {**vegetable_prices, **fruit_prices}
{'pepper': 0.2, 'onion': 0.55, 'apple': 0.4, 'orange': 0.35}
>>> # You can use this feature to iterate through multiple dictionaries
>>> for k, v in {**vegetable_prices, **fruit_prices}.items():
...     print(k, '->', v)
...
pepper -> 0.2
onion -> 0.55
apple -> 0.4
orange -> 0.35
```

字典解包操作符(`**`)确实是 Python 中一个很棒的特性。它允许你将多个字典合并成一个新的，就像你在使用`vegetable_prices`和`fruit_prices`的例子中所做的那样。一旦用解包操作符合并了字典，就可以像往常一样遍历新字典。

需要注意的是，如果您试图合并的字典有重复或公共的关键字，那么最右边的字典的值将占优势:

>>>

```py
>>> vegetable_prices = {'pepper': 0.20, 'onion': 0.55}
>>> fruit_prices = {'apple': 0.40, 'orange': 0.35, 'pepper': .25}
>>> {**vegetable_prices, **fruit_prices}
{'pepper': 0.25, 'onion': 0.55, 'apple': 0.4, 'orange': 0.35}
```

两个字典中都有`pepper`键。合并后，`pepper` ( `0.25`)的`fruit_prices`值占优势，因为`fruit_prices`是最右边的字典。

## 结论

您现在已经了解了如何在 Python 中迭代字典的基础知识，以及一些更高级的技术和策略！

**你已经学会:**

*   什么是字典，以及它们的一些主要特性和实现细节
*   Python 中遍历字典的基本方法有哪些
*   在 Python 中通过遍历字典可以完成什么样的任务
*   如何使用一些更复杂的技术和策略来遍历 Python 中的字典

您拥有充分利用 Python 词典所需的工具和知识。这将有助于您在将来使用字典迭代时更加高效和有效。

***参加测验:****通过我们的交互式“Python 字典迭代”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-dictionary-iteration/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 字典迭代:进阶提示&招数**](/courses/python-dictionary-iteration/)*************
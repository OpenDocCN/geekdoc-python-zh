# 对 Python 字典进行排序:值、键等等

> 原文：<https://realpython.com/sort-python-dictionary/>

您已经有了一个[字典](https://realpython.com/python-dicts/)，但是您想要对键值对进行排序。也许你已经尝试过将字典传递给`sorted()` [函数](https://realpython.com/defining-your-own-python-function/)，但是没有得到你期望的结果。在本教程中，如果您想用 Python 对字典进行排序，您将了解您需要知道的一切。

**在本教程中，您将**:

*   复习如何使用 **`sorted()`** 功能
*   学习如何让字典**视图**到**迭代**
*   了解字典在排序过程中是如何被转换成列表的
*   了解如何指定一个**排序键**来按照值、键或嵌套属性对字典进行排序
*   审查字典**的理解**和`dict()` **的构造**来重建你的字典
*   为您的**键值数据**考虑替代的**数据结构**

在这个过程中，您还将使用`timeit`模块对代码进行计时，并通过比较不同的键-值数据排序方法获得切实的结果。你还会考虑[一个排序字典是否真的是你的最佳选择](#judging-whether-you-want-to-use-a-sorted-dictionary)，因为它不是一个特别常见的模式。

为了充分利用本教程，您应该了解字典、列表、元组和函数。有了这些知识，在本教程结束时，您将能够对字典进行排序。一些高阶函数，比如[λ](https://realpython.com/python-lambda/)函数，也会派上用场，但不是必需的。

**免费下载:** [点击这里下载代码](https://realpython.com/bonus/sort-python-dictionary-code/)，您将在本教程中使用它对键-值对进行排序。

首先，在尝试用 Python 对字典进行排序之前，您将学习一些基础知识。

## 在 Python 中重新发现字典顺序

在 Python 3.6 之前，字典本来就**无序**。Python 字典是[散列表](https://realpython.com/python-hash-table/)的一个实现，散列表传统上是一种无序的数据结构。

作为 Python 3.6 中[紧凑字典](https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-compactdict)实现的一个副作用，字典开始保留[插入顺序](https://mail.python.org/pipermail/python-dev/2016-September/146327.html)。从 3.7 开始，插入顺序已经由 [*保证为*](https://realpython.com/python37-new-features/#the-order-of-dictionaries-is-guaranteed) 。

如果你想在压缩字典之前保持一个有序的字典作为数据结构，那么你可以使用 [`collections`模块](https://realpython.com/python-collections-module/)中的 [`OrderedDict`](https://realpython.com/python-ordereddict/) 。类似于现代的压缩字典，它也保持插入顺序，但是这两种类型的字典都不会自己排序。

存储有序键值对数据的另一种方法是将这些对存储为元组列表。正如您将在教程的后面看到的[，使用元组列表可能是您的数据的最佳选择。](#judging-whether-you-want-to-use-a-sorted-dictionary)

在对字典进行排序时，需要理解的一个要点是，即使它们保持插入顺序，它们也不会被视为一个[序列](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range)。字典就像键值对的[集合](https://realpython.com/python-sets/)，集合是无序的。

字典也没有太多的重新排序功能。它们不像列表，你可以在任何位置插入元素。在下一节中，您将进一步探索这种限制的后果。

[*Remove ads*](/account/join/)

## 理解字典排序的真正含义

因为字典没有太多的重新排序功能，所以在对字典进行排序时，很少会**就地**完成。事实上，没有方法可以显式地移动字典中的条目。

如果您想对字典进行就地排序，那么您必须使用`del`关键字从字典中删除一个条目，然后再添加它。删除然后再次添加实际上将键-值对移动到末尾。

`OrderedDict`类有一个[特定的方法](https://docs.python.org/3/library/collections.html#collections.OrderedDict.move_to_end)来将一个项目移动到末尾或开始，这可能使`OrderedDict`更适合保存一个排序的字典。然而，至少可以说，它仍然不是很常见，也不是很有性能。

对字典进行排序的典型方法是获取一个字典**视图**，对其进行排序，然后将结果列表转换回字典。所以你可以有效地从字典到列表，再回到字典。根据您的用例，您可能不需要将列表转换回字典。

**注意:**排序字典不是一种很常见的模式。在教程的后面，你会探索更多关于这个话题[的内容。](#judging-whether-you-want-to-use-a-sorted-dictionary)

有了这些准备工作，您将在下一部分开始对字典进行排序。

## Python 中的字典排序

在本节中，您将把字典排序的组件放在一起，以便最终掌握字典排序的最常用方法:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}

>>> # Sort by key
>>> dict(sorted(people.items()))
{1: 'Jill', 2: 'Jack', 3: 'Jim', 4: 'Jane'}

>>> # Sort by value
>>> dict(sorted(people.items(), key=lambda item: item[1]))
{2: 'Jack', 4: 'Jane', 1: 'Jill', 3: 'Jim'}
```

如果您不理解上面的片段，请不要担心，您将在接下来的部分中一步一步地回顾它。在这个过程中，您将学习如何使用带有排序键的`sorted()`函数、`lambda`函数和字典构造函数。

### 使用`sorted()`功能

您将用来对字典进行排序的关键函数是内置的 [`sorted()`](https://realpython.com/python-sort/) 函数。该函数将一个[可迭代的](https://realpython.com/python-for-loop/#iterables)作为主参数，并带有两个可选的[仅关键字参数](https://peps.python.org/pep-3102/)——一个`key`函数和一个`reverse`布尔值。

为了孤立地说明`sorted()`函数的行为，检查它在数字的[列表](https://realpython.com/python-lists-tuples/#python-lists)上的使用:

>>>

```py
>>> numbers = [5, 3, 4, 3, 6, 7, 3, 2, 3, 4, 1]
>>> sorted(numbers)
[1, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7]
```

如您所见，`sorted()`函数接受一个 iterable，对类似数字的**可比**元素进行排序，排序顺序为**升序**，并返回一个新列表。对于字符串，它按照字母顺序排序:

>>>

```py
>>> words = ["aa", "ab", "ac", "ba", "cb", "ca"]
>>> sorted(words)
['aa', 'ab', 'ac', 'ba', 'ca', 'cb']
```

按数字或字母顺序排序是最常见的元素排序方式，但也许您需要更多的控制。

假设您想要对上一个示例中每个单词的第二个字符*进行排序。为了定制`sorted()`函数用来排序元素的内容，您可以将一个[回调](https://en.wikipedia.org/wiki/Callback_(computer_programming))函数传递给`key`参数。*

回调函数是作为参数传递给另一个函数的函数。对于`sorted()`，您传递给它一个充当排序键的函数。然后`sorted()`函数将*回调*每个元素的排序键。

在下面的示例中，作为键传递的函数接受一个字符串，并将返回该字符串的第二个字符:

>>>

```py
>>> def select_second_character(word):
...     return word[1]
...
>>> sorted(words, key=select_second_character)
['aa', 'ba', 'ca', 'ab', 'cb', 'ac']
```

`sorted()`函数将`words` iterable 的每个元素传递给`key`函数，并使用返回值进行比较。使用键意味着`sorted()`函数将比较第二个字母，而不是直接比较整个字符串。

在教程的[后面的](#using-the-key-parameter-and-lambda-functions)中，当你使用参数按照值或嵌套元素对字典进行排序时，会有更多关于参数`key`的例子和解释。

如果你再看一下最后一次排序的结果，你可能会注意到`sorted()`函数的[稳定性](https://en.wikipedia.org/wiki/Sorting_algorithm#Stability)。这三个元素，`aa`、`ba`和`ca`，当按它们的第二个字符排序时是等价的。因为它们相等，`sorted()`函数保留了它们的*原始顺序*。Python 保证了这种稳定性。

**注意:**每个列表也有一个 [`.sort()`](https://docs.python.org/3/library/stdtypes.html#list.sort) 方法，与`sorted()`函数的签名相同。主要区别在于，`.sort()`方法对列表**就地**排序。相反，`sorted()`函数返回一个新的列表，不修改原来的列表。

你也可以通过 [`reverse=True`](https://realpython.com/python-reverse-list/) 向排序函数或方法返回相反的顺序。或者，您可以使用`reversed()`函数在排序后反转 iterable:

>>>

```py
>>> list(reversed([3, 2, 1]))
[1, 2, 3]
```

如果您想更深入地了解 Python 中的排序机制，并学习如何对字典以外的数据类型进行排序，那么请查看关于如何使用`sorted()`和`.sort()`T3[的教程](https://realpython.com/python-sort/)

那么，字典怎么样？实际上，您可以将字典直接输入到`sorted()`函数中:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}
>>> sorted(people)
[1, 2, 3, 4]
```

但是将字典直接传递给`sorted()`函数的默认行为是获取字典的**个键**，对它们进行排序，并返回一个键*的**列表**，只有*。这可能不是你想要的行为！为了保存字典中的所有信息，您需要熟悉**字典视图**。

[*Remove ads*](/account/join/)

### 从字典中获取键、值或两者

如果您想在对字典进行排序时保留字典中的所有信息，典型的第一步是调用字典上的 [`.items()`](https://docs.python.org/3/library/stdtypes.html#dict.items) 方法。在字典上调用`.items()`将提供一个表示键值对的[元组](https://realpython.com/python-lists-tuples/#python-tuples)的 iterable:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}
>>> people.items()
dict_items([(3, 'Jim'), (2, 'Jack'), (4, 'Jane'), (1, 'Jill')])
```

`.items()`方法返回一个只读的[字典视图对象](https://docs.python.org/3/library/stdtypes.html#dict-views)，作为进入字典的窗口。这个视图*不是*副本或列表——它是一个只读的[可迭代的](https://realpython.com/iterate-through-dictionary-python/)，它实际上*链接*到生成它的字典:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}
>>> view = people.items()
>>> people[2] = "Elvis"
>>> view
dict_items([(3, 'Jim'), (2, 'Elvis'), (4, 'Jane'), (1, 'Jill')])
```

您会注意到对字典的任何更新也会反映在视图中，因为它们是链接的。视图代表了一种轻量级的方式来迭代字典，而不需要首先生成列表。

**注意:**您可以使用`.values()`只获得值的视图，使用`.keys()`只获得键的视图。

至关重要的是，您可以对字典视图使用`sorted()`函数。您调用`.items()`方法，并将结果用作`sorted()`函数的参数。使用`.items()`保留字典中的所有信息:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}
>>> sorted(people.items())
[(1, 'Jill'), (2, 'Jack'), (3, 'Jim'), (4, 'Jane')]
```

这个例子产生一个元组的排序列表，每个元组代表字典的一个键值对。

如果您想最终得到一个按值排序的字典，那么还有两个问题。默认行为似乎仍然是按*键*而不是值排序。另一个问题是，你最终得到的是元组的*列表*，而不是字典。首先，您将了解如何按值排序。

### 理解 Python 如何排序元组

当在字典上使用`.items()`方法并将其输入到`sorted()`函数中时，您传递的是元组的 iterable，而`sorted()`函数直接比较整个元组。

在比较元组时，Python 的行为很像是按字母顺序对字符串进行排序。也就是说，它按字典顺序对它们进行排序。

**字典式排序**是指如果你有两个元组，`(1, 2, 4)`和`(1, 2, 3)`，那么你从比较每个元组的第一项开始。第一项在两种情况下都是`1`，这是相等的。第二个元素`2`在两种情况下也是相同的。第三要素分别是`4`和`3`。由于`3`小于`4`，您已经发现哪个项目比另一个项目少*。

因此，为了按字典顺序排列元组`(1, 2, 4)`和`(1, 2, 3)`，您可以将它们的顺序切换为`(1, 2, 3)`和`(1, 2, 4)`。

由于 Python 对元组的字典排序行为，使用带有`sorted()`函数的`.items()`方法将总是按键排序，除非您使用额外的东西。

### 使用`key`参数和λ函数

例如，如果您想按值排序，那么您必须指定一个**排序键**。排序关键字是提取可比值的一种方式。例如，如果您有一堆书，那么您可以使用作者的姓氏作为排序关键字。使用`sorted()`函数，您可以通过传递回调函数作为`key`参数来指定排序键。

**注意:**`key`参数与字典键无关！

要查看实际的排序键，请看这个例子，它类似于您在介绍`sorted()`函数的[部分看到的例子:](#using-the-sorted-function)

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}

>>> # Sort key
>>> def value_getter(item):
...     return item[1]
...

>>> sorted(people.items(), key=value_getter)
[(2, 'Jack'), (4, 'Jane'), (1, 'Jill'), (3, 'Jim')]

>>> # Or with a lambda function
>>> sorted(people.items(), key=lambda item: item[1])
[(2, 'Jack'), (4, 'Jane'), (1, 'Jill'), (3, 'Jim')]
```

在这个例子中，您尝试了两种传递`key`参数的方法。`key`参数接受一个回调函数。该函数可以是一个普通的函数标识符或一个[λ函数](https://realpython.com/python-lambda/)。示例中的 lambda 函数与`value_getter()`函数完全等价。

**注意:** Lambda 函数也称为[匿名函数](https://en.wikipedia.org/wiki/Anonymous_function)，因为它们没有名字。Lambda 函数是在代码中只使用一次的标准函数。

Lambda 函数除了让事情变得更紧凑，消除了单独定义函数的需要之外，没有带来任何好处。它们很好地将事物包含在同一行中:

```py
# With a normal function
def value_getter(item):
    return item[1]

sorted(people.items(), key=value_getter)

# With a lambda function
sorted(people.items(), key=lambda item: item[1])
```

对于示例中的基本 getter 函数，lambdas 可以派上用场。但是 lambdas 会使您的代码对于任何更复杂的东西来说可读性更差，所以要小心使用它们。

Lambdas 也只能包含一个[表达式](https://docs.python.org/3/glossary.html#term-expression)，使得任何多行[语句](https://docs.python.org/3/glossary.html#term-statement)如`if`语句或`for`循环都被禁止。例如，你可以通过使用理解和`if`表达式来解决这个问题，但是这可能会导致冗长而晦涩的一行程序。

`key`回调函数将接收它正在排序的 iterable 的每个元素。回调函数的工作是*返回可以比较的东西*，比如一个数字或者一个字符串。在这个例子中，您将函数命名为`value_getter()`,因为它所做的只是从一个键值元组中获取值。

因为带有元组的`sorted()`的默认行为是按字典顺序排序，所以`key`参数允许您从它比较的元素中选择一个值。

在下一节中，您将进一步学习排序键，并使用它们按嵌套值进行排序。

[*Remove ads*](/account/join/)

### 使用排序关键字选择嵌套值

您还可以更进一步，使用排序键选择可能存在或不存在的嵌套值，如果不存在，则返回默认值:

```py
data = {
    193: {"name": "John", "age": 30, "skills": {"python": 8, "js": 7}},
    209: {"name": "Bill", "age": 15, "skills": {"python": 6}},
    746: {"name": "Jane", "age": 58, "skills": {"js": 2, "python": 5}},
    109: {"name": "Jill", "age": 83, "skills": {"java": 10}},
    984: {"name": "Jack", "age": 28, "skills": {"c": 8, "assembly": 7}},
    765: {"name": "Penelope", "age": 76, "skills": {"python": 8, "go": 5}},
    598: {"name": "Sylvia", "age": 62, "skills": {"bash": 8, "java": 7}},
    483: {"name": "Anna", "age": 24, "skills": {"js": 10}},
    277: {"name": "Beatriz", "age": 26, "skills": {"python": 2, "js": 4}},
}

def get_relevant_skills(item):
    """Get the sum of Python and JavaScript skill"""
    skills = item[1]["skills"]

    # Return default value that is equivalent to no skill
    return skills.get("python", 0) + skills.get("js", 0)

print(sorted(data.items(), key=get_relevant_skills, reverse=True))
```

在本例中，您有一个带有数字键的字典和一个作为值的嵌套字典。您希望按照组合的 Python 和 JavaScript 技能、在`skills`子字典中找到的属性进行排序。

让综合技能排序变得棘手的部分原因是`python`和`js`键并不存在于所有人的`skills`字典中。`skills`字典也是嵌套的。您使用 [`.get()`](https://realpython.com/python-dicts/#dgetkey-default) 来读取密钥，并提供`0`作为缺省值，用于缺少的技能。

您还使用了`reverse`参数，因为您希望顶级 Python 技能首先出现。

**注意:**在这个例子中没有使用 lambda 函数。虽然这是可能的，但它会产生一长串潜在的加密代码:

```py
sorted(
    data.items(),
    key=lambda item: (
        item[1]["skills"].get("python", 0)
        + item[1]["skills"].get("js", 0)
    ),
    reverse=True,
)
```

一个 lambda 函数只能包含一个表达式，所以要在嵌套的`skills`子字典中重复完整的查找。这大大增加了线路长度。

lambda 函数还需要多个链接的方括号(`[]`)索引，这使得阅读起来更加困难。在这个例子中使用 lambda 只节省了几行代码，性能差异可以忽略不计。因此，在这些情况下，使用普通函数通常更有意义。

您已经成功地使用了一个高阶函数作为排序键来按值对字典视图进行排序。那是困难的部分。现在只剩下一个问题需要解决——将`sorted()`生成的列表转换回字典。

### 转换回字典

使用默认行为`sorted()`要解决的唯一问题是它返回一个列表，而不是一个字典。有几种方法可以将元组列表转换回字典。

您可以用一个`for`循环迭代结果，并在每次迭代中填充一个字典:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}
>>> sorted_people = sorted(people.items(), key=lambda item: item[1])

>>> sorted_people_dict = {}
>>> for key, value in sorted_people:
...     sorted_people_dict[key] = value
...

>>> sorted_people_dict
{2: 'Jack', 4: 'Jane', 1: 'Jill', 3: 'Jim'}
```

这种方法让您在决定如何构建词典时拥有绝对的控制权和灵活性。不过，这个方法输入起来可能会很长。如果您对构造字典没有任何特殊要求，那么您可能希望使用[字典构造器](https://docs.python.org/3/library/stdtypes.html#dict):

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}
>>> sorted_people = sorted(people.items(), key=lambda item: item[1])
>>> dict(sorted_people)
{2: 'Jack', 4: 'Jane', 1: 'Jill', 3: 'Jim'}
```

那个好看又小巧！你也可以使用[字典理解](https://realpython.com/iterate-through-dictionary-python/#using-comprehensions)，但是这只在你想要改变字典的形状或者交换键和值的时候才有意义。在下面的理解中，您交换了键和值:

>>>

```py
>>> {
...     value: key ...     for key, value in sorted(people.items(), key=lambda item: item[1]) ... }
...
{'Jack': 2, 'Jane': 4, 'Jill': 1, 'Jim': 3}
```

根据你或你的团队对理解的熟悉程度，这可能比仅仅使用一个普通的`for`循环可读性差。

恭喜你，你已经得到了你的分类词典！你现在可以根据你喜欢的任何标准对它进行分类。

现在您可以对字典进行排序了，您可能想知道使用排序的字典是否会对性能产生影响，或者对于键值数据是否有替代的数据结构。

## 考虑战略和性能问题

在这一节中，您将快速浏览一些性能调整、策略考虑以及关于如何使用键值数据的问题。

**注意**:如果你决定去订购集合，检查一下[分类容器](https://pypi.org/project/sortedcontainers/)包，其中包括一个 [`SortedDict`](https://grantjenks.com/docs/sortedcontainers/sorteddict.html) 。

您将利用 [`timeit`](https://realpython.com/python-timer/#estimating-running-time-with-timeit) 模块来获取一些指标。重要的是要记住，要对性能做出任何可靠的结论，您需要在各种硬件上进行测试，并使用各种类型和大小的样本。

最后，请注意，您不会详细讨论如何使用`timeit`。为此，请查看关于 Python 定时器的[教程。不过，您将有一些示例可以使用。](https://realpython.com/python-timer/)

[*Remove ads*](/account/join/)

### 使用特殊的 Getter 函数来提高性能和可读性

您可能已经注意到，到目前为止，您使用的大多数排序键功能都没有发挥多大作用。这个函数所做的就是从一个元组中获取一个值。创建 getter 函数是一种非常常见的模式，Python 有一种特殊的方法来创建比常规函数更快获取值的特殊函数。

[`itemgetter()`](https://docs.python.org/3/library/operator.html#operator.itemgetter) 函数可以产生高效版本的 getter 函数。

您传递给`itemgetter()`一个参数，它通常是您想要选择的键或索引位置。然后，`itemgetter()`函数将返回一个 getter 对象，您可以像调用函数一样调用它。

没错，就是返回函数的函数。使用`itemgetter()`函数是使用高阶函数的另一个例子。

来自`itemgetter()`的 getter 对象将在传递给它的项目上调用`.__getitem__()`方法。当某个东西调用`.__getitem__()`时，它需要传入要获取什么的键或索引。用于`.__getitem__()`的参数与传递给`itemgetter()`的参数相同:

>>>

```py
>>> item = ("name", "Guido")

>>> from operator import itemgetter

>>> getter = itemgetter(0)
>>> getter(item)
'name'
>>> getter = itemgetter(1)
>>> getter(item)
'Guido'
```

在这个例子中，我们从一个 tuple 开始，类似于作为字典视图的一部分得到的 tuple。

您通过将`0`作为参数传递给`itemgetter()`来创建第一个 getter。当结果 getter 接收到元组时，它返回元组中的第一项——索引`0`处的值。如果你用一个参数`1`调用`itemgetter()`，那么它会得到索引位置`1`的值。

您可以使用这个 itemgetter 作为`sorted()`函数的键:

>>>

```py
>>> from operator import itemgetter

>>> fruit_inventory = [
...     ("banana", 5), ("orange", 15), ("apple", 3), ("kiwi", 0)
... ]

>>> # Sort by key
>>> sorted(fruit_inventory, key=itemgetter(0))
[('apple', 3), ('banana', 5), ('kiwi', 0), ('orange', 15)]

>>> # Sort by value
>>> sorted(fruit_inventory, key=itemgetter(1))
[('kiwi', 0), ('apple', 3), ('banana', 5), ('orange', 15)]

>>> sorted(fruit_inventory, key=itemgetter(2))
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    sorted(fruit_inventory, key=itemgetter(2))
IndexError: tuple index out of range
```

在这个例子中，首先使用带有`0`的`itemgetter()`作为参数。因为它对来自`fruit_inventory`变量的每个元组进行操作，所以它从每个元组中获取第一个元素。然后这个例子演示了用`1`作为参数初始化一个`itemgetter`，它选择了元组中的第二项。

最后，这个例子展示了如果您将`itemgetter()`和`2`一起用作参数会发生什么。因为这些元组只有两个索引位置，所以试图获取索引为`2`的第三个元素会导致一个`IndexError`。

您可以使用由`itemgetter()`产生的函数来代替您到目前为止一直使用的 getter 函数:

>>>

```py
>>> people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}

>>> from operator import itemgetter
>>> sorted(people.items(), key=itemgetter(1))
[(2, 'Jack'), (4, 'Jane'), (1, 'Jill'), (3, 'Jim')]
```

`itemgetter()`函数产生的函数与前面章节中的`value_getter()`函数具有完全相同的效果。你想使用`itemgetter()`的函数的主要原因是因为它更有效。在下一节中，您将开始给出一些数字，说明它的效率究竟有多高。

### 使用`itemgetter()`和测量性能

因此，您最终得到了一个行为类似于前面章节中的原始`value_getter()`的函数，除了从`itemgetter()`返回的版本更有效。您可以使用`timeit`模块来比较它们的性能:

```py
# compare_lambda_vs_getter.py

from timeit import timeit

dict_to_order = {
    1: "requests",
    2: "pip",
    3: "jinja",
    4: "setuptools",
    5: "pandas",
    6: "numpy",
    7: "black",
    8: "pillow",
    9: "pyparsing",
    10: "boto3",
    11: "botocore",
    12: "urllib3",
    13: "s3transfer",
    14: "six",
    15: "python-dateutil",
    16: "pyyaml",
    17: "idna",
    18: "certifi",
    19: "typing-extensions",
    20: "charset-normalizer",
    21: "awscli",
    22: "wheel",
    23: "rsa",
}

sorted_with_lambda = "sorted(dict_to_order.items(), key=lambda item: item[1])"
sorted_with_itemgetter = "sorted(dict_to_order.items(), key=itemgetter(1))"

sorted_with_lambda_time = timeit(stmt=sorted_with_lambda, globals=globals())
sorted_with_itemgetter_time = timeit(
    stmt=sorted_with_itemgetter,
    setup="from operator import itemgetter",
    globals=globals(),
)

print(
    f"""\
{sorted_with_lambda_time=:.2f} seconds
{sorted_with_itemgetter_time=:.2f} seconds
itemgetter is {(
    sorted_with_lambda_time / sorted_with_itemgetter_time
):.2f} times faster"""
)
```

这段代码使用`timeit`模块来比较来自`itemgetter()`的函数和 lambda 函数的排序过程。

从 shell 中运行这个脚本应该会得到与下面类似的结果:

```py
$ python compare_lambda_vs_getter.py
sorted_with_lambda_time=1.81 seconds
sorted_with_itemgetter_time=1.29 seconds
itemgetter is 1.41 times faster
```

大约 40%的节约意义重大！

请记住，在对代码执行进行计时时，系统之间的时间可能会有很大差异。也就是说，在这种情况下，比率应该在系统间相对稳定。

从这个测试的结果可以看出，从性能的角度来看，使用`itemgetter()`更好。另外，它是 Python 标准库的一部分，所以使用它是免费的。

**注意:**在这个测试中，使用 lambda 和普通函数作为排序关键字的区别可以忽略不计。

要不要比较一下这里没有涉及到的一些操作的性能？请务必在评论中分享结果！

现在，您可以从字典排序中获得更多的性能，但是值得后退一步，考虑使用排序的字典作为您的首选数据结构是否是最佳选择。毕竟，排序字典不是一种非常常见的模式。

接下来，你会问自己一些问题，关于你想用你的排序字典做什么，以及它是否是你的用例的最佳数据结构。

[*Remove ads*](/account/join/)

### 判断是否要使用分类词典

如果您正在考虑创建一个排序的键值数据结构，那么您可能需要考虑一些事情。

如果您要将数据添加到字典中，并且希望数据保持有序，那么您最好使用元组列表或字典列表这样的结构:

```py
# Dictionary
people = {3: "Jim", 2: "Jack", 4: "Jane", 1: "Jill"}

# List of tuples
people = [
    (3, "Jim"),
    (2, "Jack"),
    (4, "Jane"),
    (1, "Jill")
]

# List of dictionaries
people = [
    {"id": 3, "name": "Jim"},
    {"id": 2, "name": "Jack"},
    {"id": 4, "name": "Jane"},
    {"id": 1, "name": "Jill"},
]
```

字典列表是最普遍的模式，因为它的跨语言兼容性，被称为[语言互操作性](https://en.wikipedia.org/wiki/Language_interoperability)。

例如，如果您创建了一个 [HTTP REST API](https://realpython.com/api-integration-in-python/) ，那么语言互操作性尤其重要。让你的数据在互联网上可用很可能意味着在 [JSON](https://realpython.com/python-json/) 中序列化它。

如果有人使用 JavaScript 来消费 REST API 中的 JSON 数据，那么等价的数据结构就是一个[对象](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)。有趣的是，JavaScript 对象是没有顺序的*，所以顺序会被打乱！

这种混乱的行为对许多语言来说都是真实的，对象甚至在 JSON 规范中被定义为一种无序的数据结构。因此，如果您在序列化到 JSON 之前仔细订购了字典，那么当它进入大多数其他环境时，这就无关紧要了。

**注意:**标记一个有序的键值对序列可能不仅仅与将 Python 字典序列化为 JSON 相关。想象一下，你的团队中有人习惯了其他语言。有序字典对他们来说可能是一个陌生的概念，所以您可能需要明确您已经创建了一个有序的数据结构。

在 Python 中明确拥有有序字典的一种方式是使用恰当命名的 [`OrderedDict`](https://realpython.com/python-ordereddict/) 。

另一个选择是，如果不需要，就不要担心数据的排序。包括每个对象的`id`、`priority`或其他等价属性足以表达顺序。如果由于某种原因，排序混淆了，那么总会有一种明确的方法来排序:

```py
people = {
    3: {"priority": 2, "name": "Jim"},
    2: {"priority": 4, "name": "Jack"},
    4: {"priority": 1, "name": "Jane"},
    1: {"priority": 2, "name": "Jill"}
}
```

例如，对于一个`priority`属性，很明显`Jane`应该排在第一位。明确你想要的顺序很好地符合了古老的 Python 格言*显式比隐式好*，来自 Python 的[禅。](https://realpython.com/lessons/zen-of-python/)

但是，使用字典列表和字典词典在性能上有什么权衡呢？在下一节中，您将开始获得关于这个问题的一些数据。

### 比较不同数据结构的性能

如果性能是一个考虑因素—例如，也许您将使用大型数据集—那么您应该仔细考虑您将使用字典做什么。

在接下来的几节中，您将寻求回答的两个主要问题是:

1.  你会排序一次，然后进行大量的查找吗？
2.  你会进行多次排序而很少查找吗？

一旦您决定了您的数据结构将遵循什么样的使用模式，那么您就可以使用`timeit`模块来测试性能。这些测量值会随着被测数据的确切形状和大小而有很大变化。

在这个例子中，您将比较字典的字典和字典的列表，看看它们在性能方面有什么不同。您将使用以下示例数据对排序操作和查找操作进行计时:



```py
# samples.py

dictionary_of_dictionaries = {
    1: {"first_name": "Dorthea", "last_name": "Emmanuele", "age": 29},
    2: {"first_name": "Evelina", "last_name": "Ferras", "age": 91},
    3: {"first_name": "Frederica", "last_name": "Livesay", "age": 99},
    4: {"first_name": "Murray", "last_name": "Linning", "age": 36},
    5: {"first_name": "Annette", "last_name": "Garioch", "age": 93},
    6: {"first_name": "Rozamond", "last_name": "Todd", "age": 36},
    7: {"first_name": "Tiffi", "last_name": "Varian", "age": 28},
    8: {"first_name": "Noland", "last_name": "Cowterd", "age": 51},
    9: {"first_name": "Dyana", "last_name": "Fallows", "age": 100},
    10: {"first_name": "Diahann", "last_name": "Cutchey", "age": 44},
    11: {"first_name": "Georgianne", "last_name": "Steinor", "age": 32},
    12: {"first_name": "Sabina", "last_name": "Lourens", "age": 31},
    13: {"first_name": "Lynde", "last_name": "Colbeck", "age": 35},
    14: {"first_name": "Abdul", "last_name": "Crisall", "age": 84},
    15: {"first_name": "Quintus", "last_name": "Brando", "age": 95},
    16: {"first_name": "Rowena", "last_name": "Geraud", "age": 21},
    17: {"first_name": "Maurice", "last_name": "MacAindreis", "age": 83},
    18: {"first_name": "Pall", "last_name": "O'Cullinane", "age": 79},
    19: {"first_name": "Kermie", "last_name": "Willshere", "age": 20},
    20: {"first_name": "Holli", "last_name": "Tattoo", "age": 88}
}

list_of_dictionaries = [
    {"id": 1, "first_name": "Dorthea", "last_name": "Emmanuele", "age": 29},
    {"id": 2, "first_name": "Evelina", "last_name": "Ferras", "age": 91},
    {"id": 3, "first_name": "Frederica", "last_name": "Livesay", "age": 99},
    {"id": 4, "first_name": "Murray", "last_name": "Linning", "age": 36},
    {"id": 5, "first_name": "Annette", "last_name": "Garioch", "age": 93},
    {"id": 6, "first_name": "Rozamond", "last_name": "Todd", "age": 36},
    {"id": 7, "first_name": "Tiffi", "last_name": "Varian", "age": 28},
    {"id": 8, "first_name": "Noland", "last_name": "Cowterd", "age": 51},
    {"id": 9, "first_name": "Dyana", "last_name": "Fallows", "age": 100},
    {"id": 10, "first_name": "Diahann", "last_name": "Cutchey", "age": 44},
    {"id": 11, "first_name": "Georgianne", "last_name": "Steinor", "age": 32},
    {"id": 12, "first_name": "Sabina", "last_name": "Lourens", "age": 31},
    {"id": 13, "first_name": "Lynde", "last_name": "Colbeck", "age": 35},
    {"id": 14, "first_name": "Abdul", "last_name": "Crisall", "age": 84},
    {"id": 15, "first_name": "Quintus", "last_name": "Brando", "age": 95},
    {"id": 16, "first_name": "Rowena", "last_name": "Geraud", "age": 21},
    {"id": 17, "first_name": "Maurice", "last_name": "MacAindreis", "age": 83},
    {"id": 18, "first_name": "Pall", "last_name": "O'Cullinane", "age": 79},
    {"id": 19, "first_name": "Kermie", "last_name": "Willshere", "age": 20},
    {"id": 20, "first_name": "Holli", "last_name": "Tattoo", "age": 88}
]
```

每个数据结构都有相同的信息，除了一个是字典的字典结构，另一个是字典的列表。首先，您将获得对这两种数据结构进行排序的一些性能指标。

[*Remove ads*](/account/join/)

### 比较排序的性能

在下面的代码中，您将使用`timeit`来比较通过`age`属性对两个数据结构进行排序所花费的时间:

```py
# compare_sorting_dict_vs_list.py

from timeit import timeit
from samples import dictionary_of_dictionaries, list_of_dictionaries

sorting_list = "sorted(list_of_dictionaries, key=lambda item:item['age'])"
sorting_dict = """
dict(
 sorted(
 dictionary_of_dictionaries.items(), key=lambda item: item[1]['age']
 )
)
"""

sorting_list_time = timeit(stmt=sorting_list, globals=globals())
sorting_dict_time = timeit(stmt=sorting_dict, globals=globals())

print(
    f"""\
{sorting_list_time=:.2f} seconds
{sorting_dict_time=:.2f} seconds
list is {(sorting_dict_time/sorting_list_time):.2f} times faster"""
)
```

这段代码导入样本数据结构，用于对`age`属性进行排序。看起来你好像没有使用来自`samples`的导入，但是这些样本必须在全局[名称空间](https://realpython.com/python-namespaces-scope/)中，这样`timeit`上下文才能访问它们。

在命令行上运行这个测试的代码应该会为您提供一些有趣的结果:

```py
$ python compare_sorting_dict_vs_list.py
sorting_list_time=1.15 seconds
sorting_dict_time=2.26 seconds
list is 1.95 times faster
```

对列表进行排序的速度几乎是对字典视图进行排序然后创建新的排序字典的速度的两倍。因此，如果您计划非常有规律地对数据进行排序，那么元组列表可能比字典更适合您。

**注意:**从这样的单一数据集无法得出多少可靠的结论。此外，对于不同大小或形状的数据，结果可能会有很大差异。

这些例子是让你接触`timeit`模块的一种方式，并开始了解如何以及为什么你可能会使用它。这将为您提供一些测试数据结构所必需的工具，帮助您决定为您的键值对选择哪种数据结构。

如果您需要额外的性能，那么就继续为您的特定数据结构计时。也就是说，当心[过早优化](https://xkcd.com/1445/)！

与列表相比，对字典进行排序的主要开销之一是在排序后重建字典。如果您去掉了外部的`dict()`构造函数，那么您将大大减少执行时间。

在下一节中，您将看到在字典和字典列表中查找值所花费的时间。

### 比较查找的性能

但是，如果您计划使用字典对数据进行一次排序，并且主要使用字典进行查找，那么字典肯定比列表更有意义:

```py
# compare_lookup_dict_vs_list.py

from timeit import timeit
from samples import dictionary_of_dictionaries, list_of_dictionaries

lookups = [15, 18, 19, 16, 6, 12, 5, 3, 9, 20, 2, 10, 13, 17, 4, 14, 11, 7, 8]

list_setup = """
def get_key_from_list(key):
 for item in list_of_dictionaries:
 if item["id"] == key:
 return item
"""

lookup_list = """
for key in lookups:
 get_key_from_list(key)
"""

lookup_dict = """
for key in lookups:
 dictionary_of_dictionaries[key]
"""

lookup_list_time = timeit(stmt=lookup_list, setup=list_setup, globals=globals())
lookup_dict_time = timeit(stmt=lookup_dict, globals=globals())

print(
    f"""\
{lookup_list_time=:.2f} seconds
{lookup_dict_time=:.2f} seconds
dict is {(lookup_list_time / lookup_dict_time):.2f} times faster"""
)
```

这段代码对列表和字典进行了一系列的查找。您会注意到，对于这个列表，您必须编写一个特殊的函数来进行查找。进行列表查找的函数涉及到逐个检查所有列表元素，直到找到目标元素，这并不理想。

从命令行运行这个比较脚本应该会产生一个结果，显示字典查找明显更快:

```py
$ python compare_lookup_dict_vs_list.py
lookup_list_time=6.73 seconds
lookup_dict_time=0.38 seconds
dict is 17.83 times faster
```

快了将近十八倍！那是一大堆。因此，您肯定希望权衡字典查找的高速和数据结构的慢速排序。请记住，这个比率在不同的系统之间可能会有很大的差异，更不用说不同大小的字典或列表可能带来的差异了。

不过，无论你如何分割，字典查找肯定会更快。也就是说，如果你只是在做查找，那么你可以用一个普通的未排序的字典来做。在这种情况下，为什么需要一个分类词典呢？在评论中留下你的用例吧！

**注意:**您可以尝试优化列表查找，例如通过实现[二分搜索法算法](https://realpython.com/binary-search-python/)来减少列表查找的时间。然而，只有在列表很大的情况下，好处才会变得明显。

对于这里测试的列表大小，使用带有`bisect`模块的二分搜索法要比常规的`for`循环慢得多。

现在您应该对存储键值数据的两种方法之间的一些权衡有了一个相对较好的想法。您可以得出的结论是，大多数时候，如果您想要一个排序的数据结构，那么您可能应该避开字典，主要是出于语言互操作性的原因。

也就是说，给[格兰特·詹克斯的](https://grantjenks.com/)前面提到的[排序词典](https://grantjenks.com/docs/sortedcontainers/sorteddict.html)一个尝试。它使用一些巧妙的策略来规避典型的性能缺陷。

对于排序的键值数据结构，你有什么有趣的或者高性能的实现吗？在评论中分享它们，以及你的排序字典的用例！

[*Remove ads*](/account/join/)

## 结论

您已经从对字典进行排序的最基本方法发展到了一些考虑对键-值对进行排序的高级方法。

**在本教程中，您已经**:

*   复习了 **`sorted()`** 功能
*   发现的字典**视图**
*   了解字典在排序过程中如何转换为列表
*   指定的**排序关键字**按值、关键字或嵌套属性对字典进行排序
*   使用字典**的理解**和`dict()` **的构造器**来重建你的字典
*   考虑一个排序的字典是否是你的**键值数据**的正确的**数据结构**

现在，您不仅可以根据您可能想到的任何标准对词典进行排序，还可以判断排序后的词典是否是您的最佳选择。

在下面的评论中分享你的排序字典用例以及性能比较！

**免费下载:** [点击这里下载代码](https://realpython.com/bonus/sort-python-dictionary-code/)，您将在本教程中使用它对键-值对进行排序。*********
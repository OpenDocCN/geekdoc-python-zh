# 如何在 Python 中使用 sorted()和 sort()

> 原文：<https://realpython.com/python-sort/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解: [**用 Python 排序数据**](/courses/python-sorting-data/)

在某些时候，所有程序员都必须编写代码来对项目或数据进行排序。排序对于应用程序中的用户体验至关重要，无论是按时间戳对用户最近的活动进行排序，还是按姓氏的字母顺序排列电子邮件收件人列表。Python 排序功能提供了强大的特性，可以进行基本排序或在粒度级别定制排序。

在本指南中，您将学习如何在不同的[数据结构](https://realpython.com/python-data-structures/)中对各种类型的数据进行排序，定制顺序，并使用 Python 中两种不同的排序方法。

**本教程结束时，你将知道如何:**

*   在数据结构上实现基本的 Python 排序和排序
*   区分`sorted()`和`.sort()`
*   根据独特的要求在代码中自定义复杂的排序顺序

对于本教程，你需要对[列表和元组](https://realpython.com/python-lists-tuples/)以及[集合](https://realpython.com/python-sets/)有一个基本的了解。这些数据结构将在本教程中使用，并将对它们执行一些基本操作。此外，本教程使用 Python 3，因此如果您使用 Python 2，本教程中的示例输出可能会略有不同。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 用`sorted()` 对数值进行排序

要开始使用 Python 排序，首先要了解如何对数字数据和[字符串](https://realpython.com/python-strings/)数据进行排序。

[*Remove ads*](/account/join/)

### 分类编号

您可以使用 Python 通过使用`sorted()`来对列表进行排序。在这个例子中，定义了一个整数列表，然后用`numbers` [变量](https://realpython.com/python-variables/)作为参数调用`sorted()`:

>>>

```py
>>> numbers = [6, 9, 3, 1]
>>> sorted(numbers)
[1, 3, 6, 9]
>>> numbers
[6, 9, 3, 1]
```

这段代码的输出是一个新的排序列表。打印原始变量时，初始值不变。

这个例子展示了`sorted()`的四个重要特征:

1.  不必定义函数`sorted()`。这是一个内置函数，在 Python 的标准安装中可用。
2.  没有附加参数的`sorted()`以升序对`numbers`中的值进行排序，即从最小到最大。
3.  原始的`numbers`变量没有改变，因为`sorted()`提供了排序的输出，并且没有就地改变原始值。
4.  当调用`sorted()`时，它提供一个有序列表作为返回值。

这最后一点意味着`sorted()`可以用在一个列表上，并且输出可以立即赋给一个变量:

>>>

```py
>>> numbers = [6, 9, 3, 1]
>>> numbers_sorted = sorted(numbers)
>>> numbers_sorted
[1, 3, 6, 9]
>>> numbers
[6, 9, 3, 1]
```

在这个例子中，现在有一个新的变量`numbers_sorted`存储了`sorted()`的输出。

您可以通过在`sorted()`上调用`help()`来确认所有这些观察结果。可选参数`key`和`reverse`将在后面的教程中介绍:

>>>

```py
>>> # Python 3
>>> help(sorted)
Help on built-in function sorted in module builtins:

sorted(iterable, /, *, key=None, reverse=False)
 Return a new list containing all items from the iterable in ascending order.

 A custom key function can be supplied to customize the sort order, and the
 reverse flag can be set to request the result in descending order.
```

**技术细节:**如果您正在从 Python 2 过渡，并且熟悉它的同名功能，那么您应该知道 Python 3 中的几个重要变化:

1.  Python 3 的`sorted()`没有`cmp`参数。相反，只有`key`用于引入定制排序逻辑。
2.  `key`和`reverse`必须作为关键字参数传递，不像在 Python 2 中，它们可以作为位置参数传递。

如果你需要将 Python 2 的`cmp`函数转换成`key`函数，那么就去看看`functools.cmp_to_key()`。本教程不会涵盖任何使用 Python 2 的例子。

`sorted()`可以非常相似地用于元组和集合:

>>>

```py
>>> numbers_tuple = (6, 9, 3, 1)
>>> numbers_set = {5, 5, 10, 1, 0}
>>> numbers_tuple_sorted = sorted(numbers_tuple)
>>> numbers_set_sorted = sorted(numbers_set)
>>> numbers_tuple_sorted
[1, 3, 6, 9]
>>> numbers_set_sorted
[0, 1, 5, 10]
```

注意即使输入是一个集合和一个元组，输出也是一个列表，因为`sorted()`根据定义返回一个新列表。如果返回的对象需要匹配输入类型，则可以将其转换为新类型。如果试图将结果列表转换回集合，请小心，因为根据定义，集合是无序的:

>>>

```py
>>> numbers_tuple = (6, 9, 3, 1)
>>> numbers_set = {5, 5, 10, 1, 0}
>>> numbers_tuple_sorted = sorted(numbers_tuple)
>>> numbers_set_sorted = sorted(numbers_set)
>>> numbers_tuple_sorted
[1, 3, 6, 9]
>>> numbers_set_sorted
[0, 1, 5, 10]
>>> tuple(numbers_tuple_sorted)
(1, 3, 6, 9)
>>> set(numbers_set_sorted)
{0, 1, 10, 5}
```

转换为`set`时的`numbers_set_sorted`值没有像预期的那样排序。另一个变量，`numbers_tuple_sorted`，保留了排序的顺序。

### 排序字符串

类型的排序类似于列表和元组等其他可迭代对象。下面的例子显示了`sorted()`如何遍历传递给它的值中的每个字符，并在输出中对它们进行排序:

>>>

```py
>>> string_number_value = '34521'
>>> string_value = 'I like to sort'
>>> sorted_string_number = sorted(string_number_value)
>>> sorted_string = sorted(string_value)
>>> sorted_string_number
['1', '2', '3', '4', '5']
>>> sorted_string
[' ', ' ', ' ', 'I', 'e', 'i', 'k', 'l', 'o', 'o', 'r', 's', 't', 't']
```

`sorted()`将把一个`str`当作一个列表，遍历每个元素。在`str`中，每个元素代表`str`中的每个字符。`sorted()`不会区别对待一个句子，它会对每个字符进行排序，包括空格。

`.split()`可以改变这种行为并清理输出，`.join()`可以把它们全部放回一起。我们将很快介绍输出的具体顺序以及为什么会这样:

>>>

```py
>>> string_value = 'I like to sort'
>>> sorted_string = sorted(string_value.split())
>>> sorted_string
['I', 'like', 'sort', 'to']
>>> ' '.join(sorted_string)
'I like sort to'
```

这个例子中的原始句子被转换成一个单词列表，而不是将其作为一个`str`。该列表然后被排序和组合以再次形成一个`str`而不是一个列表。

[*Remove ads*](/account/join/)

## Python 排序的局限性和陷阱

值得注意的是，在使用 Python 对整数以外的值进行排序时，会出现一些限制和奇怪的行为。

### 具有不可比数据类型的列表不能是`sorted()`

有些数据类型仅仅使用`sorted()`是无法相互比较的，因为它们差异太大。如果您试图在包含不可比数据的列表上使用`sorted()`，Python 将返回一个错误。在本例中，同一列表中的一个 [`None`](https://realpython.com/null-in-python/) 和一个`int`由于不兼容而无法排序:

>>>

```py
>>> mixed_types = [None, 0]
>>> sorted(mixed_types)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'int' and 'NoneType'
```

这个错误说明了为什么 Python 不能对给它的值进行排序。它试图通过使用小于运算符(`<`)来确定哪个值在排序顺序中较低，从而对值进行排序。您可以通过手动比较这两个值来复制此错误:

>>>

```py
>>> None < 0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

当您试图在不使用`sorted()`的情况下比较两个不可比较的值时，会抛出相同的`TypeError`。

如果列表中的值可以比较并且不会抛出`TypeError`，那么列表就可以排序。这可以防止对具有本质上不可排序的值的可迭代项进行排序，并产生可能没有意义的输出。

比如数字`1`应该在单词`apple`之前吗？然而，如果 iterable 包含的整数和字符串组合都是[数字](https://realpython.com/python-numbers/)，那么可以通过使用[列表理解](https://realpython.com/list-comprehension-python/)将它们转换为可比较的数据类型:

>>>

```py
>>> mixed_numbers = [5, "1", 100, "34"]
>>> sorted(mixed_numbers)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'str' and 'int'
>>> # List comprehension to convert all values to integers
>>> [int(x) for x in mixed_numbers]
[5, 1, 100, 34]
>>> sorted([int(x) for x in mixed_numbers])
[1, 5, 34, 100]
```

`mixed_numbers`中的每个元素都调用了`int()`来[将任何`str`值转换为`int`值](https://realpython.com/convert-python-string-to-int/)。然后调用`sorted()`，它可以成功地比较每个元素，并提供一个排序后的输出。

Python 也可以隐式地将一个值转换成另一种类型。在下面的例子中，`1 <= 0`的求值是假语句，所以求值的输出将是`False`。数字`1`可以转换为`True`作为`bool`类型，而`0`转换为`False`。

尽管列表中的元素看起来不同，但它们都可以被转换成[布尔值](https://realpython.com/python-boolean/) ( `True`或`False`)，并使用`sorted()`进行相互比较:

>>>

```py
>>> similar_values = [False, 0, 1, 'A' == 'B', 1 <= 0]
>>> sorted(similar_values)
[False, 0, False, False, 1]
```

`'A' == 'B'`和`1 <= 0`被转换为`False`并在有序输出中返回。

这个例子说明了排序的一个重要方面:**排序稳定性**。在 Python 中，当您对相等的值进行排序时，它们将在输出中保持其原始顺序。即使`1`移动了，所有其他的值都是相等的，所以它们保持它们相对于彼此的原始顺序。在下面的示例中，所有值都被视为相等，并将保留其原始位置:

>>>

```py
>>> false_values = [False, 0, 0, 1 == 2, 0, False, False]
>>> sorted(false_values)
[False, 0, 0, False, 0, False, False]
```

如果检查原始顺序和排序后的输出，您会看到`1 == 2`被转换为`False`，所有排序后的输出都是原始顺序。

### 对字符串排序时，大小写很重要

`sorted()`可用于字符串列表，按升序对值进行排序，默认情况下按字母顺序排列:

>>>

```py
>>> names = ['Harry', 'Suzy', 'Al', 'Mark']
>>> sorted(names)
['Al', 'Harry', 'Mark', 'Suzy']
```

然而，Python 使用每个字符串中第一个字母的 [Unicode 码位](https://en.wikipedia.org/wiki/Code_point)来确定升序排序。这意味着`sorted()`不会将名字`Al`和`al`视为相同。这个例子使用`ord()`返回每个字符串中第一个字母的 Unicode 码位:

>>>

```py
>>> names_with_case = ['harry', 'Suzy', 'al', 'Mark']
>>> sorted(names_with_case)
['Mark', 'Suzy', 'al', 'harry']
>>> # List comprehension for Unicode Code Point of first letter in each word
>>> [(ord(name[0]), name[0]) for name in sorted(names_with_case)]
[(77, 'M'), (83, 'S'), (97, 'a'), (104, 'h')]
```

`name[0]`正在返回`sorted(names_with_case)`的每个元素的第一个字符，`ord()`正在提供 Unicode 码位。尽管在字母表中`a`在`M`之前，但是`M`的码位在`a`之前，所以排序后的输出首先是`M`。

如果第一个字母相同，那么`sorted()`将使用第二个字符来确定顺序，如果第三个字符相同，则使用第三个字符，依此类推，直到字符串结束:

>>>

```py
>>> very_similar_strs = ['hhhhhd', 'hhhhha', 'hhhhhc','hhhhhb']
>>> sorted(very_similar_strs)
['hhhhha', 'hhhhhb', 'hhhhhc', 'hhhhhd']
```

除了最后一个字符，`very_similar_strs`的每个值都是相同的。`sorted()`会比较字符串，由于前五个字符相同，所以输出会以第六个字符为基础。

包含相同值的字符串将按从短到长的顺序排序，因为较短的字符串没有可与较长的字符串进行比较的元素:

>>>

```py
>>> different_lengths = ['hhhh', 'hh', 'hhhhh','h']
>>> sorted(different_lengths)
['h', 'hh', 'hhhh', 'hhhhh']
```

最短的字符串`h`排在最前面，最长的字符串`hhhhh`排在最后。

[*Remove ads*](/account/join/)

## 使用带有`reverse`参数的`sorted()`

如`sorted()`的`help()`文档所示，有一个名为`reverse`的可选关键字参数，它将根据分配给它的布尔值改变排序行为。如果`reverse`被指定为`True`，那么排序将按降序进行:

>>>

```py
>>> names = ['Harry', 'Suzy', 'Al', 'Mark']
>>> sorted(names)
['Al', 'Harry', 'Mark', 'Suzy']
>>> sorted(names, reverse=True)
['Suzy', 'Mark', 'Harry', 'Al']
```

排序逻辑保持不变，这意味着姓名仍然按其首字母排序。但是输出已经反转，关键字`reverse`设置为`True`。

分配`False`时，排序将保持升序。使用`True`或`False`可以使用前面的任何例子来查看反向的行为:

>>>

```py
>>> names_with_case = ['harry', 'Suzy', 'al', 'Mark']
>>> sorted(names_with_case, reverse=True)
['harry', 'al', 'Suzy', 'Mark']
>>> similar_values = [False, 1, 'A' == 'B', 1 <= 0]
>>> sorted(similar_values, reverse=True)
[1, False, False, False]
>>> numbers = [6, 9, 3, 1]
>>> sorted(numbers, reverse=False)
[1, 3, 6, 9]
```

## `sorted()`带着`key`的论调

`sorted()`最强大的组件之一是名为`key`的关键字参数。该参数需要一个函数传递给它，该函数将用于排序列表中的每个值，以确定结果顺序。

为了演示一个基本的例子，让我们假设对一个特定列表进行排序的要求是列表中字符串的长度，从最短到最长。返回字符串长度的函数`len()`将与`key`参数一起使用:

>>>

```py
>>> word = 'paper'
>>> len(word)
5
>>> words = ['banana', 'pie', 'Washington', 'book']
>>> sorted(words, key=len)
['pie', 'book', 'banana', 'Washington']
```

得到的顺序是一个从最短到最长的字符串顺序的列表。列表中每个元素的长度由`len()`决定，然后按升序返回。

当情况不同时，让我们回到前面按首字母排序的例子。`key`可以通过将整个字符串转换成小写来解决这个问题:

>>>

```py
>>> names_with_case = ['harry', 'Suzy', 'al', 'Mark']
>>> sorted(names_with_case)
['Mark', 'Suzy', 'al', 'harry']
>>> sorted(names_with_case, key=str.lower)
['al', 'harry', 'Mark', 'Suzy']
```

输出值没有被转换成小写，因为`key`没有操作原始列表中的数据。在排序期间，传递给`key`的函数在每个元素上被调用，以确定排序顺序，但是原始值将出现在输出中。

当您使用带有`key`参数的函数时，有两个主要限制。

首先，传递给`key`的函数中所需参数的数量必须是 1。

下面的示例显示了一个带两个参数的加法函数的定义。当在数字列表的`key`中使用该函数时，它会失败，因为它缺少第二个参数。在排序过程中，每次调用`add()`时，它每次只接收列表中的一个元素:

>>>

```py
>>> def add(x, y):
...     return x + y
... 
>>> values_to_add = [1, 2, 3]
>>> sorted(values_to_add, key=add)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: add() missing 1 required positional argument: 'y'
```

第二个限制是与`key`一起使用的函数必须能够处理 iterable 中的所有值。例如，您有一个用字符串表示的数字列表，将在`sorted()`中使用，而`key`将尝试使用`int`将它们转换成数字。如果 iterable 中的值不能转换为整数，那么函数将失败:

>>>

```py
>>> values_to_cast = ['1', '2', '3', 'four']
>>> sorted(values_to_cast, key=int)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: invalid literal for int() with base 10: 'four'
```

每个数值作为一个`str`可以转换成`int`，但是`four`不能。这导致一个`ValueError`被引发，并解释`four`不能被转换为`int`，因为它是无效的。

`key`功能非常强大，因为几乎任何内置或用户定义的函数都可以用来操纵输出顺序。

如果排序要求是根据每个字符串中的最后一个字母对 iterable 排序(如果这个字母是相同的，那么使用下一个字母)，那么可以定义一个[函数](https://realpython.com/lessons/example-function/)，然后在排序中使用。下面的例子定义了一个函数，该函数反转传递给它的字符串，然后该函数被用作`key`的参数:

>>>

```py
>>> def reverse_word(word):
...     return word[::-1]
...
>>> words = ['banana', 'pie', 'Washington', 'book']
>>> sorted(words, key=reverse_word)
['banana', 'pie', 'book', 'Washington']
```

`word[::-1]` slice 语法用于反转字符串。每个元素都将应用`reverse_word()`，排序顺序将基于倒排单词中的字符。

您可以使用在`key`参数中定义的 [`lambda`函数](https://realpython.com/python-lambda/)，而不是编写一个独立的函数。

`lambda`是一个匿名函数，它:

1.  必须内联定义
2.  没有名字
3.  不能包含[语句](https://docs.python.org/3/reference/simple_stmts.html)
4.  将像函数一样执行

在下面的例子中，`key`被定义为一个没有名字的`lambda`,`lambda`采用的参数是`x`,`x[::-1]`是将对参数执行的操作:

>>>

```py
>>> words = ['banana', 'pie', 'Washington', 'book']
>>> sorted(words, key=lambda x: x[::-1])
['banana', 'pie', 'book', 'Washington']
```

在每个元素上调用`x[::-1]`,并反转单词。反转后的输出将用于排序，但仍会返回原始单词。

如果需求改变，顺序也应该颠倒，那么可以在`key`参数旁边使用`reverse`关键字:

>>>

```py
>>> words = ['banana', 'pie', 'Washington', 'book']
>>> sorted(words, key=lambda x: x[::-1], reverse=True)
['Washington', 'book', 'pie', 'banana']
```

当您需要根据属性对`class`对象进行排序时，函数也很有用。如果你有一组学生，需要按照他们的最终成绩从高到低进行排序，那么可以使用一个`lambda`从`class`那里获得`grade`属性:

>>>

```py
>>> from collections import namedtuple

>>> StudentFinal = namedtuple('StudentFinal', 'name grade')
>>> bill = StudentFinal('Bill', 90)
>>> patty = StudentFinal('Patty', 94)
>>> bart = StudentFinal('Bart', 89)
>>> students = [bill, patty, bart]
>>> sorted(students, key=lambda x: getattr(x, 'grade'), reverse=True)
[StudentFinal(name='Patty', grade=94), StudentFinal(name='Bill', grade=90), StudentFinal(name='Bart', grade=89)]
```

这个例子使用 [`namedtuple`](https://realpython.com/python-namedtuple/) 来产生具有`name`和`grade`属性的类。`lambda`在每个元素上调用`getattr()`，并返回`grade`的值。

`reverse`设置为`True`使升序输出翻转为降序，最高等级先排序。

当您在`sorted()`上利用`key`和`reverse`关键字参数时，排序的可能性是无限的。当你为一个小函数使用一个基本的`lambda`时，代码可以保持简洁，或者你可以写一个全新的函数，导入它，并在关键参数中使用它。

[*Remove ads*](/account/join/)

## 用`.sort()` 对数值进行排序

名称非常相似的`.sort()`与内置的`sorted()`有很大不同。它们完成的事情或多或少是一样的，但是`list.sort()`的`help()`文档强调了`.sort()`和`sorted()`之间的两个最关键的区别:

>>>

```py
>>> # Python2
Help on method_descriptor:

sort(...)
 L.sort(cmp=None, key=None, reverse=False) -- stable sort *IN PLACE*;
 cmp(x, y) -> -1, 0, 1

>>> # Python3
>>> help(list.sort)
Help on method_descriptor:

sort(...)
 L.sort(key=None, reverse=False) -> None -- stable sort *IN PLACE*
```

首先，sort 是`list`类的一个方法，只能用于列表。它不是传递了 iterable 的内置函数。

第二，`.sort()`返回`None`并就地修改值。让我们来看看这两种代码差异的影响:

>>>

```py
>>> values_to_sort = [5, 2, 6, 1]
>>> # Try to call .sort() like sorted()
>>> sort(values_to_sort)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sort' is not defined

>>> # Try to use .sort() on a tuple
>>> tuple_val = (5, 1, 3, 5)
>>> tuple_val.sort()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'tuple' object has no attribute 'sort'

>>> # Sort the list and assign to new variable
>>> sorted_values = values_to_sort.sort()
>>> print(sorted_values)
None

>>> # Print original variable
>>> print(values_to_sort)
[1, 2, 5, 6]
```

在这个代码示例中，`.sort()`的运行方式与`sorted()`相比有一些非常显著的不同:

1.  `.sort()`没有有序输出，所以对新变量的赋值只传递了一个`None`类型。
2.  `values_to_sort`列表已经更改到位，不以任何方式维持原有顺序。

这些行为上的差异使得`.sort()`和`sorted()`在代码中绝对不可互换，如果其中一个被错误地使用，它们会产生意想不到的结果。

`.sort()`具有与`sorted()`相同的`key`和`reverse`可选关键字参数，产生相同的健壮功能。在这里，您可以按第三个单词的第二个字母对短语列表进行排序，并反向返回列表:

>>>

```py
>>> phrases = ['when in rome', 
...     'what goes around comes around', 
...     'all is fair in love and war'
...     ]
>>> phrases.sort(key=lambda x: x.split()[2][1], reverse=True)
>>> phrases
['what goes around comes around', 'when in rome', 'all is fair in love and war']
```

在此示例中，使用了一个`lambda`来执行以下操作:

1.  将每个短语分成一个单词列表
2.  找到第三个元素，或者这个例子中的单词
3.  找出这个单词的第二个字母

## 什么时候用`sorted()`什么时候用`.sort()`

你已经看到了`sorted()`和`.sort()`的区别，但是什么时候用哪个呢？

假设即将有一场 5k 比赛:第一届年度 Python 5k。来自比赛的数据需要被捕获和分类。需要捕获的数据是跑步者的围兜号码和完成比赛所用的秒数:

>>>

```py
>>> from collections import namedtuple

>>> Runner = namedtuple('Runner', 'bibnumber duration')
```

随着赛跑者越过终点线，每个`Runner`将被添加到一个名为`runners`的列表中。在 5 公里赛跑中，并非所有选手都同时冲过起跑线，所以第一个冲过终点线的人可能并不是最快的人:

>>>

```py
>>> runners = []
>>> runners.append(Runner('2528567', 1500))
>>> runners.append(Runner('7575234', 1420))
>>> runners.append(Runner('2666234', 1600))
>>> runners.append(Runner('2425234', 1490))
>>> runners.append(Runner('1235234', 1620))
>>> # Thousands and Thousands of entries later...
>>> runners.append(Runner('2526674', 1906))
```

每一次跑步者越过终点线，他们的号码和他们的总持续时间(以秒计)就会被添加到`runners`中。

现在，负责处理结果数据的尽职的程序员看到这个列表，知道前 5 名最快的参与者是获得奖励的获胜者，其余的跑步者将按最快时间排序。

不需要根据各种属性进行多种类型的排序。这份名单的规模是合理的。没有提到将列表存储在某个地方。只需按持续时间排序，抓住持续时间最短的五名参与者:

>>>

```py
>>> runners.sort(key=lambda x: getattr(x, 'duration'))
>>> top_five_runners = runners[:5]
```

程序员选择在`key`参数中使用`lambda`来从每个跑步者那里获得`duration`属性，并使用`.sort()`对`runners`进行排序。`runners`排序后，前 5 个元素存储在`top_five_runners`中。

任务完成！比赛总监过来告诉程序员，由于 Python 的当前版本是 3.7，他们决定每 37 个冲过终点线的人将获得一个免费的运动包。

此时，程序员开始冒汗，因为跑步者的列表已经被不可逆转地改变了。没有办法恢复跑步者的原始名单，按照他们完成的顺序，找到每第三十七个人。

如果您正在处理重要数据，并且原始数据需要恢复的可能性极小，那么`.sort()`不是最佳选择。如果数据是副本，如果它是不重要的工作数据，如果没有人会介意丢失它，因为它可以被检索，那么`.sort()`可能是一个不错的选择。

或者，可以使用`sorted()`并使用相同的`lambda`对跑步者进行分类:

>>>

```py
>>> runners_by_duration = sorted(runners, key=lambda x: getattr(x, 'duration'))
>>> top_five_runners = runners_by_duration[:5]
```

在这个使用`sorted()`的场景中，最初的跑步者列表仍然是完整的，没有被覆盖。找到每 37 个人中穿过终点线的即席要求可以通过与原始值交互来实现:

>>>

```py
>>> every_thirtyseventh_runners = runners[::37]
```

`every_thirtyseventh_runners`是通过使用`runners`上的列表片语法中的步幅创建的，它仍然包含跑步者穿过终点线的原始顺序。

[*Remove ads*](/account/join/)

## 如何在 Python 中排序:结论

如果将`.sort()`和`sorted()`与可选关键字参数`reverse`和`key`一起正确使用，它们可以提供您需要的排序顺序。

当涉及到输出和就地修改时，两者都有非常不同的特征，所以请确保您考虑过任何将使用`.sort()`的应用程序功能或程序，因为它可能会不可挽回地覆盖数据。

对于热衷于寻找排序挑战的 Pythonistas 来说，可以尝试在排序中使用更复杂的数据类型:嵌套的 iterables。此外，您可以随意深入研究内置的开源 Python 代码实现，并阅读 Python 中使用的名为 [Timsort](https://en.wikipedia.org/wiki/Timsort) 的排序算法。如果你想对字典进行排序，那么看看[对 Python 字典进行排序:值、键等等](https://realpython.com/sort-python-dictionary/)。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解: [**用 Python 排序数据**](/courses/python-sorting-data/)*******
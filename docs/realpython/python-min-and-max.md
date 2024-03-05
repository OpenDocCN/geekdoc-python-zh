# Python 的 min()和 max():查找最小和最大值

> 原文：<https://realpython.com/python-min-and-max/>

当你需要在一个[可迭代](https://docs.python.org/3/glossary.html#term-iterable)或一系列**常规参数**中找到**最小**和**最大**值时，Python 的内置`min()`和`max()`函数就派上了用场。尽管这些看起来是相当基本的计算，但它们在现实世界的编程中有许多有趣的用例。您将在这里尝试其中的一些用例。

**在本教程中，您将学习如何:**

*   使用 Python 的`min()`和`max()`来查找数据中的**最小**和**最大**值
*   用单个**可迭代**或任意数量的**常规参数**调用`min()`和`max()`
*   将`min()`和`max()`与**字符串**和**字典**一起使用
*   用 **`key`** 和 **`default`** 参数调整`min()`和`max()`的行为
*   使用**理解**和**生成器表达式**作为`min()`和`max()`的参数

一旦你掌握了这些知识，你就可以准备写一堆展示`min()`和`max()`有用性的实例了。最后，您将用纯 Python 编写自己版本的`min()`和`max()`，这可以帮助您理解这些函数在内部是如何工作的。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

为了最大限度地利用本教程，您应该有一些 Python 编程的前期知识，包括像 [`for`循环](https://realpython.com/python-for-loop/)、[函数](https://realpython.com/defining-your-own-python-function/)、[列表理解](https://realpython.com/list-comprehension-python/)和[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)这样的主题。

## Python 的`min()`和`max()`函数入门

Python 包括几个[内置函数](https://docs.python.org/3/library/functions.html)，让你的生活更加愉快和富有成效，因为这意味着你不需要重新发明轮子。这些功能的两个例子是`min()`和`max()`。它们大多适用于[的可迭代对象](https://docs.python.org/3/glossary.html#term-iterable)，但是你也可以将它们与多个常规参数一起使用。他们的工作是什么？他们负责*在他们的输入数据中找到最小和最大的值*。

无论您使用的是 Python 的`min()`还是`max()`，您都可以使用该函数来实现两种略有不同的行为。每个的标准行为是[通过直接比较输入数据返回](https://realpython.com/python-return-statement/)最小值或最大值。另一种行为是在找到最小和最大值之前，使用单参数函数来修改比较标准。

为了探究`min()`和`max()`的标准行为，您可以通过使用单个 iterable 作为参数或者使用两个或更多常规参数来调用每个函数。这就是你马上要做的。

[*Remove ads*](/account/join/)

### 用一个可迭代的参数调用`min()`和`max()`

内置的`min()`和`max()`有两个不同的签名，允许你用一个 iterable 作为它们的第一个参数或者用两个或更多的常规参数来调用它们。接受单个可迭代参数的签名如下所示:

```py
min(iterable, *[, default, key]) -> minimum_value

max(iterable, *[, default, key]) -> maximum_value
```

这两个函数都需要一个名为`iterable`的参数，并分别返回最小值和最大值。他们还接受两个可选的[关键字-唯一的](https://realpython.com/defining-your-own-python-function/#keyword-only-arguments)参数:`default`和`key`。

**注意:**在上述签名中，星号(`*`)表示后面的参数是仅关键字的参数，而方括号(`[]`)表示包含的内容是可选的。

下面是对`min()`和`max()`的参数的总结:

| 争吵 | 描述 | 需要 |
| --- | --- | --- |
| `iterable` | 接受一个可迭代对象，比如一个[列表、元组](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/python-dicts/)或[字符串](https://realpython.com/python-strings/) | 是 |
| `default` | 保存输入 iterable 为空时要返回的值 | 不 |
| `key` | 接受单参数函数来自定义比较标准 | 不 |

在本教程的后面，您将了解更多关于可选的`default`和`key`参数。现在，只关注`iterable`参数，这是一个必需的参数，它利用了 Python 中`min()`和`max()`的标准行为:

>>>

```py
>>> min([3, 5, 9, 1, -5])
-5

>>> min([])
Traceback (most recent call last):
    ...
ValueError: min() arg is an empty sequence

>>> max([3, 5, 9, 1, -5])
9

>>> max([])
Traceback (most recent call last):
    ...
ValueError: max() arg is an empty sequence
```

在这些例子中，你用一个整数列表[和一个空列表](https://realpython.com/python-numbers/)调用`min()`和`max()`。对`min()`的第一个调用返回输入列表中最小的数字`-5`。相反，对`max()`的第一次调用返回列表中最大的数字，即`9`。如果您将一个空迭代器传递给`min()`或`max()`，那么您会得到一个`ValueError`，因为在空迭代器上没有任何事情可做。

关于`min()`和`max()`需要注意的一个重要细节是，输入 iterable 中的所有值必须是可比较的。否则，您会得到一个错误。例如，数值工作正常:

>>>

```py
>>> min([3, 5.0, 9, 1.0, -5])
-5

>>> max([3, 5.0, 9, 1.0, -5])
9
```

这些例子结合了对`min()`和`max()`的调用中的`int`和`float`号码。在这两种情况下，您都会得到预期的结果，因为这些数据类型是可比较的。

但是，如果把[字符串](https://realpython.com/python-strings/)和数字混在一起会怎么样？看看下面的例子:

>>>

```py
>>> min([3, "5.0", 9, 1.0, "-5"])
Traceback (most recent call last):
    ...
TypeError: '<' not supported between instances of 'str' and 'int'

>>> max([3, "5.0", 9, 1.0, "-5"])
Traceback (most recent call last):
    ...
TypeError: '>' not supported between instances of 'str' and 'int'
```

不能用不可比较类型的 iterable 作为参数调用`min()`或`max()`。在这个例子中，一个函数试图比较一个数字和一个字符串，这就像比较苹果和橘子一样。最后的结果是你得到了一个`TypeError`。

### 使用多个参数调用`min()`和`max()`

`min()`和`max()`的第二个签名允许您使用任意数量的参数调用它们，前提是您至少使用两个参数。该签名具有以下形式:

```py
min(arg_1, arg_2[, ..., arg_n], *[, key]) -> minimum_value

max(arg_1, arg_2[, ..., arg_n], *[, key]) -> maximum_value
```

同样，这些函数分别返回最小值和最大值。以下是上述签名中参数的含义:

| 争吵 | 描述 | 需要 |
| --- | --- | --- |
| `arg_1, arg_2, ..., arg_n` | 接受任意数量的常规参数进行比较 | 是(至少两个) |
| `key` | 采用单参数函数来自定义比较标准 | 不 |

这个`min()`或`max()`的变体没有`default`自变量。您必须在调用中提供至少两个参数，函数才能正常工作。因此，不需要一个`default`值，因为为了找到最小值或最大值，你总是有至少两个值要比较。

要尝试这种替代签名，请运行以下示例:

>>>

```py
>>> min(3, 5, 9, 1, -5)
-5

>>> max(3, 5, 9, 1, -5)
9
```

可以用两个或多个常规参数调用`min()`或`max()`。同样，您将分别获得输入数据中的最小值或最大值。唯一的条件是参数必须具有可比性。

[*Remove ads*](/account/join/)

## 将`min()`和`max()`与字符串和字符串的可重复项一起使用

默认情况下，`min()`和`max()`可以处理具有可比性的值。否则，你会得到一个`TypeError`，你已经知道了。到目前为止，您已经看到了在 iterable 中或者作为多个常规参数使用数值的例子。

使用带有数值的`min()`和`max()`可以说是这些函数最常见和最有用的用例。但是，您也可以将函数用于字符串和字符串的可重复项。在这些情况下，字符的字母顺序将决定最终结果。

例如，您可以使用`min()`和`max()`在一些文本中查找最小和最大的字母。在此上下文中，*最小*表示最接近字母表的开头，*最大*表示最接近字母表的结尾:

>>>

```py
>>> min("abcdefghijklmnopqrstuvwxyz")
'a'

>>> max("abcdefghijklmnopqrstuvwxyz")
'z'

>>> min("abcdWXYZ")
'W'

>>> max("abcdWXYZ")
'd'
```

如前所述，在前两个例子中，`min()`返回`'a'`，`max()`返回`'z'`。然而，在第二对例子中，`min()`返回`'W'`，而`max()`返回`'d'`。为什么？因为在 Python 的[默认字符集](https://docs.python.org/3/howto/unicode.html#the-string-type)、 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) 中，大写字母排在小写字母之前。

**注意:** Python 内部将字符串视为字符的可重复项。因此，用字符串作为参数调用`min()`或`max()`就像用单个字符的 iterable 调用函数一样。

使用带有字符串的`min()`或`max()`作为参数不仅限于字母。您可以使用包含当前字符集中任何可能字符的字符串。例如，如果您只处理一组 [ASCII](https://en.wikipedia.org/wiki/ASCII) 字符，那么最小的字符就是最接近 [ASCII 表](https://en.wikipedia.org/wiki/ASCII#Control_code_chart)开头的字符。相比之下，最大的字符是最靠近表尾的字符。

对于其他字符集，如 UTF-8，`min()`和`max()`的行为类似:

>>>

```py
>>> # UTF-8 characters

>>> min("abc123ñ")
'1'

>>> max("abc123ñ")
'ñ'
```

在后台，`min()`和`max()`使用字符的数值来查找输入字符串中的最小和最大字符。例如，在 [Unicode](https://realpython.com/python-encodings-guide/) 字符表中，大写`A`的数值小于小写`a`:

>>>

```py
>>> ord("A")
65

>>> ord("a")
97
```

Python 内置的 [`ord()`](https://docs.python.org/3/library/functions.html#ord) 函数接受一个 Unicode 字符，并返回一个表示该字符的 **Unicode 码位**的整数。在这些例子中，大写`"A"`的码位低于小写`"a"`的码位。

这样，当您用两个字母调用`min()`和`max()`时，您会得到与这些字母的基本 Unicode 码位顺序相匹配的结果:

>>>

```py
>>> min("aA")
'A'

>>> max("aA")
'a'
```

是什么让`"A"`比`"a"`小？最简单的答案是字母的 Unicode 码位。可以在键盘上键入的所有字符以及许多其他字符在 Unicode 表中都有自己的代码点。在使用`min()`和`max()`时，Python 使用这些代码点来确定最小和最大字符。

最后，还可以用字符串的 iterables 或多个字符串参数调用`min()`和`max()`。同样，两个函数都将通过按字母顺序比较字符串来确定它们的返回值:

>>>

```py
>>> min(["Hello", "Pythonista", "and", "welcome", "world"])
'Hello'

>>> max(["Hello", "Pythonista", "and", "welcome", "world"])
'world'
```

为了在一个可迭代的字符串中找到最小或最大的字符串，`min()`和`max()`根据首字符的代码点按字母顺序比较所有的字符串。

在第一个例子中，大写的`"H"`出现在 Unicode 表中的`"P"`、`"a"`和`"w"`之前。所以，`min()`马上断定`"Hello"`是最小的字符串。在第二个例子中，小写的`"w"`出现在所有其他字符串的首字母之后。

注意有两个单词是以`"w"`、`"welcome"`和`"world"`开头的。因此，Python 开始查看每个单词的第二个字母。结果是`max()`返回`"world"`，因为`"o"`在`"e"`之后。

[*Remove ads*](/account/join/)

## 用`min()`和`max()` 处理字典

当使用`min()`和`max()`处理 Python 字典时，您需要考虑如果您直接使用字典，那么这两个函数都将在键上操作:

>>>

```py
>>> prices = {
...    "banana": 1.20,
...    "pineapple": 0.89,
...    "apple": 1.57,
...    "grape": 2.45,
... }

>>> min(prices)
'apple'

>>> max(prices)
'pineapple'
```

在这些例子中，`min()`返回`prices`中按字母顺序最小的键，`max()`返回最大的键。您可以在输入词典上使用 [`.keys()`](https://realpython.com/python-dicts/#dkeys) 方法获得相同的结果:

>>>

```py
>>> min(prices.keys())
'apple'

>>> max(prices.keys())
'pineapple'
```

后一个例子和前一个例子之间的唯一区别是，这里的代码更加清晰明了地说明了你在做什么。任何阅读您的代码的人都会很快意识到您想在输入字典中找到最小和最大的键。

另一个常见的需求是在字典中找到最小和最大的值。继续`prices`的例子，假设你想知道最小和最大价格。在这种情况下，可以使用 [`.values()`](https://realpython.com/python-dicts/#dvalues) 的方法:

>>>

```py
>>> min(prices.values())
0.89

>>> max(prices.values())
2.45
```

在这些示例中，`min()`遍历`prices`中的所有值，并找到最低价格。类似地，`max()`遍历`prices`的值并返回最高价格。

最后，您还可以使用输入字典上的 [`.items()`](https://realpython.com/python-dicts/#ditems) 方法来查找最小和最大键-值对:

>>>

```py
>>> min(prices.items())
('apple', 1.57)

>>> max(prices.items())
('pineapple', 2.45)
```

在这种情况下，`min()`和`max()`使用 Python 的内部规则来比较元组，找到输入字典中最小和最大的条目。

Python 逐项比较元组。例如，为了确定`(x1, x2)`是否大于`(y1, y2`，Python 测试了`x1 > y1`。如果这个条件是`True`，那么 Python 断定第一个元组大于第二个元组，而不检查其余的项。相反，如果`x1 < y1`，那么 Python 会得出第一个元组小于第二个元组的结论。

最后，如果`x1 == y1`，那么 Python 使用相同的规则比较第二对条目。注意，在这个上下文中，每个元组的第一项来自字典键，因为字典键是惟一的，所以这些项不能相等。所以，Python 永远不会比较第二个值。

## 用`key`和`default` 调整`min()`和`max()`的标准行为

到目前为止，您已经了解了`min()`和`max()`如何以它们的标准形式工作。在这一节中，您将学习如何通过使用`key`和`default` **关键字参数**来调整这两个函数的标准行为。

`min()`或`max()`的`key`参数允许您提供一个单参数函数，该函数将应用于输入数据中的每个值。目标是修改用于查找最小值或最大值的比较标准。

作为这个特性如何有用的一个例子，假设您有一个字符串形式的数字列表，并且想要找到最小和最大的数字。如果用`min()`和`max()`直接处理列表，那么会得到以下结果:

>>>

```py
>>> min(["20", "3", "35", "7"])
'20'

>>> max(["20", "3", "35", "7"])
'7'
```

这些可能不是你需要或期待的结果。您获得的最小和最大字符串是基于 Python 的字符串比较规则，而不是基于每个字符串的实际数值。

在这种情况下，解决方案是将内置的 [`int()`](https://docs.python.org/3/library/functions.html#int) 函数作为`key`参数传递给`min()`和`max()`，如下例所示:

>>>

```py
>>> min(["20", "3", "35", "7"], key=int)
'3'

>>> max(["20", "3", "35", "7"], key=int)
'35'
```

太好了！现在`min()`或`max()`的结果取决于底层字符串的数值。注意，你不需要打电话给`int()`。您只是传递了没有一对括号的`int`，因为`key`需要一个**函数对象**，或者更准确地说，一个**可调用对象**。

**注意:**Python 中的可调用对象包括函数、方法、类，以及任何提供了 [`.__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) [特殊方法](https://docs.python.org/3/glossary.html#term-special-method)的类的实例。

第二个仅使用关键字的参数是`default`，它允许您定制`min()`或`max()`的标准行为。请记住，该参数仅在使用单个 iterable 作为参数调用函数时可用。

`default`的作用是当用空的 iterable 调用`min()`或`max()`时，提供一个合适的默认值作为其返回值:

>>>

```py
>>> min([], default=42)
42

>>> max([], default=42)
42
```

在这些例子中，输入 iterable 是一个空列表。标准行为是`min()`或`max()`引发一个`ValueError`来抱怨空序列参数。但是，因为您向`default`提供了一个值，所以现在两个函数都返回这个值，而不是引发一个异常并中断您的代码。

[*Remove ads*](/account/join/)

## 将`min()`和`max()`用于理解和生成器表达式

也可以用**列表理解**或[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)作为参数调用`min()`或`max()`。当您需要在找到最小或最大转换值之前转换输入数据时，此功能非常有用。

当您将列表理解输入到`min()`或`max()`中时，结果值将来自转换后的数据，而不是原始数据:

>>>

```py
>>> letters = ["A", "B", "C", "X", "Y", "Z"]

>>> min(letters)
'A'
>>> min([letter.lower() for letter in letters]) 'a'

>>> max(letters)
'Z'
>>> max([letter.lower() for letter in letters]) 'z'
```

对`min()`的第二次调用将列表理解作为参数。这种理解通过对每个字母应用`.lower()`方法来转换`letters`中的原始数据。最终结果是小写的`"a"`，它不存在于原始数据中。关于`max()`的例子也发生了类似的事情。

注意，在列表理解中使用`min()`或`max()`类似于使用`key`参数。主要区别在于，使用 comprehensions，最终结果是转换后的值，而使用`key`，结果来自原始数据:

>>>

```py
>>> letters = ["A", "B", "C", "X", "Y", "Z"]

>>> min([letter.lower() for letter in letters])
'a'

>>> min(letters, key=str.lower)
'A'
```

在这两个例子中，`min()`使用`.lower()`以某种方式修改比较标准。不同之处在于，理解实际上是在进行计算之前转换输入数据，因此结果值来自转换后的数据，而不是原始数据。

列表理解在内存中创建一个完整的列表，这通常是一个浪费的操作。如果您的代码中不再需要结果列表，这一点尤其正确，这可能是`min()`和`max()`的情况。因此，使用一个**生成器表达式**总是更有效。

生成器表达式的语法与列表理解的语法几乎相同:

>>>

```py
>>> letters = ["A", "B", "C", "X", "Y", "Z"]

>>> min(letters)
'A'
>>> min(letter.lower() for letter in letters) 'a'

>>> max(letters)
'Z'
>>> max(letter.lower() for letter in letters) 'z'
```

主要的语法差异是生成器表达式使用圆括号而不是方括号(`[]`)。因为函数调用已经需要括号，所以您只需要从基于理解的例子中去掉方括号，就可以了。与列表理解不同，生成器表达式按需生成条目，这使得它们的内存效率更高。

## 将 Python 的`min()`和`max()`付诸行动

到目前为止，您已经学习了使用`min()`和`max()`在一个可迭代或一系列单个值中寻找最小和最大值的基本知识。您了解了`min()`和`max()`如何处理不同的内置 Python 数据类型，比如数字、字符串和字典。您还探索了如何调整这些函数的标准行为，以及如何将它们用于列表理解和生成器表达式。

现在您已经准备好开始编写一些实际的例子，向您展示如何在您自己的代码中使用`min()`和`max()`。

### 删除列表中最小和最大的数字

首先，您将从一个简短的示例开始，了解如何从一个数字列表中删除最小值和最大值。为此，您可以在输入列表上调用`.remove()`。根据您的需要，您将使用`min()`或`max()`来选择您将从底层列表中移除的值:

>>>

```py
>>> sample = [4, 5, 7, 6, -12, 4, 42]

>>> sample.remove(min(sample))
>>> sample
[4, 5, 7, 6, 4, 42]

>>> sample.remove(max(sample))
>>> sample
[4, 5, 7, 6, 4]
```

在这些示例中，`sample`中的最小值和最大值可能是您想要移除的[异常值](https://en.wikipedia.org/wiki/Outlier)数据点，以便它们不会影响您的进一步分析。这里，`min()`和`max()`向`.remove()`提供参数。

### 构建最小值和最大值列表

现在假设您有一个表示数值矩阵的列表，您需要构建包含输入矩阵中每一行的最小和最大值的列表。为此，您可以使用`min()`和`max()`以及一个列表理解:

>>>

```py
>>> matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

>>> [min(x) for x in matrix]
[1, 4, 7]

>>> [max(x) for x in matrix]
[3, 6, 9]
```

第一个理解遍历`matrix`中的子列表，并使用`min()`构建一个包含每个子列表中最小值的列表。第二个理解执行类似的任务，但是使用`max()`来创建一个包含来自`matrix`中的子列表的最大值的列表。

尽管`min()`和`max()`提供了一种快速的方法来处理本节中的例子，但是在处理 Python 中的矩阵时，强烈推荐使用 [NumPy](https://realpython.com/numpy-tutorial/) 库，因为 NumPy 有专门的优化工具来完成这项工作。

[*Remove ads*](/account/join/)

### 将值剪切到区间边缘

有时，您有一个数值列表，并希望将它们裁剪到给定区间的边缘或界限。例如，如果给定的值大于间隔的上限，那么您需要将其向下转换到该限制。要做这个操作，可以用`min()`。

等等！为什么是`min()`？你在处理大额交易，是吗？关键是您需要将每个大值与区间的上限进行比较，然后选择两者中较小的一个。实际上，您将所有大值设置为一个规定的上限:

>>>

```py
>>> # Clip values to the largest interval's edge

>>> upper = 100
>>> numbers = [42, 78, 200, -230, 25, 142]

>>> [min(number, upper) for number in numbers] [42, 78, 100, -230, 25, 100]
```

对`min()`的调用将每个数字与区间的上限进行比较。如果目标数大于极限，则`min()`返回极限。实际效果是，所有大于限制值的值现在都被限制到限制值。在这个例子中，数字`200`和`142`被裁剪为`100`，这是区间的上限。

相反，如果你想将小值限制在区间的下限，那么你可以使用`max()`，如下例所示:

>>>

```py
>>> # Clip values to the smallest interval's edge

>>> lower = 10
>>> numbers = [42, 78, 200, -230, 25, 142]

>>> [max(number, lower) for number in numbers] [42, 78, 200, 10, 25, 142]
```

对`max()`的调用将小值限制在区间的下限。为了进行这种裁剪，`max()`比较当前数字和间隔的限制，以找到最大值。在这个例子中，`-230`是唯一被截取的数字。

最后，您可以通过组合`min()`和`max()`来一次运行这两个操作。以下是如何做到这一点:

>>>

```py
>>> # Clipping values to 10 - 100

>>> lower, upper = 10, 100
>>> numbers = [42, 78, 100, -230, 25, 142]

>>> [max(min(number, upper), lower) for number in numbers] [42, 78, 100, 10, 25, 100]
```

为了截取所有超出区间限制的值，这种理解结合了`min()`和`max()`。对`min()`的调用将当前值与区间的上限进行比较，而对`max()`的调用将结果与下限进行比较。最终结果是，低于或大于相应限制的值被限制在限制本身。

这种理解类似于 NumPy 的 [`clip()`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) 函数，它采用一个[数组](https://numpy.org/doc/stable/reference/generated/numpy.array.html)和目标区间的限制，然后将区间外的所有值裁剪到区间的边缘。

### 寻找最近的点

现在假设您有一个元组列表，其中包含表示[笛卡尔](https://en.wikipedia.org/wiki/Cartesian_coordinate_system)点的值对。您希望处理所有这些点对，并找出哪一对点之间的距离最小。在这种情况下，您可以执行如下操作:

>>>

```py
>>> import math

>>> point_pairs = [
...     ((12, 5), (9, 4)),
...     ((2, 5), (3, 7)),
...     ((4, 11), (15, 2))
... ]

>>> min(point_pairs, key=lambda points: math.dist(*points))
((2, 5), (3, 7))
```

在本例中，您首先导入 [`math`](https://realpython.com/python-math-module/) 来访问 [`dist()`](https://docs.python.org/3/library/math.html#math.dist) 。该函数返回两个点 *p* 和 *q* 之间的[欧几里德距离](https://en.wikipedia.org/wiki/Euclidean_distance)，每个点都以坐标序列的形式给出。这两点必须有相同的维数。

`min()`函数通过它的`key`参数发挥它的魔力。在这个例子中，`key`使用了一个 [`lambda`](https://realpython.com/python-lambda/) 函数来计算两点之间的距离。该函数成为`min()`寻找两点间距离最小的一对点的比较标准。

在这个例子中，您需要一个`lambda`函数，因为`key`需要一个单参数函数，而`math.dist()`需要两个参数。因此，`lambda`函数接受一个参数`points`，然后将其解包成两个参数，并输入到`math.dist()`。

### 识别便宜和昂贵的产品

现在假设您有一个包含几种产品的名称和价格的字典，并且您想要确定最便宜和最贵的产品。在这种情况下，您可以使用`.items()`和一个适当的`lambda`函数作为`key`参数:

>>>

```py
>>> prices = {
...    "banana": 1.20,
...    "pineapple": 0.89,
...    "apple": 1.57,
...    "grape": 2.45,
... }

>>> min(prices.items(), key=lambda item: item[1])
('pineapple', 0.89)

>>> max(prices.items(), key=lambda item: item[1])
('grape', 2.45)
```

在这个例子中，`lambda`函数将一个键值对作为参数，并返回相应的值，这样`min()`和`max()`就有了合适的比较标准。因此，您会在输入数据中获得一个包含最便宜和最贵产品的元组。

[*Remove ads*](/account/join/)

### 寻找互质整数

另一个使用`min()`解决现实世界问题的有趣例子是，当你需要判断两个数字是否是[互质](https://en.wikipedia.org/wiki/Coprime_integers)时。换句话说，你需要知道你的数字的唯一公约数是否是`1`。

在这种情况下，您可以编写一个布尔值或**谓词**函数，如下所示:

>>>

```py
>>> def are_coprime(a, b):
...     for i in range(2, min(a, b) + 1):
...         if a % i == 0 and b % i == 0:
...             return False
...     return True
...

>>> are_coprime(2, 3)
True
>>> are_coprime(2, 4)
False
```

在这个代码片段中，您将`are_coprime()`定义为一个谓词函数，如果输入数字互质，它将返回`True`。如果这些数字不是互质的，那么函数返回`False`。

该函数的主要组件是一个`for`循环，它迭代一个 [`range`](https://realpython.com/python-range/) 值。要设置这个`range`对象的上限，您可以使用`min()`和作为参数的输入数字。同样，您使用`min()`来设置某个区间的上限。

### 为代码的不同实现计时

您还可以使用`min()`来比较您的几个算法，评估它们的执行时间，并确定哪个算法是最高效的。下面的示例使用 [`timeit.repeat()`](https://docs.python.org/3/library/timeit.html#timeit.repeat) 来测量两种不同方式构建包含从`0`到`99`的数字的平方值的列表的执行时间:

>>>

```py
>>> import timeit

>>> min(
...     timeit.repeat(
...         stmt="[i ** 2 for i in range(100)]",
...         number=1000,
...         repeat=3
...     )
... )
0.022141209003166296

>>> min(
...     timeit.repeat(
...         stmt="list(map(lambda i: i ** 2, range(100)))",
...         number=1000,
...         repeat=3
...     )
... )
0.023857666994445026
```

对`timeit.repeat()`的调用将基于字符串的语句运行给定的次数。在这些示例中，该语句重复了三次。对`min()`的调用从三次重复中返回最小的执行时间。

通过结合使用`min()`、`repeat()`和其他 Python [定时器函数](https://realpython.com/python-timer/)，您可以知道哪种算法在执行时间方面是最有效的。上面的例子表明，在构建新列表时，列表理解比内置的 [`map()`](https://realpython.com/python-map-function/) 函数要快一点。

## 《T2》和《T4》中`.__lt__()`和`.__gt__()`的角色探究

到目前为止，您已经了解到，内置的`min()`和`max()`函数足够灵活，可以处理各种数据类型的值，比如数字和字符串。这种灵活性背后的秘密是，`min()`和`max()`依靠 [`.__lt__()`](https://docs.python.org/3/reference/datamodel.html#object.__lt__) 和 [`.__gt__()`](https://docs.python.org/3/reference/datamodel.html#object.__gt__) 的特殊方法，拥抱了 Python 的[鸭子打字](https://en.wikipedia.org/wiki/Duck_typing)哲学。

这些方法是 Python 所谓的**丰富比较方法**的一部分。具体来说，`.__lt__()`和`.__gt__()`分别支持小于(`<`)和大于(`>`)运算符。这里的*支持*是什么意思？当 Python 在你的代码中发现类似于`x < y`的东西时，它会在内部做`x.__lt__(y)`。

要点是您可以将`min()`和`max()`与实现`.__lt__()`和`.__gt__()`的任何数据类型的值一起使用。这就是为什么这些函数适用于所有 Python 内置数据类型的值:

>>>

```py
>>> "__lt__" in dir(int) and "__gt__" in dir(int)
True

>>> "__lt__" in dir(float) and "__gt__" in dir(float)
True

>>> "__lt__" in dir(str) and "__gt__" in dir(str)
True

>>> "__lt__" in dir(list) and "__gt__" in dir(list)
True

>>> "__lt__" in dir(tuple) and "__gt__" in dir(tuple)
True

>>> "__lt__" in dir(dict) and "__gt__" in dir(dict)
True
```

Python 的内置数据类型实现了`.__lt__()`和`.__gt__()`特殊方法。因此，您可以将这些数据类型中的任何一种输入到`min()`和`max()`中，唯一的条件是所涉及的数据类型是可比较的。

您还可以使您的自定义类的实例与`min()`和`max()`兼容。为了实现这一点，您需要提供自己的`.__lt__()`和`.__gt__()`的实现。考虑下面的`Person`类作为这种兼容性的例子:

```py
# person.py

from datetime import date

class Person:
    def __init__(self, name, birth_date):
        self.name = name
        self.birth_date = date.fromisoformat(birth_date)

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"({self.name}, {self.birth_date.isoformat()})"
        )

 def __lt__(self, other):        return self.birth_date > other.birth_date

 def __gt__(self, other):        return self.birth_date < other.birth_date
```

注意，`.__lt__()`和`.__gt__()`的实现需要一个通常名为`other`的参数。该参数表示基础比较运算中的第二个操作数。例如，在一个类似于`x < y`的表达式中，你会发现`x`是`self`而`y`是`other`。

**注意:**对于小于的*和大于*的*比较操作，您只需要实现`.__lt__()`或`.__gt__()`中的一个即可。*

在这个例子中，`.__lt__()`和`.__gt__()`返回两个人的`.birth_date`属性的比较结果。这在实践中是如何工作的:

>>>

```py
>>> from person import Person

>>> jane = Person("Jane Doe", "2004-08-15")
>>> john = Person("John Doe", "2001-02-07")

>>> jane < john
True
>>> jane > john
False

>>> min(jane, john)
Person(Jane Doe, 2004-08-15)

>>> max(jane, john)
Person(John Doe, 2001-02-07)
```

酷！您可以用`min()`和`max()`处理`Person`对象，因为该类提供了`.__lt__()`和`.__gt__()`的实现。对`min()`的调用返回最年轻的人，对`max()`的调用返回最老的人。

**注意:**`.__lt__()`和`.__gt__()`方法只支持两个比较操作符`<`和`>`。如果你想要一个提供所有比较操作的类，但是你只想写一些特殊的方法，那么你可以使用 [`@functools.total_ordering`](https://docs.python.org/3/library/functools.html#functools.total_ordering) 。如果您有一个定义了`.__eq__()`和其他丰富的比较方法的类，那么这个[装饰器](https://realpython.com/primer-on-python-decorators/)将自动提供其余的比较方法。

注意，如果给定的自定义类不提供这些方法，那么它的实例将不支持`min()`和`max()`操作:

>>>

```py
>>> class Number:
...     def __init__(self, value):
...         self.value = value
...

>>> x = Number(21)
>>> y = Number(42)

>>> min(x, y)
Traceback (most recent call last):
    ...
TypeError: '<' not supported between instances of 'Number' and 'Number'

>>> max(x, y)
Traceback (most recent call last):
    ...
TypeError: '>' not supported between instances of 'Number' and 'Number'
```

因为这个`Number`类没有提供`.__lt__()`和`.__gt__()`的合适实现，`min()`和`max()`用一个`TypeError`来响应。错误消息告诉您当前的类不支持比较操作。

[*Remove ads*](/account/join/)

## 效仿 Python 的`min()`和`max()`

至此，您已经了解了 Python 的`min()`和`max()`函数是如何工作的。您已经使用它们在几个数字、字符串等中查找最小和最大值。您知道如何使用单个 iterable 作为参数或者使用未定义数量的常规参数来调用这些函数。最后，您已经编写了一系列使用`min()`和`max()`解决现实世界问题的实际例子。

虽然 Python 友好地为您提供了`min()`和`max()`来查找数据中的最小和最大值，但是从头开始学习如何进行这种计算是一种有益的练习，可以提高您的逻辑思维和编程技能。

在本节中，您将学习如何在数据中查找最小值和最大值。您还将学习如何实现自己版本的`min()`和`max()`。

### 理解`min()`和`max()` 背后的代码

作为一个人，要在一个小的数字列表中找到最小值，你通常会检查这些数字，并在头脑中隐式地比较它们。是的，你的大脑太神奇了！然而，计算机并没有那么聪明。他们需要详细的说明来完成任何任务。

你必须告诉你的计算机在成对比较时迭代所有的值。在这个过程中，计算机必须注意每一对中的当前最小值，直到值列表被完全处理。

这种解释可能很难形象化，因此这里有一个 Python 函数来完成这项工作:

>>>

```py
>>> def find_min(iterable):
...     minimum = iterable[0]
...     for value in iterable[1:]:
...         if value < minimum:
...             minimum = value
...     return minimum
...

>>> find_min([2, 5, 3, 1, 9, 7])
1
```

在这个代码片段中，您定义了`find_min()`。这个函数假设`iterable`不为空，并且它的值是任意顺序的。

该函数将第一个值视为暂定值`minimum`。然后`for`循环遍历输入数据中的其余元素。

[条件语句](https://realpython.com/python-conditional-statements/)将当前`value`与第一次迭代中的暂定`minimum`进行比较。如果当前的`value`小于`minimum`，则条件相应地更新`minimum`。

每次新的迭代将当前的`value`与更新的`minimum`进行比较。当函数到达`iterable`的末端时，`minimum`将保存输入数据中的最小值。

酷！您已经编写了一个函数，它在一组数字中寻找最小值。现在重温一下`find_min()`,想想如何编写一个函数来寻找最大值。对，就是这样！您只需将[比较运算符](https://realpython.com/python-operators-expressions/#comparison-operators)从小于(`<`)改为大于(`>`)，并可能重命名函数和一些[局部变量](https://realpython.com/python-namespaces-scope/#variable-scope)以防止混淆。

您的新函数可能如下所示:

>>>

```py
>>> def find_max(iterable):
...     maximum = iterable[0]
...     for value in iterable[1:]:
...         if value > maximum:
...             maximum = value
...     return maximum
...

>>> find_max([2, 5, 3, 1, 9, 7])
9
```

请注意，`find_max()`与`find_min()`共享其大部分代码。除了命名之外，最重要的区别是`find_max()`使用大于运算符(`>`)而不是小于运算符(`<`)。

作为练习，你可以按照[干(不要重复自己)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)的原则，思考如何避免`find_min()`和`find_max()`中的重复代码。通过这种方式，您将准备好使用您的 Python 技能来模拟`min()`和`max()`的完整行为，您将很快解决这些问题。

在深入研究之前，您需要了解知识要求。您将在函数中组合一些主题，如[条件语句](https://realpython.com/python-conditional-statements/)、[异常处理](https://realpython.com/python-exceptions/)、[列表理解](https://realpython.com/list-comprehension-python/)、带有 [`for`循环的确定迭代](https://realpython.com/python-for-loop/)，以及 [`*args`](https://realpython.com/python-kwargs-and-args/) 和[可选](https://realpython.com/python-optional-arguments/)参数。

如果你觉得自己对这些话题并不了解，那么也不用担心。你会边做边学。如果你被卡住了，那么你可以回头查看链接的资源。

[*Remove ads*](/account/join/)

### 规划您的定制`min()`和`max()`版本

要编写定制的`min()`和`max()`的实现，首先要编写一个助手函数，它能够根据调用中使用的参数找到输入数据中的最小值或最大值。当然，辅助函数将特别依赖于用于比较输入值的操作符。

您的助手函数将具有以下签名:

```py
min_max(*args, operator, key=None, default=None) -> extreme_value
```

下面是每个参数的作用:

| 争吵 | 描述 | 需要 |
| --- | --- | --- |
| `*args` | 允许您用一个 iterable 或任意数量的常规参数调用函数 | 是 |
| `operator` | 为手边的计算保存适当的比较运算符函数 | 是 |
| `key` | 接受单参数函数，该函数修改函数的比较标准和行为 | 不 |
| `default` | 存储当您使用空的 iterable 调用函数时要返回的默认值 | 不 |

`min_max()`的主体将通过处理`*args`来构建一个值列表。拥有一个标准化的值列表将允许您编写所需的算法来查找输入数据中的最小值和最大值。

然后函数需要在计算最小值和最大值之前处理`key`和`default`参数，这是`min_max()`中的最后一步。

有了`min_max()`,最后一步是在它的基础上定义两个独立的函数。这些函数将使用适当的比较**运算符函数**来分别找到最小值和最大值。一会儿你会学到更多关于操作函数的知识。

### 标准化来自`*args` 的输入数据

为了标准化输入数据，您需要检查用户提供的是单个 iterable 还是任意数量的常规参数。启动你最喜欢的[代码编辑器或 IDE](https://realpython.com/python-ides-code-editors-guide/) ，创建一个名为`min_max.py`的新 Python 文件。然后向其中添加以下代码:

```py
# min_max.py

def min_max(*args, operator, key=None, default=None):
    if len(args) == 1:
        try:
            values = list(args[0])  # Also check if the object is iterable
        except TypeError:
            raise TypeError(
                f"{type(args[0]).__name__} object is not iterable"
            ) from None
    else:
        values = args
```

在这里，你定义`min_max()`。该函数的第一部分将输入数据标准化，以便进一步处理。因为用户可以用一个 iterable 或者几个常规参数调用`min_max()`，所以需要检查`args`的长度。要进行这项检查，您可以使用内置的 [`len()`](https://realpython.com/len-python-function/) 功能。

如果`args`只有一个值，那么你需要检查这个参数是否是一个可迭代的对象。您使用 [`list()`](https://docs.python.org/3/library/functions.html#func-list) ，它隐式地进行检查，并将输入的 iterable 转换成一个列表。

如果`list()`引发了一个`TypeError`，那么你捕捉它并引发你自己的`TypeError`来通知用户所提供的对象是不可迭代的，就像`min()`和`max()`在它们的标准形式中所做的那样。注意，您使用了`from None`语法来隐藏原始`TypeError`的[回溯](https://realpython.com/python-traceback/)。

当`args`保存不止一个值时，`else`分支运行，这处理用户用几个常规参数而不是一个可迭代的值调用函数的情况。

如果这个条件最终没有引发一个`TypeError`，那么`values`将保存一个可能为空的值列表。即使结果列表是空的，它现在也是干净的，可以继续寻找它的最小值或最大值。

### 处理`default`自变量

为了继续编写`min_max()`，现在可以处理`default`参数。继续将以下代码添加到函数的末尾:

```py
# min_max.py
# ...

def min_max(*args, operator, key=None, default=None):
    # ...

    if not values:
        if default is None:
            raise ValueError("args is an empty sequence")
        return default
```

在这个代码片段中，您定义了一个条件来检查`values`是否持有一个空列表。如果是这种情况，那么检查`default`参数，看看用户是否为它提供了一个值。如果`default`还是 [`None`](https://realpython.com/null-in-python/) ，那么就升起一个`ValueError`。否则，返回`default`。当您用空的 iterables 调用`min()`和`max()`时，这个行为模拟了它们的标准行为。

[*Remove ads*](/account/join/)

### 处理可选的`key`功能

现在您需要处理`key`参数，并根据提供的`key`准备寻找最小和最大值的数据。继续用下面的代码更新`min_max()`:

```py
# min_max.py
# ...

def min_max(*args, operator, key=None, default=None):
    # ...

    if key is None:
        keys = values
    else:
        if callable(key):
            keys = [key(value) for value in values]
        else:
            raise TypeError(f"{type(key).__name__} object is not a callable")
```

您用一个条件来开始这个代码片段，该条件检查用户是否没有提供一个`key`函数。如果它们没有，那么您可以直接从原始的`values`创建一个键列表。在计算最小值和最大值时，您将使用这些键作为比较键。

另一方面，如果用户提供了一个`key`参数，那么你需要确保这个参数实际上是一个函数或者可调用的对象。为此，您使用内置的 [`callable()`](https://docs.python.org/3/library/functions.html#callable) 函数，如果它的参数是可调用的，则返回`True`，否则返回`False`。

一旦您确定了`key`是一个可调用的对象，那么您就可以通过将`key`应用于输入数据中的每个值来构建比较键的列表。

最后，如果`key`不是一个可调用对象，那么`else`子句运行，产生一个`TypeError`，就像`min()`和`max()`在类似情况下所做的那样。

### 寻找最小值和最大值

完成`min_max()`函数的最后一步是找到输入数据中的最小值和最大值，就像`min()`和`max()`一样。继续用下面的代码结束`min_max()`:

```py
# min_max.py
# ...

def min_max(*args, operator, key=None, default=None):
    # ...

    extreme_key, extreme_value = keys[0], values[0]
    for key, value in zip(keys[1:], values[1:]):
        if operator(key, extreme_key):
            extreme_key = key
            extreme_value = value
    return extreme_value
```

将`extreme_key`和`extreme_value` [变量](https://realpython.com/python-variables/)分别设置为`keys`和`values`中的第一个值。这些变量将为计算最小值和最大值提供初始键和值。

然后使用内置的 [`zip()`](https://realpython.com/python-zip-function/) 函数一次循环其余的键和值。这个函数将通过组合您的`keys`和`values`列表中的值来产生键值元组。

循环内部的条件调用`operator`将当前的`key`与存储在`extreme_key`中的暂定最小或最大密钥进行比较。此时，`operator`参数将保存来自`operator`模块的`lt()`或`gt()`，这取决于您是否想分别找到最小值或最大值。

比如，当你想在输入数据中寻找最小值时，`operator`会持有`lt()`函数。当你想找到最大值的时候，`operator`会按住`gt()`。

每次循环迭代将当前的`key`与暂定的最小或最大键进行比较，并相应地更新`extreme_key`和`extreme_value`的值。在循环结束时，这些变量将保存最小或最大键及其相应的值。最后，你只需要返回`extreme_value`中的值。

### 编写您的自定义`min()`和`max()`函数

有了`min_max()`助手函数，您可以定义自定义版本的`min()`和`max()`。继续将以下函数添加到您的`min_max.py`文件的末尾:

```py
# min_max.py

from operator import gt, lt

# ...

def custom_min(*args, key=None, default=None):
    return min_max(*args, operator=lt, key=key, default=default)

def custom_max(*args, key=None, default=None):
    return min_max(*args, operator=gt, key=key, default=default)
```

在这段代码中，首先从 [`operator`](https://docs.python.org/3/library/operator.html) 模块中导入 [`gt()`](https://docs.python.org/3/library/operator.html#operator.gt) 和 [`lt()`](https://docs.python.org/3/library/operator.html?highlight=gt#operator.lt) 。这些函数分别是大于(`>`)和小于(`<`)运算符的等效函数。比如[布尔](https://realpython.com/python-boolean/)表达式`x < y`等价于函数调用`lt(x, y)`。您将使用这些函数向您的`min_max()`提供`operator`参数。

与`min()`和`max()`一样，`custom_min()`和`custom_max()`以`*args`、`key`和`default`为参数，分别返回最小值和最大值。为了执行计算，这些函数使用所需的参数和适当的比较函数`operator`调用`min_max()`。

在`custom_min()`中，您使用`lt()`来查找输入数据中的最小值。在`custom_max()`中，你使用`gt()`来获得最大值。

如果您想获得`min_max.py`文件的全部内容，请点击下面的可折叠部分:



```py
# min_max.py

from operator import gt, lt

def min_max(*args, operator, key=None, default=None):
    if len(args) == 1:
        try:
            values = list(args[0])  # Also check if the object is iterable
        except TypeError:
            raise TypeError(
                f"{type(args[0]).__name__} object is not iterable"
            ) from None
    else:
        values = args

    if not values:
        if default is None:
            raise ValueError("args is an empty sequence")
        return default

    if key is None:
        keys = values
    else:
        if callable(key):
            keys = [key(value) for value in values]
        else:
            raise TypeError(f"{type(key).__name__} object is not a callable")

    extreme_key, extreme_value = keys[0], values[0]
    for key, value in zip(keys[1:], values[1:]):
        if operator(key, extreme_key):
            extreme_key = key
            extreme_value = value
    return extreme_value

def custom_min(*args, key=None, default=None):
    return min_max(*args, operator=lt, key=key, default=default)

def custom_max(*args, key=None, default=None):
    return min_max(*args, operator=gt, key=key, default=default)
```

酷！您已经完成了用 Python 编写自己版本的`min()`和`max()`的工作。现在去给他们一个尝试吧！

[*Remove ads*](/account/join/)

## 结论

现在你知道如何使用 Python 内置的`min()`和`max()`函数在一个可迭代的或者一系列两个或多个常规参数中找到最小的**和最大的**值。您还了解了`min()`和`max()`的一些其他特性，这些特性可以使它们在您的日常编程中有用。

**在本教程中，您学习了如何:**

*   分别使用 Python 的`min()`和`max()`找到**最小的**和**最大的**值
*   用一个**可迭代**和几个**常规参数**调用`min()`和`max()`
*   将`min()`和`max()`与**字符串**和**字典**一起使用
*   用 **`key`和 **`default`** 自定义`min()`和`max()`的行为**
*   将**理解**和**生成器表达式**送入`min()`和`max()`

此外，你已经编写了一些实际的例子，使用`min()`和`max()`来处理你在编写代码时可能遇到的**现实世界的问题**。您还用纯 Python 编写了定制版的`min()`和`max()`，这是一个很好的学习练习，可以帮助您理解这些内置函数背后的逻辑。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。**********
# Python 中的反向字符串:reversed()、Slicing 等等

> 原文：<https://realpython.com/reverse-string-python/>

当你经常在代码中使用 Python 字符串时，你可能需要以**逆序**处理它们。Python 包含了一些方便的工具和技术，可以在这些情况下帮到你。有了它们，您将能够快速有效地构建现有字符串的反向副本。

了解这些在 Python 中反转字符串的工具和技术将有助于您提高作为 Python 开发人员的熟练程度。

**在本教程中，您将学习如何:**

*   通过**切片**快速构建反向字符串
*   使用`reversed()`和`.join()`创建现有字符串的**反向副本**
*   使用**迭代**和**递归**手动反转现有字符串
*   对字符串执行**反向迭代**
*   **使用`sorted()`将**的字符串反向排序

为了充分利用本教程，您应该了解[琴弦](https://realpython.com/python-strings/)、 [`for`](https://realpython.com/python-for-loop/) 和 [`while`](https://realpython.com/python-while-loop/) 循环以及[递归](https://realpython.com/python-thinking-recursively/)的基础知识。

**免费下载:** [从《Python 基础:Python 3 实用入门》中获取一个示例章节](https://realpython.com/bonus/python-basics-sample-download/)，看看如何通过 Python 3.8 的最新完整课程从初级到中级学习 Python。

## 使用核心 Python 工具反转字符串

在某些特定的情况下，使用 Python [字符串](https://realpython.com/python-strings/)以**逆序**工作可能是一个需求。例如，假设您有一个字符串`"ABCDEF"`，您想要一个快速的方法来反转它以得到`"FEDCBA"`。您可以使用哪些 Python 工具来提供帮助？

在 Python 中，字符串是不可变的，所以在适当的位置反转给定的字符串是不可能的。你需要创建目标字符串的反向**副本**来满足需求。

Python 提供了两种简单的方法来反转字符串。因为字符串是序列，所以它们是**可索引**、**可切片**和**可迭代**。这些特性允许您使用[切片](https://docs.python.org/dev/whatsnew/2.3.html#extended-slices)以逆序直接生成给定字符串的副本。第二种选择是使用内置函数 [`reversed()`](https://docs.python.org/dev/library/functions.html#reversed) 创建一个迭代器，以逆序生成输入字符串的字符。

[*Remove ads*](/account/join/)

### 切片反转琴弦

切片是一种有用的技术，它允许你使用不同的整数索引组合从给定的序列中提取项目，这些整数索引被称为 T2 偏移量。在对字符串进行切片时，这些偏移量定义了切片中第一个字符的索引、停止切片的字符的索引以及一个值，该值定义了在每次迭代中要跳过多少个字符。

要分割字符串，可以使用以下语法:

```py
a_string[start:stop:step]
```

你的偏移量是`start`、`stop`和`step`。该表达式通过`step`提取从`start`到`stop − 1`的所有字符。一会儿你们会更深入地了解这一切意味着什么。

所有偏移都是可选的，它们具有以下默认值:

| 抵消 | 缺省值 |
| --- | --- |
| `start` | `0` |
| `stop` | `len(a_string)` |
| `step` | `1` |

这里，`start`表示切片中第一个字符的索引，而`stop`保存停止切片操作的索引。第三个偏移量`step`，允许您决定切片在每次迭代中跳过多少个字符。

**注:**当指数大于等于`stop`时，切片操作结束。这意味着它永远不会在最终切片中包含该索引处的项目(如果有)。

`step`偏移量允许您微调如何从字符串中提取所需字符，同时跳过其他字符:

>>>

```py
>>> letters = "AaBbCcDd"

>>> # Get all characters relying on default offsets
>>> letters[::]
'AaBbCcDd'
>>> letters[:]
'AaBbCcDd'

>>> # Get every other character from 0 to the end
>>> letters[::2]
'ABCD'

>>> # Get every other character from 1 to the end
>>> letters[1::2]
'abcd'
```

这里，您首先对`letters`进行切片，而不提供显式的偏移值，以获得原始字符串的完整副本。为此，还可以使用省略第二个冒号(`:`)的切片。当`step`等于`2`时，切片从目标字符串中获取每隔一个字符。您可以尝试不同的偏移来更好地理解切片是如何工作的。

为什么切片和第三个偏移量与 Python 中的反转字符串相关？答案在于`step`如何处理负值。如果您给`step`提供一个负值，那么切片向后运行，意味着从右到左。

例如，如果您将`step`设置为等于`-1`，那么您可以构建一个以逆序检索所有字符的切片:

>>>

```py
>>> letters = "ABCDEF"

>>> letters[::-1]
'FEDCBA'

>>> letters
'ABCDEF'
```

这个切片返回从字符串右端(索引等于`len(letters) - 1`)到字符串左端(索引为`0`)的所有字符。当你使用这个技巧时，你会得到一个原字符串逆序的副本，而不会影响`letters`的原内容。

创建现有字符串的反向副本的另一种技术是使用 [`slice()`](https://docs.python.org/dev/library/functions.html#slice) 。这个内置函数的签名如下:

```py
slice(start, stop, step)
```

该函数接受三个参数，与切片操作符中的偏移量具有相同的含义，并返回一个表示调用`range(start, stop, step)`产生的一组索引的[切片](https://docs.python.org/3/glossary.html#term-slice)对象。

您可以使用`slice()`来模拟切片`[::-1]`并快速反转您的字符串。继续运行下面对方括号内的`slice()`的调用:

>>>

```py
>>> letters = "ABCDEF"

>>> letters[slice(None, None, -1)]
'FEDCBA'
```

将 [`None`](https://realpython.com/null-in-python/) 传递给`slice()`的前两个参数，告诉函数您想要依赖它的内部默认行为，这与没有`start`和`stop`值的标准切片是一样的。换句话说，将`None`传递给`start`和`stop`意味着您想要从底层序列的左端到右端的切片。

[*Remove ads*](/account/join/)

### 用`.join()`和`reversed()` 反转琴弦

第二种也可能是最巧妙的反转字符串的方法是将`reversed()`和 [`str.join()`](https://docs.python.org/3/library/stdtypes.html#str.join) 一起使用。如果你传递一个字符串给`reversed()`，你会得到一个迭代器，它以相反的顺序产生字符:

>>>

```py
>>> greeting = reversed("Hello, World!")

>>> next(greeting)
'!'
>>> next(greeting)
'd'
>>> next(greeting)
'l'
```

当您使用`greeting`作为参数调用`next()`时，您从原始字符串的右端获取每个字符。

关于`reversed()`需要注意的重要一点是，结果迭代器直接从原始字符串产生字符。换句话说，它不会创建一个新的反向字符串，而是从现有字符串反向读取字符。这种行为在内存消耗方面是相当有效的，在某些环境和情况下，比如迭代，这是一个基本的优势。

您可以使用通过直接调用`reversed()`获得的迭代器作为`.join()`的参数:

>>>

```py
>>> "".join(reversed("Hello, World!"))
'!dlroW ,olleH'
```

在这个单行表达式中，您将调用`reversed()`的结果作为参数直接传递给`.join()`。因此，您会得到原始输入字符串的反向副本。这种`reversed()`和`.join()`的组合是换弦的绝佳选择。

## 手工生成反向字符串

到目前为止，您已经了解了快速反转字符串的核心 Python 工具和技术。大多数时候，他们会是你要走的路。但是，在编码过程中的某些时候，您可能需要手动反转字符串。

在本节中，您将学习如何使用显式循环和[递归](https://realpython.com/python-recursion/)来反转字符串。最后一种技术是借助 Python 的 [`reduce()`](https://realpython.com/python-reduce-function/) 函数使用函数式编程方法。

### 在循环中反转字符串

您将用来反转字符串的第一个技术涉及到一个`for`循环和连接操作符(`+`)。使用两个字符串作为操作数，该运算符返回一个新字符串，该新字符串是通过连接原始字符串而得到的。整个操作被称为**串联**。

**注意:**使用`.join()`是在 Python 中连接字符串的推荐方法。它干净、高效，而且[蟒蛇](https://realpython.com/learning-paths/writing-pythonic-code/)。

下面是一个函数，它获取一个字符串，并使用串联在循环中反转它:

>>>

```py
>>> def reversed_string(text):
...     result = ""
...     for char in text:
...         result = char + result
...     return result
...

>>> reversed_string("Hello, World!")
'!dlroW ,olleH'
```

在每次迭代中，循环从`text`中取出一个后续字符`char`，并将其与`result`的当前内容连接起来。注意，`result`最初保存一个空字符串(`""`)。新的中间字符串然后被重新分配给`result`。在循环结束时，`result`保存一个新字符串，作为原始字符串的反向副本。

**注意:**由于 Python 字符串是不可变的数据类型，您应该记住本节中的例子使用了一种浪费的技术。他们依赖于创建连续的中间字符串，只是为了在下一次迭代中丢弃它们。

如果你喜欢使用一个 [`while`循环](https://realpython.com/python-while-loop/)，那么你可以这样做来构建一个给定字符串的反向副本:

>>>

```py
>>> def reversed_string(text):
...     result = ""
...     index = len(text) - 1
...     while index >= 0:
...         result += text[index]
...         index -= 1
...     return result
...

>>> reversed_string("Hello, World!")
'!dlroW ,olleH'
```

这里，首先使用 [`len()`](https://realpython.com/len-python-function/) 计算输入字符串中最后一个字符的`index`。循环从`index`向下迭代到`0`。在每次迭代中，使用[增加赋值](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)操作符(`+=`)创建一个中间字符串，将`result`的内容与来自`text`的相应字符连接起来。同样，最终结果是通过反转输入字符串得到的新字符串。

[*Remove ads*](/account/join/)

### 用递归反转字符串

你也可以使用[递归](https://realpython.com/python-thinking-recursively/)来反转字符串。递归是函数在自己的体内调用自己。为了防止无限递归，你应该提供一个**基础用例**，它不需要再次调用函数就能产生结果。第二个组件是**递归用例**，它启动递归循环并执行大部分计算。

下面是如何定义一个递归函数来返回给定字符串的反向副本:

>>>

```py
>>> def reversed_string(text):
...     if len(text) == 1:
...         return text
...     return reversed_string(text[1:]) + text[:1]
...

>>> reversed_string("Hello, World!")
'!dlroW ,olleH'
```

在这个例子中，您首先检查基本情况。如果输入字符串只有一个字符，就将该字符串返回给调用者。

最后一个语句是递归情况，调用`reversed_string()`本身。该调用使用输入字符串的片段`text[1:]`作为参数。这片包含了`text`中的所有角色，除了第一个。下一步是将递归调用的结果与包含第一个字符`text`的单字符字符串`text[:1]`相加。

在上面的例子中需要注意的一个重要问题是，如果您将一个长字符串作为参数传递给`reversed_string()`，那么您将得到一个`RecursionError`:

>>>

```py
>>> very_long_greeting = "Hello, World!" * 1_000

>>> reversed_string(very_long_greeting)
Traceback (most recent call last):
    ...
RecursionError: maximum recursion depth exceeded while calling a Python object
```

达到 Python 的默认递归限制是您应该在代码中考虑的一个重要问题。但是，如果您真的需要使用递归，那么您仍然可以选择手动设置递归限制。

从 [`sys`](https://docs.python.org/3/library/sys.html#module-sys) 调用 [`getrecursionlimit()`](https://docs.python.org/3/library/sys.html#sys.getrecursionlimit) 可以检查你当前 Python 解释器的递归极限。默认情况下，这个值通常是`1000`。您可以在同一个模块`sys`中使用 [`setrecursionlimit()`](https://docs.python.org/3/library/sys.html#sys.setrecursionlimit) 来调整这个限制。使用这些函数，您可以配置 Python 环境，以便您的递归解决方案可以工作。来吧，试一试！

### 使用`reduce()`反转字符串

如果你喜欢使用函数式编程方法，你可以使用 [`reduce()`](https://realpython.com/python-reduce-function/) 从 [`functools`](https://docs.python.org/3/library/functools.html#module-functools) 来反转字符串。Python 的`reduce()`将一个[折叠](https://en.wikipedia.org/wiki/Fold_(higher-order_function))或归约函数和一个 iterable 作为参数。然后，它将提供的函数应用于输入 iterable 中的项，并返回一个累积值。

以下是如何利用`reduce()`来反转字符串:

>>>

```py
>>> from functools import reduce

>>> def reversed_string(text):
...     return reduce(lambda a, b: b + a, text)
...

>>> reversed_string("Hello, World!")
'!dlroW ,olleH'
```

在这个例子中， [`lambda`](https://realpython.com/python-lambda/) 函数获取两个字符串，并以相反的顺序将它们连接起来。对`reduce()`的调用将`lambda`应用到循环中的`text`，并构建原始字符串的反向副本。

## 反向遍历字符串

有时，您可能希望以相反的顺序遍历现有的字符串，这种技术通常被称为**反向迭代**。根据您的具体需要，您可以使用以下选项之一对字符串进行反向迭代:

*   `reversed()`内置函数
*   `[::-1]`切片操作符

逆向迭代可以说是这些工具最常见的用例，所以在接下来的几节中，您将学习如何在迭代环境中使用它们。

### `reversed()`内置函数

以相反的顺序遍历一个字符串的可读性最好的方法是使用`reversed()`。当您使用这个函数和`.join()`一起创建反向字符串时，您已经了解了这个函数。

然而，`reversed()`的主要意图和用例是支持 Python iterables 上的反向迭代。使用字符串作为参数，`reversed()`返回一个迭代器，该迭代器以相反的顺序从输入字符串中产生字符。

下面是如何用`reversed()`以相反的顺序遍历一个字符串:

>>>

```py
>>> greeting = "Hello, World!"

>>> for char in reversed(greeting):
...     print(char)
...
!
d
l
r
o
W

,
o
l
l
e
H

>>> reversed(greeting)
<reversed object at 0x7f17aa89e070>
```

这个例子中的`for`循环可读性很强。`reversed()`的名字清楚地表达了它的意图，并传达了该函数不会对输入数据产生任何[副作用](https://en.wikipedia.org/wiki/Side_effect_(computer_science))。由于`reversed()`返回一个迭代器，这个循环在内存使用方面也是高效的。

[*Remove ads*](/account/join/)

### `[::-1]`切片操作符，

对字符串执行反向迭代的第二种方法是使用您在前面的`a_string[::-1]`示例中看到的扩展切片语法。尽管这种方法不利于提高内存效率和可读性，但它仍然提供了一种快速迭代现有字符串的反向副本的方法:

>>>

```py
>>> greeting = "Hello, World!"

>>> for char in greeting[::-1]:
...     print(char)
...
!
d
l
r
o
W

,
o
l
l
e
H

>>> greeting[::-1]
'!dlroW ,olleH'
```

在这个例子中，您对`greeting`应用切片操作符来创建它的反向副本。然后，您使用新的反向字符串来填充循环。在这种情况下，您正在迭代一个新的反向字符串，因此这种解决方案比使用`reversed()`的内存效率低。

## 创建自定义可逆字符串

如果你曾经尝试过用[反转一个 Python 列表](https://realpython.com/python-reverse-list/)，那么你会知道列表有一个叫做`.reverse()`的简便方法，可以在适当的位置反转底层列表[。因为字符串在 Python 中是不可变的，所以它们没有提供类似的方法。](https://realpython.com/python-reverse-list/)

然而，您仍然可以用模仿`list.reverse()`的`.reverse()`方法创建一个定制的 string 子类。你可以这样做:

>>>

```py
>>> from collections import UserString

>>> class ReversibleString(UserString):
...     def reverse(self):
...         self.data = self.data[::-1]
...
```

`ReversibleString`继承自 [`UserString`](https://realpython.com/inherit-python-str/) ，是 [`collections`](https://realpython.com/python-collections-module/) 模块的一个类。`UserString`是包装 [`str`](https://docs.python.org/3/library/stdtypes.html#str) 的内置数据类型。它是专门为创建`str`的子类而设计的。当您需要创建带有额外功能的自定义字符串类时,`UserString`非常方便。

`UserString`提供与常规字符串相同的功能。它还添加了一个名为`.data`的公共属性，用于保存和访问包装的字符串对象。

在`ReversibleString`内部，你创建了`.reverse()`。该方法反转`.data`中的包装字符串，并将结果重新分配给`.data`。从外部来看，调用`.reverse()`的工作原理就像将字符串反转到位。然而，它实际做的是创建一个新的字符串，以相反的顺序包含原始数据。

下面是`ReversibleString`在实践中的工作方式:

>>>

```py
>>> text = ReversibleString("Hello, World!")
>>> text
'Hello, World!'

>>> # Reverse the string in place
>>> text.reverse()
>>> text
'!dlroW ,olleH'
```

当您在`text`上调用`.reverse()`时，该方法的行为就好像您正在对底层字符串进行就地变异。然而，您实际上是在创建一个新的字符串，并将它赋回包装后的字符串。注意`text`现在以相反的顺序保存原始字符串。

因为`UserString`提供了和它的超类`str`相同的功能，你可以使用`reversed()`来执行逆向迭代:

>>>

```py
>>> text = ReversibleString("Hello, World!")

>>> # Support reverse iteration out of the box
>>> for char in reversed(text):
...     print(char)
...
!
d
l
r
o
W

,
o
l
l
e
H

>>> text
"Hello, World!"
```

在这里，您用`text`作为参数调用`reversed()`，以进入一个`for`循环。这个调用按预期工作并返回相应的迭代器，因为`UserString`从`str`继承了所需的行为。注意调用`reversed()`不会影响原来的字符串。

## 逆序排序 Python 字符串

您将学习的最后一个主题是如何对字符串中的字符进行逆序排序。当您处理没有特定顺序的字符串，并且需要按相反的字母顺序对它们进行排序时，这非常方便。

处理这个问题，可以用 [`sorted()`](https://realpython.com/python-sort/) 。这个内置函数返回一个列表，该列表按顺序包含输入 iterable 的所有项目。除了输入 iterable，`sorted()`还接受一个`reverse`关键字参数。如果希望输入 iterable 按降序排序，可以将该参数设置为`True`:

>>>

```py
>>> vowels = "eauoi"

>>> # Sort in ascending order
>>> sorted(vowels)
['a', 'e', 'i', 'o', 'u']

>>> # Sort in descending order
>>> sorted(vowels, reverse=True)
['u', 'o', 'i', 'e', 'a']
```

当您使用一个字符串作为参数调用`sorted()`并且将`reverse`设置为`True`时，您会得到一个列表，其中包含以逆序或降序排列的输入字符串的字符。因为`sorted()`返回了一个`list`对象，所以你需要一种方法把这个列表变回一个字符串。同样，您可以像在前面几节中一样使用`.join()`:

>>>

```py
>>> vowels = "eauoi"

>>> "".join(sorted(vowels, reverse=True))
'uoiea'
```

在这个代码片段中，您在一个空字符串上调用`.join()`,它扮演一个分隔符的角色。对`.join()`的参数是以`vowels`作为参数调用`sorted()`并将`reverse`设置为`True`的结果。

您还可以利用`sorted()`以排序和反转的顺序遍历一个字符串:

>>>

```py
>>> for vowel in sorted(vowels, reverse=True):
...     print(vowel)
...
...
u
o
i
e
a
```

`sorted()`的`reverse`参数允许你以降序排列可重复项，包括字符串。所以，如果你需要一个字符串的字符按照相反的字母顺序排序，那么`sorted()`就是为你准备的。

[*Remove ads*](/account/join/)

## 结论

以**逆序**反转和处理字符串可能是编程中的常见任务。Python 提供了一套工具和技术，可以帮助您快速高效地执行字符串反转。在本教程中，您了解了这些工具和技术，以及如何在字符串处理挑战中利用它们。

**在本教程中，您学习了如何:**

*   通过**切片**快速构建反向字符串
*   使用 **`reversed()`** 和 **`.join()`** 创建现有字符串的反向副本
*   使用**迭代**和**递归**手工创建反向字符串
*   以相反的顺序在你的琴弦上循环
*   使用 **`sorted()`** 对字符串进行降序排序

尽管这个主题本身可能没有很多令人兴奋的用例，但了解如何反转字符串在编写入门级职位的面试代码时会很有用。您还会发现，掌握反转字符串的不同方式可以帮助您真正理解 Python 中字符串的不变性，这是该语言的一个显著特征。*****
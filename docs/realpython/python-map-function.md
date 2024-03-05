# Python 的 map():不使用循环处理可重复项

> 原文：<https://realpython.com/python-map-function/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 的 map()函数:变换 Iterables**](/courses/map-function-transform-iterables/)

Python 的 **[`map()`](https://docs.python.org/3/library/functions.html#map)** 是一个内置函数，允许你处理和转换一个 iterable 中的所有项，而不需要使用显式的 [`for`循环](https://realpython.com/courses/python-for-loop/)，这种技术通常被称为[映射](https://en.wikipedia.org/wiki/Map_(higher-order_function))。当您需要将一个**转换函数**应用到一个可迭代对象中的每一项，并将它们转换成一个新的可迭代对象时，`map()`非常有用。`map()`是 Python 中支持[函数式编程风格](https://realpython.com/python-functional-programming/)的工具之一。

在本教程中，您将学习:

*   Python 的 **`map()`** 是如何工作的
*   如何使用`map()`将**转换成**不同类型的 Python 可迭代对象
*   如何将**`map()`与其他功能工具结合起来进行更复杂的变换**
***   你能用什么工具来**取代** `map()`并让你的代码更**python 化***

*有了这些知识，你将能够在你的程序中有效地使用`map()`,或者，使用[列表理解](https://realpython.com/list-comprehension-python/)或[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)来使你的代码更具 Pythonic 化和可读性。

为了更好地理解`map()`，一些关于如何使用[可迭代](https://docs.python.org/3/glossary.html#term-iterable)、`for`循环、[函数](https://realpython.com/defining-your-own-python-function/)和 [`lambda`函数](https://realpython.com/python-lambda/)的知识会有所帮助。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 中的函数式编码

在 [**函数式编程**](https://realpython.com/courses/functional-programming-python/) 中，计算是通过组合函数来完成的，这些函数接受参数并返回一个(或多个)具体值作为结果。这些函数不修改它们的输入参数，也不改变程序的状态。它们只是提供给定计算的结果。这几种函数俗称[纯函数](https://en.wikipedia.org/wiki/Pure_function)。

理论上，使用函数式风格构建的程序更容易:

*   开发,因为你可以独立地编码和使用每一个功能
*   **调试和测试**,因为你可以[测试](https://realpython.com/python-testing/)和[调试](https://realpython.com/courses/python-debugging-pdb/)单个功能，而不用查看程序的其余部分
*   理解,因为你不需要在整个程序中处理状态变化

函数式编程通常使用[列表](https://realpython.com/python-lists-tuples/)、数组和其他可重复项来表示数据，以及一组对数据进行操作和转换的函数。当使用函数式风格处理数据时，至少有三种常用的技术:

1.  [**映射**](https://en.wikipedia.org/wiki/Map_(higher-order_function)) 包括将一个转换函数应用于一个可迭代对象以产生一个新的可迭代对象。新 iterable 中的项目是通过对原始 iterable 中的每个项目调用转换函数来生成的。

2.  [**过滤**](https://en.wikipedia.org/wiki/Filter_(higher-order_function)) 包括对一个可迭代对象应用一个[谓词或布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)来生成一个新的可迭代对象。新 iterable 中的项是通过过滤掉原始 iterable 中使谓词函数返回 false 的任何项而产生的。

3.  [**归约**](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 包括将归约函数应用于一个迭代项，以产生一个单一的累积值。

根据[吉多·范·罗苏姆](https://en.wikipedia.org/wiki/Guido_van_Rossum)的说法，与函数式语言相比，Python 受[命令式](https://en.wikipedia.org/wiki/Imperative_programming)编程语言的影响更大:

> 我从来不认为 Python 会受到函数式语言的严重影响，不管人们怎么说或怎么想。我更熟悉 C 和 Algol 68 等命令式语言，尽管我已经将函数作为一级对象，但我并不认为 Python 是一种函数式编程语言。([来源](https://web.archive.org/web/20161104183819/http://python-history.blogspot.com.br/2009/04/origins-of-pythons-functional-features.html))

然而，回到 1993 年，Python 社区需要一些函数式编程特性。他们要求:

*   [匿名函数](https://en.wikipedia.org/wiki/Anonymous_function)
*   一个`map()`功能
*   一个`filter()`功能
*   一个`reduce()`功能

由于社区成员的[贡献，这些功能特性被添加到语言中。如今，](https://web.archive.org/web/20200709210752/http://www.artima.com/weblogs/viewpost.jsp?thread=98196) [`map()`](https://docs.python.org/3/library/functions.html#map) 、 [`filter()`](https://docs.python.org/3/library/functions.html#filter) 、 [`reduce()`](https://realpython.com/python-reduce-function/) 是 Python 中函数式编程风格的基本组成部分。

在本教程中，您将涉及其中一个功能特性，即内置函数`map()`。您还将学习如何使用[列表理解](https://realpython.com/list-comprehension-python/)和[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)以 Pythonic 化和可读的方式获得与`map()`相同的功能。

[*Remove ads*](/account/join/)

## Python 的`map()` 入门

有时，您可能会遇到这样的情况，您需要对输入 iterable 的所有项执行相同的操作来构建新的 iterable。解决这个问题的最快和最常见的方法是使用一个 [Python `for`循环](https://realpython.com/courses/python-for-loop/)。然而，您也可以通过使用`map()`在没有显式循环的情况下解决这个问题。

在接下来的三个部分中，您将了解到`map()`是如何工作的，以及如何使用它来处理和转换可重复数据而不产生循环。

### 理解`map()`

`map()`循环遍历一个输入 iterable(或多个 iterables)的项，并返回一个迭代器，该迭代器是通过对原始输入 iterable 中的每一项应用转换函数而得到的。

根据[文档](https://docs.python.org/3/library/functions.html#map) , `map()`将一个函数对象和一个可迭代对象(或多个可迭代对象)作为参数，并返回一个迭代器，该迭代器根据需要生成转换后的项。该函数的签名定义如下:

```py
map(function, iterable[, iterable1, iterable2,..., iterableN])
```

`map()`将`function`应用于循环中`iterable`中的每一项，并返回一个新的迭代器，该迭代器根据需要生成转换后的项。`function`可以是任何一个 Python 函数，它接受的参数数量等于传递给`map()`的可迭代次数。

**注意:**`map()`的第一个参数是一个**函数对象**，这意味着你需要传递一个函数而不需要调用它。也就是说，不使用一对括号。

`map()`的第一个参数是一个**转换函数**。换句话说，就是这个函数将每个原始项转换成一个新的(转换后的)项。即使 Python 文档调用这个参数`function`，它也可以是任何 Python 可调用的。这包括[内置函数](https://docs.python.org/3/library/functions.html#built-in-functions)，[类](https://realpython.com/lessons/classes-python/)，[方法](https://realpython.com/lessons/mastering-method-types-oop-pizza-example/)， [`lambda`函数](https://realpython.com/courses/python-lambda-functions/)，以及[用户自定义函数](https://realpython.com/defining-your-own-python-function/)。

`map()`执行的操作通常被称为**映射**，因为它将输入 iterable 中的每个项目映射到结果 iterable 中的一个新项目。为此，`map()`对输入 iterable 中的所有项应用一个转换函数。

为了更好地理解`map()`，假设您需要获取一个数值列表，并将其转换为包含原始列表中每个数字的平方值的列表。在这种情况下，您可以使用一个`for`循环并编写如下代码:

>>>

```py
>>> numbers = [1, 2, 3, 4, 5]
>>> squared = []

>>> for num in numbers:
...     squared.append(num ** 2)
...

>>> squared
[1, 4, 9, 16, 25]
```

当您在`numbers`上运行这个循环时，您会得到一个平方值列表。`for`循环对`numbers`进行迭代，并对每个值进行幂运算。最后，它将结果值存储在`squared`中。

通过使用`map()`，您可以在不使用显式循环的情况下获得相同的结果。看一下上面例子的重新实现:

>>>

```py
>>> def square(number):
...     return number ** 2
...

>>> numbers = [1, 2, 3, 4, 5]

>>> squared = map(square, numbers)

>>> list(squared)
[1, 4, 9, 16, 25]
```

`square()`是将数字映射到其平方值的变换函数。对`map()`的调用将`square()`应用于`numbers`中的所有值，并返回一个产生平方值的迭代器。然后在`map()`上调用`list()`来创建一个包含平方值的列表对象。

由于`map()`是用 [C](https://realpython.com/build-python-c-extension-module/) 编写的，并且经过了高度优化，其内部隐含循环可以比常规 Python `for`循环更高效。这是使用`map()`的一个优势。

使用`map()`的第二个优势与内存消耗有关。使用一个`for`循环，您需要将整个列表存储在系统内存中。使用`map()`，你可以按需获得物品，并且在给定的时间内，只有一个物品在你的系统内存中。

**注意:**在 Python 2.x 中， [`map()`](https://docs.python.org/2/library/functions.html#map) 返回一个列表。这种行为在 [Python 3.x](https://docs.python.org/3/whatsnew/3.0.html#views-and-iterators-instead-of-lists) 中有所改变。现在，`map()`返回一个 map 对象，这是一个迭代器，可以按需生成条目。这就是为什么你需要调用`list()`来创建想要的列表对象。

再举一个例子，假设您需要将列表中的所有条目从一个[字符串转换成一个整数](https://realpython.com/courses/convert-python-string-int/)。为此，您可以将`map()`与`int()`一起使用，如下所示:

>>>

```py
>>> str_nums = ["4", "8", "6", "5", "3", "2", "8", "9", "2", "5"]

>>> int_nums = map(int, str_nums)
>>> int_nums
<map object at 0x7fb2c7e34c70>

>>> list(int_nums)
[4, 8, 6, 5, 3, 2, 8, 9, 2, 5]

>>> str_nums
["4", "8", "6", "5", "3", "2", "8", "9", "2", "5"]
```

`map()`将 [`int()`](https://docs.python.org/3/library/functions.html#int) 应用于`str_nums`中的每一个值。因为`map()`返回一个迭代器(一个 map 对象)，所以你需要调用`list()`，这样你就可以用尽迭代器并把它变成一个 list 对象。请注意，原始序列在此过程中不会被修改。

[*Remove ads*](/account/join/)

### 将`map()`与不同种类的功能一起使用

您可以使用任何类型的可通过`map()`调用的 Python。唯一的条件是 callable 接受一个参数并返回一个具体的有用的值。例如，您可以使用类、实现名为 [`__call__()`](https://docs.python.org/3/reference/datamodel.html#object.__call__) 的特殊方法的实例、实例方法、[类方法、静态方法](https://realpython.com/courses/staticmethod-vs-classmethod-python/)和函数。

有一些内置函数可以和`map()`一起使用。考虑下面的例子:

>>>

```py
>>> numbers = [-2, -1, 0, 1, 2]

>>> abs_values = list(map(abs, numbers))
>>> abs_values
[2, 1, 0, 1, 2]

>>> list(map(float, numbers))
[-2.0, -1.0, 0.0, 1.0, 2.0]

>>> words = ["Welcome", "to", "Real", "Python"]

>>> list(map(len, words))
[7, 2, 4, 6]
```

您可以使用任何带有`map()`的内置函数，只要该函数接受一个参数并返回值。

使用`map()`的一个常见模式是使用`lambda`函数作为第一个参数。当您需要将基于表达式的函数传递给`map()`时，`lambda`函数非常方便。例如，您可以使用`lambda`函数重新实现平方值的示例，如下所示:

>>>

```py
>>> numbers = [1, 2, 3, 4, 5]

>>> squared = map(lambda num: num ** 2, numbers)

>>> list(squared)
[1, 4, 9, 16, 25]
```

使用`map()`时，函数非常有用。他们可以起到第一个论证`map()`的作用。你可以使用`lambda`函数和`map()`来快速处理和转换你的可重复项。

### 用`map()` 处理多个输入项

如果您向`map()`提供多个 iterables，那么转换函数必须接受与您传入的 iterables 一样多的参数。`map()`的每次迭代都会将每个 iterable 中的一个值作为参数传递给`function`。迭代在最短的迭代结束时停止。

考虑下面这个使用 [`pow()`](https://docs.python.org/3/library/functions.html#pow) 的例子:

>>>

```py
>>> first_it = [1, 2, 3]
>>> second_it = [4, 5, 6, 7]

>>> list(map(pow, first_it, second_it))
[1, 32, 729]
```

`pow()`接受两个参数`x`和`y`，并将`x`返回给`y`的幂。第一次迭代，`x`会是`1`，`y`会是`4`，结果是`1`。在第二次迭代中，`x`将是`2`，`y`将是`5`，结果将是`32`，以此类推。最终的可迭代式只有最短的可迭代式那么长，在本例中是`first_it`。

这种技术允许您使用不同种类的数学运算来合并两个或多个数值的可迭代项。下面是一些使用`lambda`函数对几个输入变量执行不同数学运算的例子:

>>>

```py
>>> list(map(lambda x, y: x - y, [2, 4, 6], [1, 3, 5]))
[1, 1, 1]

>>> list(map(lambda x, y, z: x + y + z, [2, 4], [1, 3], [7, 8]))
[10, 15]
```

在第一个示例中，您使用减法运算来合并两个各包含三项的 iterables。在第二个示例中，您将三个 iterables 的值相加。

## 用 Python 的`map()` 转换字符串的可重复项

当您处理 string 对象的 iterables 时，您可能会对使用某种转换函数转换所有对象感兴趣。在这些情况下，Python 的`map()`可以成为你的盟友。接下来的部分将带您浏览一些如何使用`map()`来转换 string 对象的 iterables 的例子。

### 使用`str`的方法

一种很常见的[字符串操作](https://realpython.com/python-strings/#string-manipulation)方法是使用类`str` 的一些[方法将一个给定的字符串转换成一个新的字符串。如果您正在处理字符串的可重复项，并且需要对每个字符串应用相同的转换，那么您可以使用`map()`和各种字符串方法:](https://docs.python.org/3/library/stdtypes.html#string-methods)

>>>

```py
>>> string_it = ["processing", "strings", "with", "map"]
>>> list(map(str.capitalize, string_it))
['Processing', 'Strings', 'With', 'Map']

>>> list(map(str.upper, string_it))
['PROCESSING', 'STRINGS', 'WITH', 'MAP']

>>> list(map(str.lower, string_it))
['processing', 'strings', 'with', 'map']
```

您可以使用`map()`和 string 方法对`string_it`中的每一项执行一些转换。大多数时候，你会使用不带附加参数的方法，比如 [`str.capitalize()`](https://docs.python.org/3/library/stdtypes.html#str.capitalize) 、 [`str.lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower) 、 [`str.swapcase()`](https://docs.python.org/3/library/stdtypes.html#str.swapcase) 、 [`str.title()`](https://docs.python.org/3/library/stdtypes.html#str.title) 和 [`str.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper) 。

您还可以使用一些方法，这些方法采用带有默认值的附加参数，例如 [`str.strip()`](https://docs.python.org/3/library/stdtypes.html#str.strip) ，它采用一个名为`char`的可选参数，默认情况下移除空白:

>>>

```py
>>> with_spaces = ["processing ", "  strings", "with   ", " map   "]

>>> list(map(str.strip, with_spaces))
['processing', 'strings', 'with', 'map']
```

当你像这样使用`str.strip()`时，你依赖于`char`的默认值。在这种情况下，使用`map()`删除`with_spaces`条目中的所有空格。

**注意:**如果你需要提供参数而不是依赖默认值，那么你可以使用`lambda`函数。

下面是一个使用`str.strip()`来删除点而不是默认空白的例子:

>>>

```py
>>> with_dots = ["processing..", "...strings", "with....", "..map.."]

>>> list(map(lambda s: s.strip("."), with_dots))
['processing', 'strings', 'with', 'map']
```

`lambda`函数调用字符串对象`s`上的`.strip()`并删除所有的前导和尾随点。

例如，当您处理文本文件时，其中的行可能有尾随空格(或其他字符)并且您需要删除它们，这种技术会很方便。如果是这种情况，那么您需要考虑在没有自定义`char`的情况下使用`str.strip()`也会删除换行符。

[*Remove ads*](/account/join/)

### 删除标点符号

在处理文本时，有时需要删除将文本拆分成单词后留下的标点符号。为了解决这个问题，您可以创建一个自定义函数，使用一个匹配最常见标点符号的[正则表达式](https://realpython.com/regex-python)来删除单个单词的标点符号。

下面是使用 [`sub()`](https://docs.python.org/3/library/re.html#re.sub) 实现该函数的一种可能，T3 是一个正则表达式函数，位于 Python 标准库中的 [`re`](https://docs.python.org/3/library/re.html#module-re) 模块中:

>>>

```py
>>> import re

>>> def remove_punctuation(word):
...     return re.sub(r'[!?.:;,"()-]', "", word)

>>> remove_punctuation("...Python!")
'Python'
```

在`remove_punctuation()`中，您使用了一个正则表达式模式，该模式匹配任何英文文本中最常见的标点符号。对`re.sub()`的调用使用空字符串(`""`)替换匹配的标点符号，并返回一个干净的`word`。

有了转换函数，您可以使用`map()`对文本中的每个单词进行转换。它是这样工作的:

>>>

```py
>>> text = """Some people, when confronted with a problem, think
... "I know, I'll use regular expressions."
... Now they have two problems. Jamie Zawinski"""

>>> words = text.split()
>>> words
['Some', 'people,', 'when', 'confronted', 'with', 'a', 'problem,', 'think'
, '"I', 'know,', "I'll", 'use', 'regular', 'expressions."', 'Now', 'they',
 'have', 'two', 'problems.', 'Jamie', 'Zawinski']

>>> list(map(remove_punctuation, words))
['Some', 'people', 'when', 'confronted', 'with', 'a', 'problem', 'think',
'I', 'know', "I'll", 'use', 'regular', 'expressions', 'Now', 'they', 'have
', 'two', 'problems', 'Jamie', 'Zawinski']
```

在这段文字中，有些单词包含标点符号。例如，你用`'people,'`代替`'people'`，用`'problem,'`代替`'problem'`，等等。对`map()`的调用将`remove_punctuation()`应用于每个单词，并删除任何标点符号。所以，在第二个`list`中，你已经清理了单词。

请注意，撇号(`'`)不在您的正则表达式中，因为您希望像`I'll`这样的缩写保持原样。

### 实现凯撒密码算法

[罗马政治家朱利叶斯·凯撒](https://en.wikipedia.org/wiki/Julius_Caesar)，曾用一种密码对他发送给他的将军们的信息进行加密保护。一个[凯撒密码](https://en.wikipedia.org/wiki/Caesar_cipher)将每个字母移动若干个字母。例如，如果将字母`a`移动三位，则得到字母`d`，依此类推。

如果移位超出了字母表的末尾，那么你只需要旋转回到字母表的开头。在旋转三次的情况下，`x`将变成`a`。这是旋转后字母表的样子:

*   **原字母表:** `abcdefghijklmnopqrstuvwxyz`
*   **字母旋转三次:** `defghijklmnopqrstuvwxyzabc`

下面的代码实现了`rotate_chr()`，这个函数获取一个字符并将其旋转三圈。`rotate_chr()`将返回旋转后的字符。代码如下:

```py
 1def rotate_chr(c):
 2    rot_by = 3
 3    c = c.lower()
 4    alphabet = "abcdefghijklmnopqrstuvwxyz"
 5    # Keep punctuation and whitespace 6    if c not in alphabet:
 7        return c
 8    rotated_pos = ord(c) + rot_by
 9    # If the rotation is inside the alphabet 10    if rotated_pos <= ord(alphabet[-1]):
11        return chr(rotated_pos)
12    # If the rotation goes beyond the alphabet 13    return chr(rotated_pos - len(alphabet))
```

在`rotate_chr()`中，你首先检查这个字符是否在字母表中。如果不是，那么你返回相同的字符。这样做的目的是保留标点符号和其他不常用的字符。在第 8 行，您计算字符在字母表中新的旋转位置。为此，您使用内置函数 [`ord()`](https://docs.python.org/3/library/functions.html#ord) 。

`ord()`接受一个 [Unicode](https://realpython.com/python-encodings-guide/) 字符并返回一个表示输入字符的 **Unicode 码位**的整数。比如`ord("a")`返回`97`，`ord("b")`返回`98`:

>>>

```py
>>> ord("a")
97
>>> ord("b")
98
```

`ord()`以字符为参数，返回输入字符的 Unicode 码位。

如果你把这个整数加到目标数字`rot_by`上，那么你将得到新字母在字母表中的旋转位置。在这个例子中，`rot_by`就是`3`。所以，字母`"a"`旋转三圈后将成为位置`100`的字母，也就是字母`"d"`。字母`"b"`旋转三个会变成位置`101`的字母，也就是字母`"e"`，以此类推。

如果字母的新位置没有超出最后一个字母(`alphabet[-1]`)的位置，那么就在这个新位置返回字母。为此，您使用内置函数 [`chr()`](https://docs.python.org/3/library/functions.html#chr) 。

`chr()`是`ord()`的逆。它接受一个表示 Unicode 字符的 Unicode 码位的整数，并返回该位置的字符。比如`chr(97)`会返回`'a'`，`chr(98)`会返回`'b'`:

>>>

```py
>>> chr(97)
'a'
>>> chr(98)
'b'
```

`chr()`取一个表示字符的 Unicode 码位的整数，并返回相应的字符。

最后，如果新的旋转位置超出了最后一个字母(`alphabet[-1]`)的位置，那么就需要旋转回到字母表的开头。为此，您需要从旋转后的位置(`rotated_pos - len(alphabet)`)减去字母表的长度，然后使用`chr()`将字母返回到新的位置。

用`rotate_chr()`作为你的变换函数，你可以用`map()`用凯撒密码算法加密任何文本。下面是一个使用 [`str.join()`](https://realpython.com/lessons/concatenating-joining-strings-python/) 连接字符串的例子:

>>>

```py
>>> "".join(map(rotate_chr, "My secret message goes here."))
'pb vhfuhw phvvdjh jrhv khuh.'
```

字符串在 Python 中也是可迭代的。因此，对`map()`的调用将`rotate_chr()`应用于原始输入字符串中的每个字符。在这种情况下，`"M"`变成了`"p"`，`"y"`变成了`"b"`，以此类推。最后，对`str.join()`的调用将最终加密消息中的每个旋转字符连接起来。

[*Remove ads*](/account/join/)

## 用 Python 的`map()` 转换数字的迭代式

`map()`在处理和转换**数值**的迭代时也有很大的潜力。您可以执行各种数学和算术运算，将字符串值转换为浮点数或整数，等等。

在接下来的几节中，您将会看到一些如何使用`map()`来处理和转换数字的迭代的例子。

### 使用数学运算

使用数学运算转换可迭代数值的一个常见例子是使用[幂运算符(`**` )](https://docs.python.org/3/reference/expressions.html#the-power-operator) 。在下面的示例中，您编写了一个转换函数，该函数接受一个数字并返回该数字的平方和立方:

>>>

```py
>>> def powers(x):
...     return x ** 2, x ** 3
...

>>> numbers = [1, 2, 3, 4]

>>> list(map(powers, numbers))
[(1, 1), (4, 8), (9, 27), (16, 64)]
```

`powers()`取一个数`x`并返回它的平方和立方。由于 Python 将多个返回值作为元组来处理[，所以每次调用`powers()`都会返回一个包含两个值的元组。当您使用`powers()`作为参数调用`map()`时，您会得到一个元组列表，其中包含输入 iterable 中每个数字的平方和立方。](https://realpython.com/python-return-statement/#returning-multiple-values)

使用`map()`可以执行许多与数学相关的转换。您可以向每个值中添加常数，也可以从每个值中减去常数。您还可以使用 [`math`模块](https://realpython.com/python-math-module/)中的一些功能，如 [`sqrt()`](https://realpython.com/python-square-root-function/) 、 [`factorial()`](https://realpython.com/python-math-module/#find-factorials-with-python-factorial) 、 [`sin()`](https://docs.python.org/3/library/math.html#math.sin) 、 [`cos()`](https://docs.python.org/3/library/math.html#math.cos) 等等。这里有一个使用`factorial()`的例子:

>>>

```py
>>> import math

>>> numbers = [1, 2, 3, 4, 5, 6, 7]

>>> list(map(math.factorial, numbers))
[1, 2, 6, 24, 120, 720, 5040]
```

在这种情况下，您将`numbers`转换成一个新列表，其中包含原始列表中每个数字的阶乘。

您可以使用`map()`对可迭代的数字执行各种数学转换。你能在这个话题上走多远将取决于你的需求和你的想象力。考虑一下，编写你自己的例子！

### 转换温度

`map()`的另一个用例是在**测量单位**之间进行转换。假设您有一个以摄氏度或华氏度测量的温度列表，您需要将它们转换为以华氏度或摄氏度为单位的相应温度。

您可以编写两个转换函数来完成这项任务:

```py
def to_fahrenheit(c):
    return 9 / 5 * c + 32

def to_celsius(f):
    return (f - 32) * 5 / 9
```

`to_fahrenheit()`以摄氏度为单位进行温度测量，并将其转换为华氏度。类似地，`to_celsius()`采用华氏温度并将其转换为摄氏温度。

这些函数就是你的转换函数。您可以将它们与`map()`一起使用，分别将温度测量值转换为华氏温度和摄氏温度:

>>>

```py
>>> celsius_temps = [100, 40, 80]
>>> # Convert to Fahrenheit
>>> list(map(to_fahrenheit, celsius_temps))
[212.0, 104.0, 176.0]

>>> fahr_temps = [212, 104, 176]
>>> # Convert to Celsius
>>> list(map(to_celsius, fahr_temps))
[100.0, 40.0, 80.0]
```

如果你用`to_fahrenheit()`和`celsius_temps`调用`map()`，那么你会得到一个华氏温度的度量列表。如果您用`to_celsius()`和`fahr_temps`调用`map()`，那么您会得到一个以摄氏度为单位的温度测量列表。

为了扩展这个例子并涵盖任何其他类型的单位转换，您只需要编写一个适当的转换函数。

[*Remove ads*](/account/join/)

### 将字符串转换为数字

当处理数字数据时，您可能会遇到所有数据都是字符串值的情况。要做进一步的计算，您需要将字符串值转换成数值。对这些情况也有帮助。

如果你确定你的数据是干净的，没有包含错误的值，那么你可以根据你的需要直接使用 [`float()`](https://docs.python.org/3/library/functions.html#float) 或者`int()`。以下是一些例子:

>>>

```py
>>> # Convert to floating-point
>>> list(map(float, ["12.3", "3.3", "-15.2"]))
[12.3, 3.3, -15.2]

>>> # Convert to integer
>>> list(map(int, ["12", "3", "-15"]))
[12, 3, -15]
```

在第一个例子中，您使用`float()`和`map()`将所有值从字符串值转换为[浮点值](https://realpython.com/python-numbers/#floating-point-numbers)。在第二种情况下，使用`int()`将字符串转换为[整数](https://realpython.com/python-numbers/#integers)。注意，如果其中一个值不是有效的数字，那么您将得到一个 [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) 。

如果您不确定您的数据是否干净，那么您可以使用一个更复杂的转换函数，如下所示:

>>>

```py
>>> def to_float(number):
...     try:
...         return float(number.replace(",", "."))
...     except ValueError:
...         return float("nan")
...

>>> list(map(to_float, ["12.3", "3,3", "-15.2", "One"]))
[12.3, 3.3, -15.2, nan]
```

在`to_float()`中，您使用了一个 [`try`语句](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions)，如果在转换`number`时`float()`失败，该语句将捕获一个`ValueError`。如果没有错误发生，那么您的函数返回转换成有效浮点数的`number`。否则，您会得到一个 [`nan`(不是数字)值](https://docs.python.org/3/library/functions.html#float)，这是一个特殊的`float`值，您可以用它来表示不是有效数字的值，就像上面例子中的`"One"`一样。

可以根据需要定制`to_float()`。例如，您可以用语句`return 0.0`替换语句`return float("nan")`，等等。

## 将`map()`与其他功能工具结合

到目前为止，您已经讲述了如何使用`map()`来完成不同的涉及 iterables 的任务。但是，如果您使用`map()`和其他功能工具，如 [`filter()`](https://docs.python.org/3/library/functions.html#filter) 和 [`reduce()`](https://realpython.com/python-reduce-function/) ，那么您可以对您的可迭代对象执行更复杂的转换。这就是您将在接下来的两个部分中涉及的内容。

### `map()`和`filter()`

有时，您需要处理一个输入可迭代对象，并返回另一个可迭代对象，该对象是通过过滤掉输入可迭代对象中不需要的值而得到的。那样的话，Python 的 [`filter()`](https://realpython.com/python-filter-function/) 可以是你不错的选择。`filter()`是一个内置函数，接受两个位置参数:

1.  **`function`** 将是一个[谓词或布尔值函数](https://en.wikipedia.org/wiki/Boolean-valued_function)，一个根据输入数据返回`True`或`False`的函数。
2.  **`iterable`** 将是任何 Python 可迭代的。

`filter()`产生`function`返回`True`的输入`iterable`的项目。如果您将`None`传递给`function`，那么`filter()`将使用 identity 函数。这意味着`filter()`将检查`iterable`中每个项目的真值，并过滤掉所有为[假值](https://realpython.com/python-operators-expressions/#evaluation-of-non-boolean-values-in-boolean-context)的项目。

为了说明如何使用`map()`和`filter()`，假设您需要计算列表中所有值的[平方根](https://realpython.com/python-square-root-function/)。因为您的列表可能包含负值，所以您会得到一个错误，因为平方根不是为负数定义的:

>>>

```py
>>> import math

>>> math.sqrt(-16)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    math.sqrt(-16)
ValueError: math domain error
```

以负数为自变量，`math.sqrt()`引出一个`ValueError`。为了避免这个问题，你可以使用`filter()`来过滤掉所有的负值，然后找到剩余正值的平方根。看看下面的例子:

>>>

```py
>>> import math

>>> def is_positive(num):
...     return num >= 0
...

>>> def sanitized_sqrt(numbers):
...     cleaned_iter = map(math.sqrt, filter(is_positive, numbers))
...     return list(cleaned_iter)
...

>>> sanitized_sqrt([25, 9, 81, -16, 0])
[5.0, 3.0, 9.0, 0.0]
```

`is_positive()`是一个谓词函数，它将一个数字作为参数，如果该数字大于或等于零，则返回`True`。你可以通过`is_positive()`到`filter()`来清除`numbers`的所有负数。因此，对`map()`的调用将只处理正数，而`math.sqrt()`不会给你一个`ValueError`。

[*Remove ads*](/account/join/)

### `map()`和`reduce()`

Python 的 [`reduce()`](https://realpython.com/python-reduce-function/) 是一个函数，驻留在 Python 标准库中一个名为 [`functools`](https://realpython.com/lessons/functools-module/) 的模块中。`reduce()`是 Python 中的另一个核心函数工具，当你需要将一个函数应用于一个可迭代对象并将其简化为一个累积值时，它非常有用。这种操作俗称 [**缩小或**](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) 。`reduce()`需要两个参数:

1.  **`function`** 可以是任何接受两个参数并返回值的 Python 可调用函数。
2.  **`iterable`** 可以是任何 Python 可迭代的。

`reduce()`将`function`应用于`iterable`中的所有项目，并累计计算出最终值。

下面这个例子结合了`map()`和`reduce()`来计算您的主目录中所有文件的累积总大小:

>>>

```py
>>> import functools
>>> import operator
>>> import os
>>> import os.path

>>> files = os.listdir(os.path.expanduser("~"))

>>> functools.reduce(operator.add, map(os.path.getsize, files))
4377381
```

在这个例子中，您调用 [`os.path.expanduser("~")`](https://docs.python.org/3/library/os.path.html#os.path.expanduser) 来获得您的主目录的路径。然后你调用这个路径上的 [`os.listdir()`](https://docs.python.org/3/library/os.html#os.listdir) 来获得一个包含所有文件路径的列表。

对`map()`的调用使用 [`os.path.getsize()`](https://docs.python.org/3/library/os.path.html#os.path.getsize) 来获取每个文件的大小。最后，您使用`reduce()`和 [`operator.add()`](https://docs.python.org/3/library/operator.html#operator.add) 来获得每个文件大小的累积和。最终结果是您的主目录中所有文件的总大小，以字节为单位。

**注:**几年前，谷歌开发并开始使用一种编程模型，他们称之为 [MapReduce](https://en.wikipedia.org/wiki/MapReduce) 。这是一种新的数据处理方式，旨在使用[集群](https://en.wikipedia.org/wiki/Cluster_(computing))上的[并行](https://en.wikipedia.org/wiki/Parallel_computing)和[分布式计算](https://en.wikipedia.org/wiki/Distributed_computing)来管理[大数据](https://en.wikipedia.org/wiki/Big_data)。

这个模型的灵感来自于函数式编程中常用的**映射**和**归约**操作的组合。

MapReduce 模型对谷歌在合理的时间内处理大量数据的能力产生了巨大的影响。然而，到 2014 年，谷歌不再使用 MapReduce 作为他们的主要处理模型。

如今，你可以找到一些 MapReduce 的替代实现，比如 Apache Hadoop T1，这是一个使用 MapReduce 模型的开源软件工具的集合。

尽管您可以使用`reduce()`来解决本节中涉及的问题，但是 Python 提供了其他工具来实现更 Python 化和更高效的解决方案。例如，您可以使用内置函数`sum()`来计算您的主目录中文件的总大小:

>>>

```py
>>> import os
>>> import os.path

>>> files = os.listdir(os.path.expanduser("~"))

>>> sum(map(os.path.getsize, files))
4377381
```

这个例子比你之前看到的例子可读性更强，效率更高。如果您想更深入地了解如何使用`reduce()`以及可以使用哪些替代工具以 Python 的方式取代`reduce()`，那么请查看 [Python 的 reduce():从函数式到 Python 式](https://realpython.com/python-reduce-function/)。

## 用`starmap()` 处理基于元组的可重复项

Python 的 **`itertools.starmap()`** 构造了一个迭代器，它将函数应用于从元组的可迭代对象中获得的参数，并产生结果。当您处理已经分组为元组的可重复项时，这很有用。

`map()`和`starmap()`的主要区别在于，后者使用[解包操作符(`*` )](https://www.python.org/dev/peps/pep-0448) 调用其转换函数，将每个参数元组解包成几个位置参数。因此，转换函数被称为`function(*args)`，而不是`function(arg1, arg2,... argN)`。

`starmap()` 的[官方文档称该函数大致相当于以下 Python 函数:](https://docs.python.org/3/library/itertools.html#itertools.starmap)

```py
def starmap(function, iterable):
    for args in iterable:
        yield function(*args)
```

该函数中的`for`循环迭代`iterable`中的项目，并产生转换后的项目。对`function(*args)`的调用使用解包操作符将元组解包成几个位置参数。下面是一些`starmap()`如何工作的例子:

>>>

```py
>>> from itertools import starmap

>>> list(starmap(pow, [(2, 7), (4, 3)]))
[128, 64]

>>> list(starmap(ord, [(2, 7), (4, 3)]))
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    list(starmap(ord, [(2, 7), (4, 3)]))
TypeError: ord() takes exactly one argument (2 given)
```

在第一个示例中，您使用 [`pow()`](https://docs.python.org/3/library/functions.html#pow) 来计算每个元组中第一个值的第二次幂。元组将采用`(base, exponent)`的形式。

如果 iterable 中的每个元组都有两个条目，那么`function`也必须有两个参数。如果元组有三项，那么`function`必须有三个参数，依此类推。否则，你会得到一个 [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError) 。

如果您使用`map()`而不是`starmap()`，那么您将得到不同的结果，因为`map()`从每个元组中取出一个项目:

>>>

```py
>>> list(map(pow, (2, 7), (4, 3)))
[16, 343]
```

注意`map()`采用两个元组，而不是一个元组列表。`map()`在每次迭代中也从每个元组中取一个值。为了让`map()`返回与`starmap()`相同的结果，您需要交换值:

>>>

```py
>>> list(map(pow, (2, 4), (7, 3)))
[128, 64]
```

在这种情况下，您有两个元组，而不是一个元组列表。你还交换了`7`和`4`。现在，第一个元组提供了基数，第二个元组提供了指数。

[*Remove ads*](/account/join/)

## Pythonic 风格编码:替换`map()`

像`map()`、`filter()`和`reduce()`这样的函数式编程工具已经存在很久了。然而，[列表理解](https://realpython.com/courses/using-list-comprehensions-effectively/)和[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)几乎在每个用例中都成为它们的自然替代品。

例如，`map()`提供的功能几乎总是用列表理解或生成器表达式来更好地表达。在接下来的两节中，您将学习如何用列表理解或生成器表达式替换对`map()`的调用，以使您的代码更具可读性和 Pythonic 性。

### 使用列表理解

有一个通用的模式，你可以用列表理解来代替对`map()`的调用。方法如下:

```py
# Generating a list with map
list(map(function, iterable))

# Generating a list with a list comprehension
[function(x) for x in iterable]
```

注意，列表理解几乎总是比调用`map()`读起来更清楚。因为列表理解在 Python 开发人员中非常流行，所以到处都能找到它们。因此，用列表理解替换对`map()`的调用将使您的代码对其他 Python 开发人员来说更熟悉。

Here’s an example of how to replace `map()` with a list comprehension to build a list of square numbers:

>>>

```py
>>> # Transformation function
>>> def square(number):
...     return number ** 2

>>> numbers = [1, 2, 3, 4, 5, 6]

>>> # Using map()
>>> list(map(square, numbers))
[1, 4, 9, 16, 25, 36]

>>> # Using a list comprehension
>>> [square(x) for x in numbers]
[1, 4, 9, 16, 25, 36]
```

如果你比较两种解决方案，那么你可能会说，使用列表理解的解决方案更具可读性，因为它读起来几乎像普通英语。此外，列表理解避免了显式调用`map()`上的`list()`来构建最终列表的需要。

### 使用生成器表达式

`map()`返回一个**地图对象**，它是一个迭代器，根据需要生成项目。因此，`map()`的自然替代物是[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)，因为生成器表达式返回生成器对象，这些对象也是按需生成项目的迭代器。

众所周知，Python 迭代器在内存消耗方面非常高效。这就是为什么`map()`现在返回迭代器而不是`list`的原因。

列表理解和生成器表达式之间有微小的语法差异。第一个使用一对方括号(`[]`)来分隔表达式。第二种使用一对括号(`()`)。因此，要将列表理解转化为生成器表达式，只需用圆括号替换方括号。

您可以使用生成器表达式来编写比使用`map()`的代码更清晰的代码。看看下面的例子:

>>>

```py
>>> # Transformation function
>>> def square(number):
...     return number ** 2

>>> numbers = [1, 2, 3, 4, 5, 6]

>>> # Using map()
>>> map_obj = map(square, numbers)
>>> map_obj
<map object at 0x7f254d180a60>

>>> list(map_obj)
[1, 4, 9, 16, 25, 36]

>>> # Using a generator expression
>>> gen_exp = (square(x) for x in numbers)
>>> gen_exp
<generator object <genexpr> at 0x7f254e056890>

>>> list(gen_exp)
[1, 4, 9, 16, 25, 36]
```

这段代码与上一节中的代码有一个主要区别:您将方括号改为一对圆括号，将列表理解转换为生成器表达式。

生成器表达式通常用作函数调用中的参数。在这种情况下，您不需要使用括号来创建生成器表达式，因为用于调用函数的括号也提供了构建生成器的语法。有了这个想法，你可以像这样调用`list()`得到和上面例子一样的结果:

>>>

```py
>>> list(square(x) for x in numbers)
[1, 4, 9, 16, 25, 36]
```

如果在函数调用中使用生成器表达式作为参数，那么就不需要额外的一对括号。用于调用函数的括号提供了构建生成器的语法。

生成器表达式在内存消耗方面和`map()`一样高效，因为它们都返回按需生成条目的迭代器。然而，生成器表达式几乎总能提高代码的可读性。在其他 Python 开发人员看来，它们也使您的代码更加 Python 化。

[*Remove ads*](/account/join/)

## 结论

Python 的 [**`map()`**](https://docs.python.org/3/library/functions.html#map) 可以让你对 iterables 进行 [**映射**](https://en.wikipedia.org/wiki/Map_(higher-order_function)) 操作。映射操作包括将**转换函数**应用于 iterable 中的项目，以生成转换后的 iterable。一般来说，`map()`将允许您在不使用显式循环的情况下处理和转换可迭代对象。

在本教程中，您已经学习了`map()`是如何工作的，以及如何使用它来处理 iterables。您还了解了一些可以用来替换代码中的`map()`的[python 化的](https://realpython.com/learning-paths/writing-pythonic-code/)工具。

**你现在知道如何:**

*   用 Python 创作的 **`map()`**
*   使用`map()`到**处理**和**转换**迭代，而不使用显式循环
*   将`map()`与 **`filter()`** 和 **`reduce()`** 等函数结合起来执行复杂的变换
*   用类似于**列表理解**和**生成器表达式**的工具替换`map()`

有了这些新知识，你将能够在你的代码中使用`map()`，并且用[函数式编程风格](https://realpython.com/courses/functional-programming-python/)来处理你的代码。您还可以通过用一个[列表理解](https://realpython.com/list-comprehension-python/)或一个[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)替换`map()`来切换到一个更 Pythonic 和现代的风格。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**Python 的 map()函数:变换 Iterables**](/courses/map-function-transform-iterables/)************
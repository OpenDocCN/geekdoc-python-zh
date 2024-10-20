# python 201:ITER tools 简介

> 原文：<https://www.blog.pythonlibrary.org/2016/04/20/python-201-an-intro-to-itertools/>

Python 为创建自己的迭代器提供了一个很好的模块。我所指的模块是 **itertools** 。itertools 提供的工具速度快，内存效率高。您将能够利用这些构建块来创建您自己的专用迭代器，这些迭代器可用于高效的循环。在这一章中，我们将会看到每个构建模块的例子，这样到最后你就会明白如何在你自己的代码库中使用它们。

让我们从看一些无限迭代器开始吧！

* * *

### 无限迭代器

itertools 包附带了三个迭代器，可以无限迭代。这意味着，当你使用它们的时候，你需要明白你最终将需要脱离这些迭代器，否则你将会有一个无限循环。

例如，这对于生成数字或在未知长度的迭代上循环很有用。让我们开始了解这些有趣的项目吧！

#### 计数(开始=0，步进=1)

**count** 迭代器将从您作为其 **start** 参数传入的数字开始返回均匀间隔的值。计数也接受一个**步**参数。让我们看一个简单的例子:

```py

>>> from itertools import count
>>> for i in count(10):
...     if i > 20: 
...         break
...     else:
...         print(i)
... 
10
11
12
13
14
15
16
17
18
19
20

```

这里我们从 itertools 导入**计数**，并为循环创建一个**。我们添加了一个条件检查，如果迭代器超过 20，它将退出循环，否则它将打印出我们在迭代器中的位置。你会注意到输出从 10 开始，因为这是我们传递给 **count** 的初始值。**

限制这个无限迭代器输出的另一种方法是使用 itertools 的另一个子模块，即 **islice** 。方法如下:

```py

>>> from itertools import islice
>>> for i in islice(count(10), 5):
...     print(i)
... 
10
11
12
13
14

```

在这里，我们导入**is ice**并循环**计数**，从 10 开始，在 5 个项目后结束。您可能已经猜到，islice 的第二个参数是何时停止迭代。但这并不意味着“当我到达数字 5 时停止”。相反，它意味着“当我们达到五次迭代时停止”。

#### 周期(可迭代)

itertools 的**循环**迭代器允许你创建一个迭代器，它将无限循环遍历一系列值。让我们给它传递一个 3 个字母的字符串，看看会发生什么:

```py

>>> from itertools import cycle
>>> count = 0
>>> for item in cycle('XYZ'):
...     if count > 7:
...         break
...     print(item)
...     count += 1
... 
X
Y
Z
X
Y
Z
X
Y

```

这里我们为创建一个**循环来循环三个字母:XYZ 的无限循环。当然，我们不想真的永远循环下去，所以我们添加了一个简单的计数器来打破循环。**

您还可以使用 Python 的 **next** 内置来迭代您用 itertools 创建的迭代器:

```py

>>> polys = ['triangle', 'square', 'pentagon', 'rectangle']
 >>> iterator = cycle(polys)
 >>> next(iterator)
 'triangle'
 >>> next(iterator)
 'square'
 >>> next(iterator)
 'pentagon'
 >>> next(iterator)
 'rectangle'
 >>> next(iterator)
 'triangle'
 >>> next(iterator)
 'square'

```

在上面的代码中，我们创建了一个简单的多边形列表，并将它们传递给**周期**。我们将新的迭代器保存到一个变量中，然后将该变量传递给下一个函数**。每次我们调用 next，它都会返回迭代器中的下一个值。由于这个迭代器是无限的，我们可以整天调用 next，永远不会用完所有的项。**

#### 重复(对象[，次])

**重复**迭代器将一次又一次地返回一个对象，除非你设置它的**乘以**参数。它与**周期**非常相似，除了它不在一组值上重复循环。让我们看一个简单的例子:

```py

>>> from itertools import repeat
>>> repeat(5, 5)
repeat(5, 5)
>>> iterator = repeat(5, 5)
>>> next(iterator)
5
>>> next(iterator)
5
>>> next(iterator)
5
>>> next(iterator)
5
>>> next(iterator)
5
>>> next(iterator)
Traceback (most recent call last):
  Python Shell, prompt 21, line 1
builtins.StopIteration:

```

这里我们导入 **repeat** 并告诉它重复数字 5 五次。然后我们在新的迭代器上调用 **next** 六次，看看它是否工作正常。当您运行这段代码时，您会看到 **StopIteration** 被抛出，因为我们已经用完了迭代器中的值。

* * *

### 终止的迭代器

用 itertools 创建的大多数迭代器都不是无限的。在这一节中，我们将学习 itertools 的有限迭代器。为了获得可读的输出，我们将使用 Python 内置的 **list** 类型。如果你不使用**列表**，那么你只会得到一个打印出来的 itertools 对象。

#### 累加(iterable[，func])

**accumulate** 迭代器将返回两个参数函数的累加和或累加结果，您可以将它们传递给 **accumulate** 。accumulate 的默认值是加法，所以让我们快速尝试一下:

```py

>> from itertools import accumulate
>>> list(accumulate(range(10)))
[0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

```

这里我们导入**累加**并传递给它一个范围为 0-9 的 10 个数字。它依次将它们相加，因此第一个是 0，第二个是 0+1，第三个是 1+2，依此类推。现在让我们导入**操作符**模块，并将其添加到组合中:

```py

>>> import operator
>>> list(accumulate(range(1, 5), operator.mul))
[1, 2, 6, 24]

```

这里我们将数字 1-4 传递给我们的**累积**迭代器。我们也给它传递了一个函数: **operator.mul** 。这个函数接受要相乘的参数。因此，对于每次迭代，它都是乘法而不是加法(1x1=1，1x2=2，2x3=6，等等)。

accumulate 的文档显示了一些其他有趣的例子，如贷款的分期偿还或混乱的递归关系。你一定要看看这些例子，因为它们值得你花时间去做。

#### 链(*iterables)

**链**迭代器将接受一系列可迭代对象，并基本上将它们展平成一个长的可迭代对象。事实上，我最近正在帮助的一个项目需要它的帮助。基本上，我们有一个已经包含一些条目的列表和另外两个想要添加到原始列表中的列表，但是我们只想添加每个列表中的条目。最初我尝试这样做:

```py

>>> my_list = ['foo', 'bar']
>>> numbers = list(range(5))
>>> cmd = ['ls', '/some/dir']
>>> my_list.extend(cmd, numbers)
>>> my_list
['foo', 'bar', ['ls', '/some/dir'], [0, 1, 2, 3, 4]]

```

嗯，这并不完全是我想要的方式。itertools 模块提供了一种更加优雅的方式，使用**链**将这些列表合并成一个列表:

```py

>>> from itertools import chain
>>> my_list = list(chain(['foo', 'bar'], cmd, numbers))
>>> my_list
['foo', 'bar', 'ls', '/some/dir', 0, 1, 2, 3, 4]

```

我的更精明的读者可能会注意到，实际上有另一种方法可以在不使用 itertools 的情况下完成同样的事情。您可以这样做来获得相同的效果:

```py

>>> my_list = ['foo', 'bar']
>>> my_list += cmd + numbers
>>> my_list
['foo', 'bar', 'ls', '/some/dir', 0, 1, 2, 3, 4]

```

这两种方法当然都是有效的，在我了解链之前，我可能会走这条路，但我认为在这种特殊情况下，链是一种更优雅、更容易理解的解决方案。

#### chain.from_iterable(iterable)

您也可以使用**链**的方法，从可迭代的调用**。这种方法的工作原理与直接使用链略有不同。您必须传入嵌套列表，而不是传入一系列 iterables。让我们看看:**

```py

>>> from itertools import chain
>>> numbers = list(range(5))
>>> cmd = ['ls', '/some/dir']
>>> chain.from_iterable(cmd, numbers)
Traceback (most recent call last):
  Python Shell, prompt 66, line 1
builtins.TypeError: from_iterable() takes exactly one argument (2 given)
>>> list(chain.from_iterable([cmd, numbers]))
['ls', '/some/dir', 0, 1, 2, 3, 4]

```

这里我们像以前一样导入链。我们尝试传入两个列表，但最终得到了一个 **TypeError** ！为了解决这个问题，我们稍微修改了一下我们的调用，将 **cmd** 和**数字**放在一个**列表**中，然后将这个嵌套列表传递给 **from_iterable** 。这是一个微妙的区别，但仍然易于使用！

#### 压缩(数据，选择器)

**compress** 子模块对于过滤第一个 iterable 和第二个 iterable 非常有用。这是通过使第二个 iterable 成为一个布尔列表(或者等同于同一事物的 1 和 0)来实现的。它是这样工作的:

```py

>>> from itertools import compress
>>> letters = 'ABCDEFG'
>>> bools = [True, False, True, True, False]
>>> list(compress(letters, bools))
['A', 'C', 'D']

```

在这个例子中，我们有一组七个字母和一个五个布尔的列表。然后我们将它们传递给压缩函数。compress 函数将遍历每个相应的 iterable，并对照第二个 iterable 检查第一个 iterable。如果第二个有匹配的 True，那么它将被保留。如果它是假的，那么这个项目将被删除。因此，如果你研究上面的例子，你会注意到我们在第一，第三和第四个位置有一个 True，分别对应于 A，C 和 d。

#### dropwhile(谓词，可迭代)

itertools 中有一个简洁的小迭代器叫做 **dropwhile** 只要过滤条件为真，这个有趣的小迭代器就会删除元素。因此，在谓词变为 False 之前，您不会看到这个迭代器的任何输出。这可能会使启动时间变得很长，因此需要注意这一点。

让我们看看 Python 文档中的一个例子:

```py

>>> from itertools import dropwhile
>>> list(dropwhile(lambda x: x<5, [1,4,6,4,1]))
[6, 4, 1]

```

这里我们导入了 **dropwhile** ，然后我们给它传递了一个简单的 **lambda** 语句。如果 **x** 小于 5，该函数将返回 True。否则它将返回 False。dropwhile 函数将遍历列表并将每个元素传递给 lambda。如果 lambda 返回 True，那么该值将被删除。一旦我们到达数字 6，lambda 返回 False，我们保留数字 6 和它后面的所有值。

当我学习新东西时，我发现在 lambda 上使用常规函数很有用。因此，让我们颠倒一下，创建一个如果数字大于 5 则返回 True 的函数。

```py

>>> from itertools import dropwhile
>>> def greater_than_five(x):
...     return x > 5 
... 
>>> list(dropwhile(greater_than_five, [6, 7, 8, 9, 1, 2, 3, 10]))
[1, 2, 3, 10]

```

这里我们在 Python 的解释器中创建了一个简单的函数。这个函数是我们的谓词或过滤器。如果我们传递给它的值为真，那么它们将被丢弃。一旦我们找到一个小于 5 的值，那么在这个值之后的所有值都将被保留，这在上面的例子中可以看到。

#### filterfalse(谓词，可迭代)

itertools 的 **filterfalse** 函数与 **dropwhile** 非常相似。然而，filterfalse 不会删除与 True 匹配的值，而只会返回那些计算结果为 false 的值。让我们使用上一节中的函数来说明:

```py

>>> from itertools import filterfalse
>>> def greater_than_five(x):
...     return x > 5 
... 
>>> list(filterfalse(greater_than_five, [6, 7, 8, 9, 1, 2, 3, 10]))
[1, 2, 3]

```

这里我们传递 filterfalse 我们的函数和一个整数列表。如果整数小于 5，则保留该整数。否则就扔掉。你会注意到我们的结果只有 1，2 和 3。与 dropwhile 不同，filterfalse 将根据我们的谓词检查每个值。

#### groupby(iterable，key=None)

groupby 迭代器将从 iterable 中返回连续的键和组。如果没有例子，你很难理解这一点。所以我们来看一个吧！将以下代码放入您的解释器或保存在文件中:

```py

from itertools import groupby

vehicles = [('Ford', 'Taurus'), ('Dodge', 'Durango'),
            ('Chevrolet', 'Cobalt'), ('Ford', 'F150'),
            ('Dodge', 'Charger'), ('Ford', 'GT')]

sorted_vehicles = sorted(vehicles)

for key, group in groupby(sorted_vehicles, lambda make: make[0]):
    for make, model in group:
        print('{model} is made by {make}'.format(model=model,
                                                 make=make))
	print ("**** END OF GROUP ***\n")

```

这里我们导入 **groupby** ，然后创建一个元组列表。然后，我们对数据进行排序，以便在输出时更有意义，它还让 groupby 正确地对项目进行分组。接下来，我们实际上是在 groupby 返回的迭代器上循环，这给了我们键和组。然后我们循环遍历这个组，并打印出其中的内容。如果您运行这段代码，您应该会看到类似这样的内容:

```py

Cobalt is made by Chevrolet
**** END OF GROUP ***

Charger is made by Dodge
Durango is made by Dodge
**** END OF GROUP ***

F150 is made by Ford
GT is made by Ford
Taurus is made by Ford
**** END OF GROUP ***

```

只是为了好玩，试着修改一下代码，让你传入的是**辆车**而不是**辆车**。如果您这样做，您将很快了解为什么应该在通过 groupby 运行数据之前对其进行排序。

#### islice(可重复，开始，停止[，步进])

我们实际上在**计数**部分提到过 **islice** 。但在这里，我们将更深入地研究它。islice 是一个迭代器，它从 iterable 中返回选定的元素。这是一种不透明的说法。基本上，islice 所做的是通过索引获取 iterable(你迭代的对象)的一部分，并以迭代器的形式返回所选择的项。islice 实际上有两个实现。有 **itertools.islice(iterable，stop)** 还有更接近常规 Python 切片的 islice 版本: **islice(iterable，start，stop[，step])** 。

让我们看看第一个版本，看看它是如何工作的:

```py

>>> from itertools import islice
>>> iterator = islice('123456', 4)
>>> next(iterator)
'1'
>>> next(iterator)
'2'
>>> next(iterator)
'3'
>>> next(iterator)
'4'
>>> next(iterator)
Traceback (most recent call last):
  Python Shell, prompt 15, line 1
builtins.StopIteration:

```

在上面的代码中，我们向 islice 传递了一个由六个字符组成的字符串，以及作为停止参数的数字 4。这意味着 islice 返回的迭代器将包含字符串的前 4 项。我们可以通过在迭代器上调用**next**四次来验证这一点，这就是我们上面所做的。Python 足够聪明，知道如果只有两个参数传递给 islice，那么第二个参数就是 **stop** 参数。

让我们试着给它三个参数，来证明你可以给它一个开始参数和一个停止参数。itertools 的 **count** 工具可以帮助我们说明这个概念:

```py

>>> from itertools import islice
>>> from itertools import count
>>> for i in islice(count(), 3, 15):
...     print(i)
... 
3
4
5
6
7
8
9
10
11
12
13
14

```

在这里，我们只是调用 count，告诉它从数字 3 开始，到 15 时停止。这就像做切片一样，只不过你是对迭代器做切片，然后返回一个新的迭代器！

#### 星图(函数，可迭代)

工具将创建一个迭代器，它可以使用提供的函数和 iterable 进行计算。正如文档中提到的，“map()和 starmap()之间的区别类似于*函数(a，b)* 和*函数(*c)* 。”

让我们看一个简单的例子:

```py

>>> from itertools import starmap
>>> def add(a, b):
...     return a+b
... 
>>> for item in starmap(add, [(2,3), (4,5)]):
...     print(item)
... 
5
9

```

这里我们创建了一个简单的加法函数，它接受两个参数。接下来，我们为循环创建一个**并调用 **starmap** ，将函数作为第一个参数，并为 iterable 创建一个元组列表。然后，starmap 函数会将每个元组元素传递到函数中，并返回结果的迭代器，我们会打印出来。**

#### takewhile(谓词，可迭代)

**takewhile** 模块基本上与我们之前看到的 **dropwhile** 迭代器相反。takewhile 将创建一个迭代器，只要我们的谓词或过滤器为真，它就从 iterable 返回元素。让我们尝试一个简单的例子来看看它是如何工作的:

```py

>>> from itertools import takewhile
>>> list(takewhile(lambda x: x<5, [1,4,6,4,1]))
[1, 4]

```

这里我们使用 lambda 函数和一个列表运行 takewhile。输出只是 iterable 的前两个整数。原因是 1 和 4 都小于 5，但 6 更大。因此，一旦 takewhile 看到 6，条件就变为 False，它将忽略 iterable 中的其余项。

#### 三通(可迭代，n=2)

**tee** 工具将从单个可迭代对象中创建*n*个迭代器。这意味着你可以从一个 iterable 创建多个迭代器。让我们来看一些解释其工作原理的代码:

```py

>>> from itertools import tee
>>> data = 'ABCDE'
>>> iter1, iter2 = tee(data)
>>> for item in iter1:
...     print(item)
... 
A
B
C
D
E
>>> for item in iter2:
...     print(item)
... 
A
B
C
D
E

```

这里我们创建一个 5 个字母的字符串，并将其传递给 **tee** 。因为 tee 默认为 2，所以我们使用多重赋值来获取从 tee 返回的两个迭代器。最后，我们遍历每个迭代器并打印出它们的内容。如你所见，它们的内容是一样的。

#### zip_longest(*iterables，fillvalue=None)

**zip_longest**迭代器可用于将两个可迭代对象压缩在一起。如果 iterables 的长度不一样，那么也可以传入一个**fillvalue**。让我们看一个基于这个函数的文档的愚蠢的例子:

```py

>>> from itertools import zip_longest
>>> for item in zip_longest('ABCD', 'xy', fillvalue='BLANK'):
...     print (item)
... 
('A', 'x')
('B', 'y')
('C', 'BLANK')
('D', 'BLANK')

```

在这段代码中，我们导入 zip_longest，然后向它传递两个字符串以压缩在一起。您会注意到第一个字符串有 4 个字符长，而第二个只有 2 个字符长。我们还设置了一个填充值“空白”。当我们循环并打印出来时，你会看到我们得到了返回的元组。前两个元组分别是每个字符串的第一个和第二个字母的组合。最后两个插入了我们的填充值。

应该注意的是，如果传递给 zip_longest 的 iterable 可能是无限的，那么应该用 islice 之类的东西包装这个函数，以限制调用的次数。

* * *

### 组合生成器

itertools 库包含四个迭代器，可用于创建数据的组合和排列。我们将在这一节讨论这些有趣的迭代器。

#### 组合(iterable，r)

如果您需要创建组合，Python 已经为您提供了 **itertools.combinations** 。组合可以让你从一个一定长度的 iterable 中创建一个迭代器。让我们来看看:

```py

>>>from itertools import combinations
>>>list(combinations('WXYZ', 2))
[('W', 'X'), ('W', 'Y'), ('W', 'Z'), ('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]

```

当您运行它时，您会注意到 combinations 返回元组。为了使这个输出更具可读性，让我们循环遍历迭代器，并将元组连接成一个字符串:

```py

>>> for item in combinations('WXYZ', 2):
...     print(''.join(item))
... 
WX
WY
WZ
XY
XZ
YZ

```

现在更容易看到各种组合了。注意，combinations 函数的组合是按字典顺序排序的，所以如果 iterable 排序了，那么组合元组也会排序。同样值得注意的是，如果所有输入元素都是唯一的，组合不会在组合中产生重复值。

#### combinations with _ replacement(iterable，r)

带有迭代器的* * combinations _ with _ replacement * *与**combinations**非常相似。唯一的区别是，它实际上会创建元素重复的组合。让我们用上一节中的一个例子来说明:

```py

>>> from itertools import combinations_with_replacement
>>> for item in combinations_with_replacement('WXYZ', 2):
...     print(''.join(item))
... 
WW
WX
WY
WZ
XX
XY
XZ
YY
YZ
ZZ

```

如您所见，我们现在有四个新的输出项:WW、XX、YY 和 ZZ。

#### 产品(*iterables，repeat=1)

itertools 包有一个简洁的小函数，用于从一系列输入的可迭代对象中创建笛卡尔乘积。没错，那个功能就是**产品* *。让我们看看它是如何工作的！

```py

>>> from itertools import product
>>> arrays = [(-1,1), (-3,3), (-5,5)]
>>> cp = list(product(*arrays))
>>> cp
[(-1, -3, -5),
 (-1, -3, 5),
 (-1, 3, -5),
 (-1, 3, 5),
 (1, -3, -5),
 (1, -3, 5),
 (1, 3, -5),
 (1, 3, 5)]

```

在这里，我们导入 product，然后建立一个元组列表，并将其分配给变量**arrays**。接下来，我们称之为具有这些阵列的产品。你会注意到我们用***数组**来调用它。这将导致列表被“展开”或按顺序应用于产品功能。这意味着你要传入三个参数而不是一个。如果你愿意的话，试着在数组前面加上星号来调用它，看看会发生什么。

#### 排列

itertools 的**置换**子模块将从您给定的 iterable 返回连续的 *r* 长度的元素置换。与 combinations 函数非常相似，排列是按字典排序顺序发出的。让我们来看看:

```py

>>> from itertools import permutations
>>> for item in permutations('WXYZ', 2):
...     print(''.join(item))
... 
WX
WY
WZ
XW
XY
XZ
YW
YX
YZ
ZW
ZX
ZY

```

你会注意到输出比组合的输出要长很多。当您使用置换时，它将遍历字符串的所有置换，但如果输入元素是唯一的，它不会重复值。

* * *

### 包扎

itertools 是一套非常通用的工具，用于创建迭代器。您可以使用它们来创建您自己的迭代器，无论是单独使用还是相互组合使用。Python 文档中有很多很棒的例子，您可以研究这些例子来了解如何使用这个有价值的库。

* * *

### 相关阅读

*   关于 itertools 的 Python 文档
*   itertools 简介
*   看看 Python 的一些有用的工具 [itertools](http://naiquevin.github.io/a-look-at-some-of-pythons-useful-itertools.html)
*   本周 Python 模块: [itertools](https://pymotw.com/2/itertools/)
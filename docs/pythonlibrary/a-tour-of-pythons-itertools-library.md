# Python 的 itertools 库之旅

> 原文：<https://www.blog.pythonlibrary.org/2021/12/07/a-tour-of-pythons-itertools-library/>

Python 为创建自己的迭代器提供了一个很好的模块。我所指的模块是 **itertools** 。itertools 提供的工具速度快，内存效率高。您将能够利用这些构建块来创建您自己的专用迭代器，这些迭代器可用于高效的循环。

在本文中，您将会看到每个构建块的示例，这样到最后您就会理解如何在自己的代码库中使用它们。

让我们从看一些无限迭代器开始吧！

## 无限迭代器

itertools 包附带了三个迭代器，可以无限迭代。这意味着当你使用它们的时候，你需要
理解你最终将需要脱离这些迭代器，否则你将会有一个无限循环。

例如，这对于生成数字或在未知长度的迭代上循环很有用。让我们开始了解这些有趣的项目吧！

### 计数(开始=0，步进=1)

count 迭代器将从您作为起始参数传入的数字开始返回等距值。Count 也接受一个步长参数。

让我们看一个简单的例子:

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

在这里，您从 **itertools** 导入**计数**，并为循环创建一个**。您添加了一个条件检查，如果迭代器超过 20，它将中断循环，否则，它将打印出您在迭代器中的位置。您会注意到输出从 10 开始，因为这是您传递来作为起始值计数的值。**

限制这个无限迭代器输出的另一种方法是使用来自 **itertools** 的另一个名为 **islice** 的子模块。

方法如下:

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

在本例中，您导入了 **islice** 并循环遍历 **count** ，从 10 开始，在 5 个项目后结束。你可能已经猜到了， **islice** 的第二个参数是何时停止迭代。但这并不意味着“当我到达数字 5 时停止”。相反，它意味着“当你达到五次迭代时停止”。

### 周期(可迭代)

itertools 中的循环迭代器允许你创建一个迭代器，它将无限循环遍历一系列值。让我们给它传递一个 3 个字母的字符串，看看会发生什么:

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
```

在这里，您为循环创建了一个**来循环三个字母:XYZ 的无限循环。当然，您不希望实际上永远循环下去，所以您添加了一个简单的计数器来打破循环。**

您还可以使用 Python 的 next 内置函数来迭代您用 **itertools** 创建的迭代器:

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

在上面的代码中，您创建了一个简单的多边形列表，并将它们传递给**循环**。你把我们新的迭代器保存到一个变量中，然后把这个变量传递给下一个函数。每次调用 next，它都会返回迭代器中的下一个值。因为这个迭代器是无限的，所以你可以整天调用 next，永远不会用完所有的条目。

### 重复(对象)

**repeat** 迭代器将一次又一次地返回一个对象，除非你设置它的 times 参数。它与 cycle 非常相似，只是它不重复循环一组值。

让我们看一个简单的例子:

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

这里你导入**重复**并告诉它重复数字 5 五次。然后在我们的新迭代器上调用 next 六次，看看它是否工作正常。当您运行这段代码时，您会看到 **StopIteration** 被抛出，因为您已经用完了迭代器中的值。

## 终止的迭代器

用 itertools 创建的大多数迭代器都不是无限的。在本节中，您将学习 **itertools** 的有限迭代器。为了获得可读的输出，您将使用 Python 内置的**列表**类型。

如果你不使用**列表**，那么你只会得到一个 **itertools** 对象的打印结果。

### 累积(可迭代)

**accumulate** 迭代器将返回累加和或一个双参数函数的累加结果，您可以传递给 accumulate。accumulate 的默认值是加法，所以让我们快速尝试一下:

```py
>>> from itertools import accumulate
>>> list(accumulate(range(10)))
[0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
```

这里我们导入 accumulate 并传递给它一个范围为 0-9 的 10 个数字。它依次将它们相加，因此第一个是 0，第二个是 0+1，第三个是 1+2，依此类推。

现在让我们导入**操作符**模块，并将其添加到 mix 中:

```py
>>> import operator
>>> list(accumulate(range(1, 5), operator.mul))
[1, 2, 6, 24]
```

这个函数接受两个要相乘的参数。因此，对于每次迭代，它都是乘法而不是加法(1x1=1，1x2=2，2x3=6，等等)。

**accumulate** 的文档显示了一些其他有趣的例子，如贷款的分期偿还或混乱的递归关系。你一定要看看这些例子，因为它们非常值得你花时间去做。

### 链(*iterables)

**链**迭代器将接受一系列可迭代对象，并基本上将它们展平成一个长的可迭代对象。事实上，我最近正在帮助的一个项目需要它的帮助。基本上，您有一个已经包含一些项目的列表和两个想要添加到原始列表中的其他列表，但是您只想将每个列表中的项目添加到原始列表中，而不是创建一个列表列表。

最初我尝试这样做:

```py
>>> my_list = ['foo', 'bar']
>>> numbers = list(range(5))
>>> cmd = ['ls', '/some/dir']
>>> my_list.extend(cmd, numbers)
>>> my_list
['foo', 'bar', ['ls', '/some/dir'], [0, 1, 2, 3, 4]]
```

嗯，这并不完全是我想要的方式。 **itertools** 模块提供了一种更加优雅的方式，使用**链**将这些列表整合成一个列表:

```py
>>> from itertools import chain
>>> my_list = list(chain(['foo', 'bar'], cmd, numbers))
>>> my_list
['foo', 'bar', 'ls', '/some/dir', 0, 1, 2, 3, 4]
```

我的更敏锐的读者可能会注意到，实际上有另一种方法可以在不使用 **itertools** 的情况下完成同样的事情。您可以这样做来获得相同的效果:

```py
>>> my_list = ['foo', 'bar']
>>> my_list += cmd + numbers
>>> my_list
['foo', 'bar', 'ls', '/some/dir', 0, 1, 2, 3, 4]
```

这两种方法当然都是有效的，在我了解 chain 之前，我可能会走这条路，但我认为在这种特殊情况下, **chain** 是更优雅、更容易理解的解决方案。

### chain.from_iterable(iterable)

你也可以使用一种叫做**的**链**的方法 from_iterable** 。这种方法与直接使用**链条**略有不同。不是传入一系列的 iterables，而是传入一个嵌套的 list。

让我们来看看:

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

在这里，您像以前一样导入链。您尝试传入我们的两个列表，但最终得到的是一个 **TypeError** ！为了解决这个问题，您稍微修改一下您的调用，将 cmd 和 numbers 放在一个列表中，然后将这个嵌套列表传递给 **from_iterable** 。这是一个微妙的区别，但仍然易于使用！

### 压缩(数据，选择器)

**compress** 子模块对于过滤第一个 iterable 和第二个 iterable 非常有用。这是通过使第二个 iterable 成为一个布尔列表(或者等同于同一事物的 1 和 0)来实现的。

它是这样工作的:

```py
>>> from itertools import compress
>>> letters = 'ABCDEFG'
>>> bools = [True, False, True, True, False]
>>> list(compress(letters, bools))
['A', 'C', 'D']
```

在本例中，您有一组七个字母和一个五个布尔的列表。然后将它们传递给压缩函数。compress 函数将遍历每个相应的 iterable，并对照第二个 iterable 检查第一个 iterable。如果第二个有匹配的 True，那么它将被保留。如果它是假的，那么这个项目将被删除。

因此，如果你研究上面的例子，你会注意到在第一、第三和第四个位置上有一个 True，分别对应于 A、C 和 d。

### dropwhile(谓词，可迭代)

itertools 中有一个简洁的小迭代器，叫做 **dropwhile** 。这个有趣的小迭代器将删除元素，只要过滤标准是**真**。因此，在谓词变为 False 之前，您不会看到这个迭代器的任何输出。这可能会使启动时间变得很长，因此需要注意这一点。

让我们看看 Python 文档中的一个例子:

```py
>>> from itertools import dropwhile
>>> list(dropwhile(lambda x: x<5, [1,4,6,4,1]))
[6, 4, 1]
```

在这里，您导入 **dropwhile** ，然后传递给它一个简单的 lambda 语句。如果 x 小于 5，该函数将返回**真值**。否则将返回**假**。 **dropwhile** 函数将遍历列表并将每个元素传递给 lambda。如果 lambda 返回 True，那么该值将被删除。一旦你到达数字 6，lambda 返回 **False** ，你保留数字 6 和它后面的所有值。

当我学习新东西时，我发现在λ上使用常规函数很有用。所以让我们颠倒一下，创建一个函数，如果数字大于 5，返回 **True** 。

```py
>>> from itertools import dropwhile
>>> def greater_than_five(x):
...     return x > 5 
... 
>>> list(dropwhile(greater_than_five, [6, 7, 8, 9, 1, 2, 3, 10]))
[1, 2, 3, 10]
```

这里您在 Python 的解释器中创建了一个简单的函数。这个函数是你的谓词或过滤器。如果你传递给它的值是真的，那么它们将被丢弃。一旦您找到一个小于 5 的值，那么在该值之后的所有值(包括该值)都将被保留，您可以在上面的示例中看到这一点。

### filterfalse(谓词，可迭代)

来自 **itertools** 的 **filterfalse** 函数与 **dropwhile** 非常相似。然而， **filterfalse** 将只返回那些被评估为 false 的值，而不是删除与 True 匹配的值。

让我们使用上一节中的函数来说明:

```py
>>> from itertools import filterfalse
>>> def greater_than_five(x):
...     return x > 5 
... 
>>> list(filterfalse(greater_than_five, [6, 7, 8, 9, 1, 2, 3, 10]))
[1, 2, 3]
```

在这里，你传递给你的函数和一个整数列表。如果整数小于 5，则保留该整数。否则，就扔掉。你会注意到我们的结果只有 1，2 和 3。与 **dropwhile** 不同， **filterfalse** 将根据我们的谓词检查每一个值。

### groupby(iterable，key=None)

groupby 迭代器将从 iterable 中返回连续的键和组。如果没有例子，你很难理解这一点。所以我们来看一个吧！

将以下代码放入您的解释器或保存在文件中:

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

在这里，您导入 **groupby** ，然后创建一个元组列表。然后你对数据进行排序，这样当你输出它时更有意义，它也让 **groupby** 实际上正确地对项目进行分组。接下来，你实际上循环遍历由 **groupby** 返回的迭代器，它给你键和组。最后，循环遍历该组并打印出其中的内容。

如果您运行这段代码，您应该会看到类似这样的内容:

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

只是为了好玩，试着修改代码，让你传入 vehicles 而不是 sorted_vehicles。如果您这样做，您将很快了解为什么应该在通过 groupby 运行数据之前对其进行排序。

### islice(可迭代、启动、停止)

你实际上早在计数部分就已经知道了。但是在这里你会看得更深入一点。islice 是一个迭代器，它从 iterable 中返回选定的元素。这是一种不透明的说法。

基本上 **islice** 所做的是通过 iterable(你迭代的对象)的索引获取一个切片，并以迭代器的形式返回所选择的项目。实际上， **islice** 有两个实现。有 **itertools.islice(iterable，stop)** 还有更接近常规 Python 切片的 **islice** 版本: **islice(iterable，start，stop[，step])** 。

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

在上面的代码中，您将一个由六个字符组成的字符串连同数字 4(即停止参数)一起传递给了 islice 。这意味着 **islice** 返回的迭代器将包含字符串中的前 4 项。您可以通过在迭代器上调用 next 四次来验证这一点，就像上面所做的那样。Python 足够聪明，知道如果只有两个参数传递给 **islice** ，那么第二个参数就是 stop 参数。

让我们试着给它三个参数，来证明你可以给它一个开始参数和一个停止参数。来自 **itertools** 的**计数**工具可以帮助我们说明这个概念:

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

在这里，您只需调用 count 并告诉 **islice** 从数字 3 开始，到 15 时停止。这就像做切片一样，只不过你是对迭代器做切片，然后返回一个新的迭代器！

### 星图(函数，可迭代)

工具将创建一个迭代器，它可以使用提供的函数和 iterable 进行计算。正如文档中提到的，“T2 地图()和 T4 星图()之间的区别类似于 T6 函数(a，b) 和 T8 函数(*c) ”

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

这里您创建了一个简单的加法函数，它接受两个参数。接下来，创建一个 for 循环，调用 **starmap** ，将函数作为第一个参数，并为 iterable 提供一个元组列表。然后， **starmap** 函数会将每个元组元素传递到函数中，并返回结果的迭代器，结果将打印出来。

### takewhile(谓词，可迭代)

**takewhile** 函数基本上与您之前看到的 **dropwhile** 迭代器相反。 **takewhile** 将创建一个迭代器，只要我们的谓词或过滤器为真，它就从 iterable 返回元素。

让我们尝试一个简单的例子来看看它是如何工作的:

```py
>>> from itertools import takewhile
>>> list(takewhile(lambda x: x<5, [1,4,6,4,1]))
[1, 4]
```

在这里，您使用一个**λ**函数和一个链表来运行 **takewhile** 。输出只是 iterable 的前两个整数。原因是 1 和 4 都小于 5，但 6 更大。因此，一旦 **takewhile** 看到 6，条件就变为假，它将忽略 iterable 中的其余项。

### 三通(可迭代，n=2)

函数将从一个 iterable 创建 n 个迭代器。这意味着你可以从一个 iterable 创建多个迭代器。让我们来看一些解释其工作原理的代码:

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

在这里，您创建了一个 5 个字母的字符串，并将其传递给 tee。因为 **tee** 默认为 2，所以使用多重赋值来获取从 **tee** 返回的两个迭代器。最后，循环遍历每个迭代器并打印出它们的内容。如你所见，它们的内容是一样的。

### zip_longest(*iterables，fillvalue=None)

**zip_longest** 迭代器可以用来将两个可迭代对象压缩在一起。如果可重复项的长度不同，那么你也可以传入一个**填充值**。让我们看一个基于这个函数的文档的愚蠢的例子:

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

在这段代码中，您导入 zip_longest，然后向它传递两个字符串以压缩在一起。您可以看到第一个字符串有 4 个字符长，而第二个只有 2 个字符长。您还将填充值设置为“空白”。当您循环并打印出来时，您会看到返回了元组。

前两个元组分别是每个字符串的第一个和第二个字母的组合。最后两个插入了您的填充值。

应该注意的是，如果传递给 **zip_longest** 的 iterable 有可能是无限的，那么你应该用类似于 **islice** 的东西来包装这个函数，以限制调用的次数。

## 组合生成器

itertools 库包含四个迭代器，可用于创建数据的组合和排列。在本节中，您将会看到这些有趣的迭代器。

### 组合(iterable，r)

如果您需要创建组合，Python 已经为您提供了 **itertools.combinations** 。**组合**允许你做的是从一个一定长度的可迭代对象创建一个迭代器。

让我们来看看:

```py
>>> from itertools import combinations
>>> list(combinations('WXYZ', 2))
[('W', 'X'), ('W', 'Y'), ('W', 'Z'), ('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
```

当您运行它时，您可能会注意到**组合**返回元组。为了使这个输出更具可读性，让我们循环遍历迭代器，并将元组连接成一个字符串:

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

### combinations with _ replacement(iterable，r)

**combinations _ with _ replacement**迭代器与**组合**非常相似。唯一的区别是，它实际上会创建元素重复的组合。让我们用上一节中的一个例子来说明:

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

如您所见，现在我们的输出中有四个新条目:WW、XX、YY 和 ZZ。

### 产品(*iterables，repeat=1)

itertools 包有一个简洁的小函数，可以从一系列输入的可迭代对象中创建笛卡尔乘积。是的，那个功能是**产品**。

让我们看看它是如何工作的！

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

在这里，您导入**产品**,然后设置一个元组列表，并将其分配给变量数组。接下来，使用这些数组调用 product。你会注意到你用***数组来调用它。**这将导致列表被“展开”或按顺序应用于产品功能。这意味着你要传入三个参数而不是一个。如果你愿意的话，试着在数组前面加上星号来调用它，看看会发生什么。

### 排列

**itertools** 的**置换**子模块将从你给定的 iterable 返回连续 r 长度的元素置换。与**组合**函数非常相似，排列是按照字典排序顺序发出的。

让我们来看看:

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

您会注意到输出比组合的输出要长得多。当您使用**置换**时，它将遍历字符串的所有**置换**，但是如果输入元素是唯一的，它不会重复值。

## 包扎

itertools 是一套非常通用的创建迭代器的工具。您可以使用它们来创建您自己的迭代器，无论是单独使用还是相互组合使用。Python 文档中有很多很棒的例子，您可以研究这些例子来了解如何使用这个有价值的库。
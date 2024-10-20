# 如何使用 Python Zip 函数(快速简单)

> 原文：<https://www.pythoncentral.io/how-to-use-the-python-zip-function-fast-and-easy/>

Python 的 zip()函数可以帮助开发人员从多种数据结构中对数据进行分组。元素按照它们出现的顺序连接；有了它，您可以同时迭代不同的数据结构。所有这些都没有使用一个嵌套循环！

如果这是你想要实现的目标，你需要学会使用 zip()。该功能最大的优点是使用起来并不复杂。

在本指南中，我们将带您了解如何使用 zip()并并行迭代不同的数据结构。

## **Python 中的 zip()是什么？**

在 Python 中，zip()同时合并几个被称为 iterables 的对象。它通过创建一个新对象来实现这一点，该对象的元素都是元组。

可迭代的元素保存在这些元组中。让我们看一个简单的例子来理解 zip()函数。

假设我们有两个包含名字的元组，就像这样:

```py
names1 = ("Jack", "Theo", "Jose")

names2 = ("Duncan", "Chris", "Anthony")
```

所以，如果要对这些元组的元素进行配对，可以使用 zip()。函数是这样循环的:

```py
print(zip(names1,names2))
```

当然，打印功能将帮助我们了解使用 zip 时会发生什么。运行这段代码，您会看到它返回一个特定的 zip 对象。代码的输出看起来会像这样:

```py
<zip object at 0x7f7991fc4cc0>
```

现在，让我们来看看如何将这个 zip 对象转换成一个列表。就写这么简单:

```py
print(list(zip(names1,names2)))
```

再次运行代码，您将看到输出:

| [( 【杰克】 ， 【邓肯】 )，( 【西奥】 ， 【克里斯】 )，( 【何塞】【安东尼】)】 |

到目前为止，我们已经介绍了这个函数的基本工作原理。我们将在接下来的章节中讨论 zip 函数的几个细节。

如果你想看看官方对 zip()函数的定义， [这里是官方 Python 文档页面的链接](https://docs.python.org/3/library/functions.html#zip) 。请注意，您还可以在文档中找到 zip()函数的源代码，如果您想了解该函数的技术工作方式，这将很有帮助。

## **对各种数据结构使用 zip()函数**

在上一节的示例程序中，我们探讨了如何使用 zip()函数将两个列表合并成一个对象——zip 对象。对象的元素是长度为 2 的元组。

要利用 zip()函数的功能，您必须理解该函数可以处理所有类型的数据结构。在这一节中，我们将探索 zip()是如何做到这一点的。

### **使用 zip()和元组**

向 zip()函数传递两个元组类似于向它传递列表。因此，代码应该是这样的:

```py
names1 = ("Jack", "Theo", "Jose")
names2 = ("Duncan", "Chris", "Anthony")
print(list(zip(names1,names2)))
```

运行代码，输出将是:

| [( 【杰克】 ， 【邓肯】 )，( 【西奥】 ， 【克里斯】 )，( 【何塞】【安东尼】)】 |

很明显，对列表和元组使用 zip()函数实际上没有区别。方便！

### **使用 zip()和字典**

虽然您可以将 zip()与字典一起使用，但您需要注意一些限制。

让我们通过一个例子来深入了解本质:

```py
names1 = {"mom":"Sonia", "dad":"Jesse"}
names2 = {"firstChild":"Andrew", "secondChild": "Kayla"}
print(list(zip(names1,names2)))
```

花一分钟思考一下代码输出应该是什么。

默认情况下，zip()函数将字典的键缝合在一起。因此，运行代码将给出以下输出:

| [( 【妈妈】 ， 【第一胎】 )，( 【爸爸】 ， 【二胎】 )] |

然而，zip()函数也允许您“压缩”字典的值。要做到这一点，您必须在将 dictionary 对象传递给 zip()时调用它们的 values 方法。

让我们继续前面的例子，看看如何做到这一点:

```py
names1 = {"mom":"Sonia", "dad":"Jesse"}
names2 = {"firstChild":"Andrew", "secondChild": "Kayla"}
print(list(zip(names1.values(),names2.values())))
```

代码的输出现在看起来像这样:

| 【(【索尼娅】【安德鲁】)【杰西】【凯拉】)】 |

### **从 zip()返回不同的数据结构**

zip()的一个好处是，除了允许你传递不同的数据结构，它还允许你导出不同的数据类型。

向后滚动一点，回想一下 zip()的默认输出是一个特殊的 zip 对象。

之前，我们在 list 函数中包装了 zip()函数来输出一个列表。所以，您也可以使用 dict 和 tuple 来返回字典和元组，这并不奇怪。

我们来看一个例子:

```py
names1 = ("Jack", "Theo", "Jose")
names2 = ("Duncan", "Chris", "Anthony")
print(tuple(zip(names1,names2)))
```

运行这段代码的结果看起来会像这样:

| (【杰克】 、 【邓肯】 )、( 【西奥】 、 【克里斯】 )、( 【何塞】 、 【安东尼】) |

我们也可以使用 dict 函数，就像这样

```py
names1 = ("Jack", "Theo", "Jose")
names2 = ("Duncan", "Chris", "Anthony")
print(dict(zip(names1,names2)))
```

输出将如下所示:

| 【杰克】 : 【邓肯】【西奥】 : 【克里斯】【何塞】 : 【安东尼】 } |

### **使用带有两个以上参数的 zip()函数**

本指南中的所有示例都包含两个可迭代变量，但是 zip()理论上可以使用无限个变量。在这一节中，我们将探索 zip()函数的这一功能。

让我们从定义三个变量来创建一个 zip 对象开始:

```py
india_cities = ['Mumbai', 'Delhi', 'Bangalore']
africa_cities = ['Lagos', 'Cape Town', 'Tunis']
russia_cities = ['Moscow', 'Samara', 'Omsk']
```

现在，将三个对象压缩在一起就像将它们传递到 zip()中一样简单，就像这样:

```py
print(list(zip(india_cities, africa_cities, russia_cities)))
```

代码的输出将如下所示:

| (【孟买】 ， 【拉各斯】【莫斯科】 )， 【德里】【开普敦】【萨马拉】 ， 【班加罗尔】 |

显然，向 zip()传递三个参数会创建一个新的数据结构。该结构的元素是长度为 3 的元组。

请记住，这延伸到更大的数字。简单地说，如果您向 zip()传递“n”个参数，它将创建一个新的数据结构，其中包含长度为“n”的元组元素

### **zip()如何处理不同长度的参数**

在本指南中，我们只涉及了 zip()函数如何处理相同长度的数据结构。首先，我们处理了长度为 2 的结构，在上一节中，处理了长度为 3 的列表。

但是当我们将不同长度的参数传递给 zip()函数时会发生什么呢？

Python zip()函数非常灵活，接受不同长度的参数。让我们在这一节中探索函数的这一能力。

如果你回过头来打开 zip()函数的官方 Python 文档，函数的定义会让事情变得清楚。功能描述说:

*“当最短的输入 iterable 用尽时，迭代器停止。”*

换句话说，zip()函数输出的长度将总是等于 最小的 自变量的长度。

这里有一个例子来验证这一点。假设我们有两组人:一组是男人，一组是女人。你想让他们两人一组，这样他们就可以上双人舞蹈课——也许是萨尔萨舞。

现在，让我们用 zip()函数写一些代码，看看它如何帮助我们:

```py
menGroup = ['Devin', 'Bruce', 'Jimmy', 'Eddie']
womenGroup = ['Gina', 'Mila', 'Friday']
print(list(zip(menGroup,womenGroup)))
```

很明显，menGroup 变量比 womenGroup 变量多包含一个元素。因此，我们可以预期 zip()函数会删除 menGroup 变量中的最后一个元素。

运行这段代码，我们将得到以下结果:

| [( 【德文】 ， 【吉娜】 )，( 【布鲁斯】 ， 【米拉】 )，( 【吉米】【星期五】) |

不出所料，埃迪没有得到萨尔萨舞比赛的参赛资格。让我们期待另一位女士的加入吧！

### **用 zip()函数循环多个对象**

zip()函数最有用的一个方面是它允许你并行迭代多个对象。更重要的是，所涉及的语法容易记忆和使用！

我们用一个例子来理解用 zip()循环。假设你们两个列表:一个有学生的名字，另一个有成绩。下面是代码的样子:

```py
studentList = ['Devin', 'Bruce', 'Jimmy'] 
gradeList = [97, 71, 83]
```

在本例中，让我们假设学生列表的排序位置与他们的成绩列表相同。现在，让我们编写代码来打印每个学生的姓名及其相应的成绩。

```py
for student, grade in zip(studentList, gradeList):
    print(f"{student}'s grade is {grade}")
```

运行这段代码的结果如下:

| 德文的成绩是 97布鲁斯的成绩是 71 分吉米的成绩是 83 分 |

## **Python 3 和 2 在 zip()函数上的区别**

需要注意的是，zip()在 Python 2 和 3 中的工作方式不同。zip()函数返回 Python 2 中的元组列表，这个列表被截断为最短输入 iterable 的长度。

因此，如果您调用一个没有参数的 zip()函数，输出将是一个空列表，如下所示:

```py
zipping = zip(range(3), 'WXYZ')
zipping  # Holding the list object
[(0, 'W'), (1, 'X'), (2, 'Y')]
type(zipping)
zipping = zip()  # Creating empty list
zipping
[]
```

正如所料，在上面的代码中调用 zip()会给出一个元组列表的输出。列表在“Y”处被截断，不带参数调用 zip()会返回一个空列表。

但是在 Python 3 中 zip()并不是这样工作的。在 Python 3 中，不带参数调用 zip()会返回一个迭代器。

如果开发者需要，迭代器对象产生元组。但是元组只能被遍历一次。迭代以 StopIteration 异常结束，该异常在最短的输入 iterable 用尽时引发。

请记住，虽然 Python 3 中返回了迭代器，但它是空的。我们来看一个例子:

```py
zipping = zip(range(3), 'WXYZ')
zipping  # Holding an iterator
type(zipping)
list(zipping)
zipping = zip()  # Create an empty iterator
zipping
next(zipping)
```

运行这段代码将产生输出:

| 回溯(最近呼叫最后一次):文件<字符串>，行 **7** ， < 模块 >**停止迭代** |

在这种情况下，调用 zip()函数会返回一个迭代器。第一次迭代在 c 处被截断，而第二次迭代引发 StopIteration 异常。

Python 3 的好处在于它可以模拟 Python 2 中 zip()的行为。它通过将返回的迭代器封装在对 list()的调用中来实现这一点。这样，它遍历迭代器，返回一个元组列表。

如果您使用的是 Python 2，请注意向 zip()函数传递长输入 iterables 会导致不合理的高内存消耗。因此，如果您需要传递长输入的 iterables，请使用 itertools.izip(*iterables)。

该函数生成一个迭代器，从可迭代对象中聚集元素。因此，使用它与在 Python 3 中使用 zip()具有相同的效果。我们来看一个例子:

```py
from itertools import izip
zipped = izip(range(3), 'WXYZ')
zipped
list(zipped)
```

在上面的代码中，调用了 itertools.izip()函数，它创建了一个迭代器。当返回的迭代器被 list()消耗时，它输出一个元组列表，就像 Python 3 中的 zip()函数一样。当它到达最短输入迭代器的末尾时，它停止。

如果你同时使用 Python 2 和 3，并且喜欢用两种语言编写代码，那么这里有一些你可以使用的代码:

```py
try:
    from itertools import izip as zip
except ImportError:
    pass
```

在这段代码中，izip()函数在 itertools 中是可用的。因此，在 Python 2 中，该函数将在别名 zip 下导入。否则，程序将引发一个 ImportError 异常，这样您就知道您使用的是 Python 3。

上面的代码使你能够在你写的所有代码中使用 zip()函数。当代码运行时，Python 会自动选择正确的版本。

既然您已经理解了 ins 和 out of zip()函数，那么是时候打开指关节，喝杯咖啡，开始编写现实世界的解决方案了！
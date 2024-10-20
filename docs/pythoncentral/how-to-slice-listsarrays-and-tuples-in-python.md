# 如何在 Python 中切片列表/数组和元组

> 原文：<https://www.pythoncentral.io/how-to-slice-listsarrays-and-tuples-in-python/>

所以你已经有了一个列表、元组或数组，你想从中获得特定的子元素集合，而不需要任何冗长的循环？

Python 有一个惊人的特性，叫做*切片*。切片不仅可以用于列表、元组或数组，还可以用于定制数据结构，使用*切片*对象，这将在本文后面使用。

## **理解 Python 切片语法**

为了理解 Python 中 slice 函数的工作原理，你有必要理解什么是索引。

在 Python 中，可以将 slice 函数用于几种不同的数据类型，包括列表、元组和字符串。这是因为 Python 在它们中使用了索引的概念。

类似列表的数据类型中的字符或元素的“索引”是字符或元素在数据类型中的位置。不管数据类型包含什么数据，索引值总是从零开始，以列表中的项数减一结束。

Python 还使用了负索引的概念，使数据更容易访问。使用负索引，Python 用户可以从容器的末尾而不是开头访问数据类型中的数据。

切片的语法为:

```py
[python]
slice(start, end, step)
[/python]
```

切片功能同时接受多达三个参数。start 参数是可选的，它指示要开始对数据类型进行切片的容器的索引。如果没有提供值，则 start 的值默认为 *【无】* 。

另一方面，输入 stop 参数是强制性的，因为它向 Python 指示切片应该发生的索引。默认情况下，Python 将在值“stop - 1”处停止切片

换句话说，如果使用 slice 函数时停止值为 5，Python 将只返回值，直到索引值为 4。

Python 接受的最后一个参数叫做 step，可以根据需要随意输入。它是一个整数值，向 Python 表明当对列表进行切片时，用户希望在索引之间增加多少。默认的增量是 1，但是用户可以在 step 中放入任何适用的正整数。

## **切片 Python 列表/数组和元组语法**

让我们从一个正常的日常清单开始。

```py
[python]
>>> a = [1, 2, 3, 4, 5, 6, 7, 8]
[/python]
```

没什么疯狂的，只是一个从 1 到 8 的普通列表。现在假设我们真的希望子元素 2、3 和 4 在一个新的列表中返回。我们如何做到这一点？

不是用一个`for`循环，就是这样。这是 Pythonic 式的做事方式:

```py
[python]
>>> a[1:4]
[2, 3, 4]
[/python]
```

这正是我们想要的结果。那个语法到底是什么意思？好问题。

我来解释一下。`1`表示从列表中的第二个元素开始(注意切片索引从`0`开始)。`4`表示在列表的第五个元素处结束，但不包括它。中间的冒号是 Python 的列表识别我们想要使用切片来获取列表中的对象的方式。

### **高级 Python 切片(列表、元组和数组)增量**

我们还可以添加一个可选的第二子句，它允许我们设置列表的索引在我们设置的索引之间如何递增。

在上面的例子中，假设我们不希望返回难看的`3`,我们只希望列表中有好的偶数。很简单。

```py
[python]
>>> a[1:4:2]
[2, 4]
[/python]
```

看，就这么简单。最后一个冒号告诉 Python 我们想要选择切片增量。默认情况下，Python 将这个增量设置为`1`，但是数字末尾的额外冒号允许我们指定它是什么。

### **Python 逆向切片(列表、元组和数组)**

好吧，如果我们想让我们的列表倒过来，只是为了好玩呢？让我们尝试一些疯狂的事情。

```py
[python]
>>> a[::-1]
[8, 7, 6, 5, 4, 3, 2, 1]
[/python]
```

什么？！那是什么鬼东西？

列表在切片时有一个默认的功能。如果第一个冒号前没有值，则意味着从列表的起始索引开始。如果第一个冒号后没有值，就意味着一直到列表的末尾。这为我们节省了时间，这样我们就不必手动指定`len(a)`作为结束索引。

好吧。我最后偷偷溜了进去？这意味着每次增加索引`-1`，意味着它将通过向后遍历列表。如果想让偶数索引倒退，可以跳过每隔一个元素，将迭代设置为`-2`。简单。

### **切片 Python 元组**

上面的一切也适用于元组。完全相同的语法和完全相同的做事方式。我举个例子只是为了说明确实是一样的。

```py
[python]
>>> t = (2, 5, 7, 9, 10, 11, 12)
>>> t[2:4]
(7, 9)
[/python]
```

仅此而已。简单、优雅、强大。

### **使用 Python 切片对象**

还有一种更容易理解的语法形式可以使用。Python 中有称为`slice`对象的对象，可以用来代替上面的冒号语法。

这里有一个例子:

```py
[python]
>>> a = [1, 2, 3, 4, 5]
>>> sliceObj = slice(1, 3)
>>> a[sliceObj]
[2, 3]
[/python]
```

`slice`对象初始化接受 3 个参数，最后一个是可选的索引增量。一般的方法格式是:`slice(start, stop, increment)`，应用于列表或元组时类似于`start:stop:increment`。

就是这样！

## **Python 切片示例**

为了帮助你理解如何使用 slice，这里有一些例子:

### **#1 基本切片示例**

让我们假设你有一个字符串列表，如下:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']
[/python]
```

下面的代码使用 slice 函数从条目列表中导出一个子列表:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

sub_items = items[1:4]

print(sub_items)
[/python]
```

代码的输出是:

```py
[python]
['bike', 'house', 'bank']
[/python]
```

由于起始索引是 1，Python 以‘bike’开始切片另一方面，结束索引是 4，所以切片在“bank”处停止列表

由于这个原因，代码的最终结果是一个包含三项的列表:自行车、房子和银行。因为这段代码没有使用 step，所以 slice 函数接受该范围内的所有值，并且不跳过任何元素。

### **#2 用 Python Slice 切片列表中的前“N”个元素**

使用 slice 从列表中获取前“N”个元素就像使用下面的语法一样简单:

```py
[python]
list[:n]
[/python]
```

我们以之前的例子同一个列表为例，通过切片输出列表的前三个元素:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

sub_items = items[:3]

print(sub_items)
[/python]
```

代码将输出以下结果:

```py
[python]
['car', 'bike', 'house']
[/python]
```

注意项目[:3]如何与项目[0:3] 相同

**#3 用 Python Slice 从列表中获取最后“N”个元素**

这个和上一个例子差不多，只是我们可以要求后三个要素，而不是要求前三个要素。

下面是我们如何处理上一个例子中的列表:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

sub_items = items[-3:]

print(sub_items)
[/python]
```

该代码将为我们提供以下结果:

```py
[python]
['purse', 'photo', 'box']
[/python]
```

**#4 使用 Python 对列表中的每“n”个元素进行切片**

为此，我们必须使用 step 参数并返回我们想要的子列表。使用与之前相同的列表，并假设我们希望代码每隔一个元素返回一次:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

sub_items = items[::2]

print(sub_items)
[/python]
```

该代码将显示以下结果:

```py
[python]
['car', 'house', 'purse', 'box']
[/python]
```

**#5 使用 Python 切片反转列表**

slice 函数使得反转列表变得异常简单。你只需要给 slice 函数一个负的步长，函数就会从列表的末尾到开头列出元素——反过来。

继续我们之前使用的示例列表:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

reversed_items = items[::-1]

print(reversed_items)
[/python]
```

这将为我们提供输出:

```py
[python]
['box', 'photo', 'purse', 'bank', 'house', 'bike', 'car']
[/python]
```

**#6 使用 Python 切片替换列表的一部分**

除了可以提取列表的一部分，slice 函数还可以让你很容易地改变列表中的元素。

在下面的代码中，我们将列表的前两个元素更改为两个新值:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

items[0:2] = ['new car', 'new bike']

print(items)
[/python]
```

代码的输出是:

```py
[python]
['new car', 'new bike', 'house', 'bank', 'purse', 'photo', 'box']
[/python]
```

**#7 用切片**部分替换列表并调整其大小

使用 slice 函数改变元素和向列表中添加新元素非常容易:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

print("The list has {len{items)} elements")

items[0:2] = ['new car', 'new bike', 'new watch']

print(items)

print ("The list now has {len(items)} elements")
[/python]
```

该代码给出以下输出:

```py
[python]
The list has 7 elements

['new car', 'new bike', 'new watch', 'house', 'bank', 'purse', 'photo', 'box']

The list has 8 elements
[/python]
```

**#8 使用切片从列表中删除元素**

使用我们一直在使用的示例列表，下面是如何使用 slice 函数从列表中删除元素:

```py
[python]
items = ['car', 'bike', 'house', 'bank', 'purse', 'photo', 'box']

del items[2:4]
print(items)
[/python]
```

代码的输出是:

```py
[python]
['car', 'bike', 'purse', 'photo', 'box']
[/python]
```

如果你想知道如何分割字符串，在另一篇名为[如何在 Python 中从字符串中获取子字符串——分割字符串](https://www.pythoncentral.io/how-to-get-a-substring-from-a-string-in-python-slicing-strings/ "How to Get a Sub-string From a String in Python – Slicing Strings")的文章中有所涉及。

目前就这些。希望你喜欢学习切片，我希望它会帮助你的追求。

再见。
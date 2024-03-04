# Python 中的命名元组

> 原文：<https://www.pythonforbeginners.com/data-types/namedtuple-in-python>

你必须在你的程序中使用一个元组或者一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)。尽管它们是非常有用的数据结构，但是它们也有一些缺点。在这篇文章中，我们将学习什么是 python 中的命名元组，以及我们如何用它来代替元组或字典。

## 元组和字典的问题

假设您正在创建一个 python 应用程序，并且您必须指定图形界面的颜色。在这种情况下，您可以使用具有(红、绿、蓝)值的元组，如下所示。

```py
color=(100,125,130)
```

但是，这里有一个问题。假设你的队友读了你的代码，他不明白上面例子中的元组意味着什么。可能是(红、绿、蓝)值，也可能是(色调、饱和度、亮度)值，依他而定。为了避免这种混乱，你可以使用一本以红绿蓝为基调的字典。

```py
color={"red":100, "green":125, "blue": 130}
```

同样，当我们使用字典时，我们必须为图形界面中使用的每个组件创建一个字典。这将使我们的代码变得多余和不可读。为了解决这些问题，我们可以在 python 程序中使用 namedtuple。让我们讨论它是什么以及我们将如何使用它。

## python 中的命名元组是什么？

您可以将命名元组视为元组和字典之间的中间数据结构。顾名思义，python 中的 namedtuple 是带有命名字段的元组。通常，我们使用元组的索引来访问它们的元素。但是在 namedtuple 中，我们可以指定每个索引的名称，然后我们可以使用这些名称来访问元素。

让我们在下面的章节中详细讨论这一点。

## 如何在 Python 中创建 namedtuple？

集合模块中定义了 Namedtuple。您可以按如下方式导入它。

```py
from collections import namedtuple
```

导入 namedtuple 模块后，可以使用 namedtuple()构造函数创建一个 namedtuple。

namedtuple()构造函数采用要创建的对象类的名称及其字段名称作为输入参数。然后它创建 tuple 的一个子类，其中的字段被命名。

例如，我们可以用字段名“红”、“绿”和“蓝”定义一个“颜色”名元组，如下所示。

```py
Color = namedtuple("Color", ["red", "green", "blue"])
```

在定义了 namedtuple 子类之后，我们可以通过给定字段值来创建命名元组，如下所示。

```py
color=Color(100,125,130)
```

我们还可以使用如下的属性符号来访问 namedtuple 的不同字段中的值。

```py
from collections import namedtuple

Color = namedtuple("Color", ["red", "green", "blue"])
color = Color(100, 125, 130)

print("The namedtuple is:", color)
print("The red value is:", color.red)
print("The green value is:", color.green)
print("The blue value is:", color.blue) 
```

输出:

```py
The namedtuple is: Color(red=100, green=125, blue=130)
The red value is: 100
The green value is: 125
The blue value is: 130
```

如果需要，您还可以找到 namedtuple 的字段名称。为此，您可以使用包含字段名称的 _fields 属性，如下所示。

```py
from collections import namedtuple

Color = namedtuple("Color", ["red", "green", "blue"])
color = Color(100, 125, 130)

print("The namedtuple is:", color)
print("The fields of the namedtuple are:", color._fields)
```

输出:

```py
The namedtuple is: Color(red=100, green=125, blue=130)
The fields of the namedtuple are: ('red', 'green', 'blue')
```

## 在 Python 中使用命名元组的好处

与元组和字典相比，使用命名元组有几个好处。

*   与元组不同，命名元组允许索引和字段名。这使得我们的代码更容易理解。
*   与字典不同，命名元组是不可变的，可以存储在一个集合中。此外，使用 namedtuple 代替字典需要我们编写更少的代码。
*   尽管有这些优点，命名元组与元组相比使用几乎相似的内存。

## 如何修改命名元组？

namedtuple 是 tuple 的子类，被认为是不可变的。它可以存储在一个集合中，并显示不可变对象的每个属性。但是，我们可以修改命名元组中的值。为此，我们可以使用 _replace()方法，该方法采用带有字段名和新值的关键字参数。之后，它返回修改后的 namedtuple，如下所示。

```py
from collections import namedtuple

Color = namedtuple("Color", ["red", "green", "blue"])
color = Color(100, 125, 130)

print("The original namedtuple is:", color)
color = color._replace(red=200)
print("The modified namedtuple is:", color) 
```

输出:

```py
The original namedtuple is: Color(red=100, green=125, blue=130)
The modified namedtuple is: Color(red=200, green=125, blue=130) 
```

## 结论

在本文中，我们讨论了 python 中的 namedtuple。我们还讨论了它的使用以及相对于元组和字典的好处。要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 Python 中[链表的文章。](https://www.pythonforbeginners.com/lists/linked-list-in-python)
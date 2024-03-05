# Python 对象的浅层与深层复制

> 原文：<https://realpython.com/copying-python-objects/>

Python 中的赋值语句并不创建对象的副本，它们只是将名字绑定到一个对象上。对于[不可变的](https://realpython.com/courses/immutability-python/)对象，这通常没什么区别。

但是对于处理可变对象或可变对象的集合，您可能需要寻找一种方法来创建这些对象的“真实副本”或“克隆”。

本质上，你有时会想要可以修改*的副本，而不需要*同时自动修改原件。在本文中，我将向您简要介绍如何在 Python 3 中复制或“克隆”对象，以及一些相关的注意事项。

> **注意:**本教程是在考虑 Python 3 的情况下编写的，但是在复制对象方面，Python 2 和 Python 3 几乎没有区别。当有不同之处时，我会在文中指出来。

让我们先来看看如何复制 Python 的内置集合。Python 内置的可变集合，如[列表、字典和集合](https://realpython.com/learn/python-first-steps/)可以通过在现有集合上调用它们的工厂函数来复制:

```py
new_list = list(original_list)
new_dict = dict(original_dict)
new_set = set(original_set)
```

然而，这个方法对定制对象不起作用，除此之外，它只创建了*浅拷贝*。对于像[列表](https://realpython.com/python-lists-tuples/)、[字典](https://realpython.com/python-dicts/)、[集合](https://realpython.com/python-sets/)这样的复合对象来说，*浅*和*深*复制有一个重要的区别:

*   一个**浅拷贝**意味着构造一个新的集合对象，然后用在原始对象中找到的子对象的引用填充它。从本质上来说，一个浅的副本只比*深一级*。复制过程不会[递归](https://realpython.com/python-thinking-recursively/)，因此不会创建子对象本身的副本。

*   一个**深度复制**使得复制过程[递归](https://realpython.com/python-recursion/)。它意味着首先构造一个新的集合对象，然后用原始集合中找到的子对象的副本递归地填充它。以这种方式复制对象会遍历整个对象树，从而创建原始对象及其所有子对象的完全独立的克隆。

我知道，这有点拗口。因此，让我们看一些例子来说明深层拷贝和浅层拷贝之间的差异。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 制作浅拷贝

在下面的例子中，我们将创建一个新的嵌套列表，然后*用`list()`工厂函数简单地*复制它:

>>>

```py
>>> xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> ys = list(xs)  # Make a shallow copy
```

这意味着`ys`现在将是一个新的独立对象，其内容与`xs`相同。您可以通过检查两个对象来验证这一点:

>>>

```py
>>> xs
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> ys
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

为了证实`ys`真的独立于原始数据，让我们设计一个小实验。您可以尝试向原始列表(`xs`)添加一个新的子列表，然后检查以确保这个修改没有影响副本(`ys`):

>>>

```py
>>> xs.append(['new sublist'])
>>> xs
[[1, 2, 3], [4, 5, 6], [7, 8, 9], ['new sublist']]
>>> ys
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

如您所见，这达到了预期的效果。在“肤浅”的层次上修改复制的列表没有任何问题。

然而，因为我们只创建了原始列表的*浅层*副本，`ys`仍然包含对存储在`xs`中的原始子对象的引用。

这些孩子不是被复制的。它们只是在复制的列表中被再次引用。

因此，当你修改`xs`中的一个子对象时，这个修改也会反映在`ys`中——这是因为*两个列表共享相同的子对象*。该副本只是一个浅的、一级深的副本:

>>>

```py
>>> xs[1][0] = 'X'
>>> xs
[[1, 2, 3], ['X', 5, 6], [7, 8, 9], ['new sublist']]
>>> ys
[[1, 2, 3], ['X', 5, 6], [7, 8, 9]]
```

在上面的例子中，我们(似乎)只对`xs`做了一个修改。但结果是*在`xs` *和* `ys`中索引为 1 的*子列表都被修改了。同样，这是因为我们只创建了原始列表的一个*浅层*副本。

如果我们在第一步中创建了`xs`的*深度*副本，那么这两个对象将是完全独立的。这就是对象的浅拷贝和深拷贝的实际区别。

现在您知道了如何创建一些内置集合类的浅层拷贝，并且知道了浅层拷贝和深层拷贝的区别。我们仍然希望得到答案的问题是:

*   如何创建内置集合的深层副本？
*   如何创建任意对象(包括自定义类)的副本(浅层和深层)？

这些问题的答案就在 Python 标准库中的`copy`模块中。这个[模块](https://realpython.com/python-modules-packages/)为创建任意 Python 对象的浅层和深层副本提供了一个简单的接口。

[*Remove ads*](/account/join/)

## 制作深层副本

让我们重复前面的列表复制示例，但是有一个重要的区别。这次我们将使用在`copy`模块中定义的`deepcopy()`函数创建一个*深度*副本:

>>>

```py
>>> import copy
>>> xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> zs = copy.deepcopy(xs)
```

当你检查我们用`copy.deepcopy()`创建的`xs`和它的克隆`zs`时，你会看到它们看起来又是一样的——就像前面的例子一样:

>>>

```py
>>> xs
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> zs
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

然而，如果您对原始对象(`xs`)中的一个子对象进行修改，您将会看到该修改不会影响深层副本(`zs`)。

两个对象，原始对象和副本，这次是完全独立的。`xs`被递归克隆，包括它的所有子对象:

>>>

```py
>>> xs[1][0] = 'X'
>>> xs
[[1, 2, 3], ['X', 5, 6], [7, 8, 9]]
>>> zs
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

现在，您可能想花些时间坐下来和 Python 解释器一起研究这些例子。当您直接体验和使用这些示例时，就更容易理解如何复制对象。

顺便说一下，您还可以使用`copy`模块中的函数创建浅层副本。`copy.copy()`函数创建对象的浅层副本。

如果您需要清楚地表明您正在代码中的某个地方创建一个浅层副本，这是非常有用的。使用`copy.copy()`可以让你指出这个事实。然而，对于内置集合，简单地使用 list、dict 和 set factory 函数来创建浅层副本被认为是更 Pythonic 化的。

## 复制任意 Python 对象

我们仍然需要回答的问题是，我们如何创建任意对象的副本(浅层和深层)，包括自定义类。现在让我们来看看。

再次`copy`模块来救我们了。它的`copy.copy()`和`copy.deepcopy()`功能可以用来复制任何对象。

同样，理解如何使用这些的最好方法是通过一个简单的实验。我将以前面的列表复制示例为基础。让我们从定义一个简单的 2D 点类开始:

```py
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point({self.x!r}, {self.y!r})'
```

我希望你同意这很简单。我添加了一个`__repr__()`实现，这样我们可以很容易地在 Python 解释器中检查从这个类创建的对象。

> **注意:**上面的例子使用了一个 [Python 3.6 f-string](https://dbader.org/blog/python-string-formatting) 来构造`__repr__`返回的字符串。在 Python 2 和 Python 3.6 之前的版本中，您可以使用不同的字符串格式表达式，例如:
> 
> ```py
> `def __repr__(self):
>     return 'Point(%r, %r)' % (self.x, self.y)` 
> ```

接下来，我们将创建一个`Point`实例，然后使用`copy`模块(浅显地)复制它:

>>>

```py
>>> a = Point(23, 42)
>>> b = copy.copy(a)
```

如果我们检查原始`Point`对象及其(浅层)克隆的内容，我们会看到我们所期望的:

>>>

```py
>>> a
Point(23, 42)
>>> b
Point(23, 42)
>>> a is b
False
```

还有一些事情需要记住。因为我们的 point 对象使用不可变类型(int)作为它的坐标，所以在这种情况下，浅拷贝和深拷贝没有区别。但是我马上会扩展这个例子。

让我们来看一个更复杂的例子。我将定义另一个类来表示 2D 矩形。我将以一种允许我们创建一个更复杂的对象层次的方式来完成它——我的矩形将使用`Point`对象来表示它们的坐标:

```py
class Rectangle:
    def __init__(self, topleft, bottomright):
        self.topleft = topleft
        self.bottomright = bottomright

    def __repr__(self):
        return (f'Rectangle({self.topleft!r}, '
                f'{self.bottomright!r})')
```

同样，首先我们将尝试创建一个矩形实例的浅层副本:

```py
rect = Rectangle(Point(0, 1), Point(5, 6))
srect = copy.copy(rect)
```

如果您检查原始矩形和它的副本，您将看到`__repr__()`覆盖工作得多么好，并且浅层复制过程如预期那样工作:

>>>

```py
>>> rect
Rectangle(Point(0, 1), Point(5, 6))
>>> srect
Rectangle(Point(0, 1), Point(5, 6))
>>> rect is srect
False
```

还记得上一个 list 例子是如何说明深层和浅层拷贝之间的区别的吗？我将在这里使用相同的方法。我将修改对象层次中更深层次的对象，然后您将看到这一变化也反映在(浅层)副本中:

>>>

```py
>>> rect.topleft.x = 999
>>> rect
Rectangle(Point(999, 1), Point(5, 6))
>>> srect
Rectangle(Point(999, 1), Point(5, 6))
```

我希望这是你所期望的。接下来，我将创建一个原始矩形的深度副本。然后，我将应用另一个修改，您将看到哪些对象受到影响:

>>>

```py
>>> drect = copy.deepcopy(srect)
>>> drect.topleft.x = 222
>>> drect
Rectangle(Point(222, 1), Point(5, 6))
>>> rect
Rectangle(Point(999, 1), Point(5, 6))
>>> srect
Rectangle(Point(999, 1), Point(5, 6))
```

瞧啊。这一次深层拷贝(`drect`)完全独立于原始拷贝(`rect`)和浅层拷贝(`srect`)。

我们在这里已经讨论了很多内容，但是仍然有一些关于复制对象的细节。

深入是值得的(哈！)关于这个话题，所以你可能要好好研究一下 [`copy`模块文档](https://docs.python.org/3/library/copy.html)。例如，对象可以通过定义特殊的方法`__copy__()`和`__deepcopy__()`来控制它们如何被复制。

[*Remove ads*](/account/join/)

## 需要记住的 3 件事

*   制作对象的浅层副本不会克隆子对象。因此，副本并不完全独立于原件。
*   对象的深层副本将递归克隆子对象。克隆完全独立于原始副本，但是创建深层副本的速度较慢。
*   您可以使用`copy`模块复制任意对象(包括自定义类)。

如果您想更深入地了解其他中级 Python 编程技术，请查看这个免费赠品:

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。**
# Python 101:平等 vs 身份

> 原文：<https://www.blog.pythonlibrary.org/2017/02/28/python-101-equality-vs-identity/>

刚接触 Python 编程语言的人可能会对 **"=="** (相等)和 Python 的关键字**"是"**(身份)之间的区别感到有点困惑。我甚至见过一些有经验的程序员，他们会发现这种差别非常微妙，以至于他们会在代码中引入逻辑错误，从而造成两者之间的误解。在这篇文章中，我们将看看这个有趣的话题。

* * *

### Python 中的等式

许多编程语言都有**相等**的概念，一些语言使用双等号(“==”)来表示这个概念。让；让我们来看看行动中的平等:

```py
>>> num = 1
>>> num_two = num
>>> num == num_two
True

```

这里我们创建了一个名为 **num** 的变量，并将它赋给整数 1。接下来，我们创建第二个变量 **num_two** ，并将其赋值给 **num** 的值。最后我们问 Python num 和 num_two 是否相等。在这种情况下，Python 告诉我们这个表达式是**真**。

另一种思考等式的方式是，我们询问 Python 两个变量是否包含相同的东西。在上面的例子中，它们都包含整数 1。让我们看看当我们创建两个具有相同值的列表时会发生什么:

```py
>>> list_one = [1, 2, 3]
>>> list_two = [1, 2, 3]
>>> list_one == list_two
True

```

这正如我们所料。

现在让我们看看如果我们询问 Python 他们的身份会发生什么:

```py
>>> num is num_two
True
>>> list_one is list_two
False

```

这里发生了什么？第一个示例返回 True，但第二个返回 False！我们将在下一节研究这个问题。

* * *

### Python 中的身份

当你问 Python 一个对象**是否与另一个对象**相同时，你是在问它们是否有相同的身份。它们实际上是同一个对象吗？在 num 和 num_two 的情况下，答案是肯定的。Python 通过其内置的 **id()** 函数提供了一种简单的证明方法:

```py
>>> id(num)
10914368
>>> id(num_two)
10914368

```

这两个变量共享相同标识的原因是，当我们将 num 赋值给 num_two(即 num_two = num)时，我们告诉 Python 它们应该返回。如果你来自 C 或 C++，你可以把标识看作一个指针，其中 num 和 num_two 都指向内存中的同一个位置。如果您在两个列表对象上使用 Python 的 id()函数，您会很快发现它们具有不同的身份:

```py
>>> id(list_one)
140401050827592
>>> id(list_two)
140401050827976

```

因此，当你问 Python“list _ one 是 list_two”这个问题时，你会得到 False。请注意，您也可以询问 Python 一个对象是否不是另一个对象:

```py
>>> list_one = [1, 2, 3]
>>> list_two = [1, 2, 3]
>>> list_one is not list_two
True

```

让我们花一点时间来看看当你混淆平等和身份时会发生什么。

* * *

### 混合起来

我知道当我开始做 Python 程序员时，这类事情会导致愚蠢的错误。原因是我会看到这样的推荐语句:

```py
if obj is None: 
    # do something
    call_function()

```

所以我会天真地认为你可以这样做:

```py
>>> def func():
    return [1, 2, 3]

>>> list_one = [1, 2, 3]
>>> list_two = func()
>>> list_one is list_two
False

```

当然，这不起作用，因为我现在有两个不同的对象，它们有不同的身份。我想做的是:

```py
>>> list_one == list_two
True

```

与这个问题无关的另一个问题是，当您创建指向同一个对象的两个变量时，您认为您可以独立地处理它们:

```py
>>> list_one = list_two = [1, 2, 3]
>>> list_one == list_two
True
>>> list_one is list_two
True
>>> list_two.append(5)
>>> list_one
[1, 2, 3, 5]

```

在这个例子中，我创建了两个指向一个对象的变量。然后我试着添加一个元素到两个列表中。很多初学者没有意识到的是，他们刚刚把那个元素也添加到了**列表中。原因是 list_one 和 list_two 都指向完全相同的对象。这一点在我们问 Python 是 **list_one 是 list_two** 时得到了证明，它返回了 **True** 。**

* * *

### 包扎

希望现在您已经理解了 Python 中等式(==)和等式(is)之间的区别。相等基本上就是询问两个对象的内容是否相同，在列表的情况下，也需要相同的顺序。Python 中的身份指的是你所指的对象。在 Python 中，对象的标识是一个惟一的常量整数(或长整数)，它在对象的生命周期中一直存在。

* * *

### 附加阅读

*   Python 文档中关于 [id()函数](https://docs.python.org/2/library/functions.html#id)
*   [的 Python 文档是](https://docs.python.org/2/reference/expressions.html#is)
*   StackOverflow: [理解 Python 的“is”操作符](http://stackoverflow.com/questions/13650293/understanding-pythons-is-operator)
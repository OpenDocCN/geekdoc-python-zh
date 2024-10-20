# Python 中的类型检查

> 原文：<https://www.blog.pythonlibrary.org/2020/04/15/type-checking-in-python/>

类型检查或提示是 Python 3.5 中新增的一个新特性。类型提示也被称为**类型注释**。类型提示是向函数和变量声明中添加特殊的语法，告诉开发人员参数或变量是什么类型。

Python 不强制类型提示。因此，在 Python 中仍然可以随意更改类型。然而，一些集成开发环境，如 PyCharm，支持类型提示，并会突出显示输入错误。你也可以使用一个叫做 [**Mypy**](https://mypy.readthedocs.io) 的工具来帮你检查你的打字情况。在本文的后面，您将了解到关于该工具的更多信息。

您将了解以下内容:

*   类型提示的利与弊
*   内置类型提示/变量注释
*   集合类型提示
*   可以是`None`的暗示值
*   类型提示功能
*   当事情变得复杂时该怎么办
*   班级
*   装修工
*   错认假频伪信号
*   其他类型提示
*   键入注释
*   静态类型检查

我们开始吧！

### 类型提示的利与弊

谈到 Python 中的类型提示，有几件事情需要提前了解。让我们先来看看类型提示的优点:

*   除了文档字符串之外，类型提示也是记录代码的好方法
*   类型提示可以让 ide 和 linters 给出更好的反馈和更好的自动完成
*   添加类型提示迫使您考虑类型，这可能有助于您在设计应用程序时做出正确的决策。

添加类型提示并不都是彩虹和玫瑰。有一些缺点:

*   代码更加冗长，也更难编写
*   类型提示增加了开发时间
*   类型提示仅在 Python 3.5+中有效。在那之前，你必须使用类型注释
*   类型提示在使用它的代码中会有一个较小的启动时间损失，特别是如果您导入了`typing`模块。

那么什么时候应该使用类型提示呢？以下是一些例子:

*   如果您计划编写简短的代码片段或一次性脚本，则不需要包含类型提示。
*   初学者在学习 Python 时也不需要添加类型提示
*   如果您正在设计一个供其他开发人员使用的库，添加类型提示可能是一个好主意
*   大型 Python 项目(即数千行代码)也可以从类型提示中受益
*   如果你要写单元测试，一些核心开发人员建议添加类型提示

在 Python 中，类型提示是一个颇有争议的话题。您不需要一直使用它，但是在某些情况下，类型提示会有所帮助。

让我们用这篇文章的剩余部分来学习如何使用类型提示！

### 内置类型提示/变量注释

您可以使用下列内置类型添加类型提示:

*   `int`
*   `float`
*   `bool`
*   `str`
*   `bytes`

这些既可以用在函数中，也可以用在变量注释中。变量注释的概念是在 3.6 版本中添加到 Python 语言中的。变量批注允许您向变量添加类型提示。

以下是一些例子:

```py
x: int  # a variable named x without initialization
y: float = 1.0  # a float variable, initialized to 1.0
z: bool = False
a: str = 'Hello type hinting'
```

您可以在根本不初始化变量的情况下向变量添加类型提示，第一行代码就是这种情况。另外 3 行代码展示了如何注释每个变量并适当地初始化它们。

接下来让我们看看如何为集合添加类型提示！

### 序列类型提示

在 Python 中，集合是一组项目。常见的集合或序列有`list`、`dict`、`tuple`和`set`。但是，您不能使用这些内置类型来注释变量。相反，你必须使用`typing`模块。

让我们看几个例子:

```py
>>> from typing import List
>>> names: List[str] = ['Mike']
>>> names
['Mike']
```

这里您创建了一个只有一个`str`的`list`。这表明您正在创建一个字符串的`list`。如果您知道列表总是相同的大小，您可以在列表中指定每个项目的类型:

```py
>>> from typing import List
>>> names: List[str, str] = ['Mike', 'James']
```

提示元组非常相似:

```py
>>> from typing import Tuple
>>> s: Tuple[int, float, str] = (5, 3.14, 'hello')
```

字典略有不同，因为您应该提示键和值的类型是:

```py
>>> from typing import Dict
>>> d: Dict[str, int] = {'one': 1}
```

如果您知道集合的大小可变，您可以使用省略号:

```py
>>> from typing import Tuple
>>> t: Tuple[int, ...] = (4, 5, 6)
```

现在让我们来学习如果一个项目是类型`None`该怎么做！

### 暗示值可能是零

有时一个值需要被初始化为`None`，但是当它被稍后设置时，你希望它是别的值。

为此，您可以使用`Optional`:

```py
>>> from typing import Optional
>>> result: Optional[str] = my_function()
```

另一方面，如果值永远不能是`None`，那么您应该在代码中添加一个`assert`:

```py
>>> assert result is not None
```

接下来让我们看看如何注释函数！

### 类型提示功能

类型提示函数类似于类型提示变量。主要的区别是你也可以给函数添加一个返回类型。

让我们来看一个例子:

```py
def adder(x: int, y: int) -> None:
    print(f'The total of {x} + {y} = {x+y}')
```

这个例子向您展示了`adder()`有两个参数，`x`和`y`，它们都应该是整数。返回类型是`None`，您可以在结束括号之后冒号之前使用`->`来指定。

假设您想将`adder()`函数赋给一个变量。您可以像这样将变量注释为`Callable`:

```py
from typing import Callable

def adder(x: int, y: int) -> None:
    print(f'The total of {x} + {y} = {x+y}')

a: Callable[[int, int], None] = adder
```

`Callable`接受函数的参数列表。它还允许您指定返回类型。

让我们再看一个例子，在这个例子中，你可以传递更复杂的参数:

```py
from typing import Tuple, Optional

def some_func(x: int, y: Tuple[str, str], 
              z: Optional[float]: = None): -> Optional[str]:
    if x > 10:
        return None
    return 'You called some_func'
```

对于这个例子，您创建了接受 3 个参数的`some_func()`:

*   一个`int`
*   两个项目的字符串
*   默认为`None`的可选`float`

注意，当你在函数中使用默认值时，在使用类型提示时，你应该在等号前后加一个空格。

它也返回`None`或一个字符串。

让我们继续前进，发现在更复杂的情况下该怎么做！

### 当事情变得复杂时该怎么办

你已经学会了当一个值可以是`None`时该怎么做，但是当事情变得复杂时你还能做什么？例如，如果传入的参数可以是多种不同的类型，您会怎么做？

对于特定用例，您可以使用`Union`:

```py
>>> from typing import Union
>>> z: Union[str, int]
```

这个类型提示的意思是，变量`z`可以是字符串，也可以是整数。

也有函数接受一个对象的情况。如果该对象可以是几个不同对象中的一个，那么您可以使用`Any`。

```py
x: Any = some_function()
```

小心使用`Any`,因为你不能真正说出你要返回的是什么。由于它可以是“任何”类型，这就像用一个简单的`except`来捕捉所有异常。当你使用`Any`的时候，你不知道你在捕捉什么异常，你也不知道你在暗示什么类型。

### 班级

如果你已经写了一个`class`，你也可以为它创建一个注释。

```py
>>> class Test:
...     pass
... 
>>> t: Test = Test()
```

如果您在函数或方法之间传递类的实例，这将非常有用。

### 装修工

装修工是一种特殊的野兽。它们是接受其他函数并修改它们的函数。在这本书的后面你会学到关于装饰者的知识。

给装饰者添加类型提示有点难看。

让我们来看看:

```py
>>> from typing import Any, Callable, TypeVar, cast
>>> F = TypeVar('F', bound=Callable[..., Any])
>>> def my_decorator(func: F) -> F:
        def wrapper(*args, **kwds):
            print("Calling", func)
            return func(*args, **kwds)
        return cast(F, wrapper)
```

`TypeVar`是一种指定自定义类型的方式。您正在创建一个定制的`Callable`类型，它可以接受任意数量的参数并返回`Any`。然后创建一个装饰器，并添加新类型作为第一个参数的类型提示以及返回类型。

仅静态代码检查器实用程序 Mypy 使用了`cast`函数。它用于将值强制转换为指定的类型。在这种情况下，您将把`wrapper`函数转换为类型`F`。

### 错认假频伪信号

您可以为类型创建新名称。例如，让我们将`List`类型重命名为`Vector`:

```py
>>> from typing import List
>>> Vector = List[int]
>>> def some_function(a: Vector) -> None:
...     print(a)
```

现在`Vector`和`List`指的是同一类型的提示。混淆类型提示对于复杂类型很有用。

`typing`文档中有一个很好的例子，复制如下:

```py
from typing import Dict, Tuple

ConnectionOptions = Dict[str, str]
Address = Tuple[str, int]
Server = Tuple[Address, ConnectionOptions]
```

这段代码允许您将类型嵌套在其他类型中，同时仍然能够编写适当的类型提示。

### 其他类型提示

您还可以使用其他几种类型提示。例如，有一些通用的可变类型，比如`MutableMapping`，您可以将它们用于定制的可变字典。

还有一种`ContextManager`类型可以用于上下文管理器。

查看所有各种类型的所有细节的完整文档:

*   [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)

### 键入注释

Python 2.7 开发于 2020 年 1 月 1 日结束。然而，在未来的几年里，人们将不得不使用许多遗留的 Python 2 代码。Python 2 中从未添加类型提示。但是您可以使用类似的语法作为注释。

这里有一个例子:

```py
def some_function(a):
    # type: str -> None
    print(a)
```

要实现这一点，您需要让注释以`type:`开头。这一行必须位于它所提示的代码的同一行或下一行。如果函数有多个参数，那么可以用逗号分隔提示:

```py
def some_function(a, b, c):
    # type: (str, int, int) -> None
    print(a)
```

一些 Python IDEs 可能支持 docstring 中的类型提示。例如，PyCharm 允许您执行以下操作:

```py
def some_function(a, b):
    """
    @type a: int
    @type b: float
    """
```

Mypy 将处理其他评论，但不会处理这些评论。如果您使用 PyCharm，您可以使用任何一种类型提示。

如果你的公司想使用类型提示，你应该提倡升级到 Python 3 来充分利用它。

### 静态类型检查

你已经多次看到有人提到我的 py。你可以在这里阅读所有相关内容:

*   [http://mypy-lang.org/](http://mypy-lang.org/)

如果您想在自己的代码上运行 Mypy，您需要使用`pip`来安装它:

```py
$ pip install mypy
```

一旦安装了`mypy`,就可以像这样运行这个工具:

```py
$ mypy my_program.py
```

Mypy 将针对您的代码运行，并打印出它发现的任何类型错误。当 Mypy 运行时，它不运行您的代码。这很像棉绒的工作原理。linter 是一个静态检查代码错误的工具。

如果你的程序中没有类型提示，Mypy 将不会打印任何错误报告。

让我们编写一个错误类型的提示函数，并将其保存到一个名为`bad_type_hinting.py`的文件中:

```py
# bad_type_hinting.py

def my_function(a: str, b: str) -> None:
    return a.keys() + b.keys()
```

现在您有了一些代码，您可以对它运行 Mypy:

```py
$ mypy bad_type_hinting.py 
bad_type_hinting.py:4: error: "str" has no attribute "keys"
Found 1 error in 1 file (checked 1 source file)
```

这个输出告诉您第 4 行有一个问题。字符串没有`keys()`属性。

让我们更新代码，删除对不存在的`keys()`方法的调用。您可以将这些更改保存到名为`bad_type_hinting2.py`的新文件中:

```py
# bad_type_hinting2.py

def my_function(a: str, b: str) -> None:
    return a + b
```

现在，您应该针对您的更改运行 Mypy，看看您是否修复了它:

```py
$ mypy bad_type_hinting2.py 
bad_type_hinting2.py:4: error: No return value expected
Found 1 error in 1 file (checked 1 source file)
```

哎呦！仍然有一个错误。这一次你知道你不期望这个函数返回任何东西。您可以修改代码，使其不返回任何内容，或者您可以修改类型提示，使其返回一个`str`。

您应该尝试后一种方法，并将下面的代码保存到`good_type_hinting.py`:

```py
# good_type_hinting.py

def my_function(a: str, b: str) -> str:
    return a + b
```

现在对这个新文件运行 Mypy:

```py
$ mypy good_type_hinting.py 
Success: no issues found in 1 source file

```

这一次你的代码没有问题！

您可以针对多个文件甚至整个文件夹运行 Mypy。如果您致力于在代码中使用类型提示，那么您应该经常在代码中运行 Mypy，以确保您的代码没有错误。

### 包扎

您现在知道了什么是类型提示或注释，以及如何操作。事实上，您已经学习了有效进行类型提示所需的所有基础知识。

在本文中，您了解了:

*   类型提示的利与弊
*   内置类型提示/变量注释
*   集合类型提示
*   可以是`None`的暗示值
*   类型提示功能
*   当事情变得复杂时该怎么办
*   班级
*   装修工
*   错认假频伪信号
*   其他类型提示
*   键入注释
*   静态类型检查

如果遇到困难，您应该查看以下资源寻求帮助:

*   来自 Mypy 的类型提示 [Cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
*   [排版](https://github.com/python/typeshed)
*   Python 的[类型模块](https://docs.python.org/3/library/typing.html)的文档

Python 中不需要类型提示。您可以编写所有代码，而无需在代码中添加任何注释。但是类型提示很好理解，并且可能证明在您的工具箱中有它很方便。
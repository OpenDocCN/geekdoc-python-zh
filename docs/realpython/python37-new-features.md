# Python 3.7:很酷的新特性供您尝试

> 原文：<https://realpython.com/python37-new-features/>

python 3.7[正式发布](https://www.python.org/downloads/release/python-370/)！这个新的 Python 版本自 2016 年 9 月开始开发，现在我们都开始享受核心开发者辛勤工作的成果。

Python 新版本带来了什么？虽然[文档](https://docs.python.org/3.7/whatsnew/3.7.html)很好地概述了新特性，但本文将深入探讨一些最重要的新闻。其中包括:

*   通过新的内置`breakpoint()`更容易访问调试器
*   使用数据类创建简单的类
*   对模块属性的定制访问
*   改进了对类型提示的支持
*   更高精度的定时功能

更重要的是，Python 3.7 速度快。

在本文的最后几节，您将会读到更多关于这个速度的内容，以及 Python 3.7 的一些其他很酷的特性。您还将获得一些关于升级到新版本的建议。

## `breakpoint()`内置

虽然我们可能努力写出完美的代码，但简单的事实是我们从来没有这样做过。调试是编程的一个重要部分。Python 3.7 引入了新的内置函数`breakpoint()`。这并没有给 Python 添加任何新的功能，但是它使得调试器的使用更加灵活和直观。

假设您在文件`bugs.py`中有以下错误代码:

```py
def divide(e, f):
    return f / e

a, b = 0, 1
print(divide(a, b))
```

运行代码会在`divide()`函数中产生一个`ZeroDivisionError`。假设你想中断你的代码，进入`divide()`顶部的[调试器](https://realpython.com/python-debugging-pdb/)。您可以通过在代码中设置一个所谓的“断点”来做到这一点:

```py
def divide(e, f):
    # Insert breakpoint here
    return f / e
```

断点是代码内部的一个信号，表示执行应该暂时停止，以便您可以查看程序的当前状态。如何放置断点？在 Python 3.6 和更低版本中，您使用了这一行有点神秘的代码:

```py
def divide(e, f):
    import pdb; pdb.set_trace()
    return f / e
```

这里， [`pdb`](https://docs.python.org/library/pdb.html) 是来自标准库的 Python 调试器。在 Python 3.7 中，您可以使用新的`breakpoint()`函数调用作为快捷方式:

```py
def divide(e, f):
    breakpoint()
    return f / e
```

在后台，`breakpoint()`是先导入`pdb`再给你调用`pdb.set_trace()`。明显的好处是`breakpoint()`更容易记忆，你只需要输入 12 个字符而不是 27 个。然而，使用`breakpoint()`的真正好处是它的可定制性。

使用`breakpoint()`运行您的`bugs.py`脚本:

```py
$ python3.7 bugs.py 
> /home/gahjelle/bugs.py(3)divide()
-> return f / e
(Pdb)
```

脚本将在到达`breakpoint()`时中断，并让您进入 PDB 调试会话。您可以键入`c`并点击 `Enter` 继续脚本。如果你想了解更多关于 PDB 和调试的知识，请参考[内森·詹宁斯的 PDB 指南](https://realpython.com/python-debugging-pdb/)。

现在，假设你认为你已经修复了这个错误。您希望再次运行脚本，但不在调试器中停止。当然，您可以注释掉`breakpoint()`行，但是另一个选择是使用`PYTHONBREAKPOINT`环境变量。这个变量控制`breakpoint()`的行为，设置`PYTHONBREAKPOINT=0`意味着任何对`breakpoint()`的调用都被忽略:

```py
$ PYTHONBREAKPOINT=0 python3.7 bugs.py
ZeroDivisionError: division by zero
```

哎呀，看来你还是没有修复这个错误…

另一个选择是使用`PYTHONBREAKPOINT`来指定一个除 PDB 之外的调试器。例如，要使用 [PuDB](https://pypi.org/project/pudb/) (控制台中的一个可视化调试器)，您可以:

```py
$ PYTHONBREAKPOINT=pudb.set_trace python3.7 bugs.py
```

为此，您需要安装`pudb`(`pip install pudb`)。不过 Python 会为你负责导入`pudb`。这样，您也可以设置您的默认调试器。只需将`PYTHONBREAKPOINT`环境变量设置为您喜欢的调试器。参见[本指南](https://www.schrodinger.com/kb/1842)了解如何在您的系统上设置环境变量。

新的`breakpoint()`函数不仅适用于调试器。一个方便的选择是在代码中简单地启动一个交互式 shell。例如，要启动 IPython 会话，可以使用以下命令:

```py
$ PYTHONBREAKPOINT=IPython.embed python3.7 bugs.py 
IPython 6.3.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: print(e / f)
0.0
```

你也可以创建自己的函数并让`breakpoint()`调用它。下面的代码打印局部范围内的所有变量。将其添加到名为`bp_utils.py`的文件中:

```py
from pprint import pprint
import sys

def print_locals():
    caller = sys._getframe(1)  # Caller is 1 frame up.
    pprint(caller.f_locals)
```

要使用该功能，如前所述设置`PYTHONBREAKPOINT`，用`<module>.<function>`符号表示:

```py
$ PYTHONBREAKPOINT=bp_utils.print_locals python3.7 bugs.py 
{'e': 0, 'f': 1}
ZeroDivisionError: division by zero
```

正常情况下，`breakpoint()`会被用来调用不需要参数的函数和方法。然而，也可以传递参数。将`bugs.py`中的`breakpoint()`行改为:

```py
breakpoint(e, f, end="<-END\n")
```

**注意:**默认的 PDB 调试器将在这一行抛出一个`TypeError`，因为`pdb.set_trace()`不接受任何位置参数。

用伪装成 [`print()`](https://realpython.com/python-print/) 函数的`breakpoint()`运行这段代码，查看一个传递参数的简单示例:

```py
$ PYTHONBREAKPOINT=print python3.7 bugs.py 
0 1<-END
ZeroDivisionError: division by zero
```

更多信息参见 [PEP 553](https://www.python.org/dev/peps/pep-0553/) 以及 [`breakpoint()`](https://docs.python.org/3.7/library/functions.html#breakpoint) 和 [`sys.breakpointhook()`](https://docs.python.org/3.7/library/sys.html#sys.breakpointhook) 的文档。

[*Remove ads*](/account/join/)

## 数据类别

新的 [`dataclasses`](https://realpython.com/python-data-classes/) 模块使得编写自己的类更加方便，因为像`.__init__()`、`.__repr__()`和`.__eq__()`这样的特殊方法是自动添加的。使用`@dataclass`装饰器，您可以编写如下代码:

```py
from dataclasses import dataclass, field

@dataclass(order=True)
class Country:
    name: str
    population: int
    area: float = field(repr=False, compare=False)
    coastline: float = 0

    def beach_per_person(self):
        """Meters of coastline per person"""
        return (self.coastline * 1000) / self.population
```

这九行代码代表了相当多的样板代码和最佳实践。考虑一下将`Country`实现为一个常规类需要什么:一个`.__init__()`方法，一个`repr`，六个不同的比较方法以及一个`.beach_per_person()`方法。您可以展开下面的框来查看大约相当于数据类的`Country`的实现:



```py
class Country:

    def __init__(self, name, population, area, coastline=0):
        self.name = name
        self.population = population
        self.area = area
        self.coastline = coastline

    def __repr__(self):
        return (
            f"Country(name={self.name!r}, population={self.population!r},"
            f" coastline={self.coastline!r})"
        )

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (
                (self.name, self.population, self.coastline)
                == (other.name, other.population, other.coastline)
            )
        return NotImplemented

    def __ne__(self, other):
        if other.__class__ is self.__class__:
            return (
                (self.name, self.population, self.coastline)
                != (other.name, other.population, other.coastline)
            )
        return NotImplemented

    def __lt__(self, other):
        if other.__class__ is self.__class__:
            return ((self.name, self.population, self.coastline) < (
                other.name, other.population, other.coastline
            ))
        return NotImplemented

    def __le__(self, other):
        if other.__class__ is self.__class__:
            return ((self.name, self.population, self.coastline) <= (
                other.name, other.population, other.coastline
            ))
        return NotImplemented

    def __gt__(self, other):
        if other.__class__ is self.__class__:
            return ((self.name, self.population, self.coastline) > (
                other.name, other.population, other.coastline
            ))
        return NotImplemented

    def __ge__(self, other):
        if other.__class__ is self.__class__:
            return ((self.name, self.population, self.coastline) >= (
                other.name, other.population, other.coastline
            ))
        return NotImplemented

    def beach_per_person(self):
        """Meters of coastline per person"""
        return (self.coastline * 1000) / self.population
```

创建后，数据类就是普通类。例如，您可以以正常方式从数据类继承。数据类的主要目的是使编写健壮的类变得快速和容易，特别是主要存储数据的小类。

您可以像使用任何其他类一样使用`Country`数据类:

>>>

```py
>>> norway = Country("Norway", 5320045, 323802, 58133)
>>> norway
Country(name='Norway', population=5320045, coastline=58133)

>>> norway.area
323802

>>> usa = Country("United States", 326625791, 9833517, 19924)
>>> nepal = Country("Nepal", 29384297, 147181)
>>> nepal
Country(name='Nepal', population=29384297, coastline=0)

>>> usa.beach_per_person()
0.06099946957342386

>>> norway.beach_per_person()
10.927163210085629
```

注意，初始化类时使用所有字段`.name`、`.population`、`.area`和`.coastline`(尽管`.coastline`是可选的，如内陆尼泊尔的例子所示)。`Country`类有一个合理的 [`repr`](https://dbader.org/blog/python-repr-vs-str) ，而定义方法的工作方式与普通类相同。

默认情况下，可以比较数据类是否相等。因为我们在`@dataclass`装饰器中指定了`order=True`，所以`Country`类也可以被排序:

>>>

```py
>>> norway == norway
True

>>> nepal == usa
False

>>> sorted((norway, usa, nepal))
[Country(name='Nepal', population=29384297, coastline=0),
 Country(name='Norway', population=5320045, coastline=58133),
 Country(name='United States', population=326625791, coastline=19924)]
```

对字段值进行排序，首先是`.name`，然后是`.population`，依此类推。然而，如果您使用`field()`，您可以[定制](https://realpython.com/python-data-classes/#advanced-default-values)哪些字段将用于比较。在这个例子中，`.area`字段被排除在`repr`和比较之外。

**注:**国家数据来自[中情局世界概况](https://www.cia.gov/library/publications/the-world-factbook/)，人口数字估计为 2017 年 7 月。

在你们预订下一次挪威海滩度假之前，这里是关于挪威气候的实况报道:“沿海地区气候温和，受北大西洋洋流影响；降水量增加，夏季更冷，内陆更冷；西海岸全年多雨。”

数据类做一些与 [`namedtuple`](https://dbader.org/blog/writing-clean-python-with-namedtuples) 相同的事情。然而，他们从 [`attrs`项目](http://www.attrs.org/)中获得了最大的灵感。参见我们的[完整的数据类指南](https://realpython.com/python-data-classes/)以获得更多的例子和进一步的信息，以及 [PEP 557](https://www.python.org/dev/peps/pep-0557/) 的官方描述。

## 模块属性定制

属性在 Python 中无处不在！虽然类属性可能是最著名的，但属性实际上可以放在任何东西上——包括函数和模块。Python 的几个基本特性被实现为属性:大部分自省功能、文档字符串和名称空间。模块内部的函数可以作为模块属性使用。

最常使用点符号来检索属性:`thing.attribute`。然而，您也可以使用`getattr()`获得在运行时命名的属性:

```py
import random

random_attr = random.choice(("gammavariate", "lognormvariate", "normalvariate"))
random_func = getattr(random, random_attr)

print(f"A {random_attr} random value: {random_func(1, 1)}")
```

运行这段代码将会产生如下结果:

```py
A gammavariate random value: 2.8017715125270618
```

对于类，调用`thing.attr`将首先寻找在`thing`上定义的`attr`。如果没有找到，那么调用特殊方法`thing.__getattr__("attr")`。(这是一种简化。详见[本文](http://blog.lerner.co.il/python-attributes/)。)方法`.__getattr__()`可用于定制对对象属性的访问。

在 Python 3.7 之前，同样的定制不容易用于模块属性。然而， [PEP 562](https://www.python.org/dev/peps/pep-0562/) 在模块上引入了`__getattr__()`，以及相应的`__dir__()`功能。`__dir__()`特殊功能允许定制模块上调用[T3 的结果。](https://realpython.com/python-modules-packages/#the-dir-function)

PEP 本身给出了一些如何使用这些函数的例子，包括向函数中添加弃用警告，以及延迟加载沉重的子模块。下面，我们将构建一个简单的插件系统，允许功能被动态地添加到一个模块中。这个例子利用了 Python 包。如果你需要关于软件包的复习，请参见本文。

创建一个新目录`plugins`，并将以下代码添加到文件`plugins/__init__.py`:

```py
from importlib import import_module
from importlib import resources

PLUGINS = dict()

def register_plugin(func):
    """Decorator to register plug-ins"""
    name = func.__name__
    PLUGINS[name] = func
    return func

def __getattr__(name):
    """Return a named plugin"""
    try:
        return PLUGINS[name]
    except KeyError:
        _import_plugins()
        if name in PLUGINS:
            return PLUGINS[name]
        else:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None

def __dir__():
    """List available plug-ins"""
    _import_plugins()
    return list(PLUGINS.keys())

def _import_plugins():
    """Import all resources to register plug-ins"""
    for name in resources.contents(__name__):
        if name.endswith(".py"):
            import_module(f"{__name__}.{name[:-3]}")
```

在我们看这段代码做什么之前，在`plugins`目录中再添加两个文件。首先来看看`plugins/plugin_1.py`:

```py
from . import register_plugin

@register_plugin
def hello_1():
    print("Hello from Plugin 1")
```

接下来，在文件`plugins/plugin_2.py`中添加类似的代码:

```py
from . import register_plugin

@register_plugin
def hello_2():
    print("Hello from Plugin 2")

@register_plugin
def goodbye():
    print("Plugin 2 says goodbye")
```

这些插件现在可以如下使用:

>>>

```py
>>> import plugins
>>> plugins.hello_1()
Hello from Plugin 1

>>> dir(plugins)
['goodbye', 'hello_1', 'hello_2']

>>> plugins.goodbye()
Plugin 2 says goodbye
```

这可能看起来不那么具有革命性(很可能不是)，但让我们看看这里实际发生了什么。通常情况下，为了能够调用`plugins.hello_1()`，`hello_1()`函数必须在`plugins`模块中定义，或者在`plugins`包的`__init__.py`中显式导入。在这里，两者都不是！

相反，`hello_1()`被定义在`plugins`包内的任意文件中，`hello_1()`通过使用`@register_plugin` [装饰器](https://realpython.com/primer-on-python-decorators/)注册自己而成为`plugins`包的一部分。

差别是微妙的。不同于软件包规定哪些功能可用，单个功能将自己注册为软件包的一部分。这为您提供了一个简单的结构，您可以独立于代码的其余部分添加函数，而不必保留可用函数的集中列表。

让我们快速回顾一下`__getattr__()`在`plugins/__init__.py`代码中做了什么。当您请求`plugins.hello_1()`时，Python 首先在`plugins/__init__.py`文件中寻找一个`hello_1()`函数。因为不存在这样的函数，所以 Python 调用了`__getattr__("hello_1")`。记住`__getattr__()`函数的源代码:

```py
def __getattr__(name):
    """Return a named plugin"""
    try:
        return PLUGINS[name]        # 1) Try to return plugin
    except KeyError:
        _import_plugins()           # 2) Import all plugins
        if name in PLUGINS:
            return PLUGINS[name]    # 3) Try to return plugin again
        else:
            raise AttributeError(   # 4) Raise error
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
```

`__getattr__()`包含以下步骤。下表中的数字对应于代码中的编号注释:

1.  首先，该函数乐观地尝试从`PLUGINS`字典中返回已命名的插件。如果名为`name`的插件存在并且已经被导入，这将会成功。
2.  如果在`PLUGINS`字典中没有找到指定的插件，我们确保所有的插件都被导入。
3.  如果导入后可用，则返回指定的插件。
4.  如果在导入所有插件后，插件不在`PLUGINS`字典中，我们抛出一个`AttributeError`,表示`name`不是当前模块的属性(插件)。

然而，字典是如何填充的呢？`_import_plugins()`函数导入了`plugins`包中的所有 Python 文件，但似乎没有触及`PLUGINS`:

```py
def _import_plugins():
    """Import all resources to register plug-ins"""
    for name in resources.contents(__name__):
        if name.endswith(".py"):
            import_module(f"{__name__}.{name[:-3]}")
```

别忘了每个插件函数都是由`@register_plugin`装饰器装饰的。这个装饰器在插件被导入时被调用，并且是真正填充`PLUGINS`字典的那个。如果您手动导入其中一个插件文件，就会看到这种情况:

>>>

```py
>>> import plugins
>>> plugins.PLUGINS
{}

>>> import plugins.plugin_1
>>> plugins.PLUGINS
{'hello_1': <function hello_1 at 0x7f29d4341598>}
```

继续这个例子，注意在模块上调用`dir()`也会导入剩余的插件:

>>>

```py
>>> dir(plugins)
['goodbye', 'hello_1', 'hello_2']

>>> plugins.PLUGINS
{'hello_1': <function hello_1 at 0x7f29d4341598>,
 'hello_2': <function hello_2 at 0x7f29d4341620>,
 'goodbye': <function goodbye at 0x7f29d43416a8>}
```

`dir()`通常列出一个对象的所有可用属性。通常，在一个模块上使用`dir()`会产生类似这样的结果:

>>>

```py
>>> import plugins
>>> dir(plugins)
['PLUGINS', '__builtins__', '__cached__', '__doc__',
 '__file__', '__getattr__', '__loader__', '__name__',
 '__package__', '__path__', '__spec__', '_import_plugins',
 'import_module', 'register_plugin', 'resources']
```

虽然这可能是有用的信息，但我们更感兴趣的是公开可用的插件。在 Python 3.7 中，可以通过添加一个`__dir__()`特殊函数来自定义在模块上调用`dir()`的结果。对于`plugins/__init__.py`，这个函数首先确定所有插件都已经导入，然后列出它们的名称:

```py
def __dir__():
    """List available plug-ins"""
    _import_plugins()
    return list(PLUGINS.keys())
```

在离开这个例子之前，请注意我们还使用了 Python 3.7 的另一个很酷的新特性。为了导入`plugins`目录中的所有模块，我们使用了新的 [`importlib.resources`](https://docs.python.org/3.7/library/importlib.html#module-importlib.resources) 模块。这个模块提供了对模块和包内部的文件和资源的访问，而不需要`__file__`黑客(这并不总是有效)或者`pkg_resources`(这很慢)。`importlib.resources`的其他特点将在后面中[强调。](#other-pretty-cool-features)

[*Remove ads*](/account/join/)

## 打字增强功能

[类型提示和注释](https://realpython.com/python-type-checking/)在 Python 3 系列版本中一直在不断发展。Python 的打字系统现在已经相当稳定了。尽管如此，Python 3.7 还是带来了一些改进:更好的性能、核心支持和前向引用。

Python 在运行时不做任何类型检查(除非你显式地使用像 [`enforce`](https://pypi.org/project/enforce/) 这样的包)。因此，向代码中添加类型提示不会影响其性能。

不幸的是，这并不完全正确，因为大多数类型提示都需要`typing`模块。`typing`模块是标准库中[最慢的模块](https://www.python.org/dev/peps/pep-0560/#performance)之一。 [PEP 560](https://www.python.org/dev/peps/pep-0560) 在 Python 3.7 中增加了一些对类型的核心支持，显著加快了`typing`模块的速度。这方面的细节一般来说没有必要知道。只需向后一靠，享受更高的性能。

虽然 Python 的类型系统表达能力相当强，但有一个问题很让人头疼，那就是前向引用。在导入模块时，会计算类型提示，或者更一般的注释。因此，所有名称在使用之前必须已经定义。以下情况是不可能的:

```py
class Tree:
    def __init__(self, left: Tree, right: Tree) -> None:
        self.left = left
        self.right = right
```

运行代码会引发一个`NameError`，因为在`.__init__()`方法的定义中还没有(完全)定义类`Tree`:

```py
Traceback (most recent call last):
  File "tree.py", line 1, in <module>
    class Tree:
  File "tree.py", line 2, in Tree
    def __init__(self, left: Tree, right: Tree) -> None:
NameError: name 'Tree' is not defined
```

为了克服这一点，您可能需要将`"Tree"`写成字符串文字:

```py
class Tree:
    def __init__(self, left: "Tree", right: "Tree") -> None:
        self.left = left
        self.right = right
```

原讨论见 [PEP 484](https://www.python.org/dev/peps/pep-0484/#forward-references) 。

在未来的 [Python 4.0](http://www.curiousefficiency.org/posts/2014/08/python-4000.html) 中，这种所谓的向前引用将被允许。这将通过在明确要求之前不评估注释来处理。 [PEP 563](https://www.python.org/dev/peps/pep-0563/) 描述了该提案的细节。在 Python 3.7 中，前向引用已经可以作为 [`__future__`导入](https://docs.python.org/library/__future__.html)使用。您现在可以编写以下内容:

```py
from __future__ import annotations

class Tree:
    def __init__(self, left: Tree, right: Tree) -> None:
        self.left = left
        self.right = right
```

请注意，除了避免有些笨拙的`"Tree"`语法之外，[延迟的注释求值](https://realpython.com/python-news-april-2021/#what-pep-563-proposed-to-improve-type-annotations)也将加速您的代码，因为不执行类型提示。 [`mypy`](http://mypy-lang.org/) 已经支持正向引用。

到目前为止，注释最常见的用途是类型提示。尽管如此，您在运行时仍然可以完全访问注释，并且可以在您认为合适的时候使用它们。如果您直接处理注释，您需要显式地处理可能的前向引用。

让我们创建一些公认的愚蠢的例子来显示注释何时被求值。首先我们用老方法，所以注释在导入时被评估。让`anno.py`包含以下代码:

```py
def greet(name: print("Now!")):
    print(f"Hello {name}")
```

注意`name`的标注是`print()`。这只是为了查看注释何时被求值。导入新模块:

>>>

```py
>>> import anno
Now!

>>> anno.greet.__annotations__
{'name': None}

>>> anno.greet("Alice")
Hello Alice
```

如您所见，注释是在导入时进行评估的。注意，`name`以`None`结束注释，因为那是`print()`的返回值。

添加`__future__`导入以启用注释的延期评估:

```py
from __future__ import annotations

def greet(name: print("Now!")):
    print(f"Hello {name}")
```

导入此更新的代码将不会评估批注:

>>>

```py
>>> import anno

>>> anno.greet.__annotations__
{'name': "print('Now!')"}

>>> anno.greet("Marty")
Hello Marty
```

注意，`Now!`永远不会被打印，注释作为字符串保存在`__annotations__`字典中。为了评价注释，使用`typing.get_type_hints()`或 [`eval()`](https://realpython.com/python-eval-function/) :

>>>

```py
>>> import typing
>>> typing.get_type_hints(anno.greet)
Now!
{'name': <class 'NoneType'>}

>>> eval(anno.greet.__annotations__["name"])
Now!

>>> anno.greet.__annotations__
{'name': "print('Now!')"}
```

注意到`__annotations__`字典从不更新，所以每次使用时都需要评估注释。

[*Remove ads*](/account/join/)

## 计时精度

在 Python 3.7 中， [`time`模块](https://realpython.com/python-time-module/)获得了一些新功能，如 [PEP 564](https://www.python.org/dev/peps/pep-0564/) 中所述。特别是增加了以下六个功能:

*   **`clock_gettime_ns()` :** 返回指定时钟的时间
*   **`clock_settime_ns()` :** 设置指定时钟的时间
*   **`monotonic_ns()` :** 返回不能倒退的相对时钟的时间(例如由于夏令时)
*   **`perf_counter_ns()`** :返回性能计数器的值，该计数器是一种专门用于测量短时间间隔的时钟
*   **`process_time_ns()` :** 返回当前进程的系统和用户 CPU 时间之和(不包括睡眠时间)
*   **`time_ns()`** :返回自 1970 年 1 月 1 日以来的纳秒数

从某种意义上说，没有添加新的功能。每个函数都类似于一个没有`_ns`后缀的现有函数。不同之处在于，新函数返回纳秒数作为`int`，而不是秒数作为`float`。

对大多数应用来说，这些新的毫微秒函数和它们的旧对应物之间的差别是不明显的。然而，新函数更容易推理，因为它们依赖于`int`而不是`float`。浮点数本质上是不准确的:

>>>

```py
>>> 0.1 + 0.1 + 0.1
0.30000000000000004

>>> 0.1 + 0.1 + 0.1 == 0.3
False
```

这不是 Python 的问题，而是计算机需要用有限的位数表示无限的十进制数的结果。

Python `float`遵循 [IEEE 754 标准](https://en.wikipedia.org/wiki/IEEE_754)，使用 53 个有效位。结果是，任何大于大约 104 天(2⁵或大约 [9 千万亿纳秒](https://en.wikipedia.org/wiki/Names_of_large_numbers))的时间都不能用纳秒精度的浮点数表示。相比之下，Python [`int`是无限的](https://stackoverflow.com/a/9860611)，因此整数纳秒的精度总是与时间值无关。

例如，`time.time()`返回自 1970 年 1 月 1 日以来的秒数。这个数字已经相当大了，所以这个数字的精度在微秒级。这个函数是其`_ns`版本中改进最大的一个。`time.time_ns()`的分辨率大约是[的 3 倍](https://www.python.org/dev/peps/pep-0564/#analysis)比`time.time()`好。

顺便问一下，纳秒是什么？从技术上讲，它是十亿分之一秒，或者如果你更喜欢科学记数法的话，是`1e-9`秒。这些只是数字，并不真正提供任何直觉。想要更好的视觉帮助，请看[格蕾丝·赫柏的](https://en.wikipedia.org/wiki/Grace_Hopper#Anecdotes)纳秒的精彩演示[。](https://www.youtube.com/watch?v=9eyFDBPk4Yw)

顺便说一句，如果你需要处理纳秒精度的日期时间， [`datetime`标准库](https://realpython.com/python-datetime/)不会满足你的要求。它只显式处理微秒:

>>>

```py
>>> from datetime import datetime, timedelta
>>> datetime(2018, 6, 27) + timedelta(seconds=1e-6)
datetime.datetime(2018, 6, 27, 0, 0, 0, 1)

>>> datetime(2018, 6, 27) + timedelta(seconds=1e-9)
datetime.datetime(2018, 6, 27, 0, 0)
```

相反，你可以使用 [`astropy`项目](http://www.astropy.org/)。它的 [`astropy.time`](http://docs.astropy.org/en/stable/time/) 包使用两个`float`对象表示日期时间，这保证了“跨越宇宙年龄的亚纳秒精度”

>>>

```py
>>> from astropy.time import Time, TimeDelta
>>> Time("2018-06-27")
<Time object: scale='utc' format='iso' value=2018-06-27 00:00:00.000>

>>> t = Time("2018-06-27") + TimeDelta(1e-9, format="sec")
>>> (t - Time("2018-06-27")).sec
9.976020010071807e-10
```

最新版本的`astropy`在 Python 3.5 及更高版本中可用。

## 其他非常酷的功能

到目前为止，您已经看到了关于 Python 3.7 新特性的头条新闻。然而，还有许多其他的变化也很酷。在本节中，我们将简要介绍其中的一些。

### 字典的顺序是有保证的

Python 3.6 的 CPython 实现已经对字典进行了排序。( [PyPy](https://realpython.com/pypy-faster-python/) 也有这个。)这意味着字典中的条目按照它们被插入的顺序被迭代。第一个例子是使用 Python 3.5，第二个例子是使用 Python 3.6:

>>>

```py
>>> {"one": 1, "two": 2, "three": 3}  # Python <= 3.5
{'three': 3, 'one': 1, 'two': 2}

>>> {"one": 1, "two": 2, "three": 3}  # Python >= 3.6
{'one': 1, 'two': 2, 'three': 3}
```

在 Python 3.6 中，这种排序只是实现`dict`的一个很好的结果。然而，在 Python 3.7 中，保留插入顺序的字典是[语言规范](https://mail.python.org/pipermail/python-dev/2017-December/151283.html)的一部分。因此，现在可以在只支持 Python > = 3.7(或 CPython > = 3.6)的项目中依赖它。

[*Remove ads*](/account/join/)

### “`async`”和“`await`”是关键词

Python 3.5 引入了带有`async`和`await`语法的[协程。为了避免向后兼容的问题，`async`和`await`没有被添加到保留的](https://www.python.org/dev/peps/pep-0492/)[关键字](https://realpython.com/python-keywords/)列表中。换句话说，仍然可以定义名为`async`和`await`的变量或函数。

在 Python 3.7 中，这不再可能:

>>>

```py
>>> async = 1
  File "<stdin>", line 1
    async = 1
          ^
SyntaxError: invalid syntax

>>> def await():
  File "<stdin>", line 1
    def await():
            ^
SyntaxError: invalid syntax
```

### `asyncio`整容

标准库最初是在 Python 3.4 中引入的，使用事件循环、协程和未来以现代方式处理并发性。下面是[温柔介绍](https://hackernoon.com/asyncio-for-the-working-python-developer-5c468e6e2e8e)。

在 Python 3.7 中，`asyncio`模块得到了[的重大改进](https://docs.python.org/3.7/whatsnew/3.7.html#asyncio)，包括许多新功能、对上下文变量的支持(参见下面的[)和性能改进。特别值得注意的是`asyncio.run()`，它简化了从同步代码调用协程。使用](#context-variables) [`asyncio.run()`](https://docs.python.org/3.7/library/asyncio-task.html#asyncio.run) ，你不需要显式创建事件循环。现在可以编写一个[异步](https://realpython.com/python-async-features/) Hello World 程序:

```py
import asyncio

async def hello_world():
    print("Hello World!")

asyncio.run(hello_world())
```

### 上下文变量

上下文变量是根据上下文可以有不同值的变量。它们类似于线程本地存储，其中每个执行线程可能有一个不同的变量值。然而，对于上下文变量，一个执行线程中可能有几个上下文。上下文变量的主要用例是在并发异步任务中跟踪变量。

下面的例子构造了三个上下文，每个上下文都有自己的值`name`。`greet()`函数稍后能够在每个上下文中使用`name`的值:

```py
import contextvars

name = contextvars.ContextVar("name")
contexts = list()

def greet():
    print(f"Hello {name.get()}")

# Construct contexts and set the context variable name
for first_name in ["Steve", "Dina", "Harry"]:
    ctx = contextvars.copy_context()
    ctx.run(name.set, first_name)
    contexts.append(ctx)

# Run greet function inside each context
for ctx in reversed(contexts):
    ctx.run(greet)
```

运行该脚本以相反的顺序问候史蒂夫、迪娜和哈利:

```py
$ python3.7 context_demo.py
Hello Harry
Hello Dina
Hello Steve
```

### 导入带有`importlib.resources`的数据文件

打包 Python 项目时的一个挑战是决定如何处理项目资源，如项目所需的数据文件。有几个常用的选项:

*   硬编码数据文件的路径。
*   将数据文件放入包中，并使用`__file__`找到它。
*   使用 [`setuptools.pkg_resources`](https://setuptools.readthedocs.io/en/latest/pkg_resources.html) 访问数据文件资源。

这些都有其缺点。第一种选择是不可移植的。使用`__file__`更具可移植性，但是如果安装了 Python 项目，它可能会在 zip 中结束，并且没有`__file__`属性。第三种选择解决了这个问题，但不幸的是非常慢。

比较好的解决方案是在标准库中新增 [`importlib.resources`](https://docs.python.org/3.7/library/importlib.html#module-importlib.resources) 模块。它使用 Python 现有的导入功能来导入数据文件。假设您在 Python 包中有一个资源，如下所示:

```py
data/
│
├── alice_in_wonderland.txt
└── __init__.py
```

注意`data`需要是一个 [Python 包](https://realpython.com/python-modules-packages/)。也就是说，目录需要包含一个`__init__.py`文件(可能是空的)。然后你可以如下阅读`alice_in_wonderland.txt`文件:

>>>

```py
>>> from importlib import resources
>>> with resources.open_text("data", "alice_in_wonderland.txt") as fid:
...     alice = fid.readlines()
... 
>>> print("".join(alice[:7]))
CHAPTER I. Down the Rabbit-Hole

Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, "and what is the use of a book," thought Alice "without pictures or
conversations?"
```

类似的 [`resources.open_binary()`](https://docs.python.org/3.7/library/importlib.html#importlib.resources.open_binary) 功能也可用于以二进制模式打开文件。在前面的[“插件作为模块属性”示例](#customization-of-module-attributes)中，我们使用`importlib.resources`通过`resources.contents()`来发现可用的插件。更多信息，请参见[巴里华沙 PyCon 2018 演讲](https://www.youtube.com/watch?v=ZsGFU2qh73E)。

在 Python 2.7 和 Python 3.4+中可以通过一个[反向端口](https://pypi.org/project/importlib_resources/)来使用`importlib.resources`。从`pkg_resources`迁移到`importlib.resources`T6 的[指南可用。](http://importlib-resources.readthedocs.io/en/latest/migration.html)

[*Remove ads*](/account/join/)

### 开发者招数

Python 3.7 增加了几个针对开发人员的特性。你已经见过[新的`breakpoint()`内置](#the-breakpoint-built-in)。此外，Python 解释器中增加了一些新的 [`-X`命令行选项](https://docs.python.org/3.7/using/cmdline.html#id5)。

使用`-X importtime`，您可以很容易地知道脚本中的导入需要多少时间:

```py
$ python3.7 -X importtime my_script.py
import time: self [us] | cumulative | imported package
import time:      2607 |       2607 | _frozen_importlib_external
...
import time:       844 |      28866 |   importlib.resources
import time:       404 |      30434 | plugins
```

`cumulative`列显示导入的累计时间(以微秒计)。在这个例子中，导入`plugins`花费了大约 0.03 秒，其中大部分时间用于导入`importlib.resources`。`self`列显示不包括嵌套导入的导入时间。

您现在可以使用`-X dev`来激活“开发模式”开发模式将添加某些调试功能和运行时检查，这些功能被认为太慢，默认情况下无法启用。这些包括启用 [`faulthandler`](https://docs.python.org/library/faulthandler.html#module-faulthandler) 来显示对严重崩溃的追溯，以及更多的警告和调试挂钩。

最后，`-X utf8`启用 [UTF-8 模式](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONUTF8)。(参见 [PEP 540](https://www.python.org/dev/peps/pep-0540/) 。)在这种模式下，`UTF-8`将被用于文本编码，而不管当前的语言环境。

### 优化

Python 的每个新版本都有一组优化。在 Python 3.7 中，有一些显著的加速，包括:

*   在标准库中调用许多方法的开销更少。
*   一般来说，方法调用要快 20%。
*   Python 本身的启动时间减少了 10-30%。
*   导入`typing`快 7 倍。

此外，还包括许多更专业的优化。详细概述见[该列表](https://docs.python.org/3.7/whatsnew/3.7.html#optimizations)。

所有这些优化的结果是 [Python 3.7 更快](https://speed.python.org/)。它简直是迄今为止发布的 CPython 的[最快版本。](https://hackernoon.com/which-is-the-fastest-version-of-python-2ae7c61a6b2b)

## 那么，我该不该升级？

先说简单的答案。如果您想尝试一下您在这里看到的任何新特性，那么您确实需要能够使用 Python 3.7。使用诸如 [`pyenv`](https://github.com/pyenv/pyenv) 或 [Anaconda](https://www.anaconda.com/download/) 之类的工具，可以很容易地同时安装多个版本的 Python。安装 Python 3.7 并试用它没有什么坏处。

现在，对于更复杂的问题。您是否应该将生产环境升级到 Python 3.7？您是否应该让自己的项目依赖于 Python 3.7 来利用这些新特性？

显而易见，在升级您的生产环境之前，您应该总是进行彻底的测试，Python 3.7 中很少有东西会破坏早期的代码(虽然`async`和`await`成为关键字就是一个例子)。如果你已经在使用现代 Python，升级到 3.7 应该会相当顺利。如果你想保守一点，你可能想等待第一个维护版本的发布——Python 3 . 7 . 1——[暂定 2018 年 7 月的某个时间](https://www.python.org/dev/peps/pep-0537/#maintenance-releases)。

争论你应该只让你的项目 3.7 更难。Python 3.7 中的许多新特性要么可以作为 Python 3.6 的反向移植(数据类，`importlib.resources`)，要么很方便(更快的启动和方法调用，更容易的调试，以及`-X`选项)。后者，您可以通过自己运行 Python 3.7 来利用，同时保持代码与 Python 3.6(或更低版本)兼容。

将你的代码锁定到 Python 3.7 的主要特性是模块上的 [`__getattr__()`、类型提示中的](#customization-of-module-attributes)[前向引用](#typing-enhancements)，以及[纳秒`time`函数](#timing-precision)。如果你真的需要这些，你应该继续前进，提高你的要求。否则，如果您的项目可以在 Python 3.6 上运行一段时间，它可能会对其他人更有用。

有关升级时需要注意的详细信息，请参见[移植到 Python 3.7 指南](https://docs.python.org/3.7/whatsnew/3.7.html#porting-to-python-37)。*****
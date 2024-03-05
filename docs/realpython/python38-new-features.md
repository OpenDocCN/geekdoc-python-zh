# Python 3.8:很酷的新特性供您尝试

> 原文：<https://realpython.com/python38-new-features/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 3.8 中很酷的新特性**](/courses/cool-new-features-python-38/)

Python 最新版本发布！Python 3.8 自夏季以来一直有测试版，但在 2019 年【2019 月 14 日第一个正式版本已经准备好了。现在，我们都可以开始使用新功能，并从最新的改进中受益。

Python 3.8 带来了什么？[文档](https://docs.python.org/3.8/whatsnew/3.8.html)很好地概述了新特性。然而，本文将更深入地讨论一些最大的变化，并向您展示如何利用 Python 3.8。

在这篇文章中，你将了解到:

*   使用赋值表达式简化一些代码结构
*   在您自己的函数中强制使用仅位置参数
*   指定更精确的类型提示
*   使用 f 字符串简化调试

除了少数例外，Python 3.8 包含了许多对早期版本的小改进。在本文的结尾，您将看到许多不太引人注目的变化，以及关于使 Python 3.8 比其前身更快的一些优化的讨论。最后，你会得到一些关于升级到新版本的建议。

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 房间里的海象:赋值表达式

Python 3.8 最大的变化是引入了**赋值表达式**。它们是用一种新的符号(`:=`)写的。这种操作者通常被称为**海象操作者**，因为它像海象侧面的眼睛和长牙。

赋值表达式允许您在同一个表达式中赋值和返回值。例如，如果你想给一个[变量赋值](https://realpython.com/python-variables/)并且[打印](https://realpython.com/python-print/)它的值，那么你通常会这样做:

>>>

```py
>>> walrus = False
>>> print(walrus)
False
```

在 Python 3.8 中，您可以使用 walrus 运算符将这两个语句合并为一个:

>>>

```py
>>> print(walrus := True)
True
```

赋值表达式允许您将`True`赋值给`walrus`，并立即打印该值。但是请记住，没有它，海象运营商不会*而不是*做任何不可能的事情。它只是使某些构造更加方便，有时可以更清楚地传达代码的意图。

显示 walrus 操作符的一些优点的一个模式是 [`while`循环](https://realpython.com/python-while-loop/)，其中您需要初始化和更新一个变量。例如，下面的代码要求用户输入，直到他们键入`quit`:

```py
inputs = list()
current = input("Write something: ")
while current != "quit":
    inputs.append(current)
    current = input("Write something: ")
```

这段代码不太理想。您在重复`input()`语句，不知何故，您需要将`current`添加到列表*中，然后在*之前向用户请求。更好的解决方案是建立一个无限的`while`循环，并使用`break`来停止循环:

```py
inputs = list()
while True:
    current = input("Write something: ")
    if current == "quit":
        break
    inputs.append(current)
```

这段代码相当于上面的代码，但是避免了重复，并且以某种方式保持了更符合逻辑的顺序。如果使用赋值表达式，可以进一步简化这个循环:

```py
inputs = list()
while (current := input("Write something: ")) != "quit":
    inputs.append(current)
```

这将测试移回到`while`行，它应该在那里。然而，现在在那一行发生了几件事，所以要正确地阅读它需要更多的努力。对于 walrus 操作符何时有助于提高代码的可读性，请做出最佳判断。

PEP 572 描述了赋值表达式的所有细节，包括将它们引入语言的一些基本原理，以及如何使用 walrus 运算符的几个例子[。](https://www.python.org/dev/peps/pep-0572/#examples)

[*Remove ads*](/account/join/)

## 仅位置参数

内置函数`float()`可用于将[文本串](https://realpython.com/python-strings/)和数字转换为`float`对象。考虑下面的例子:

>>>

```py
>>> float("3.8")
3.8

>>> help(float)
class float(object)
 |  float(x=0, /) | 
 |  Convert a string or number to a floating point number, if possible.

[...]
```

仔细看`float()`的签名。注意参数后面的斜杠(`/`)。这是什么意思？

**注:**关于`/`符号的深入讨论，参见 [PEP 457 -仅位置参数符号](https://www.python.org/dev/peps/pep-0457/)。

原来，虽然`float()`的一个参数被称为`x`，但是不允许使用它的名称:

>>>

```py
>>> float(x="3.8")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: float() takes no keyword arguments
```

当使用`float()`时，你只能通过位置指定参数，而不能通过关键字。在 Python 3.8 之前，这种**仅位置的**参数只可能用于内置函数。没有简单的方法来指定参数应该是位置性的——仅在您自己的函数中:

>>>

```py
>>> def incr(x):
...     return x + 1
... 
>>> incr(3.8)
4.8

>>> incr(x=3.8)
4.8
```

使用`*args` 可以模拟的仅位置参数[，但是这不太灵活，可读性差，并且迫使您实现自己的参数解析。在 Python 3.8 中，可以使用`/`来表示它之前的所有参数都必须由位置指定。你可以重写`incr()`来只接受位置参数:](https://realpython.com/python-kwargs-and-args/)

>>>

```py
>>> def incr(x, /):
...     return x + 1
... 
>>> incr(3.8)
4.8

>>> incr(x=3.8)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: incr() got some positional-only arguments passed as
 keyword arguments: 'x'
```

通过在`x`后添加`/`，您可以指定`x`是一个只有位置的参数。通过将常规参数放在斜杠后，可以将常规参数与仅限位置的参数组合在一起:

>>>

```py
>>> def greet(name, /, greeting="Hello"):
...     return f"{greeting}, {name}"
... 
>>> greet("Łukasz")
'Hello, Łukasz'

>>> greet("Łukasz", greeting="Awesome job")
'Awesome job, Łukasz'

>>> greet(name="Łukasz", greeting="Awesome job")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: greet() got some positional-only arguments passed as
 keyword arguments: 'name'
```

在`greet()`中，斜线位于`name`和`greeting`之间。这意味着`name`是一个只有位置的参数，而`greeting`是一个可以通过位置或关键字传递的常规参数。

乍一看，只有位置的参数似乎有点限制，并且违背了 Python 关于可读性重要性的口头禅。您可能会发现，仅有位置的参数改善代码的情况并不多见。

然而，在正确的情况下，只有位置的参数可以在设计函数时给你一些灵活性。首先，当参数有自然的顺序，但是很难给它们起一个好的、描述性的名字时，只有位置的参数是有意义的。

使用仅位置参数的另一个好处是可以更容易地重构函数。特别是，您可以更改参数的名称，而不必担心其他代码依赖于这些名称。

只有位置的参数很好地补充了**只有关键字的**参数。在 Python 3 的任何版本中，都可以使用星号(`*`)指定仅关键字参数。 `*`后的任何参数*必须使用关键字指定:*

>>>

```py
>>> def to_fahrenheit(*, celsius):
...     return 32 + celsius * 9 / 5
... 
>>> to_fahrenheit(40)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: to_fahrenheit() takes 0 positional arguments but 1 was given

>>> to_fahrenheit(celsius=40)
104.0
```

`celsius`是一个只有关键字的参数，所以如果您试图在没有关键字的情况下基于位置来指定它，Python 会引发一个错误。

通过以`/`和`*`分隔的顺序指定，您可以组合仅位置、常规和仅关键字参数。在下面的示例中，`text`是仅位置参数，`border`是具有默认值的常规参数，`width`是具有默认值的仅关键字参数:

>>>

```py
>>> def headline(text, /, border="♦", *, width=50):
...     return f" {text} ".center(width, border)
...
```

因为`text`是位置唯一的，所以不能使用关键字`text`:

>>>

```py
>>> headline("Positional-only Arguments")
'♦♦♦♦♦♦♦♦♦♦♦ Positional-only Arguments ♦♦♦♦♦♦♦♦♦♦♦♦'

>>> headline(text="This doesn't work!")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: headline() got some positional-only arguments passed as
 keyword arguments: 'text'
```

另一方面，`border`既可以用关键字指定，也可以不用关键字指定:

>>>

```py
>>> headline("Python 3.8", "=")
'=================== Python 3.8 ==================='

>>> headline("Real Python", border=":")
':::::::::::::::::: Real Python :::::::::::::::::::'
```

最后，`width`必须使用关键字指定:

>>>

```
>>> headline("Python", "🐍", width=38)
'🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍 Python 🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍'

>>> headline("Python", "🐍", 38)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: headline() takes from 1 to 2 positional arguments
 but 3 were given
```py

你可以在 [PEP 570](https://www.python.org/dev/peps/pep-0570/) 中读到更多关于位置唯一参数的内容。

[*Remove ads*](/account/join/)

## 更精确的类型

Python 的类型系统在这一点上已经相当成熟了。然而，在 Python 3.8 中，一些新特性被添加到了`typing`中，以允许更精确的输入:

*   文字类型
*   打字词典
*   最终对象
*   协议

Python 支持可选的**类型提示**，通常作为代码的注释:

```
def double(number: float) -> float:
    return 2 * number
```py

在这个例子中，你说`number`应该是一个`float`，`double()`函数也应该返回一个`float`。然而，Python 将这些注释视为*提示*。它们不会在运行时强制执行:

>>>

```
>>> double(3.14)
6.28

>>> double("I'm not a float")
"I'm not a floatI'm not a float"
```py

`double()`愉快地接受`"I'm not a float"`作为参数，尽管那不是`float`。有[库可以在运行时使用类型](https://realpython.com/python-type-checking/#using-types-at-runtime)，但这不是 Python 类型系统的主要用例。

相反，类型提示允许[静态类型检查器](https://realpython.com/python-type-checking/#other-static-type-checkers)对您的 Python 代码进行类型检查，而无需实际运行您的脚本。这让人想起编译器捕捉其他语言中的类型错误，如 [Java](https://www.java.com) 、 [Rust](https://www.rust-lang.org/) 和 [Crystal](https://crystal-lang.org/) 。此外，类型提示充当代码的[文档](https://realpython.com/documenting-python-code/)，使其更容易阅读，以及[改进 IDE](https://realpython.com/python-type-checking/#pros-and-cons) 中的自动完成。

**注:**有几种静态类型的跳棋可供选择，包括 [Pyright](https://github.com/Microsoft/pyright) 、 [Pytype](https://google.github.io/pytype/) 和 [Pyre](https://pyre-check.org/) 。在本文中，您将使用 [Mypy](http://mypy-lang.org/) 。您可以使用 [`pip`](https://realpython.com/what-is-pip/) 从 [PyPI](https://pypi.org/project/mypy/) 安装 Mypy:

```
$ python -m pip install mypy
```py

在某种意义上，Mypy 是 Python 的类型检查器的参考实现，并且正在 Jukka Lehtasalo 的领导下由 Dropbox 开发。Python 的创造者吉多·范·罗苏姆是 Mypy 团队的一员。

你可以在[原始 PEP 484](https://www.python.org/dev/peps/pep-0484/) 以及 [Python 类型检查(指南)](https://realpython.com/python-type-checking/)中找到更多关于 Python 中类型提示的信息。

Python 3.8 中已经接受并包含了四个关于类型检查的新 pep。你会看到每个例子的简短例子。

[PEP 586](https://www.python.org/dev/peps/pep-0586/) 介绍一下 **[`Literal`](https://docs.python.org/3.8/library/typing.html#typing.Literal)** 型。`Literal`有点特殊，代表一个或几个特定值。`Literal`的一个用例是能够精确地添加类型，当字符串参数被用来描述特定的行为时。考虑下面的例子:

```
# draw_line.py

def draw_line(direction: str) -> None:
    if direction == "horizontal":
        ...  # Draw horizontal line

    elif direction == "vertical":
        ...  # Draw vertical line

    else:
        raise ValueError(f"invalid direction {direction!r}")

draw_line("up")
```py

程序将通过静态类型检查，即使`"up"`是一个无效的方向。类型检查器只检查`"up"`是一个字符串。在这种情况下，更准确的说法是`direction`必须是字符串`"horizontal"`或字符串`"vertical"`。使用`Literal`，您可以做到这一点:

```
# draw_line.py

from typing import Literal

def draw_line(direction: Literal["horizontal", "vertical"]) -> None:
    if direction == "horizontal":
        ...  # Draw horizontal line

    elif direction == "vertical":
        ...  # Draw vertical line

    else:
        raise ValueError(f"invalid direction {direction!r}")

draw_line("up")
```py

通过将允许的值`direction`暴露给类型检查器，您现在可以得到关于错误的警告:

```
$ mypy draw_line.py 
draw_line.py:15: error:
 Argument 1 to "draw_line" has incompatible type "Literal['up']";
 expected "Union[Literal['horizontal'], Literal['vertical']]"
Found 1 error in 1 file (checked 1 source file)
```py

基本语法是`Literal[<literal>]`。例如，`Literal[38]`表示文字值 38。您可以使用`Union`来表示几个文字值中的一个:

```
Union[Literal["horizontal"], Literal["vertical"]]
```py

由于这是一个相当常见的用例，您可以(并且可能应该)使用更简单的符号`Literal["horizontal", "vertical"]`来代替。在向`draw_line()`添加类型时，您已经使用了后者。如果仔细观察上面 Mypy 的输出，可以看到它在内部将更简单的符号翻译成了`Union`符号。

有些情况下，函数返回值的类型取决于输入参数。一个例子是`open()`，它可能根据`mode`的值返回一个文本字符串或一个字节数组。这可以通过[超载](https://mypy.readthedocs.io/en/latest/more_types.html#function-overloading)来处理。

下面的例子展示了一个计算器的框架，它可以以普通数字(`38`)或[罗马数字](http://code.activestate.com/recipes/81611-roman-numerals/) ( `XXXVIII`)的形式返回答案:

```
# calculator.py

from typing import Union

ARABIC_TO_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                   (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                   (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

def _convert_to_roman_numeral(number: int) -> str:
    """Convert number to a roman numeral string"""
    result = list()
    for arabic, roman in ARABIC_TO_ROMAN:
        count, number = divmod(number, arabic)
        result.append(roman * count)
    return "".join(result)

def add(num_1: int, num_2: int, to_roman: bool = True) -> Union[str, int]:
    """Add two numbers"""
    result = num_1 + num_2

    if to_roman:
        return _convert_to_roman_numeral(result)
    else:
        return result
```py

代码有正确的类型提示:`add()`的结果将是`str`或`int`。然而，通常调用这段代码时会使用文字`True`或`False`作为`to_roman`的值，在这种情况下，您会希望类型检查器准确推断出返回的是`str`还是`int`。这可以通过使用`Literal`和`@overload`来完成:

```
# calculator.py

from typing import Literal, overload, Union

ARABIC_TO_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                   (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                   (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

def _convert_to_roman_numeral(number: int) -> str:
    """Convert number to a roman numeral string"""
    result = list()
    for arabic, roman in ARABIC_TO_ROMAN:
        count, number = divmod(number, arabic)
        result.append(roman * count)
    return "".join(result)

@overload
def add(num_1: int, num_2: int, to_roman: Literal[True]) -> str: ...
@overload
def add(num_1: int, num_2: int, to_roman: Literal[False]) -> int: ...

def add(num_1: int, num_2: int, to_roman: bool = True) -> Union[str, int]:
    """Add two numbers"""
    result = num_1 + num_2

    if to_roman:
        return _convert_to_roman_numeral(result)
    else:
        return result
```py

添加的`@overload`签名将帮助您的类型检查器根据`to_roman`的文字值推断出`str`或`int`。注意省略号(`...`)是代码的一部分。它们代表重载签名中的函数体。

作为对`Literal`、 [PEP 591](https://www.python.org/dev/peps/pep-0591/) 的补充介绍 **[`Final`](https://docs.python.org/3.8/library/typing.html#typing.Final)** 。此限定符指定变量或属性不应被重新分配、重新定义或重写。以下是一个打字错误:

```
from typing import Final

ID: Final = 1

...

ID += 1
```py

Mypy 将突出显示行`ID += 1`，并注意到您`Cannot assign to final name "ID"`。这为您提供了一种方法来确保代码中的常量永远不会改变它们的值。

此外，还有一个可以应用于类和方法的 **[`@final`](https://docs.python.org/3.8/library/typing.html#typing.final)** 装饰器。用`@final`修饰的[类](https://realpython.com/courses/python-decorators-101/)不能被子类化，而`@final`方法不能被子类覆盖:

```
from typing import final

@final
class Base:
    ...

class Sub(Base):
    ...
```py

Mypy 将用错误消息`Cannot inherit from final class "Base"`标记这个例子。要了解更多关于`Final`和`@final`的信息，请参见 [PEP 591](https://www.python.org/dev/peps/pep-0591/) 。

第三个允许更具体类型提示的 PEP 是 [PEP 589](https://www.python.org/dev/peps/pep-0589/) ，它引入了 **[`TypedDict`](https://docs.python.org/3.8/library/typing.html#typing.TypedDict)** 。这可用于指定字典中键和值的类型，使用类似于键入的 [`NamedTuple`](https://docs.python.org/library/typing.html#typing.NamedTuple) 的符号。

传统上，词典都是用 [`Dict`](https://docs.python.org/library/typing.html#typing.Dict) 来注释的。问题是这只允许一种类型的键和一种类型的值，经常导致类似于`Dict[str, Any]`的注释。例如，考虑一个注册 Python 版本信息的字典:

```
py38 = {"version": "3.8", "release_year": 2019}
```py

`version`对应的值是一个字符串，而`release_year`是一个整数。这不能用`Dict`来精确描述。使用新的`TypedDict`，您可以执行以下操作:

```
from typing import TypedDict

class PythonVersion(TypedDict):
    version: str
    release_year: int

py38 = PythonVersion(version="3.8", release_year=2019)
```py

类型检查器将能够推断出`py38["version"]`具有类型`str`，而`py38["release_year"]`是一个`int`。在运行时，`TypedDict`是一个常规的`dict`，类型提示照常被忽略。您也可以将`TypedDict`纯粹用作注释:

```
py38: PythonVersion = {"version": "3.8", "release_year": 2019}
```py

Mypy 会让你知道你的值是否有错误的类型，或者你是否使用了一个没有声明的键。更多例子见 [PEP 589](https://www.python.org/dev/peps/pep-0589/) 。

Mypy 支持 [**协议**](https://realpython.com/python-type-checking/#duck-types-and-protocols) 已经有一段时间了。然而，[官方验收](https://mail.python.org/archives/list/typing-sig@python.org/message/FDO4KFYWYQEP3U2HVVBEBR3SXPHQSHYR/)却发生在 2019 年 5 月。

协议是一种形式化 Python 对 duck 类型支持的方式:

> 当我看到一只像鸭子一样走路、像鸭子一样游泳、像鸭子一样嘎嘎叫的鸟时，我就把那只鸟叫做鸭子。([来源](https://en.wikipedia.org/wiki/Duck_test#History))

例如，Duck typing 允许您读取任何具有`.name`属性的对象上的`.name`，而不必真正关心对象的类型。打字系统支持这一点似乎有悖常理。通过[结构分型](https://en.wikipedia.org/wiki/Structural_type_system)，还是有可能搞清楚鸭子分型的。

例如，您可以定义一个名为`Named`的协议，该协议可以识别具有`.name`属性的所有对象:

```
from typing import Protocol

class Named(Protocol):
    name: str

def greet(obj: Named) -> None:
    print(f"Hi {obj.name}")
```py

这里，`greet()`接受任何对象，只要它定义了一个`.name`属性。有关协议的更多信息，请参见 [PEP 544](https://www.python.org/dev/peps/pep-0544/) 和[Mypy 文档](https://mypy.readthedocs.io/en/latest/protocols.html)。

[*Remove ads*](/account/join/)

## 使用 f 弦进行更简单的调试

f 弦是在 Python 3.6 中引入的，并且变得非常流行。这可能是 Python 库仅在 3.6 版及更高版本中受支持的最常见原因。f 字符串是格式化的字符串文字。你可以通过主角`f`认出来:

>>>

```
>>> style = "formatted"
>>> f"This is a {style} string"
'This is a formatted string'
```py

当你使用 f 字符串时，你可以用花括号把变量甚至表达式括起来。然后，它们将在运行时被计算并包含在字符串中。一个 f 字符串中可以有多个表达式:

>>>

```
>>> import math
>>> r = 3.6

>>> f"A circle with radius {r} has area {math.pi * r * r:.2f}"
'A circle with radius 3.6 has area 40.72'
```py

在最后一个表达式`{math.pi * r * r:.2f}`中，还使用了格式说明符。格式说明符用冒号与表达式分开。

`.2f`表示该区域被格式化为具有 2 位小数的浮点数。格式说明符同 [`.format()`](https://docs.python.org/library/stdtypes.html#str.format) 。参见[官方文档](https://docs.python.org/library/string.html#format-specification-mini-language)获得允许格式说明符的完整列表。

在 Python 3.8 中，可以在 f 字符串中使用赋值表达式。只需确保用括号将赋值表达式括起来:

>>>

```
>>> import math
>>> r = 3.8

>>> f"Diameter {(diam := 2 * r)} gives circumference {math.pi * diam:.2f}"
'Diameter 7.6 gives circumference 23.88'
```py

然而，Python 3.8 中真正的新闻是新的调试说明符。您现在可以在表达式的末尾添加`=`，它将打印表达式及其值:

>>>

```
>>> python = 3.8
>>> f"{python=}"
'python=3.8'
```py

这是一个简写，通常在交互工作或添加打印语句来调试脚本时最有用。在 Python 的早期版本中，您需要两次拼出变量或表达式才能获得相同的信息:

>>>

```
>>> python = 3.7
>>> f"python={python}"
'python=3.7'
```py

您可以在`=`周围添加空格，并照常使用格式说明符:

>>>

```
>>> name = "Eric"
>>> f"{name = }"
"name = 'Eric'"

>>> f"{name = :>10}"
'name =       Eric'
```py

`>10`格式说明符指出`name`应该在 10 个字符串内右对齐。`=`也适用于更复杂的表达式:

>>>

```
>>> f"{name.upper()[::-1] = }"
"name.upper()[::-1] = 'CIRE'"
```py

有关 f 字符串的更多信息，请参见 [Python 3 的 f 字符串:改进的字符串格式化语法(指南)](https://realpython.com/python-f-strings/)。

## Python 指导委员会

从技术上来说， [Python 的**治理**](https://www.python.org/dev/peps/pep-0013/) 并不是语言特性。然而，Python 3.8 是第一个不是在**仁慈的独裁统治**和[吉多·范·罗苏姆](https://gvanrossum.github.io/)下开发的版本。Python 语言现在由五个核心开发者组成的**指导委员会**管理:

*   [巴里华沙](https://twitter.com/pumpichank)
*   布雷特·卡农
*   [卡罗尔心甘情愿](https://twitter.com/WillingCarol)
*   [圭多·范罗斯](https://twitter.com/gvanrossum)
*   尼克·科格兰

Python 的新治理模型之路是自组织中一项有趣的研究。吉多·范·罗苏姆在 20 世纪 90 年代初创造了 Python，并被亲切地称为 Python 的[**【BDFL】**](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life)**。这些年来，越来越多关于 Python 语言的决定是通过 [**Python 增强提案** (PEPs)](https://www.python.org/dev/peps/pep-0001/) 做出的。尽管如此，Guido 还是对任何新的语言特性拥有最终决定权。*

*在关于[任务表达](#the-walrus-in-the-room-assignment-expressions)的漫长讨论之后，圭多[于 2018 年 7 月宣布](https://mail.python.org/pipermail/python-committers/2018-July/005664.html)他将从 BDFL 的角色中退休(这次是真正的)。他故意没有指定继任者。相反，他要求核心开发人员团队找出 Python 今后应该如何治理。

幸运的是，PEP 流程已经很好地建立起来了，所以使用 PEP 来讨论和决定新的治理模型是很自然的。在 2018 年秋季，[提出了几种模式](https://www.python.org/dev/peps/pep-8000/)，包括[选举新的 BDFL](https://www.python.org/dev/peps/pep-8010/) (更名为亲切的裁判影响决策官:圭多)，或者转向基于共识和投票的[社区模式](https://www.python.org/dev/peps/pep-8012/)，没有集中的领导。2018 年 12 月，[指导委员会型号](https://www.python.org/dev/peps/pep-8016/)在核心开发者中投票选出。

[![The Python Steering Council at PyCon 2019](img/d7c233a4afe1c85ed9cf90a885c7cbf6.png)](https://files.realpython.com/media/steering_council.1aae31a91dad.jpg)

<figcaption class="figure-caption text-center">The Python Steering Council at PyCon 2019\. From left to right: Barry Warsaw, Brett Cannon, Carol Willing, Guido van Rossum, and Nick Coghlan (Image: Geir Arne Hjelle)</figcaption>

指导委员会由 Python 社区的五名成员组成，如上所列。在 Python 的每一个主要版本发布后，都会选举一个新的指导委员会。换句话说，Python 3.8 发布后会有一次选举。

虽然这是一次公开选举，但预计首届指导委员会的大部分成员(如果不是全部的话)将会改选。指导委员会拥有广泛的权力来决定 Python 语言，但是应该尽可能少的行使这些权力。

你可以在 [PEP 13](https://www.python.org/dev/peps/pep-0013/) 中阅读关于新治理模式的所有信息，而决定新模式的过程在 [PEP 8000](https://www.python.org/dev/peps/pep-8000/) 中描述。欲了解更多信息，请参见 [PyCon 2019 主题演讲](https://pyvideo.org/pycon-us-2019/python-steering-council-keynote-pycon-2019.html)，并聆听 Brett Cannon 在[与我谈论 Python](https://talkpython.fm/episodes/show/209/inside-python-s-new-governance-model)和[Changelog 播客](https://changelog.com/podcast/348)上的演讲。你可以在 [GitHub](https://github.com/python/steering-council) 上关注指导委员会的更新。

[*Remove ads*](/account/join/)

## 其他非常酷的功能

到目前为止，您已经看到了关于 Python 3.8 新特性的头条新闻。然而，还有许多其他的变化也很酷。在本节中，您将快速浏览其中一些。

### `importlib.metadata`

Python 3.8 中的标准库中新增了一个模块: [`importlib.metadata`](https://importlib-metadata.readthedocs.io) 。通过此模块，您可以访问 Python 安装中已安装包的相关信息。与它的同伴模块[`importlib.resources`](https://realpython.com/python37-new-features/#importing-data-files-with-importlibresources)`importlib.metadata`一起，改进了老款 [`pkg_resources`](https://setuptools.readthedocs.io/en/latest/pkg_resources.html) 的功能。

举个例子，你可以得到一些关于 [`pip`](https://realpython.com/courses/what-is-pip/) 的信息:

>>>

```
>>> from importlib import metadata
>>> metadata.version("pip")
'19.2.3'

>>> pip_metadata = metadata.metadata("pip")
>>> list(pip_metadata)
['Metadata-Version', 'Name', 'Version', 'Summary', 'Home-page', 'Author',
 'Author-email', 'License', 'Keywords', 'Platform', 'Classifier',
 'Classifier', 'Classifier', 'Classifier', 'Classifier', 'Classifier',
 'Classifier', 'Classifier', 'Classifier', 'Classifier', 'Classifier',
 'Classifier', 'Classifier', 'Requires-Python']

>>> pip_metadata["Home-page"]
'https://pip.pypa.io/'

>>> pip_metadata["Requires-Python"]
'>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*'

>>> len(metadata.files("pip"))
668
```py

目前安装的`pip`版本是 19.2.3。`metadata()`提供您在 [PyPI](https://pypi.org/project/pip/) 上看到的大部分信息。例如，你可以看到这个版本的`pip`需要 Python 2.7，或者 Python 3.5 或更高版本。使用`files()`，您将获得组成`pip`包的所有文件的列表。在这种情况下，有将近 700 个文件。

`files()`返回一个 [`Path`](https://realpython.com/python-pathlib/) 对象的[列表](https://realpython.com/python-lists-tuples/)。这些给了你一个方便的方法来查看一个包的源代码，使用`read_text()`。以下示例从 [`realpython-reader`](https://pypi.org/project/realpython-reader/) 包中打印出`__init__.py`:

>>>

```
>>> [p for p in metadata.files("realpython-reader") if p.suffix == ".py"]
[PackagePath('reader/__init__.py'), PackagePath('reader/__main__.py'),
 PackagePath('reader/feed.py'), PackagePath('reader/viewer.py')]

>>> init_path = _[0]  # Underscore access last returned value in the REPL
>>> print(init_path.read_text()) """Real Python feed reader

Import the `feed` module to work with the Real Python feed:

 >>> from reader import feed
 >>> feed.get_titles()
 ['Logging in Python', 'The Best Python Books', ...]

See https://github.com/realpython/reader/ for more information
"""

# Version of realpython-reader package
__version__ = "1.0.0"

...
```py

您还可以访问软件包相关性:

>>>

```
>>> metadata.requires("realpython-reader")
['feedparser', 'html2text', 'importlib-resources', 'typing']
```py

列出一个包的依赖关系。您可以看到，`realpython-reader`在后台使用 [`feedparser`](https://pypi.org/project/feedparser/) 来读取和解析文章提要。

PyPI 上有一个对早期版本 Python 有效的`importlib.metadata` [的反向移植。您可以使用`pip`安装它:](https://pypi.org/project/importlib-metadata/)

```
$ python -m pip install importlib-metadata
```py

您可以在代码中使用 PyPI 反向端口，如下所示:

```
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

...
```py

有关`importlib.metadata`的更多信息，请参见[文档](https://importlib-metadata.readthedocs.io)

### 新的和改进的`math`和`statistics`功能

Python 3.8 对现有的标准库包和模块进行了许多改进。`math`在标准库中有一些新的功能。`math.prod()`的工作方式与内置的`sum()`类似，但对于乘法运算:

>>>

```
>>> import math
>>> math.prod((2, 8, 7, 7))
784

>>> 2 * 8 * 7 * 7
784
```py

这两种说法是等价的。当你已经将因子存储在一个 iterable 中时，将会更容易使用。

另一个新功能是`math.isqrt()`。可以用`isqrt()`求[平方根](https://realpython.com/python-square-root-function/)的整数部分:

>>>

```
>>> import math
>>> math.isqrt(9)
3

>>> math.sqrt(9)
3.0

>>> math.isqrt(15)
3

>>> math.sqrt(15)
3.872983346207417
```py

9 的平方根是 3。可以看到`isqrt()`返回一个整数结果，而 [`math.sqrt()`](https://realpython.com/python-square-root-function/) 总是返回一个`float`。15 的平方根差不多是 3.9。请注意，`isqrt()` [将答案截断到下一个整数](https://realpython.com/python-rounding/#truncation)，在本例中为 3。

最后，你现在可以更容易地使用标准库中的 *n* 维点和向量。用`math.dist()`可以求出两点之间的距离，用`math.hypot()`可以求出一个矢量的长度:

>>>

```
>>> import math
>>> point_1 = (16, 25, 20)
>>> point_2 = (8, 15, 14)

>>> math.dist(point_1, point_2)
14.142135623730951

>>> math.hypot(*point_1)
35.79106033634656

>>> math.hypot(*point_2)
22.02271554554524
```py

这使得使用标准库处理点和向量变得更加容易。然而，如果你要对点或向量做很多计算，你应该检查一下 [NumPy](https://realpython.com/numpy-array-programming/) 。

`statistics`模块还有几个新功能:

*   [`statistics.fmean()`](https://docs.python.org/3.8/library/statistics.html#statistics.fmean) 计算`float`数字的平均值。
*   [`statistics.geometric_mean()`](https://docs.python.org/3.8/library/statistics.html#statistics.geometric_mean) 计算`float`个数字的几何平均值。
*   [`statistics.multimode()`](https://docs.python.org/3.8/library/statistics.html#statistics.multimode) 查找序列中出现频率最高的值。
*   [`statistics.quantiles()`](https://docs.python.org/3.8/library/statistics.html#statistics.quantiles) 计算分割点，将数据等概率分割成 *n 个*连续区间。

以下示例显示了正在使用的函数:

>>>

```
>>> import statistics
>>> data = [9, 3, 2, 1, 1, 2, 7, 9]
>>> statistics.fmean(data)
4.25

>>> statistics.geometric_mean(data)
3.013668912157617

>>> statistics.multimode(data)
[9, 2, 1]

>>> statistics.quantiles(data, n=4)
[1.25, 2.5, 8.5]
```py

在 Python 3.8 中，有一个新的 [`statistics.NormalDist`](https://docs.python.org/3.8/library/statistics.html#statistics.NormalDist) 类，使得[使用高斯正态分布](https://docs.python.org/3.8/library/statistics.html#normaldist-examples-and-recipes)更加方便。

要看使用`NormalDist`的例子，可以试着比较一下新`statistics.fmean()`和传统`statistics.mean()`的速度:

>>>

```
>>> import random
>>> import statistics
>>> from timeit import timeit

>>> # Create 10,000 random numbers
>>> data = [random.random() for _ in range(10_000)]

>>> # Measure the time it takes to run mean() and fmean()
>>> t_mean = [timeit("statistics.mean(data)", number=100, globals=globals())
...           for _ in range(30)]
>>> t_fmean = [timeit("statistics.fmean(data)", number=100, globals=globals())
...            for _ in range(30)]

>>> # Create NormalDist objects based on the sampled timings
>>> n_mean = statistics.NormalDist.from_samples(t_mean)
>>> n_fmean = statistics.NormalDist.from_samples(t_fmean)

>>> # Look at sample mean and standard deviation
>>> n_mean.mean, n_mean.stdev
(0.825690647733245, 0.07788573997674526)

>>> n_fmean.mean, n_fmean.stdev
(0.010488564966666065, 0.0008572332785645231)

>>> # Calculate the lower 1 percentile of mean
>>> n_mean.quantiles(n=100)[0]
0.6445013221202459
```py

在这个例子中，您使用 [`timeit`](https://docs.python.org/library/timeit.html) 来测量`mean()`和`fmean()`的执行时间。为了获得可靠的结果，您让`timeit`执行每个函数 100 次，并为每个函数收集 30 个这样的时间样本。基于这些样本，你创建两个`NormalDist`对象。注意，如果您自己运行代码，可能需要一分钟来收集不同的时间样本。

`NormalDist`有很多方便的属性和方法。完整列表见[文档](https://docs.python.org/3.8/library/statistics.html#normaldist-objects)。考察`.mean`和`.stdev`，你看到老款`statistics.mean()`跑 0.826±0.078 秒，新款`statistics.fmean()`花 0.0105±0.0009 秒。换句话说，`fmean()`对于这些数据来说大约快了 80 倍。

如果您需要 Python 中比标准库提供的更高级的统计，请查看 [`statsmodels`](https://www.statsmodels.org/) 和 [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html) 。

[*Remove ads*](/account/join/)

### 关于危险语法的警告

Python 有一个 [`SyntaxWarning`](https://docs.python.org/3/library/exceptions.html#SyntaxWarning) ，它可以警告可疑的语法，这通常不是一个 [`SyntaxError`](https://realpython.com/invalid-syntax-python/) 。Python 3.8 增加了一些新功能，可以在编码和调试过程中帮助你。

`is`和`==`的区别可能会让人混淆。后者检查值是否相等，而只有当对象相同时，`is`才是`True`。Python 3.8 将试图警告你应该使用`==`而不是`is`的情况:

>>>

```
>>> # Python 3.7
>>> version = "3.7"
>>> version is "3.7"
False

>>> # Python 3.8
>>> version = "3.8"
>>> version is "3.8"
<stdin>:1: SyntaxWarning: "is" with a literal. Did you mean "=="? False

>>> version == "3.8"
True
```py

当你写一个很长的列表时，很容易漏掉一个逗号，尤其是当它是垂直格式的时候。忘记元组列表中的逗号会给出一个混乱的错误消息，说明元组不可调用。Python 3.8 还发出了一个警告，指出了真正的问题:

>>>

```
>>> [
...   (1, 3)
...   (2, 4)
... ]
<stdin>:2: SyntaxWarning: 'tuple' object is not callable; perhaps
 you missed a comma? Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: 'tuple' object is not callable
```py

该警告正确地将丢失的逗号识别为真正的原因。

### 优化

Python 3.8 进行了多项优化。一些能让代码运行得更快。其他的可以减少内存占用。例如，与 Python 3.7 相比，Python 3.8 在 [`namedtuple`](https://realpython.com/python-namedtuple/) 中查找字段要快得多:

>>>

```
>>> import collections
>>> from timeit import timeit
>>> Person = collections.namedtuple("Person", "name twitter")
>>> raymond = Person("Raymond", "@raymondh")

>>> # Python 3.7
>>> timeit("raymond.twitter", globals=globals())
0.05876131607996285

>>> # Python 3.8
>>> timeit("raymond.twitter", globals=globals())
0.0377705999400132
```py

你可以看到在 Python 3.8 中，在`namedtuple`上查找`.twitter`要快 30-40%。当列表从已知长度的 iterables 初始化时，可以节省一些空间。这可以节省内存:

>>>

```
>>> import sys

>>> # Python 3.7
>>> sys.getsizeof(list(range(20191014)))
181719232

>>> # Python 3.8
>>> sys.getsizeof(list(range(20191014)))
161528168
```

在这种情况下，Python 3.8 中的列表使用的内存比 Python 3.7 少 11%。

其他优化包括 [`subprocess`](https://docs.python.org/library/subprocess.html) 更好的性能、 [`shutil`](https://docs.python.org/library/shutil.html) 更快的文件复制、 [`pickle`](https://realpython.com/python-pickle-module/) 更好的默认性能、更快的 [`operator.itemgetter`](https://docs.python.org/library/operator.html#operator.itemgetter) 操作。有关优化的完整列表，请参见[官方文档](https://docs.python.org/3.8/whatsnew/3.8.html#optimizations)。

## 那么，应该升级到 Python 3.8 吗？

先说简单的答案。如果您想尝试这里看到的任何新特性，那么您确实需要能够使用 Python 3.8。像 [`pyenv`](https://realpython.com/intro-to-pyenv/) 和 [Anaconda](https://realpython.com/python-windows-machine-learning-setup/#introducing-anaconda-and-conda) 这样的工具使得并排安装几个版本的 Python 变得很容易。或者，可以运行[官方 Python 3.8 Docker 容器](https://hub.docker.com/_/python/)。亲自尝试 Python 3.8 没有任何坏处。

现在，对于更复杂的问题。您是否应该将生产环境升级到 Python 3.8？您是否应该让自己的项目依赖于 Python 3.8 来利用这些新特性？

在 Python 3.8 中运行 Python 3.7 代码应该没什么问题。因此，升级您的环境以运行 Python 3.8 是非常安全的，并且您将能够利用新版本中的[优化](#optimizations)。Python 3.8 的不同测试版本已经发布了好几个月了，所以希望大多数错误已经被解决了。然而，如果你想保守一点，你可以坚持到第一个维护版本(Python 3.8.1)发布。

一旦您升级了您的环境，您就可以开始尝试 Python 3.8 中才有的特性，比如[赋值表达式](#the-walrus-in-the-room-assignment-expressions)和[仅位置参数](#positional-only-arguments)。但是，您应该注意其他人是否依赖您的代码，因为这将迫使他们也升级他们的环境。流行的库可能会在相当长的一段时间内至少支持 Python 3.6。

有关为 Python 3.8 准备代码的更多信息，请参见[移植到 Python 3.8](https://docs.python.org/3.8/whatsnew/3.8.html#porting-to-python-3-8) 。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 3.8 中很酷的新特性**](/courses/cool-new-features-python-38/)*********
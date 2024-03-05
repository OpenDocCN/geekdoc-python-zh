# Python 3.11:很酷的新特性供您尝试

> 原文：<https://realpython.com/python311-new-features/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 3.11 中很酷的新特性**](/courses/new-features-python-311/)

[Python 3.11](https://www.python.org/downloads/release/python-3110/) 发布于[2022 年 10 月 24 日](https://peps.python.org/pep-0664/)。Python 的这个最新版本更快，也更用户友好。经过 17 个月的开发，它现在已经可以投入使用了。

和每个版本一样，Python 3.11 也有很多改进和变化。你可以在[文档](https://docs.python.org/3.11/whatsnew/3.11.html)中看到它们的列表。在这里，您将探索最酷、最具影响力的新功能。

**在本教程中，您将了解到新的特性和改进，例如:**

*   更好的**错误消息**,提供更多信息的回溯
*   **更快的代码执行**得益于在**更快的 CPython** 项目中付出的巨大努力
*   简化异步代码工作的任务和异常组
*   几个新的类型特性改进了 Python 的静态类型支持
*   本地的 TOML 支持使用配置文件

如果你想尝试本教程中的任何例子，那么你需要使用 Python 3.11。Python 3 的[安装&安装指南](https://realpython.com/installing-python/)和[如何安装 Python 的预发布版本？](https://realpython.com/python-pre-release/)向您介绍向系统添加新版本 Python 的几个选项。

除了了解更多关于该语言的新特性，您还将获得一些关于升级到新版本之前需要考虑的事项的建议。单击以下链接下载演示 Python 3.11 新功能的代码示例:

**免费下载:** [点击这里下载免费的示例代码](https://realpython.com/bonus/python-311-examples/)，它展示了 Python 3.11 的一些新特性。

## 更多信息的错误回溯

Python 通常被认为是很好的初学者编程语言，它的[可读](https://realpython.com/python-pep8/)语法和强大的[数据结构](https://realpython.com/python-data-structures/)。对所有人来说，尤其是对 Python 新手来说，一个挑战是如何解释 Python 遇到错误时显示的[回溯](https://realpython.com/python-traceback/)。

在 [Python 3.10](https://realpython.com/python310-new-features/) 中，Python 的错误消息被[大幅改进](https://realpython.com/python310-new-features/#better-error-messages)。同样，Python 3.11 的[最令人期待的特性之一](https://twitter.com/pyblogsal/status/1569366992758398977)也将提升你的开发者体验。装饰性注释被添加到回溯中，可以帮助您更快地解释错误信息。

要查看增强回溯的快速示例，请将以下代码添加到名为`inverse.py`的文件中:

```py
# inverse.py

def inverse(number):
    return 1 / number

print(inverse(0))
```

你可以用`inverse()`来计算一个数的[乘逆](https://en.wikipedia.org/wiki/Multiplicative_inverse)。`0`没有乘法逆运算，所以当您运行它时，您的代码会产生一个错误:

```py
$ python inverse.py
Traceback (most recent call last):
 File "/home/realpython/inverse.py", line 6, in <module>
 print(inverse(0))
 ^^^^^^^^^^ File "/home/realpython/inverse.py", line 4, in inverse
 return 1 / number
 ~~^~~~~~~~ ZeroDivisionError: division by zero
```

注意回溯中嵌入的`^`和`~`符号。它们用于引导您注意导致错误的代码。像通常的回溯一样，你应该从底层开始，一步步向上。在这个例子中，`ZeroDivisionError`是由分割`1 / number`引起的。真正的罪魁祸首是召唤`inverse(0)`，因为`0`没有逆。

在发现错误方面获得这种额外的帮助是有用的。然而，如果您的代码更复杂，带注释的回溯甚至更强大。它们也许能够传达你以前无法从回溯本身获得的信息。

为了体会改进的回溯的威力，您将构建一个关于几个程序员的信息的小型解析器。假设您有一个名为`programmers.json`的文件，其内容如下:

```py
[ {"name":  {"first":  "Uncle Barry"}}, { "name":  {"first":  "Ada",  "last":  "Lovelace"}, "birth":  {"year":  1815}, "death":  {"month":  11,  "day":  27} }, { "name":  {"first":  "Grace",  "last":  "Hopper"}, "birth":  {"year":  1906,  "month":  12,  "day":  9}, "death":  {"year":  1992,  "month":  1,  "day":  1} }, { "name":  {"first":  "Ole-Johan",  "last":  "Dahl"}, "birth":  {"year":  1931,  "month":  10,  "day":  12}, "death":  {"year":  2002,  "month":  6,  "day":  29} }, { "name":  {"first":  "Guido",  "last":  "Van Rossum"}, "birth":  {"year":  1956,  "month":  1,  "day":  31}, "death":  null } ]
```

注意，关于程序员的信息是相当不一致的。虽然关于[格蕾丝·赫柏](https://en.wikipedia.org/wiki/Grace_Hopper)和[奥利·约翰·达尔](https://en.wikipedia.org/wiki/Ole-Johan_Dahl)的信息是完整的，但是你遗漏了[阿达·洛芙莱斯的](https://en.wikipedia.org/wiki/Ada_Lovelace)出生日期和月份以及她的死亡年份。自然，你只有关于吉多·范·罗苏姆的出生信息。更重要的是，你只记录了巴里叔叔的名字。

您将创建一个可以包装这些信息的类。首先从 JSON 文件中读取信息:

```py
# programmers.py

import json
import pathlib

programmers = json.loads(
    pathlib.Path("programmers.json").read_text(encoding="utf-8")
)
```

您使用 [`pathlib`](https://realpython.com/python-pathlib/) 读取 JSON 文件，使用 [`json`](https://realpython.com/python-json/) 将信息解析到一个 Python 字典列表中。

接下来，您将使用一个[数据类](https://realpython.com/python-data-classes/)来封装关于每个程序员的信息:

```py
# programmers.py

from dataclasses import dataclass

# ...

@dataclass
class Person:
    name: str
    life_span: tuple[int, int]

    @classmethod
    def from_dict(cls, info):
        return cls(
            name=f"{info['name']['first']}  {info['name']['last']}",
            life_span=(info["birth"]["year"], info["death"]["year"]),
        )
```

每个`Person`将有一个`name`和一个`life_span`属性。此外，您添加了一个方便的[构造函数](https://realpython.com/python-multiple-constructors/)，它可以根据 JSON 文件中的信息和结构初始化`Person`。

您还将添加一个可以一次性初始化两个`Person`对象的函数:

```py
# programmers.py

# ...

def convert_pair(first, second):
    return Person.from_dict(first), Person.from_dict(second)
```

`convert_pair()`函数两次使用`.from_dict()`构造函数将一对程序员从 JSON 结构转换成`Person`对象。

是时候探索您的代码了，尤其是看看一些回溯。运行带有`-i`标志的程序，打开 Python 的交互式 REPL，其中包含所有可用的变量、类和函数:

```py
$ python -i programmers.py
>>> Person.from_dict(programmers[2]) Person(name='Grace Hopper', life_span=(1906, 1992))
```

Grace 的信息是完整的，因此您可以将她的全名和寿命信息封装到一个`Person`对象中。

要查看新的回溯功能，请尝试转换 Barry 叔叔:

>>>

```py
>>> programmers[0]
{'name': {'first': 'Uncle Barry'}}

>>> Person.from_dict(programmers[0])
Traceback (most recent call last):
  File "/home/realpython/programmers.py", line 17, in from_dict
    name=f"{info['name']['first']}  {info['name']['last']}",
                                    ~~~~~~~~~~~~^^^^^^^^
KeyError: 'last'
```

你得到一个 [`KeyError`](https://realpython.com/python-keyerror/) ，因为`last`不见了。虽然您可能记得`last`是`name`中的一个子字段，但是注释立即为您指出了这一点。

类似地，回想一下关于 Ada 的寿命信息是不完整的。您不能为她创建`Person`对象:

>>>

```py
>>> programmers[1]
{
 'name': {'first': 'Ada', 'last': 'Lovelace'},
 'birth': {'year': 1815},
 'death': {'month': 11, 'day': 27}
}

>>> Person.from_dict(programmers[1])
Traceback (most recent call last):
  File "/home/realpython/programmers.py", line 18, in from_dict
    life_span=(info["birth"]["year"], info["death"]["year"]),
                                      ~~~~~~~~~~~~~^^^^^^^^
KeyError: 'year'
```

您将获得另一个`KeyError`，这次是因为`year`丢失了。在这种情况下，回溯比前面的例子更加有用。您有两个`year`子字段，一个用于`birth`，一个用于`death`。回溯注释会立即显示您缺少了死亡年份。

圭多怎么样了？你只有关于他出生的信息:

>>>

```py
>>> programmers[4]
{
 'name': {'first': 'Guido', 'last': 'Van Rossum'},
 'birth': {'year': 1956, 'month': 1, 'day': 31},
 'death': None
}

>>> Person.from_dict(programmers[4])
Traceback (most recent call last):
  File "/home/realpython/programmers.py", line 18, in from_dict
    life_span=(info["birth"]["year"], info["death"]["year"]),
                                      ~~~~~~~~~~~~~^^^^^^^^
TypeError: 'NoneType' object is not subscriptable
```

在这种情况下，会产生一个`TypeError`。您可能以前见过这种`'NoneType'`类型的错误。众所周知，它们很难调试，因为不清楚哪个对象是意外的`None`。但是，从注释中，你会看到这个例子中的`info["death"]`是 [`None`](https://realpython.com/null-in-python/) 。

在最后一个例子中，您将探索嵌套函数调用会发生什么。记住`convert_pair()`调用`Person.from_dict()`两次。现在，尝试将 Ada 和 Ole-Johan 配对:

>>>

```py
>>> convert_pair(programmers[3], programmers[1])
Traceback (most recent call last):
  File "/home/realpython/programmers.py", line 24, in convert_pair
 return Person.from_dict(first), Person.from_dict(second)                                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/realpython/programmers.py", line 18, in from_dict
    life_span=(info["birth"]["year"], info["death"]["year"]),
                                      ~~~~~~~~~~~~~^^^^^^^^
KeyError: 'year'
```

试图封装 Ada 引发了与前面相同的`KeyError`。但是，请注意来自`convert_pair()`内部的回溯。因为该函数调用了`.from_dict()`两次，所以通常需要花费一些精力来判断在处理`first`或`second`时是否出现了错误。在 Python 的最新版本中，您会立即发现这些问题是由`second`引起的。

这些回溯使得 Python 3.11 中的调试比早期版本更容易。你可以在 Python 3.11 预览教程[中看到更多的例子，更多关于如何实现回溯的信息，以及其他可以在调试中使用的工具，甚至更好的错误消息](https://realpython.com/python311-error-messages/)。更多技术细节，请看 [PEP 657](https://peps.python.org/pep-0657/) 。

作为 Python 开发人员，带注释的回溯将有助于提高您的工作效率。另一个令人兴奋的发展是 Python 3.11 是迄今为止最快的 Python 版本。

[*Remove ads*](/account/join/)

## 更快的代码执行

Python 以缓慢的语言著称。例如，Python 中的常规循环比 c 中的类似循环要慢几个数量级。通常程序员的生产力比代码执行时间更重要。

Python 也非常能够包装用更快的语言编写的库。例如，在 NumPy 中完成的计算比在纯 Python 中完成的类似计算要快得多。与开发代码的便利性相匹配，这使得 Python 成为数据科学领域的有力竞争者。

尽管如此，还是有一股力量推动核心 Python 语言变得更快。2020 年秋天，Mark Shannon 提出了一些可以在 Python 中实现的性能改进。被称为**香农计划**的[提案](https://github.com/markshannon/faster-cpython/blob/master/plan.md)非常雄心勃勃，希望在几个版本中将 Python 速度提高五倍。

微软已经加入进来，目前正在支持一组开发人员——包括马克·香农和 Python 的创造者吉多·范·罗苏姆——从事现在所知的**更快的 CPython** 项目。基于更快的 CPython 项目，Python 3.11 有很多改进。在这一节中，您将了解到**专用自适应解释器**。在后面的章节中，您还将了解到[更快的启动时间](#faster-startup)和[零成本异常](#zero-cost-exceptions)。

描述了一个专门化的自适应解释器。主要思想是通过优化经常执行的操作来加速代码的运行。这类似于[即时](https://en.wikipedia.org/wiki/Just-in-time_compilation) (JIT)编译，除了它不影响编译。相反，Python 的字节码是动态适应或改变的。

**注意:** Python 代码在运行前被[编译](https://devguide.python.org/compiler/)成**字节码**。[字节码](https://en.wikipedia.org/wiki/Bytecode)由比常规 Python 代码更多的基本指令组成，所以 Python 的每一行都被转换成几个字节码语句。

你可以用 [`dis`](https://docs.python.org/3.11/library/dis.html) 来看看 Python 的字节码。例如，考虑一个可以从英尺转换为米的函数:

>>>

```py
 1>>> def feet_to_meters(feet):
 2...     return 0.3048 * feet
 3...
```

您可以通过调用`dis.dis()`将这个函数反汇编成字节码:

>>>

```py
>>> import dis
>>> dis.dis(feet_to_meters)
 1           0 RESUME                   0

 2           2 LOAD_CONST               1 (0.3048)
 4 LOAD_FAST                0 (feet)
 6 BINARY_OP                5 (*)
 10 RETURN_VALUE
```

每行显示一个字节码指令的信息。这五列分别是**行号**、**字节地址**、**操作码名称**、**操作参数**以及圆括号内参数的**解释。**

一般来说，写 Python 不需要了解字节码。不过，它可以帮助您理解 Python 内部是如何工作的。

字节码生成中增加了一个叫做**加快**的新步骤。这需要在运行时优化指令，并用**自适应**指令替换它们。每一条这样的指令都会考虑如何使用它，并且**可能会相应地对**进行特殊化。

一旦一个函数被调用了一定的次数，加速就开始了。在 CPython 3.11 中，这发生在八次调用之后。您可以通过调用`dis()`并设置`adaptive`参数来观察解释器如何适应字节码。首先定义一个函数，用浮点数作为参数调用它七次:

>>>

```py
>>> def feet_to_meters(feet):
...     return 0.3048 * feet
...

>>> feet_to_meters(1.1)
0.33528
>>> feet_to_meters(2.2)
0.67056
>>> feet_to_meters(3.3)
1.00584
>>> feet_to_meters(4.4)
1.34112
>>> feet_to_meters(5.5)
1.6764000000000001
>>> feet_to_meters(6.6)
2.01168
>>> feet_to_meters(7.7)
2.34696
```

接下来，看看`feet_to_meters()`的字节码:

>>>

```py
>>> import dis
>>> dis.dis(feet_to_meters, adaptive=True)
 1           0 RESUME                   0

 2           2 LOAD_CONST               1 (0.3048)
 4 LOAD_FAST                0 (feet)
 6 BINARY_OP                5 (*)
 10 RETURN_VALUE
```

你还不会观察到什么特别的东西。字节码的这个版本仍然与非自适应版本相同。当您第八次调用`feet_to_meters()`时，情况会发生变化:

>>>

```py
>>> feet_to_meters(8.8)
2.68224

>>> dis.dis(feet_to_meters, adaptive=True)
 1           0 RESUME_QUICK                 0 
 2           2 LOAD_CONST__LOAD_FAST        1 (0.3048) 4 LOAD_FAST                    0 (feet)
 6 BINARY_OP_MULTIPLY_FLOAT     5 (*) 10 RETURN_VALUE
```

现在，原来的几个说明已经被专门的取代了。例如，`BINARY_OP`已经被专门化为`BINARY_OP_MULTIPLY_FLOAT`，它在两个`float`数相乘时更快。

即使`feet_to_meters()`已经针对`feet`是一个`float`参数的情况进行了优化，它仍然通过退回到原始的字节码指令来正常工作于其他类型的参数。内部操作发生了变化，但是您的代码将与以前完全一样。

专用指令仍然是自适应的。再调用你的函数 52 次，但是现在用一个整数参数:

>>>

```py
>>> for feet in range(52):
...     feet_to_meters(feet)
...

>>> dis.dis(feet_to_meters, adaptive=True)
 1           0 RESUME_QUICK                 0

 2           2 LOAD_CONST__LOAD_FAST        1 (0.3048)
 4 LOAD_FAST                    0 (feet)
 6 BINARY_OP_MULTIPLY_FLOAT     5 (*) 10 RETURN_VALUE
```

Python 解释器仍然希望能够将两个`float`数字相乘。当您用整数再次调用`feet_to_meters()`时，它会重新提交并转换回一个非专门化的自适应指令:

>>>

```py
>>> feet_to_meters(52)
15.8496

>>> dis.dis(feet_to_meters, adaptive=True)
 1           0 RESUME_QUICK              0

 2           2 LOAD_CONST__LOAD_FAST     1 (0.3048)
 4 LOAD_FAST                 0 (feet)
 6 BINARY_OP_ADAPTIVE        5 (*) 10 RETURN_VALUE
```

在这种情况下，字节码指令被改为`BINARY_OP_ADAPTIVE`而不是`BINARY_OP_MULTIPLY_INT`，因为其中一个操作符`0.3048`总是浮点数。

整数和浮点数之间的乘法比同类型数字之间的乘法更难优化。至少目前没有专门的指令来做`float`和`int`之间的乘法。

这个例子旨在让您对自适应专门化解释器的工作原理有所了解。一般来说，您不应该担心更改现有代码来利用它。您的大部分代码实际上会运行得更快。

也就是说，在一些情况下，您可以重构您的代码，以便更有效地进行专门化。Brandt Bucher 的 [`specialist`](https://pypi.org/project/specialist/) 是一个可视化解释器如何处理你的代码的工具。教程展示了一个手工改进代码的例子。您可以在[与我谈论 Python 播客](https://talkpython.fm/episodes/show/381/python-perf-specializing-adaptive-interpreter)上了解更多信息。

更快的 CPython 项目的几个重要准则是:

*   该项目不会对 Python 引入任何突破性的改变。
*   大多数代码的性能都应该得到提高。

在基准测试中，“CPython 3.11 平均比 CPython 3.10 快 25%”([来源](https://docs.python.org/3.11/whatsnew/3.11.html#faster-cpython))。然而，你应该更感兴趣的是 Python 3.11 在你的代码上的*表现，而不是它在基准测试中的表现。展开下面的方框，了解如何衡量自己代码的性能:*



一般来说，有三种方法可以用来衡量代码性能:

1.  对程序中重要的小段代码进行基准测试。
2.  剖析你的程序，找出可以改进的瓶颈。
3.  监控整个程序的性能。

通常情况下，您希望完成所有这些任务。**基准**可以在你开发代码的时候帮助你[在不同的实现之间选择](https://realpython.com/sort-python-dictionary/#considering-strategic-and-performance-issues)。Python 内置了对带有 [`timeit`](https://docs.python.org/3/library/timeit.html) 模块的微基准测试的支持。第三方的 [`richbench`](https://pypi.org/project/richbench/) 工具对于基准函数来说是不错的。此外， [`pyperformance`](https://pyperformance.readthedocs.io/) 是更快的 CPython 项目用来衡量改进的基准套件。

如果你需要加速你的程序，并且想弄清楚[应该关注哪部分代码](https://realpython.com/python-timer/#finding-bottlenecks-in-your-code-with-profilers)，那么**代码分析器**会很有用。Python 的标准库提供了 [`cProfile`](https://docs.python.org/3.10/library/profile.html#module-cProfile) ，您可以用它来收集关于您的程序的统计数据，还提供了 [`pstats`](https://docs.python.org/3.10/library/profile.html#the-stats-class) ，您可以用它来研究那些统计数据。

第三种方法，**监控**你的程序的运行时间，这是你应该对所有运行超过几秒钟的程序做的事情。最简单的方法是在您的日志消息中添加一个[计时器](https://realpython.com/python-timer/)。第三方的 [`codetiming`](https://pypi.org/project/codetiming/) 允许你这样做，例如通过给你的主函数添加一个[装饰器](https://realpython.com/primer-on-python-decorators/)。

让 Python 变得更快的一个可行且重要的方法是分享举例说明您的用例的基准。特别是如果你没有注意到 Python 3.11 中的加速，如果你能够分享你的代码，这对核心开发者来说是有帮助的。有关更多信息，请参见 Mark Shannon 的闪电演讲[如何帮助提升 Python 的速度](https://www.youtube.com/watch?v=xQ0-aSmn9ZA&t=1189s)。

更快的 CPython 项目是一项正在进行的工作，已经有几项优化计划在 2023 年 10 月发布的 [Python 3.12](https://docs.python.org/3.12/whatsnew/3.12.html) 中发布。你可以在 [GitHub](https://github.com/faster-cpython/ideas) 上关注这个项目。要了解更多信息，您还可以查看以下讨论和演示:

*   跟我说说 Python:[用 Guido 和 Mark 让 Python 更快](https://talkpython.fm/episodes/show/339/making-python-faster-with-guido-and-mark)
*   欧洲 Python 2022: [我们如何让 Python 3.11 更快](https://www.youtube.com/watch?v=4ytEjwywUMM&t=9748s)马克·香农
*   跟我说说 Python:[Python Perf:specialized，Adaptive Interpreter](https://talkpython.fm/episodes/show/381/python-perf-specializing-adaptive-interpreter) 与 Brandt Bucher
*   PyCon 是:[更快的 CPython 项目:我们怎样使 Python 3.11 更快](https://charlas.2022.es.pycon.org/pycones2022/talk/D9ABNU/?linkId=182586474)【由 Pablo Galindo Salgado，西班牙语介绍]

更快的 CPython 是一个庞大的项目，涉及 Python 的所有部分。自适应专门化解释器就是其中的一部分。在本教程的后面，您将了解另外两种优化:[更快的启动](#faster-startup)和[零成本异常](#zero-cost-exceptions)。

[*Remove ads*](/account/join/)

## 更好的异步任务语法

Python 中对异步编程的支持已经发展了很长时间。随着[生成器](https://peps.python.org/pep-0255/)的加入，Python 2 时代奠定了基础。Python 3.4 中最初增加了 [`asyncio`](https://peps.python.org/pep-3156/) 库，Python 3.5 中又跟进了[`async``await`](https://peps.python.org/pep-0492/)关键字。

这种开发在以后的版本中继续进行，对 Python 的异步功能进行了许多小的改进。在 Python 3.11 中，您可以使用任务组，它为运行和监控异步任务提供了更清晰的语法。

**注意:**如果您还不熟悉 Python 中的异步编程，那么您可以查阅以下资源开始学习:

*   [通过并发加速您的 Python 程序](https://realpython.com/python-concurrency/)
*   [Python 异步特性入门](https://realpython.com/python-async-features/)
*   [Python 中的异步 IO:完整演练](https://realpython.com/async-io-python/)

您还可以在 [Python 3.11 预览版:任务和异常组](https://realpython.com/python311-exception-groups/)中了解关于异步任务组的更多细节。

`asyncio`库是 Python 标准库的一部分。然而，这并不是异步工作的唯一方式。有几个流行的第三方图书馆提供同样的功能，包括[三重奏](https://trio.readthedocs.io/)和[古玩](https://curio.readthedocs.io/)。此外，像 [uvloop](https://uvloop.readthedocs.io/) 、 [AnyIO](https://anyio.readthedocs.io/) 、 [Quattro](https://pypi.org/project/quattro/) 这样的包增强了`asyncio`更好的性能和更多的功能。

用`asyncio`运行几个异步任务的传统方式是用 [`create_task()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) 创建任务，然后用 [`gather()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather) 等待它们。这就完成了任务，但是有点麻烦。

为了组织孩子的任务，Curio 引入了[任务组](https://curio.readthedocs.io/en/latest/tutorial.html#task-groups)，Trio 引入了[托儿所](https://trio.readthedocs.io/en/stable/reference-core.html#tasks)作为替代。新的`asyncio`任务组深受这些的启发。

当您使用`gather()`组织异步任务时，部分代码通常如下所示:

```py
tasks = [asyncio.create_task(run_some_task(param)) for param in params]
await asyncio.gather(*tasks)
```

在将任务传递给`gather()`之前，手动跟踪列表中的所有任务。通过等待`gather()`，你可以确保在继续前进之前完成每项任务。

对于任务组，等价的代码更加简单。不使用`gather()`，而是使用上下文管理器来定义任务等待的时间:

```py
async with asyncio.TaskGroup() as tg:
    for param in params:
        tg.create_task(run_some_task(param))
```

您创建一个任务组对象，在本例中命名为`tg`，并使用它的`.create_task()`方法来创建新任务。

要查看完整的示例，请考虑下载几个文件的任务。您想下载一些历史 PEP 文档的文本，这些文档展示了 Python 的异步特性是如何发展的。为了提高效率，您将使用第三方库 [`aiohttp`](https://pypi.org/project/aiohttp/) 来异步下载文件。

首先导入必要的库，记下存储每个 PEP 文本的[库](https://github.com/python/peps/)的 URL:

```py
# download_peps_gather.py

import asyncio
import aiohttp

PEP_URL = (
    "https://raw.githubusercontent.com/python/peps/master/pep-{pep:04d}.txt"
)

async def main(peps):
    async with aiohttp.ClientSession() as session:
        await download_peps(session, peps)
```

您添加了一个初始化一个`aiohttp`会话的`main()`函数来管理可能被重用的连接池。现在，您正在调用一个名为`download_peps()`的函数，而您还没有编写这个函数。该函数将为每个需要下载的 PEP 创建一个任务:

```py
# download_peps_gather.py

# ...

async def download_peps(session, peps):
    tasks = [asyncio.create_task(download_pep(session, pep)) for pep in peps]
    await asyncio.gather(*tasks)
```

这符合您之前看到的模式。每个任务都由运行`download_pep()`组成，接下来您将对其进行定义。一旦你设置好所有的任务，你就把它们传递给`gather()`。

每个任务下载一个 PEP。您将添加几个`print()`呼叫，这样您就可以看到发生了什么:

```py
# download_peps_gather.py

# ...

async def download_pep(session, pep):
    print(f"Downloading PEP {pep}")
    url = PEP_URL.format(pep=pep)
    async with session.get(url, params={}) as response:
        pep_text = await response.text()

    title = pep_text.split("\n")[1].removeprefix("Title:").strip()
    print(f"Downloaded PEP {pep}: {title}")
```

对于每个 PEP，您可以找到它自己的 URL 并使用`session.get()`下载它。一旦有了 PEP 的文本，就可以找到 PEP 的标题并将其打印到控制台。

最后，异步运行`main()`:

```py
# download_peps_gather.py

# ...

asyncio.run(main([492, 525, 530, 3148, 3156]))
```

您正在用 PEP 编号列表调用您的代码，所有这些都与 Python 中的异步特性相关。运行您的脚本，看看它是如何工作的:

```py
$ python download_peps_gather.py
Downloading PEP 492
Downloading PEP 525
Downloading PEP 530
Downloading PEP 3148
Downloading PEP 3156
Downloaded PEP 3148: futures - execute computations asynchronously
Downloaded PEP 492: Coroutines with async and await syntax
Downloaded PEP 530: Asynchronous Comprehensions
Downloaded PEP 3156: Asynchronous IO Support Rebooted: the "asyncio" Module
Downloaded PEP 525: Asynchronous Generators
```

您可以看到所有的下载都是同时发生的，因为所有的任务都显示它们在任何任务报告完成之前开始下载 PEP。另外，请注意，任务是按照您定义的顺序启动的，pep 是按照数字顺序启动的。

相反，任务似乎是以随机的顺序完成的。对`gather()`的调用确保了所有的任务都在代码继续之前完成。

您可以更新您的代码来使用任务组而不是`gather()`。首先，将`download_peps_gather.py`复制到名为`download_peps_taskgroup.py`的新文件中。这些文件将非常相似。您只需要编辑`download_peps()`功能:

```py
# download_peps_taskgroup.py

# ...

async def download_peps(session, peps):
    async with asyncio.TaskGroup() as tg:
        for pep in peps:
            tg.create_task(download_pep(session, pep))

# ...
```

请注意，您的代码遵循示例之前概述的一般模式。首先在[上下文管理器](https://realpython.com/python-with-statement/)中建立一个任务组，然后使用该任务组创建子任务:每个 PEP 下载一个任务。运行更新后的代码，观察它的行为是否与早期版本相同。

当您处理几个异步任务时，一个挑战是它们中的任何一个都可能在任何时候引发错误。理论上，两个或更多的任务甚至可以同时引发一个错误。

像 [Trio](https://trio.readthedocs.io/en/stable/reference-core.html#working-with-multierrors) 和 [Curio](https://curio.readthedocs.io/en/latest/reference.html#exceptions) 这样的库已经用一种特殊的多错误对象处理了这个问题。这是可行的，但是有点麻烦，因为 Python 没有提供太多的内置支持。

为了正确支持任务组中的错误处理，Python 3.11 引入了[异常组](https://realpython.com/python311-exception-groups/#exception-groups-and-except-in-python-311)，用于跟踪几个并发错误。稍后在本教程中，你会学到更多关于他们[的知识。](#exception-groups)

任务组使用异常组来提供比旧方法更好的错误处理支持。关于任务组的更深入的讨论，请参见 [Python 3.11 预览版:任务和异常组](https://realpython.com/python311-exception-groups/#asynchronous-task-groups-in-python-311)。你可以在吉多·范·罗苏姆关于`asyncio.Semaphore` 的[推理中了解更多的基本原理。](https://neopythonic.blogspot.com/2022/10/reasoning-about-asynciosemaphore.html)

[*Remove ads*](/account/join/)

## 改进类型变量

Python 是一种动态类型语言，但是它通过可选的[类型提示](https://realpython.com/python-type-checking/)支持静态类型。Python 的静态类型系统的基础是在 2015 年 [PEP 484](https://peps.python.org/pep-0484) 中定义的。从 Python 3.5 开始，每个 Python 版本都引入了几个与类型相关的新提议。

Python 3.11 宣布了五个与类型相关的 pep，创历史新高:

*   PEP 646 :可变泛型
*   [PEP 655](https://peps.python.org/pep-0655) :将单个`TypedDict`项目标记为必需或潜在缺失
*   [人教版 673](https://peps.python.org/pep-0673) : `Self`类型
*   [PEP 675](https://peps.python.org/pep-0675) :任意文字字符串类型
*   [PEP 681](https://peps.python.org/pep-0681) :数据类转换

在这一节中，您将关注其中的两个:可变泛型和`Self`类型。要了解更多信息，请查看 PEP 文档和在这个 [Python 3.11 预览版](https://realpython.com/python311-tomllib/#other-new-features)中输入的覆盖范围。

**注意:**除了 Python 版本之外，对类型化特性的支持还取决于您的类型检查器。例如，在 Python 3.11 发布时， [mypy](https://mypy.readthedocs.io/) 不支持[的几个新特性。](https://github.com/python/mypy/issues/12840)

从[开始](https://peps.python.org/pep-0484/#generics)，类型变量就已经是 Python 静态类型系统的一部分。你用它们来参数化**通用**类型。换句话说，如果您有一个列表，那么您可以使用类型变量来检查列表中项目的类型:

```py
from typing import Sequence, TypeVar

T = TypeVar("T")

def first(sequence: Sequence[T]) -> T:
    return sequence[0]
```

`first()`函数从一个序列类型中挑选出第一个元素，比如一个列表。不管序列元素的类型如何，代码都是一样的。尽管如此，您仍然需要跟踪元素类型，以便知道`first()`的返回类型。

类型变量正是这样做的。例如，如果您将一个整数列表传递给`first()`，那么在类型检查期间`T`将被设置为`int`。因此，类型检查器可以推断出对`first()`的调用返回了`int`。在这个例子中，列表被称为**通用类型**，因为它可以被其他类型参数化。

随着时间的推移，发展起来的一种模式试图解决引用当前类的类型提示问题。回忆一下之前[的`Person`课](#more-informative-error-tracebacks):

```py
# programmers.py

from dataclasses import dataclass

# ...

@dataclass
class Person:
    name: str
    life_span: tuple[int, int]

    @classmethod
    def from_dict(cls, info):
        return cls(
            name=f"{info['name']['first']}  {info['name']['last']}",
            life_span=(info["birth"]["year"], info["death"]["year"]),
        )
```

`.from_dict()`构造函数返回一个`Person`对象。然而，不允许使用`-> Person`作为`.from_dict()`返回值的类型提示，因为`Person`类在你的代码中还没有完全定义。

此外，如果你被允许使用`-> Person`，那么这将不能很好地与继承一起工作。如果你创建了一个`Person`的子类，那么`.from_dict()`将返回那个子类，而不是一个`Person`对象。

这个挑战的一个解决方案是使用绑定到您的类的类型变量:

```py
# programmers.py

# ...

from typing import Any, Type, TypeVar 
TPerson = TypeVar("TPerson", bound="Person") 
@dataclass
class Person:
    name: str
    life_span: tuple[int, int]

    @classmethod
 def from_dict(cls: Type[TPerson], info: dict[str, Any]) -> TPerson:        return cls(
            name=f"{info['name']['first']}  {info['name']['last']}",
            life_span=(info["birth"]["year"], info["death"]["year"]),
        )
```

你指定了`bound`来确保`TPerson`永远只能是`Person`或者它的一个子类。这种模式是可行的，但是可读性不是特别好。它还迫使你注释`self`或`cls`，这通常是不必要的。

你现在可以使用新的 [`Self`](https://docs.python.org/3.11/library/typing.html#typing.Self) 类型来代替。它总是引用封装类，所以你不必手动定义一个类型变量。以下代码等效于前面的示例:

```py
# programmers.py

# ...

from typing import Any, Self 
@dataclass
class Person:
    name: str
    life_span: tuple[int, int]

    @classmethod
 def from_dict(cls, info: dict[str, Any]) -> Self:        return cls(
            name=f"{info['name']['first']}  {info['name']['last']}",
            life_span=(info["birth"]["year"], info["death"]["year"]),
        )
```

可以从`typing`导入`Self`。你不需要创建一个类型变量或者注释`cls`。相反，您会注意到该方法返回了`Self`，它将引用`Person`。

关于如何使用`Self`的另一个例子，参见 [Python 3.11 预览版](https://realpython.com/python311-tomllib/#self-type)。你也可以查看 [PEP 673](https://peps.python.org/pep-0673/) 了解更多详情。

类型变量的一个限制是它们一次只能代表一种类型。假设您有一个翻转二元元组顺序的函数:

```py
# pair_order.py

def flip(pair):
    first, second = pair
    return (second, first)
```

这里，`pair`假设是一个有两个元素的元组。元素可以是不同的类型，因此需要两个类型变量来注释函数:

```py
# pair_order.py

from typing import TypeVar 
T0 = TypeVar("T0") T1 = TypeVar("T1") 
def flip(pair: tuple[T0, T1]) -> tuple[T1, T0]:
    first, second = pair
    return (second, first)
```

这个写起来有点繁琐，不过还是可以的。注释是清晰易读的。如果您想要注释代码的变体，该变体适用于具有任意数量元素的元组，那么挑战就来了:

```py
# tuple_order.py

def cycle(elements):
    first, *rest = elements
    return (*rest, first)
```

使用`cycle()`，将第一个元素移动到一个包含任意数量元素的元组的末尾。如果你传入一对元素，那么这相当于`flip()`。

想想你会如何注释`cycle()`。如果`elements`是一个有 *n* 个元素的元组，那么你需要 *n* 个类型变量。但是元素的数量可以是任意的，所以你不知道你需要多少类型变量。

[PEP 646](https://peps.python.org/pep-0646/) 引入 [TypeVarTuple](https://docs.python.org/3.11/library/typing.html#typing.TypeVarTuple) 来处理这个用例。一个`TypeVarTuple`可以代表任意数量的类型。因此，您可以用它来注释带有**变量**参数的泛型类型。

您可以向`cycle()`添加类型提示，如下所示:

```py
# tuple_order.py

from typing import TypeVar, TypeVarTuple 
T0 = TypeVar("T0") Ts = TypeVarTuple("Ts") 
def cycle(elements: tuple[T0, *Ts]) -> tuple[*Ts, T0]:
    first, *rest = elements
    return (*rest, first)
```

`TypeVarTuple`将替换任意数量的类型，因此该注释将适用于具有一个、三个、十一个或任何其他数量元素的元组。

注意，`Ts`前面的星号(`*`)是语法的必要部分。它类似于你已经在代码中使用的[解包语法](https://realpython.com/python-kwargs-and-args/#unpacking-with-the-asterisk-operators)，它提醒你`Ts`代表任意数量的类型。

引入类型变量元组的激励用例是注释多维数组的形状。你可以在这个 [Python 3.11 预览版](https://realpython.com/python311-tomllib/#variadic-generic-types)和 [PEP](https://peps.python.org/pep-0646/) 中了解更多关于这个例子的信息。

在结束关于类型提示的这一节之前，请记住静态类型结合了两种不同的工具:Python 语言和类型检查器。要使用新的输入功能，您的 Python 版本必须支持它们。此外，它们需要得到您的类型检查器的支持。

在 [`typing_extensions`](https://pypi.org/project/typing-extensions/) 包中，包括`Self`和`TypeVarTuple`在内的许多输入特性都被移植到了旧版本的 Python 中。在 Python 3.10 上，您可以使用 [`pip`](https://realpython.com/what-is-pip/) 将`typing-extensions`安装到您的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中，然后实现最后一个示例，如下所示:

```py
# tuple_order.py

from typing_extensions import TypeVar, TypeVarTuple, Unpack 
T0 = TypeVar("T0")
Ts = TypeVarTuple("Ts")

def cycle(elements: tuple[T0, Unpack[Ts]]) -> tuple[Unpack[Ts], T0]:
    first, *rest = elements
 return (*rest, first)
```

只有 Python 3.11 才允许使用`*Ts`语法。一个在旧版本 Python 上工作的[等价替代](https://docs.python.org/3.11/library/typing.html#typing.Unpack)是`Unpack[Ts]`。即使你的代码可以在你的 Python 版本上运行，也不是所有的类型检查器[都支持](https://github.com/python/typeshed/issues/8708) `TypeVarTuple`。

[*Remove ads*](/account/join/)

## 支持 TOML 配置解析

TOML 是**汤姆明显最小语言**的简称。这是一种配置文件格式，在过去十年中变得很流行。当[为包和项目指定元数据](https://peps.python.org/pep-0518/)时，Python 社区已经将 [TOML](https://toml.io/en/) 作为首选格式。

TOML 被设计成易于人类阅读和计算机解析。你可以在 [Python 和 TOML: New Best Friends](https://realpython.com/python-toml/) 中了解配置文件格式本身。

虽然 TOML 已经被许多不同的工具使用了很多年，但是 Python 还没有内置的 TOML 支持。Python 3.11 中的变化是，当 [`tomllib`](https://docs.python.org/3.11/library/tomllib.html) 被添加到标准库中。这个新模块建立在流行的第三方库 [`tomli`](https://pypi.org/project/tomli/) 之上，允许你解析 TOML 文件。

下面是一个名为`units.toml`的 TOML 文件的例子:

```py
# units.toml [second] label  =  {  singular  =  "second",  plural  =  "seconds"  } aliases  =  ["s",  "sec",  "seconds"] [minute] label  =  {  singular  =  "minute",  plural  =  "minutes"  } aliases  =  ["min",  "minutes"] multiplier  =  60 to_unit  =  "second" [hour] label  =  {  singular  =  "hour",  plural  =  "hours"  } aliases  =  ["h",  "hr",  "hours"] multiplier  =  60 to_unit  =  "minute" [day] label  =  {  singular  =  "day",  plural  =  "days"  } aliases  =  ["d",  "days"] multiplier  =  24 to_unit  =  "hour" [year] label  =  {  singular  =  "year",  plural  =  "years"  } aliases  =  ["y",  "yr",  "years",  "julian_year",  "julian years"] multiplier  =  365.25 to_unit  =  "day"
```

该文件包含几个部分，标题在方括号中。每个这样的部分在 TOML 中被称为一个**表**，它的标题被称为一个**键**。表格包含**键值对**。表格可以嵌套，这样值就是新的表格。在上面的例子中，你可以看到除了`second`之外的每个表都有相同的结构，有四个键:`label`、`aliases`、`multiplier`和`to_unit`。

值可以有不同的类型。在本例中，您可以看到四种数据类型:

1.  `label`是一个**内联表**，类似于 Python 的字典。
2.  `aliases`是一个**数组**，类似于 Python 的 list。
3.  `multiplier`是一个**数**，可以是整数，也可以是浮点数。
4.  `to_unit`是一根**弦**。

TOML 支持更多的数据类型，包括布尔值和日期。请参见 [Python 和 TOML:新的最好的朋友](https://realpython.com/python-toml/#get-to-know-toml-key-value-pairs)以深入了解该格式及其语法示例。

您可以使用`tomllib`来读取一个 TOML 文件:

>>>

```py
>>> import tomllib
>>> with open("units.toml", mode="rb") as file:
...     units = tomllib.load(file)
...
>>> units
{'second': {'label': {'singular': 'second', 'plural': 'seconds'}, ... }}
```

当使用`tomllib.load()`时，你通过指定`mode="rb"`传入一个必须在**二进制模式**下打开的文件对象。或者，您可以用`tomllib.loads()`解析一个字符串:

>>>

```py
>>> import tomllib
>>> import pathlib
>>> units = tomllib.loads(
...     pathlib.Path("units.toml").read_text(encoding="utf-8")
... )
>>> units
{'second': {'label': {'singular': 'second', 'plural': 'seconds'}, ... }}
```

在这个例子中，首先使用 [`pathlib`](https://realpython.com/python-pathlib/) 将`units.toml`读入一个字符串，然后用`loads()`解析这个字符串。TOML 文件应该存储在一个 [UTF-8 编码](https://realpython.com/python-encodings-guide/)中。您应该明确地为[指定编码，以确保](https://realpython.com/python310-new-features/#default-text-encodings)您的代码在所有平台上都运行相同。

接下来，把注意力转向调用`load()`或`loads()`的结果。在上面的例子中，你可以看到`units`是一个嵌套的字典。情况总是这样:`tomllib`将 TOML 文档解析成 Python 字典。

在本节的其余部分，您将练习在 Python 中使用 TOML 数据。您将创建一个小的**单元转换器**，它解析您的 TOML 文件并使用生成的字典。

**注意:**如果你真的在做单位转换，那么你应该看看[品脱](http://pint.readthedocs.org/)。这个库可以在[数百个单位](https://github.com/hgrecco/pint/blob/master/pint/default_en.txt)之间转换，并且很好地集成到其他包中，比如 [NumPy](https://realpython.com/numpy-tutorial/) 。

将您的代码添加到名为`units.py`的文件中:

```py
# units.py

import pathlib
import tomllib

# Read units from file
with pathlib.Path("units.toml").open(mode="rb") as file:
    base_units = tomllib.load(file)
```

您希望能够通过名称或别名来查找每个单元。您可以通过复制单元信息来实现这一点，这样每个别名都可以用作字典键:

```py
# units.py

# ...

units = {}
for unit, unit_info in base_units.items():
    units[unit] = unit_info
    for alias in unit_info["aliases"]:
        units[alias] = unit_info
```

例如，您的`units`字典现在将有键`second`以及它的别名`s`、`sec`和`seconds`都指向`second`表。

接下来，您将定义`to_baseunit()`，它可以将 TOML 文件中的任何单元转换为其对应的基本单元。在本例中，基本单位始终是`second`。但是，您可以扩展该表，以包括以`meter`为基本单位的长度单位。

将`to_baseunit()`的定义添加到文件中:

```py
# units.py

# ...

def to_baseunit(value, from_unit):
    from_info = units[from_unit]
    if "multiplier" not in from_info:
        return (
            value,
            from_info["label"]["singular" if value == 1 else "plural"],
        )

    return to_baseunit(value * from_info["multiplier"], from_info["to_unit"])
```

您将`to_baseunit()`实现为一个[递归函数](https://realpython.com/python-recursion/)。如果对应于`from_unit`的表不包含`multiplier`字段，那么您将该单元视为基本单元并返回其值和名称。另一方面，如果有一个`multiplier`字段，那么您转换到链中的下一个单元，并再次调用`to_baseunit()`。

启动你的 [REPL](https://realpython.com/interacting-with-python/#using-the-python-interpreter-interactively) 。然后，导入`units`并转换几个数字:

>>>

```py
>>> import units
>>> units.to_baseunit(7, "s")
(7, 'seconds')

>>> units.to_baseunit(3.11, "minutes")
(186.6, 'seconds')
```

在第一个例子中，`"s"`被解释为`second`，因为它是一个别名。既然这是基地单位，`7`就原封不动地返回。在第二个例子中，`"minutes"`让您的函数在`minute`表中查找。它发现可以通过乘以`60`转换成`second`。

转化链可能会更长:

>>>

```py
>>> units.to_baseunit(14, "days")
(1209600, 'seconds')

>>> units.to_baseunit(1 / 12, "yr")
(2629800.0, 'seconds')
```

为了将`"days"`转换为其基本单位，您的函数首先将`day`转换为`hour`，然后将`hour`转换为`minute`，最后将`minute`转换为`second`。你发现十四天大约有 120 万秒，一年的十二分之一大约有 260 万秒。

**注意:**在这个例子中，您使用 TOML 文件来存储关于您的单元转换器所支持的单元的信息。您也可以通过将`base_units`定义为一个文字字典，将信息直接放入代码中。

但是，使用配置文件带来了几个与代码和数据分离相关的优点:

*   你的逻辑从你的数据中分离出来。
*   非开发者可以**贡献**到你的单元转换器中，而不需要接触——甚至不需要了解——Python 代码。
*   只需最少的努力，您就可以支持**额外的**单元配置文件。用户可以添加这些，以便在转换器中包含自定义单位。

您应该考虑为您使用的任何项目设置一个配置文件。

如前所述，`tomllib`基于`tomli`。如果你想在需要支持旧 Python 版本的代码中解析 TOML 文档，那么你可以[安装`tomli`](https://github.com/hukkin/tomli#installation) 并把它作为`tomllib` 的[后端口，如下所示:](https://github.com/hukkin/tomli#building-a-tomlitomllib-compatibility-layer)

```py
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
```

在 Python 3.11 上，这照常导入`tomllib`。在 Python 的早期版本中，导入会引发一个`ModuleNotFoundError`。在这里，您捕捉错误并导入`tomli`，同时将其别名化为名称`tomllib`，这样您的代码的其余部分就可以不变地工作。

你可以在 [Python 3.11 预览版:TOML 和`tomllib`](https://realpython.com/python311-tomllib/) 中了解更多关于`tomllib`的内容。此外， [PEP 680](https://peps.python.org/pep-0680/) 概述了导致`tomllib`被添加到 Python 的讨论。

[*Remove ads*](/account/join/)

## 其他非常酷的功能

到目前为止，您已经了解了 Python 3.11 中最大的变化和改进。然而，还有更多的功能需要探索。在本节中，您将了解一些可能隐藏在标题下的新特性。它们包括更多的加速、对异常的更多修改以及对字符串格式的一点改进。

### 更快启动

更快的 CPython 项目的另一个令人兴奋的结果是更快的启动时间。运行 Python 脚本时，解释器初始化时会发生几件事。这导致即使是最简单的程序也需要几毫秒才能运行:

*   [*视窗*](#windows-1)
**   [*Linux*](#linux-1)**   [*macOS*](#macos-1)**

```py
PS> Measure-Command {python -c "pass"}
...
TotalMilliseconds : 25.9823
```

```py
$ time python -c "pass"
real    0m0,020s
user    0m0,012s
sys     0m0,008s
```

```py
$ time python -c "pass"
python -c "pass"
0.02s user
0.01s system
90% cpu
0.024 total
```

您使用`-c`在命令行上直接传递一个程序。在这种情况下，您的整个程序由一个`pass`语句组成，它什么也不做。

在许多情况下，与运行代码所需的时间相比，启动程序所需的时间可以忽略不计。但是，在运行时间较短的脚本中，比如典型的命令行应用程序，启动时间可能会显著影响程序的性能。

作为一个具体的例子，考虑下面的脚本——受经典的 [cowsay](https://en.wikipedia.org/wiki/Cowsay) 程序的启发:

```py
# snakesay.py
import sys

message = " ".join(sys.argv[1:])
bubble_length = len(message) + 2
print(
    rf"""
  {"_" * bubble_length} ( {message} )
  {"‾" * bubble_length} \
 \    __
 \  [oo]
 (__)\
 λ \\
 _\\__
 (_____)_
 (________)Oo°"""
)
```

在`snakesay.py`中，你从[命令行](https://realpython.com/python-command-line-arguments/)中读到一条消息。然后，你在一个伴随着可爱的蛇的讲话泡泡里打印信息。现在，你可以让蛇说任何话:

```py
$ python snakesay.py Faster startup!
 _________________
 ( Faster startup! )
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 \
 \    __
 \  [oo]
 (__)\
 λ \\
 _\\__
 (_____)_
 (________)Oo°
```

这是一个命令行应用程序的基本示例。像许多其他命令行应用程序一样，它运行速度很快。尽管如此，它还需要几毫秒才能运行。这种开销的很大一部分是在 Python 导入模块时发生的，甚至有些模块不是您自己显式导入的。

您可以使用 [`-X importtime`选项](https://realpython.com/python37-new-features/#developer-tricks)来显示导入模块所用时间的概述:

```py
$ python -X importtime -S snakesay.py Imports are faster!
import time: self [us] | cumulative | imported package
import time:       283 |        283 |   _io
import time:        56 |         56 |   marshal
import time:       647 |        647 |   posix
import time:       587 |       1573 | _frozen_importlib_external
import time:       167 |        167 |   time
import time:       191 |        358 | zipimport
import time:        90 |         90 |     _codecs
import time:       561 |        651 |   codecs
import time:       825 |        825 |   encodings.aliases
import time:      1136 |       2611 | encodings
import time:       417 |        417 | encodings.utf_8
import time:       174 |        174 | _signal
import time:        56 |         56 |     _abc
import time:       251 |        306 |   abc
import time:       310 |        616 | io
 _____________________
 ( Imports are faster! )
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 \
 \    __
 \  [oo]
 (__)\
 λ \\
 _\\__
 (_____)_
 (________)Oo°
```

表中的数字以**微秒**为单位。注意最后一列中模块名称的格式。树形结构表明有几个顶级模块，这些模块导入其他模块。例如，`io`是顶级导入，而`abc`是由`io`导入的。

**注意:**你使用了上面的`-S`选项。根据文档，这将“禁用模块`site`的导入和它所需要的`sys.path`的站点相关操作”([源](https://docs.python.org/3/using/cmdline.html#cmdoption-S))。

对于这个简单的程序，`-S`使它运行得更快，因为需要的导入更少。然而，这不是一个可以在大多数脚本中使用的优化，因为您不能导入任何第三方库。

该示例在 Python 3.11 上运行。下表以微秒为单位将这些数字与使用 Python 3.10 运行相同命令进行了比较:

| 组件 | Python 3.11 | Python 3.10 | 加速 |
| --- | --- | --- | --- |
| `_frozen_importlib_external` | One thousand five hundred and seventy-three | Two thousand two hundred and fifty-five | 1.43 倍 |
| `zipimport` | Three hundred and fifty-eight | Five hundred and fifty-eight | 1.56 倍 |
| `encodings` | Two thousand six hundred and eleven | Three thousand and nine | 1.15 倍 |
| `encodings.utf_8` | Four hundred and seventeen | Four hundred and nine | 0.98 倍 |
| `_signal` | One hundred and seventy-four | One hundred and seventy-three | 0.99x |
| `io` | Six hundred and sixteen | One thousand two hundred and sixteen | 1.97 倍 |
| **总计** | **5749** | **7620** | **1.33x** |

您的数字会有所不同，但您应该会看到相同的模式。Python 3.11 的导入速度更快，这有助于 Python 程序更快地启动。

速度加快的一个重要原因是**缓存的**字节码是如何存储和读取的。正如您所了解的，Python 将源代码编译成由解释器运行的字节码。在[长时间](https://peps.python.org/pep-3147/)中，Python 将编译后的字节码存储在一个名为 [`__pycache__`](https://docs.python.org/3/tutorial/modules.html#compiled-python-files) 的目录中，以避免不必要的重新编译。

但是在 Python 的最新版本中，许多模块都被冻结了，并以一种更快的方式存储在内存中。你可以在[文档](https://docs.python.org/3.11/whatsnew/3.11.html#faster-startup)中了解更多关于快速启动的信息。

[*Remove ads*](/account/join/)

### 零成本例外

Python 3.11 中异常的内部表示是不同的。异常对象更加轻量级，异常处理也发生了变化，因此只要不触发`except`子句，在`try` … `except`块中就几乎没有开销。

所谓的[零成本异常](https://github.com/python/cpython/issues/84403)是受 C++和 Java 等其他语言的启发。我们的目标是快乐之路——当没有出现异常时——实际上应该是免费的。处理异常仍然需要一些时间。

当源代码被编译成字节码时，通过让编译器创建跳转表来实现零成本异常。如果出现异常，将参考这些表。如果没有异常，那么`try`块中的代码就没有运行时开销。

回想一下您之前与[合作的乘法逆运算示例](#more-informative-error-tracebacks)。您添加了一些错误处理:

>>>

```py
 1>>> def inverse(number):
 2...     try:
 3...         return 1 / number
 4...     except ZeroDivisionError:
 5...         print("0 has no inverse")
 6...
```

如果你试图计算零的倒数，那么就会产生一个`ZeroDivisionError`。在您的新实现中，您捕获这些错误并打印一条描述性消息。和以前一样，您使用`dis`来查看幕后的字节码:

>>>

```py
>>> import dis
>>> dis.dis(inverse)
 1           0 RESUME                   0

 2           2 NOP 
 3           4 LOAD_CONST               1 (1)
 6 LOAD_FAST                0 (number)
 8 BINARY_OP               11 (/)
 12 RETURN_VALUE
 >>   14 PUSH_EXC_INFO

 4          16 LOAD_GLOBAL              0 (ZeroDivisionError)
 28 CHECK_EXC_MATCH
 30 POP_JUMP_FORWARD_IF_FALSE    19 (to 70)
 32 POP_TOP

 5          34 LOAD_GLOBAL              3 (NULL + print)
 46 LOAD_CONST               2 ('0 has no inverse')
 48 PRECALL                  1
 52 CALL                     1
 62 POP_TOP
 64 POP_EXCEPT
 66 LOAD_CONST               0 (None)
 68 RETURN_VALUE

 4     >>   70 RERAISE                  0
 >>   72 COPY                     3
 74 POP_EXCEPT
 76 RERAISE                  1
ExceptionTable:
 4 to 10 -> 14 [0] 14 to 62 -> 72 [1] lasti 70 to 70 -> 72 [1] lasti
```

你不需要理解字节码的细节。但是，您可以将最左边一列中的数字与源代码中的行号进行比较。注意，第 2 行是`try:`，被翻译成单个 [`NOP`](https://docs.python.org/3/library/dis.html#opcode-NOP) 指令。这是一个**无操作**，它什么也不做。更有趣的是，反汇编的最后是一个异常表。这是解释器在需要处理异常时使用的跳转表。

在 Python 3.10 和更早的版本中，运行时有一点异常处理。例如，`try`语句被编译成包含指向第一个异常块的指针的 [`SETUP_FINALLY`](https://docs.python.org/3/library/dis.html#opcode-SETUP_FINALLY) 指令。用跳转表替换它可以在异常没有出现时加速`try`块。

零成本异常很好地适应了一种[的更容易请求宽恕而不是许可的](https://realpython.com/python-lbyl-vs-eafp/)代码风格，这种风格通常使用大量的`try` … `except`块。

### 异常组

之前，您学习了[任务组](#nicer-syntax-for-asynchronous-tasks)以及它们如何能够同时处理几个错误。他们通过一个叫做**异常组**的新特性来做到这一点。

考虑异常组的一种方式是，它们是包装几个其他常规异常的常规异常。然而，尽管异常组在许多方面表现得像常规异常，但它们也支持特殊的语法，帮助您有效地处理每个包装的异常。

通过为例外组提供描述并列出它所包装的例外，可以创建例外组:

>>>

```py
>>> ExceptionGroup("twice", [TypeError("int"), ValueError(654)])
ExceptionGroup('twice', [TypeError('int'), ValueError(654)])
```

这里您已经创建了一个描述为`"twice"`的异常组，它包装了一个`TypeError`和一个`ValueError`。如果一个异常组在没有被处理的情况下被引发，那么它会显示一个很好的回溯来说明错误的分组和嵌套:

>>>

```py
>>> raise ExceptionGroup("twice", [TypeError("int"), ValueError(654)])
 + Exception Group Traceback (most recent call last):
 |   File "<stdin>", line 1, in <module>
 | ExceptionGroup: twice (2 sub-exceptions)
 +-+---------------- 1 ----------------
 | TypeError: int
 +---------------- 2 ----------------
 | ValueError: 654
 +------------------------------------
```

此错误消息说明引发了一个包含两个子异常的异常组。每个包装的异常都显示在它自己的面板中。

除了引入异常组之外，新版本的 Python 还添加了新的语法来有效地使用它们。你*可以*做`except ExceptionGroup as eg`并且循环`eg`中的每个错误。然而，这很麻烦。相反，你应该使用新的`except*`关键字:

>>>

```py
>>> try:
...     raise ExceptionGroup("twice", [TypeError("int"), ValueError(654)])
... except* ValueError as err: ...     print(f"handling ValueError: {err.exceptions}")
... except* TypeError as err: ...     print(f"handling TypeError: {err.exceptions}")
...
handling ValueError: (ValueError(654),)
handling TypeError: (TypeError('int'),)
```

与常规的`except`语句相比，几个`except*`语句可以触发。在这个例子中，`ValueError`和`TypeError`都被处理了。

异常组中未处理的异常将停止你的程序，并照常显示回溯。请注意，由`except*`处理的错误被过滤出该组:

>>>

```py
>>> try:
...     raise ExceptionGroup("twice", [TypeError("int"), ValueError(654)])
... except* ValueError as err:
...     print(f"handling ValueError: {err.exceptions}")
...
handling ValueError: (ValueError(654),)
 + Exception Group Traceback (most recent call last):
 |   File "<stdin>", line 2, in <module>
 | ExceptionGroup: twice (1 sub-exception)
 +-+---------------- 1 ----------------
 | TypeError: int
 +------------------------------------
```

你处理了`ValueError`，但是`TypeError`没有被触动。这反映在回溯中，这里的`twice`异常组现在只有一个子异常。

异常组和`except*`语法不会取代常规异常和普通的`except`。事实上，您可能不会有很多自己创建异常组的用例。相反，他们将主要由像`asyncio`这样的图书馆抚养。

用`except*`可以捕捉常规异常。尽管如此，在大多数情况下，你还是希望坚持使用普通的`except`，并且只对实际上可能引发异常组的代码使用`except*`。

要了解更多关于异常组如何工作、如何嵌套以及`except*`的全部功能，请参见 [Python 3.11 预览版:任务和异常组](https://realpython.com/python311-exception-groups/)。Python 的核心开发者之一 Irit Katriel 在 2021 年的 [Python 语言峰会和 2022 年](https://pyfound.blogspot.com/2021/05/the-2021-python-language-summit-pep-654.html)的 [PyCon UK 上展示了异常组。](https://www.youtube.com/watch?v=uARIj9eAZcQ)

你可以在 [PEP 654](https://peps.python.org/pep-0654/) 中阅读更多关于异常组的动机和导致当前实现的讨论。

[*Remove ads*](/account/join/)

### 异常注释

常规异常的一个扩展是添加任意注释的能力。PEP 678 描述了如何使用这些注释在不同于最初引发异常的代码段中向异常添加信息。例如，像[假设](https://hypothesis.readthedocs.io/en/latest/)这样的[测试](https://realpython.com/python-testing/)库可以[添加关于哪个测试失败的信息](https://peps.python.org/pep-0678/#example-usage)。

您可以使用`.add_note()`向任何异常添加注释，并通过检查`.__notes__`属性查看现有注释:

>>>

```py
>>> err = ValueError(678)
>>> err.add_note("Enriching Exceptions with Notes")
>>> err.add_note("Python 3.11")

>>> err.__notes__
['Enriching Exceptions with Notes', 'Python 3.11']

>>> raise err
Traceback (most recent call last):
  ...
ValueError: 678
Enriching Exceptions with Notes
Python 3.11
```

如果出现错误，任何相关的注释都会打印在追溯的底部。

在下面的例子中，您将主循环包装在一个`try` … `except`块中，该块为错误添加了一个时间戳。如果您需要将错误消息与程序的运行中的[日志](https://realpython.com/python-logging/)进行比较，这可能会很有用:

```py
# timestamped_errors.py

from datetime import datetime

def main():
    inverse(0)

def inverse(number):
    return 1 / number

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
 err.add_note(f"Raised at {datetime.now()}")        raise
```

正如你之前看到的，这个程序计算乘法逆运算。这里，您添加了一个简短的 [`main()`](https://realpython.com/python-main-function/) 函数，稍后您将调用它。

您已经将对`main()`的调用包装在一个`try` … `except`块中，该块捕捉任何`Exception`。虽然您通常希望更加具体，但是您在这里使用`Exception`来有效地为主程序碰巧引发的任何异常添加上下文。

当您运行这段代码时，您将看到预期的`ZeroDivisionError`。此外，您的回溯包含一个时间戳，可能有助于您的调试工作:

```py
$ python timestamped_errors.py
Traceback (most recent call last):
 ...
ZeroDivisionError: division by zero
Raised at 2022-10-24 12:18:13.913838
```

您可以使用相同的模式向您的异常添加其他有用的信息。更多信息参见这个 [Python 3.11 预览版](https://realpython.com/python311-exception-groups/#annotate-exceptions-with-custom-notes)和 [PEP 678](https://peps.python.org/pep-0678/) 。

### 负零格式

使用浮点数进行计算时，您可能会遇到一个奇怪的概念，那就是**负零**。您可以观察到负零和常规零在您的 REPL 中呈现不同:

>>>

```py
>>> -0.0
-0.0
>>> 0.0
0.0
```

正常情况下，只有一个零，它既不是正的也不是负的。然而，当允许[符号零](https://en.wikipedia.org/wiki/Signed_zero)时，浮点数的[表示](https://en.wikipedia.org/wiki/IEEE_754#Formats)更容易。在内部，数字用它们的符号和大小作为独立的量来表示。和其他数字一样，用正号或负号来表示零更简单。

Python 知道这两种表示是相等的:

>>>

```py
>>> -0.0 == 0.0
True
```

一般来说，你在计算中不需要担心负零。尽管如此，当您显示带有四舍五入的小负数的数据时，您可能会得到一些意外的结果:

>>>

```py
>>> small = -0.00311
>>> f"A small number: {small:.2f}"
'A small number: -0.00'
```

通常，当一个数被四舍五入为零时，它将被表示为一个无符号的零。在这个例子中，当表示为 [f 字符串](https://realpython.com/python-f-strings/)时，小负数被四舍五入到两位小数。注意，在零之前显示一个负号。

[PEP 682](https://peps.python.org/pep-0682/) 对 f 弦和 [`str.format()`](https://realpython.com/python-formatted-output/) 使用的[格式迷你语言](https://docs.python.org/3/library/string.html#formatstrings)做了一个小的扩展。在 Python 3.11 中，可以在格式字符串中添加文字`z`。这将在格式化之前强制将任何零规范化为正零:

>>>

```py
>>> small = -0.00311
>>> f"A small number: {small:z.2f}"
'A small number: 0.00'
```

您已经向格式字符串`z.2f`添加了一个`z`。这确保了负零不会渗透到面向用户的数据表示中。

[*Remove ads*](/account/join/)

### 没电的电池

Python 早期的优势之一是它自带了包括 T1 在内的 T0 电池。这个有点神秘的短语用来指出编程语言本身包含了很多功能。例如，Python 是最早包含对列表、元组和字典等容器的高级支持的语言之一。

然而，真正的电池可以在 Python 的标准库中找到。这是 Python 的每个安装都包含的包的集合。例如，标准库包括以下功能:

*   解析不同的文件类型和格式，比如 [JSON](https://realpython.com/python-json/) 和 [XML](https://realpython.com/python-xml-parser/)
*   运行一个简单的[网络服务器](https://docs.python.org/3/library/http.server.html)
*   撰写和发送[封电子邮件](https://realpython.com/python-send-email/)
*   与[套接字](https://realpython.com/python-sockets/)和[其他进程](https://realpython.com/python-subprocess/)通信

总的来说，标准库由数百个模块组成:

>>>

```py
>>> import sys
>>> len(sys.stdlib_module_names)
305
```

通过检查`sys.stdlib_module_names`可以看到标准库中有哪些模块。在早期，语言内置了如此强大的功能对 Python 来说是一个福音。

随着时间的推移，标准库的用处已经减少，主要是因为第三方模块的分发和安装变得更加方便。Python 的许多最受欢迎的特性现在都存在于主发行版之外。像 [NumPy](https://realpython.com/numpy-tutorial/) 和[熊猫](https://realpython.com/pandas-dataframe/)这样的数据科学库，像 [Matplotlib](https://realpython.com/python-matplotlib-guide/) 和 [Bokeh](https://realpython.com/python-data-visualization-bokeh/) 这样的可视化工具，像 [Django](https://realpython.com/get-started-with-django-1/) 和 [Flask](https://realpython.com/python-web-applications/) 这样的 web 框架都是独立开发的。

PEP 594 描述了一项从标准库中移除废电池的计划。这个想法是不再相关的模块应该从标准库中删除。这将有助于 Python 的维护者将他们的精力集中在最需要的地方，从而获得最大的收益。此外，一个更精简的标准库使 Python 更适合替代平台，如[微控制器](https://realpython.com/micropython/)或[浏览器](https://realpython.com/pyscript-python-in-browser/)。

在这个版本的 Python 中，没有从标准库中删除任何模块。相反，在 Python 3.13 中，有几个很少使用的模块被标记为删除。在 Python 3.11 中，有问题的模块将开始发出警告:

>>>

```py
>>> import imghdr
<stdin>:1: DeprecationWarning: 'imghdr' is deprecated and slated for
 removal in Python 3.13
```

如果您的代码开始发出这种警告，那么您应该开始考虑重写您的代码。在大多数情况下，会有更现代的选择。例如，如果你正在使用`imghdr`，那么你可以重写你的代码来使用 [python-magic](https://pypi.org/project/python-magic/) 。在这里，您可以识别文件的类型:

>>>

```py
>>> import imghdr
<stdin>:1: DeprecationWarning: 'imghdr' is deprecated and slated for
 removal in Python 3.13 >>> imghdr.what("python-311.jpg")
'jpeg'

>>> import magic
>>> magic.from_file("python-311.jpg")
'JPEG image data, JFIF standard 1.02, precision 8, 1920x1080, components 3'
```

旧的、废弃的`imghdr`和第三方 python-magic 库都认为`python-311.jpg`代表 JPEG 图像文件。

**注意:**python-magic 包依赖于一个需要安装的 C 库。查看[文档](https://github.com/ahupp/python-magic#installation)了解如何在你的操作系统上安装它的细节。

你可以在电池耗尽的 PEP 中找到一个包含所有废弃模块的列表。

## 那么，该不该升级到 Python 3.11 呢？

Python 3.11 中最酷的改进和新特性之旅到此结束。一个重要的问题是，是否应该升级到 Python 的新版本。如果是的话，什么时候是升级的最佳时机？

像往常一样，这类问题的答案是一清二楚的**看情况**！

Python 3.11 最大的胜利是对开发人员体验的改进:更好的错误消息和更快的代码执行。这些都是尽快升级你用于当地发展的环境的巨大激励。这也是风险最小的一种升级，因为您遇到的任何错误都应该影响有限。

速度的提高也是更新您的**生产环境**的一个很好的理由。但是，与往常一样，在更新环境时，您应该小心，因为在这些环境中，错误和错误可能会带来严重的后果。确保在运行交换机之前进行适当的测试。作为更快的 CPython 项目的一部分，新版本的内部变化比平时更大，范围更广。发布经理 Pablo Galindo Salgado 在[的真实 Python 播客](https://realpython.com/podcasts/rpp/130/)上讲述了这些变化是如何影响发布过程的。

新版本的一个常见问题是，您所依赖的一些第三方包可能在第一天就没有为新版本做好准备。对于 Python 3.11，像 NumPy 和 SciPy 这样的大软件包在发布之前就已经开始为 3.11 [打包轮子了。希望这一次您不必等待您的依赖项为升级做好准备。](https://twitter.com/HenrySchreiner3/status/1558993585198059522)

升级的另一个方面是何时应该开始利用新语法。如果你正在**维护一个支持旧版本 Python 的库**，那么你就不能在你的代码中使用`TaskGroup()`或者像`except*`这样的语法。尽管如此，对于任何使用 Python 3.11 的人来说，你的库会更快。

相反，如果您正在**创建一个应用程序**，在那里您控制它运行的环境，那么一旦您升级了环境，您将能够使用新的特性。

[*Remove ads*](/account/join/)

## 结论

Python 的新版本总是值得庆祝的，也是对来自世界各地的志愿者投入到这门语言中的所有努力的认可。

**在本教程中，您已经看到了新的特性和改进，例如:**

*   更好的**错误消息**,提供更多信息的回溯
*   **更快的代码执行**得益于在**更快的 CPython** 项目中付出的巨大努力
*   简化异步代码工作的任务和异常组
*   几个新的类型特性改进了 Python 的静态类型支持
*   本地的 TOML 支持使用配置文件

您可能无法立即利用所有功能。尽管如此，您应该努力在 Python 3.11 上测试您的代码，以确保您的代码是面向未来的。你注意到速度加快了吗？请在下面的评论中分享你的经历。

**免费下载:** [点击这里下载免费的示例代码](https://realpython.com/bonus/python-311-examples/)，它展示了 Python 3.11 的一些新特性。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解:[**Python 3.11 中很酷的新特性**](/courses/new-features-python-311/)**************
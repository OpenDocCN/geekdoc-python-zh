# Python 3.11 预览版:更好的错误消息

> 原文：<https://realpython.com/python311-error-messages/>

Python 3.11 将于 2022 年 10 月发布。尽管十月份还有几个月的时间，但您已经可以预览一些即将到来的特性，包括 Python 3.11 将如何提供更具可读性和可操作性的错误消息。

**在本教程中，您将:**

*   **在你的电脑上安装** Python 3.11 Alpha，就在你当前安装的 Python 旁边
*   解释 Python 3.11 中**改进的错误消息**，学习**更有效地调试**你的代码
*   将这些改进与 Python 3.10 中的 **PEG 解析器**和**更好的错误消息**联系起来
*   探索提供增强错误消息的第三方包
*   测试 Python 3.11 中较小的改进，包括**新的数学函数**和更多的**可读分数**

Python 3.11 中还有许多其他的改进和特性。跟踪变更日志中的[新内容](https://docs.python.org/3.11/whatsnew/3.11.html)以获得最新列表。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## Python 3.11 Alpha

Python 的新版本在每年 10 月发布。代码是在发布日期前[经过 17 个月的时间](https://www.python.org/dev/peps/pep-0602/)开发和测试的。新功能在 [alpha 阶段](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)实现，持续到五月，大约在最终发布前五个月。

大约每月[一次](https://www.python.org/dev/peps/pep-0664/)在 alpha 阶段，Python 的核心开发者发布一个新的 **alpha 版本**来展示新特性，测试它们，并获得早期反馈。目前 Python 3.11 的最新版本是 **3.11.0 alpha 5** ，发布于 2022 年 2 月 3 日。

**注:**本教程使用的是 Python 3.11 的第五个 alpha 版本。如果您使用更高版本，可能会遇到一些小的差异。然而，你可以期望你在这里学到的大部分内容在 alpha 和 beta 阶段以及 Python 3.11 的最终版本中保持不变。

Python 3.11 的第一个**测试版**计划于 2022 年 5 月 6 日发布。通常，在[测试阶段](https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta)不会添加新功能。相反，特性冻结和发布日期之间的时间被用来测试和固化代码。

[*Remove ads*](/account/join/)

### 很酷的新功能

Python 3.11 的一些最新亮点包括:

*   **增强的错误消息**，这将帮助你更有效地调试你的代码
*   **异常组**，允许程序同时引发和处理多个异常
*   **优化**，承诺使 Python 3.11 比以前的版本明显更快
*   **静态类型**的改进，这将让你[更精确地注释](https://realpython.com/python-type-checking/#annotations)你的代码
*   **TOML 支持**，它允许你使用标准库解析 TOML 文档

Python 3.11 有很多值得期待的地方！要获得全面的概述，请查看 [Python 3.11:供您尝试的酷新功能](https://realpython.com/python311-new-features/)。您还可以在本系列的其他文章中更深入地研究上面列出的一些特性:

*   [Python 3.11 预览版:任务和异常组](https://realpython.com/python311-exception-groups/)
*   [Python 3.11 预览版:TOML 和`tomllib`](https://realpython.com/python311-tomllib/)

在本教程中，您将关注增强的错误报告如何通过让您更有效地调试代码来改善您的开发人员体验。您还将看到 Python 3.11 中其他一些更小的特性。

### 安装

要使用本教程中的代码示例，您需要在系统上安装 Python 3.11 版本。在这一节中，你将学习一些不同的方法来做到这一点:使用 **Docker** ，使用 **pyenv** ，或者从**源**安装。选择最适合您和您的系统的一个。

**注意:** Alpha 版本是即将推出的功能的预览。虽然大多数特性都可以很好地工作，但是您不应该在生产中依赖任何 Python 3.11 alpha 版本，也不应该依赖任何 bug 会带来严重后果的地方。

如果您可以在您的系统上访问 [Docker](https://docs.docker.com/get-docker/) ，那么您可以通过拉取并运行`python:3.11-rc-slim` [Docker 镜像](https://hub.docker.com/_/python)来下载最新版本的 Python 3.11:

```py
$ docker pull python:3.11-rc-slim
Unable to find image 'python:3.11-rc-slim' locally
latest: Pulling from library/python
[...]

$ docker run -it --rm python:3.11-rc-slim
```

这会将您带入 Python 3.11 REPL。查看 Docker 中的[运行 Python 版本，了解如何通过 Docker 使用 Python 的更多信息，包括如何运行脚本。](https://realpython.com/python-versions-docker/#running-python-in-a-docker-container)

[pyenv](https://realpython.com/intro-to-pyenv/) 工具非常适合管理系统上不同版本的 Python，如果你愿意，你可以用它来安装 Python 3.11 Alpha。它有两个不同的版本，一个用于 Windows，一个用于 Linux 和 macOS:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)

**在 Windows 上，你可以使用 [pyenv-win](https://pyenv-win.github.io/pyenv-win/) 。首先更新您的`pyenv`安装:

```py
C:\> pyenv update
:: [Info] ::  Mirror: https://www.python.org/ftp/python
[...]
```

进行更新可以确保您可以安装最新版本的 Python。你也可以[手动更新`pyenv`](https://pyenv-win.github.io/pyenv-win/#how-to-update-pyenv)。

在 Linux 和 macOS 上，可以使用 [pyenv](https://github.com/pyenv/pyenv) 。首先使用 [`pyenv-update`](https://github.com/pyenv/pyenv-update) 插件更新您的`pyenv`安装:

```py
$ pyenv update
Updating /home/realpython/.pyenv...
[...]
```

进行更新可以确保您可以安装最新版本的 Python。如果不想用更新插件，可以[手动更新`pyenv`](https://github.com/pyenv/pyenv#upgrading)。

使用`pyenv install --list`查看 Python 3.11 有哪些版本。然后，安装最新版本:

```py
$ pyenv install 3.11.0a5
Downloading Python-3.11.0a5.tar.xz...
[...]
```

安装可能需要几分钟时间。一旦你的新 alpha 版本安装完毕，你就可以创建一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)来玩它:

*   [*视窗*](#windows-2)
**   [**Linux + macOS**](#linux-macos-2)*

```py
C:\> pyenv local 3.11.0a5
C:\> python -m venv venv
C:\> venv\Scripts\activate.bat
```

使用`pyenv local`激活您的 Python 3.11 版本，然后使用`python -m venv`设置虚拟环境。

```py
$ pyenv virtualenv 3.11.0a5 311_preview
$ pyenv activate 311_preview
```

在 Linux 和 macOS 上，你使用 [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) 插件来设置虚拟环境并激活它。

你也可以从[python.org](https://www.python.org/)的预发布版本中安装 Python。选择[最新预发布](https://www.python.org/download/pre-releases/)，向下滚动到页面底部的*文件*部分。下载并安装与您的系统对应的文件。更多信息参见 [Python 3 安装&设置指南](https://realpython.com/installing-python/)。

在本教程的其余部分，`python3.11`用于指示您应该启动 Python 3.11 可执行文件。具体如何运行取决于您如何安装它。如果你不确定的话，可以参考关于 [Docker](https://realpython.com/python-versions-docker/#running-python-in-a-docker-container) 、 [pyenv](https://realpython.com/intro-to-pyenv/) 、[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)或者[从源码](https://realpython.com/installing-python/)安装的相关教程。

[*Remove ads*](/account/join/)

## Python 3.11 中更好的错误消息

从一开始，Python 就使用自制的、显式基本的 [LL(1)解析器](https://en.wikipedia.org/wiki/LL_parser)，带有单令牌[前瞻](https://en.wikipedia.org/wiki/Parsing#Lookahead)，没有回溯能力。根据 Python 的创造者[吉多·范·罗苏姆](https://twitter.com/gvanrossum)的说法，这是一个有意识的选择:

> Python 的解析器生成器是……蹩脚的，但这反过来又是有意的——它是如此蹩脚，以至于无法阻止我发明难以编写解析器或难以被人类读者消除歧义的语法，而人类读者在 Python 的设计中总是排在第一位。([来源](https://www.artima.com/weblogs/viewpost.jsp?thread=85551))

LL(1)解析器中的限制导致了几个变通办法，使 Python 的语法规则及其解析器生成变得复杂。最终，Guido [建议](https://medium.com/@gvanrossum_83706/peg-parsing-series-de5d41b2ed60)将 Python 的语法更新为具有无限前瞻和回溯的**解析表达式语法(PEG)** 。为 [Python 3.9](https://realpython.com/python39-new-features/#a-more-powerful-python-parser) 创建了一个新的[解析器](https://en.wikipedia.org/wiki/Parsing#Parser)。

[Python 3.10](https://realpython.com/python310-new-features/) 利用新的 [PEG 解析器](https://en.wikipedia.org/wiki/Parsing_expression_grammar)实现了[结构模式匹配](https://realpython.com/python310-new-features/#structural-pattern-matching)和[更好的错误消息](https://realpython.com/python310-new-features/#better-error-messages)。这项工作在 Python 3.11 中继续进行，对 Python 的错误消息进行了更多的改进。

### Python 3.11 面临的挑战

您将很快看到新的和改进的错误消息的例子。不过，首先，您会在 Python 3.10 或更早版本中犯一些错误，这样您就会理解当前的挑战。

假设你有一个数据集，里面有一些关于[著名科学家](https://www.famousscientists.org/)的不一致数据。对于每个科学家，他们的名字、出生日期和死亡日期都被记录下来:

```py
# scientists.py

scientists = [
    {
        "name": {"first": "Grace", "last": "Hopper"},
        "birth": {"year": 1906, "month": 12, "day": 9},
        "death": {"year": 1992, "month": 1, "day": 1},
    },
    {"name": {"first": "Euclid"}},
    {"name": {"first": "Abu Nasr", "last": "Al-Farabi"}, "birth": None},
    {
        "name": {"first": "Srinivasa", "last": "Ramanujan"},
        "birth": {"year": 1887},
        "death": {"month": 4, "day": 26},
    },
    {
        "name": {"first": "Ada", "last": "Lovelace"},
        "birth": {"year": 1815},
        "death": {"year": 1852},
    },
    {
        "name": {"first": "Charles", "last": "Babbage"},
        "birth": {"year": 1791, "month": 12, "day": 26},
        "death": {"year": 1871, "month": 10, "day": 18},
    },
]
```

注意，关于每个科学家的信息都记录在一个嵌套字典中，该字典有`name`、`birth`和`death`字段。但是，有些信息是不完整的。比如欧几里德只有一个名字，Ramanujan 缺少他的死亡年份。

为了处理这些数据，您决定创建一个名为 tuple 的[和一个可以将嵌套字典转换为命名元组的函数:](https://realpython.com/python-namedtuple/)

```py
 1# scientists.py
 2
 3from typing import NamedTuple
 4
 5class Person(NamedTuple):
 6    name: str
 7    life_span: tuple
 8
 9def dict_to_person(info):
10    """Convert a dictionary to a Person object"""
11    return Person(
12        name=f"{info['name']['first']}  {info['name']['last']}",
13        life_span=(info["birth"]["year"], info["death"]["year"]),
14    )
15
16scientists = ...  # As above
```

`Person`将一个人的信息编辑成两个字段:`name`和`life_span`。您可以通过交互运行`scientists.py`来尝试一下:

```py
$ python -i scientists.py
```

使用`-i`装载`scientists.py`并将你留在 [REPL](https://realpython.com/interacting-with-python/#using-the-python-interpreter-interactively) 继续你的探索。例如，您可以转换列出的第一位科学家[格蕾丝·赫柏](https://www.famousscientists.org/grace-murray-hopper/)的信息:

>>>

```py
>>> dict_to_person(scientists[0])
Person(name='Grace Hopper', life_span=(1906, 1992))
```

请注意，您在`dict_to_person()`中没有做任何验证或错误处理，所以当您试图处理一些数据不完整的科学家时，您会遇到问题。本节中的其余示例都是在 Python 3.10 上运行的，显示了一些模糊不清的错误消息。

为了了解处理不完整数据时会发生什么，首先尝试转换关于[欧几里德](https://www.famousscientists.org/euclid/)的信息:

>>>

```py
>>> scientists[1]
{'name': {'first': 'Euclid'}}

>>> dict_to_person(scientists[1])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 12, in dict_to_person
 name=f"{info['name']['first']}  {info['name']['last']}", KeyError: 'last'
```

正确地说，错误消息指出您缺少`last`字段。你需要在回溯或编辑器中查看你的代码，看看`last`是否应该嵌套在`name`中。尽管如此，这种反馈是相当可行的。

接下来，考虑当你处理阿布·纳斯尔·阿尔·法拉比时会发生什么:

>>>

```py
>>> scientists[2]
{'name': {'first': 'Abu Nasr', 'last': 'Al-Farabi'}, 'birth': None}

>>> dict_to_person(scientists[2])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 13, in dict_to_person
 life_span=(info["birth"]["year"], info["death"]["year"]), TypeError: 'NoneType' object is not subscriptable
```

在这种情况下，你被告知某个对象是 [`None`](https://realpython.com/null-in-python/) ，你正试图从中获取某个项目。从相关的代码中，你可以判断出`info`、`info["birth"]`或`info["death"]`中的任何一个一定是`None`，但是你没有办法知道是哪一个，直到你查看你的`scientist`字典。

[斯里尼瓦瑟·拉马努金的](https://www.famousscientists.org/srinivasa-ramanujan/)数据引发了一个类似的问题:

>>>

```py
>>> scientists[3]
{'name': {'first': 'Srinivasa', 'last': 'Ramanujan'},
 'birth': {'year': 1887},
 'death': {'month': 4, 'day': 26}}

>>> dict_to_person(scientists[3])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 13, in dict_to_person
 life_span=(info["birth"]["year"], info["death"]["year"]), KeyError: 'year'
```

在这种情况下，`birth`或`death`字段中缺少`year`。同样，您需要检查实际数据来确定错误。

当您的代码在一条语句中进行多次函数调用时，您可能会遇到一个不同但相似的问题。为了说明这一点，添加一个将一对字典转换为`Person`对象的函数:

```py
# scientists.py

# ...

def convert_pair(first, second):
    """Convert two dictionaries to Person objects"""
    return dict_to_person(first), dict_to_person(second)

# ...
```

注意，`convert_pair()`调用`dict_to_person()`两次，每个科学家一次。你可以用它来查看关于[阿达·洛芙莱斯](https://www.famousscientists.org/ada-lovelace/)和[查尔斯·巴贝奇](https://www.famousscientists.org/charles-babbage/)的信息:

>>>

```py
>>> convert_pair(scientists[4], scientists[5])
(Person(name='Ada Lovelace', life_span=(1815, 1852)),
 Person(name='Charles Babbage', life_span=(1791, 1871)))
```

不出所料，您会得到一组代表科学家的`Person`对象。接下来，看看如果你尝试将阿达·洛芙莱斯和斯里尼瓦瑟·拉马努金配对会发生什么:

>>>

```py
>>> convert_pair(scientists[4], scientists[3])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 19, in convert_pair
 return dict_to_person(first), dict_to_person(second)  File "/home/realpython/scientists.py", line 13, in dict_to_person
 life_span=(info["birth"]["year"], info["death"]["year"]), KeyError: 'year'
```

同样，你注意到`year`不见了，尽管你不能判断它是否与`birth`或`death`相关。另外，还有更多困惑:是第一次还是第二次调用`dict_to_person()`导致了 [`KeyError`](https://realpython.com/python-keyerror/) ？追溯不会告诉你。和以前一样，您需要手动跟踪输入数据，以真正理解错误的原因。

通过这些例子，您已经体验了 Python 3.10 和更早版本中错误消息的一些小麻烦。在这里，输入数据很少，您可以相当快地找出每个错误的原因。通常，您要处理更大的数据集和更复杂的代码，这使得挑战变得更加困难。

这种模糊错误消息的技术原因是 Python 在内部使用源代码中的一行作为程序中每个指令的引用，即使一行可以包含几个指令。这在 Python 3.11 中有所改变。

[*Remove ads*](/account/join/)

### Python 3.11 中的改进

Python 3.11 改进了上一节中的所有错误消息。你可以在 [PEP 657 中查看细节——包括回溯](https://www.python.org/dev/peps/pep-0657/)中的细粒度错误位置。Python 的错误消息，包括导致错误的函数调用，被称为[回溯](https://realpython.com/python-traceback/)。在这一节中，您将了解更精确的错误消息如何帮助您进行调试。

要开始探索，请将`scientists.py`交互式加载到您的 Python 3.11 解释器中:

```py
$ python3.11 --version
Python 3.11.0a5

$ python3.11 -i scientists.py
```

和上一节一样，这将您带入交互式 REPL，其中已经定义了`scientists`、`dict_to_person()`和`convert_pair()`。

只要信息格式良好，您仍然可以创建`Person`对象。但是，请观察如果遇到错误会发生什么:

>>>

```py
>>> dict_to_person(scientists[0])
Person(name='Grace Hopper', life_span=(1906, 1992))

>>> scientists[1]
{'name': {'first': 'Euclid'}}

>>> dict_to_person(scientists[1])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 12, in dict_to_person
 name=f"{info['name']['first']}  {info['name']['last']}", ~~~~~~~~~~~~^^^^^^^^ KeyError: 'last'
```

因为缺少了一个`last`字段，所以仍然得到相同的`KeyError`。但是现在一个可见的标记指向源代码行中的确切位置，所以您可以立即看到`last`是嵌套在`name`中的一个预期字段。

这已经是一种改进，因为您不需要如此仔细地研究错误消息。然而，在原始错误消息含糊不清的情况下，好处变得非常明显。现在，处理阿布·纳斯尔·阿尔·法拉比的数据:

>>>

```py
>>> scientists[2]
{'name': {'first': 'Abu Nasr', 'last': 'Al-Farabi'}, 'birth': None}

>>> dict_to_person(scientists[2])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 13, in dict_to_person
 life_span=(info["birth"]["year"], info["death"]["year"]), ~~~~~~~~~~~~~^^^^^^^^ TypeError: 'NoneType' object is not subscriptable
```

虽然消息`'NoneType' object is not subscriptable`没有告诉您数据结构的哪一部分恰好是`None`，但是标记清楚地表明了这一点。在这里，`info["birth"]`是`None`，所以你无法从中获得`year`物品。

注意如果`info`本身是`None`的区别:

>>>

```py
>>> dict_to_person(None)
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 12, in dict_to_person
 name=f"{info['name']['first']}  {info['name']['last']}", ~~~~^^^^^^^^ TypeError: 'NoneType' object is not subscriptable
```

现在，波浪号(`~`)标记指示`info`是`None`，这在试图读取`name`时会导致错误，正如卡雷茨(`^`)指示的那样。

相同的标记将区分出生和死亡年份:

>>>

```py
>>> scientists[3]
{'name': {'first': 'Srinivasa', 'last': 'Ramanujan'},
 'birth': {'year': 1887}, 'death': {'month': 4, 'day': 26}}

>>> dict_to_person(scientists[3])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 13, in dict_to_person
 life_span=(info["birth"]["year"], info["death"]["year"]), ~~~~~~~~~~~~~^^^^^^^^ KeyError: 'year'
```

不需要研究数据。错误消息和新标记立即告诉您,`death`字段缺少关于年份的信息。

最后，注意当错误发生在嵌套函数调用中时，您将获得什么信息。再次将阿达·洛芙莱斯和斯里尼瓦瑟·拉马努金配对:

>>>

```py
>>> convert_pair(scientists[4], scientists[3])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 19, in convert_pair
 return dict_to_person(first), dict_to_person(second) ^^^^^^^^^^^^^^^^^^^^^^  File "/home/realpython/scientists.py", line 13, in dict_to_person
 life_span=(info["birth"]["year"], info["death"]["year"]), ~~~~~~~~~~~~~^^^^^^^^ KeyError: 'year'
```

最后一条回溯消息仍然指向`death`缺失`year`。但是，请注意，上面的回溯信息现在清楚地显示问题出在第二位科学家 Ramanujan 身上。如此示例所示，回溯标记被添加到回溯中的每一行代码中。

错误消息中增加的清晰度将帮助您在问题出现时快速跟踪问题，以便您可以修复它们。

[*Remove ads*](/account/join/)

### 技术背景

标记一行中的哪一部分导致了错误，看起来似乎是一个快速而明显的改进。为什么 Python 之前没有包含这个？

为了理解技术细节，您应该对 CPython [如何运行您的源代码](https://devguide.python.org/compiler/)有所了解:

1.  您的代码被[标记化](https://docs.python.org/3/library/tokenize.html)。
2.  这些标记被解析成一棵[抽象语法树(AST)](https://docs.python.org/3/library/ast.html) 。
3.  AST 被转换成一个[控制流图(CFG)](https://en.wikipedia.org/wiki/Control-flow_graph) 。
4.  CFG 被转换成[字节码](https://en.wikipedia.org/wiki/Bytecode)。

在运行时，Python 解释器只关心字节码，这是从源代码中删除的几个步骤。

标准库中的几个模块允许您窥视这个过程的幕后。例如，你可以使用 [`dis`](https://docs.python.org/3/library/dis.html) 来反汇编字节码。记住`convert_pair()`的定义:

```py
17def convert_pair(first, second):
18    """Convert two dictionaries to Person objects"""
19    return dict_to_person(first), dict_to_person(second)
```

如上所述，这段代码被标记化、解析，并最终转换成字节码。您可以按如下方式研究这个函数的字节码:

>>>

```py
>>> import dis
>>> dis.dis(convert_pair)
 17           0 RESUME                   0

 19           2 LOAD_GLOBAL              0 (dict_to_person)
 4 LOAD_FAST                0 (first)
 6 PRECALL_FUNCTION         1
 8 CALL                     0
 10 LOAD_GLOBAL              0 (dict_to_person)
 12 LOAD_FAST                1 (second)
 14 PRECALL_FUNCTION         1
 16 CALL                     0
 18 BUILD_TUPLE              2
 20 RETURN_VALUE
```

这里每个指令的含义并不重要。只需注意最左边一列中的数字:17 和 19 是原始源代码的行号。您可以看到第 19 行已经被转换成十个字节码指令。如果这些指令中的任何一个失败了，Python 的早期版本只有足够的信息来断定错误发生在第 19 行的某个地方。

Python 3.11 为每个字节码指令引入了一个新的四位数元组。表示每条指令的**起始行**、**结束行**、**起始列偏移**和**结束列偏移**。您可以通过对代码对象调用新的 [`.co_positions()`](https://docs.python.org/3.11/reference/datamodel.html#codeobject.co_positions) 方法来访问这些元组:

>>>

```py
>>> list(convert_pair.__code__.co_positions())
[(17, 17, 0, 0), (19, 19, 11, 25), (19, 19, 26, 31), (19, 19, 11, 32),
 (19, 19, 11, 32), (19, 19, 34, 48), (19, 19, 49, 55), (19, 19, 34, 56),
 (19, 19, 34, 56), (19, 19, 11, 56), (19, 19, 4, 56)]
```

例如，第一个`LOAD_GLOBAL`指令具有位置`(19, 19, 11, 25)`。查看源代码的第 19 行。从 0 开始计数，你会发现`d`是这一行中的第 11 个字符。您发现列偏移量 11 到 25 对应于文本`dict_to_person`。将所有行号和列偏移量连接到源代码，并将它们与字节码指令相匹配，以创建下表:

| 字节码 | 源代码 |
| --- | --- |
| 简历 |  |
| 加载 _ 全局 | `dict_to_person` |
| 快速加载 | `first` |
| 预调用函数 | `dict_to_person(first)` |
| 呼叫 | `dict_to_person(first)` |
| 加载 _ 全局 | `dict_to_person` |
| 快速加载 | `second` |
| 预调用函数 | `dict_to_person(second)` |
| 呼叫 | `dict_to_person(second)` |
| 构建元组 | `dict_to_person(first), dict_to_person(second)` |
| 返回值 | `return dict_to_person(first), dict_to_person(second)` |

关于行号和列偏移量的新信息允许您的回溯更加详细。您已经看到 Python 3.11 中的内置回溯是如何利用这一点的。随着 Python 3.11 越来越广泛地被使用，一些第三方包可能也会使用这些信息。

**注意:**`.co_positions()`方法不仅仅支持更好、更精确的错误消息。它还可以向其他类型的工具提供信息——比如 [Coverage.py](https://coverage.readthedocs.io/) ,它测量你的代码的哪些部分被执行了。

在运行时，存储这些偏移量会占用 Python 的缓存字节码文件和内存中的一些空间。如果这是一个问题，您可以通过设置`PYTHONNODEBUGRANGES`环境变量或使用`-X no_debug_ranges`命令行选项来删除它们:

```py
$ python3.11 -X no_debug_ranges -i scientists.py
```

自然地，关闭这些会删除回溯中添加的信息:

>>>

```py
>>> dict_to_person(scientists[3])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 13, in dict_to_person
    life_span=(info["birth"]["year"], info["death"]["year"]),
KeyError: 'year'

>>> list(convert_pair.__code__.co_positions())
[(17, None, None, None), (19, None, None, None), (19, None, None, None),
 (19, None, None, None), (19, None, None, None), (19, None, None, None),
 (19, None, None, None), (19, None, None, None), (19, None, None, None),
 (19, None, None, None), (19, None, None, None)]
```

请注意，没有标记显示哪个字段丢失了`year`，并且`.co_positions()`只包含关于行号的信息。标有`None`的字段不存储在磁盘或内存中。

这样做的好处是您的`.pyc`文件更小，代码对象占用的内存空间也相应更少:

*   [*视窗*](#windows-3)
**   [**Linux + macOS**](#linux-macos-3)*

```py
C:\> python3.11 -m py_compile scientists.py
C:\> dir __pycache__\scientists.cpython-311.pyc
[...]
 1 File(s)         1,679 bytes

C:\> python3.11 -X no_debug_ranges -m py_compile scientists.py
C:\> dir __pycache__\scientists.cpython-311.pyc
[...]
 1 File(s)         1,279 bytes
```

```py
$ python3.11 -m py_compile scientists.py
$ wc -c __pycache__/scientists.cpython-311.pyc
1679 __pycache__/scientists.cpython-311.pyc

$ python3.11 -X no_debug_ranges -m py_compile scientists.py
$ wc -c __pycache__/scientists.cpython-311.pyc
1279 __pycache__/scientists.cpython-311.pyc
```

在这种情况下，您可以看到删除额外的信息节省了 400 个字节。通常情况下，这不会影响您的程序。当您在一个[受限环境](https://realpython.com/embedded-python/)中运行时，您只需要考虑关闭这个信息，在这里您确实需要优化您的内存使用。

[*Remove ads*](/account/join/)

### 甚至更好的错误消息使用第三方库

有几个第三方包可以用来增强错误消息，包括 3.11 之前的 Python 版本。这些并不依赖于你到目前为止所了解到的改进。相反，它们是对这些开发的补充，您可以使用它们为自己建立一个更好的调试工作流。

[`better_exceptions`](https://github.com/qix-/better-exceptions) 包将变量值的信息添加到回溯中。要试用它，你首先需要从 [PyPI](https://pypi.org/) 安装它:

```py
$ python -m pip install better_exceptions
```

在你自己的工作中有几种方法可以使用`better_exceptions`。例如，您可以使用环境变量来激活它:

*   [*视窗*](#windows-4)
**   [**Linux + macOS**](#linux-macos-4)*

```py
C:\> set BETTER_EXCEPTIONS=1
C:\> python -i scientists.py
```

```py
$ BETTER_EXCEPTIONS=1 python -i scientists.py
```

通过设置`BETTER_EXCEPTIONS`环境变量，您可以让包格式化您的回溯。关于调用`better_exceptions`的其他方式，可以参考[文档](https://github.com/qix-/better-exceptions)。

既然已经设置了环境变量，请注意如果您调用`convert_pair()`并尝试将欧几里德与他自己配对会发生什么:

>>>

```py
>>> convert_pair(scientists[1], scientists[1])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 19, in convert_pair
 return dict_to_person(first), dict_to_person(second) │              │       │              └ {'name': {'first': 'Euclid'}} │              │       └ <function dict_to_person at 0x7fe2f2c0c040> │              └ {'name': {'first': 'Euclid'}} └ <function dict_to_person at 0x7fe2f2c0c040>  File "/home/realpython/scientists.py", line 12, in dict_to_person
 name=f"{info['name']['first']}  {info['name']['last']}", │                       └ {'name': {'first': 'Euclid'}} └ {'name': {'first': 'Euclid'}} KeyError: 'last'
```

请注意，回溯中的每个变量名都用其对应的值进行了注释。这使您可以快速判断出`KeyError`的发生是因为欧几里德的信息缺少了`last`字段。

**注意:**`better_exceptions`的当前最新版本，版本 0.3.3，用自己的标记替换了 Python 3.11 的标记。换句话说，您在前面几节中学到的箭头不见了。希望未来版本的`better_exceptions`能够展示这两者。

友好的 T2 项目提供了一种不同的追溯方式。它的[原始目的](https://aroberge.github.io/friendly-traceback-docs/docs/html/design.html#original-purpose)是“让初学者更容易理解是什么导致程序产生回溯。”要自己尝试友好，用 [`pip`](https://realpython.com/what-is-pip/) 安装:

```py
$ python -m pip install friendly
```

正如[文档](https://friendly-traceback.github.io/docs/usage.html)所解释的，你可以在不同的环境中使用 Friendly，包括控制台、[笔记本](https://realpython.com/jupyter-notebook-introduction/)和[编辑器](https://realpython.com/python-ides-code-editors-guide/)。一个简单的选择是，在遇到错误后，您可以友好地开始:

>>>

```py
>>> dict_to_person(scientists[2])
Traceback (most recent call last):
  ...
  File "/home/realpython/scientists.py", line 13, in dict_to_person
    life_span=(info["birth"]["year"], info["death"]["year"]),
               ~~~~~~~~~~~~~^^^^^^^^
TypeError: 'NoneType' object is not subscriptable

>>> from friendly import start_console >>> start_console()
```

友好的控制台充当常规 Python REPL 的包装器。您现在可以执行几个新命令，让您更深入地了解最近的错误:

>>>

```py
>>> why()
Subscriptable objects are typically containers from which you can retrieve
item using the notation [...]. Using this notation, you attempted to
retrieve an item from an object of type NoneType which is not allowed.

Note: NoneType means that the object has a value of None.

>>> what()
A TypeError is usually caused by trying to combine two incompatible types
of objects, by calling a function with the wrong type of object, or by
trying to do an operation not allowed on a given type of object.
```

`why()`函数为您提供关于特定错误的信息，而`what()`为您遇到的错误添加一些背景信息，在本例中是一个`TypeError`。也可以试试`where()`、`explain()`、`www()`。

**注意:**友好适用于 Python 3.11。然而，当使用 Python 的开发版本时，您可能会遇到一些库支持方面的问题。请记住，本节中使用的所有库也适用于旧版本的 Python。

最近的一个选择是 [Rich](https://rich.readthedocs.io/) ，它提供了对带注释的[回溯的支持](https://rich.readthedocs.io/en/stable/traceback.html)。要试用 Rich，您应该首先安装它:

```py
$ python -m pip install rich
```

您可以通过安装 Rich 的异常钩子来激活增强的回溯。如果遇到错误，您将得到一个彩色的、格式良好的回溯，其中包含所有可用变量的值的信息，以及发生错误的行的更多上下文信息:

>>>

```py
>>> from rich import traceback
>>> traceback.install(show_locals=True)
<built-in function excepthook>

>>> dict_to_person(scientists[3])
╭───────────────── Traceback (most recent call last) ──────────────────╮
│ <stdin>:1 in <module>                                                │
│ ╭───────────────────────────── locals ─────────────────────────────╮ │
│ │ __annotations__ = {}                                             │ │
│ │    __builtins__ = <module 'builtins' (built-in)>                 │ │
│ │         __doc__ = None                                           │ │
│ │      __loader__ = <_frozen_importlib_external.SourceFileLoader   │ │
│ │                   object at 0x7f933c7b05d0>                      │ │
│ │        __name__ = '__main__'                                     │ │
│ │     __package__ = None                                           │ │
│ │        __spec__ = None                                           │ │
│ │    convert_pair = <function convert_pair at 0x7f933c628680>      │ │
│ │  dict_to_person = <function dict_to_person at 0x7f933c837380>    │ │
│ │      NamedTuple = <function NamedTuple at 0x7f933c615080>        │ │
│ │          Person = <class '__main__.Person'>                      │ │
│ │      scientists = [ ... ]                                        │ │
│ │       traceback = <module 'rich.traceback' from                  │ │
│ │                   '/home/realpython/.pyenv/versions/311_preview/…│ │
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ /home/realpython/scientists.py:13 in dict_to_person                  │
│                                                                      │
│   10 │   """Convert a dictionary to a Person object"""               │
│   11 │   return Person(                                              │
│   12 │   │   name=f"{info['name']['first']} {info['name']['last']}", │
│ ❱ 13 │   │   life_span=(info["birth"]["year"], info["death"]["year"])│
│   14 │   )                                                           │
│   15                                                                 │
│   16                                                                 │
│                                                                      │
│ ╭──────────────────────────── locals ─────────────────────────────╮  │
│ │ info = {                                                        │  │
│ │        │   'name': {                                            │  │
│ │        │   │   'first': 'Srinivasa',                            │  │
│ │        │   │   'last': 'Ramanujan'                              │  │
│ │        │   },                                                   │  │
│ │        │   'birth': {'year': 1887},                             │  │
│ │        │   'death': {'month': 4, 'day': 26}                     │  │
│ │        }                                                        │  │
│ ╰─────────────────────────────────────────────────────────────────╯  │
╰──────────────────────────────────────────────────────────────────────╯
KeyError: 'year'
```

参见[丰富的文档](https://rich.readthedocs.io/en/stable/traceback.html)以获得更多信息和其他输出示例。

还有其他项目试图改进 Python 的回溯和错误信息。其中几个在[用 Python 的异常钩子创建漂亮的回溯](https://martinheinz.dev/blog/66)和[在](https://www.youtube.com/watch?v=JJpv8-w7lG8&t=26m22s) [Python Bytes](https://pythonbytes.fm/episodes/show/270/can-errors-really-be-beautiful) 播客上讨论的中得到了强调。所有这些都适用于 Python 3.11 之前的版本。

[*Remove ads*](/account/join/)

## 其他新功能

在 Python 的每一个新版本中，少数几个特性获得了最多的关注。然而，Python 的大部分发展都是一小步一小步地发生的，通过在这里或那里添加一个功能，改进一些现有的功能，或者修复一个长期存在的错误。

Python 3.11 也不例外。本节展示了 Python 3.11 中一些较小的改进。

### 二的立方根和幂

[`math`](https://realpython.com/python-math-module/) 模块包含基本的数学函数和常数。大多数都是类似的 [C 函数](https://realpython.com/c-for-python-programmers/)的包装器。Python 3.11 给`math`增加了两个新函数:

*   [`cbrt()`](https://docs.python.org/3.11/library/math.html#math.cbrt) 计算立方根。
*   [`exp2()`](https://docs.python.org/3.11/library/math.html#math.exp2) 计算二的幂。

类似于其他的`math`函数，这些是作为相应的 C 函数的包装器实现的。例如，您可以使用`cbrt()`来确认 [Ramanujan 的观察结果](https://blogs.ams.org/mathgradblog/2013/08/15/ramanujans-taxicab-number/)，您可以用两种不同的方式将 [1729](https://en.wikipedia.org/wiki/1729_(number)) 表示为两个立方体的和:

>>>

```py
>>> import math

>>> 1 + 1728
1729
>>> math.cbrt(1)
1.0
>>> math.cbrt(1728)
12.000000000000002

>>> 729 + 1000
1729
>>> math.cbrt(729)
9.000000000000002
>>> math.cbrt(1000)
10.0
```

尽管有一些舍入误差，你注意到 1729 可以写成 *1 + 12* 或者 *9 + 10* 。换句话说，1729 可以表示为两个不同的立方数之和。

在 Python 的早期版本中，可以使用取幂(`**`或`math.pow()`)来计算立方根和 2 的幂。现在，`cbrt()`允许你在没有明确指定`1/3`的情况下找到立方根。同样，`exp2()`给你一个计算 2 的幂的捷径。在 Python 3.11 中，进行这些计算有几种选择:

>>>

```py
>>> math.cbrt(729)
9.000000000000002
>>> 729**(1/3)
8.999999999999998
>>> math.pow(729, 1/3)
8.999999999999998

>>> math.exp2(16)
65536.0
>>> 2**16
65536
>>> math.pow(2, 16)
65536.0
```

注意，由于[浮点表示错误](https://realpython.com/python-numbers/#make-python-lie-to-you)，不同的方法可能会得到稍微不同的结果。特别是在 Windows 上，`exp2()`似乎比`math.pow()`更不准确[。目前坚持旧的方法应该对你有好处。](https://bugs.python.org/issue45917#msg407321)

当计算负数的立方根时，你也会得到不同的结果:

>>>

```py
>>> math.cbrt(-8)
-2.0

>>> (-8)**(1/3)
(1.0000000000000002+1.7320508075688772j)

>>> math.pow(-8, 1/3)
Traceback (most recent call last):
  ...
ValueError: math domain error
```

任何数字都有三个[立方根](https://en.wikipedia.org/wiki/Cube_root)。对于实数，这些根中的一个将是实数，而另外两个根将是一对[复数](https://realpython.com/python-complex-numbers/#extracting-the-root-of-a-complex-number)。`cbrt()`返回主立方根，包括负数。取幂运算返回一个复数立方根，而`math.pow()`只处理整数指数的负数。

### 分数中的下划线

从 Python 3.6 开始，Python 就支持在文字数字中添加下划线。通常，您使用下划线将大量数字分组，以使它们更具可读性:

>>>

```py
>>> number = 60481729
>>> readable_number = 60_481_729
```

在这个例子中，`number`是大约 600 万还是 6000 万可能不是很明显。通过将数字分成三组，很明显`readable_number`大约是六千万。

请注意，这个特性是一种便利，可以让您的源代码更具可读性。下划线对计算或 Python 表示数字的方式没有影响，尽管您可以使用 [f 字符串](https://realpython.com/python-f-strings/)用下划线格式化数字:

>>>

```py
>>> number == readable_number
True

>>> readable_number
60481729

>>> f"{number:_}"
'60_481_729'
```

注意 Python 并不关心你把下划线放在哪里。你应该小心，不要让它们最终增加混乱:

>>>

```py
>>> confusing_number = 6_048_1729
```

`confusing_number`的价值也大约是六千万，但是你很容易认为它是六百万。如果您使用下划线来分隔千位，那么您应该知道在世界范围内有不同的[惯例](https://en.wikipedia.org/wiki/Decimal_separator#Examples_of_use)来对数字进行分组。

Python 可以用 [`fractions`](https://docs.python.org/3/library/fractions.html) 模块准确地表示[有理数](https://realpython.com/python-fractions/)。例如，您可以使用字符串文字指定 1729 的分数 6048，如下所示:

>>>

```py
>>> from fractions import Fraction
>>> print(Fraction("6048/1729"))
864/247
```

出于某种原因，在 Python 3.11 之前，下划线不允许出现在`Fraction`字符串参数中。现在，您也可以在指定分数时使用下划线:

>>>

```py
>>> print(Fraction("6_048/1_729"))
864/247
```

和其他数字一样，Python 不关心下划线放在哪里。使用下划线来提高代码的可读性取决于您。

[*Remove ads*](/account/join/)

### 对象的灵活调用

[`operator`](https://docs.python.org/3/library/operator.html) 模块包含在使用 Python 的一些[函数式编程](https://realpython.com/python-functional-programming/)特性时有用的函数。举个简单的例子，您可以使用`operator.abs`到[按照绝对值对数字](https://realpython.com/python-sort/) -3、-2、-1、0、1、2 和 3 进行排序:

>>>

```py
>>> sorted([-3, -2, -1, 0, 1, 2, 3], key=operator.abs)
[0, -1, 1, -2, 2, -3, 3]
```

通过指定`key`，首先通过计算每个项目的[绝对值](https://realpython.com/python-absolute-value/)来对列表进行排序。

Python 3.11 在`operator`上增加了`call()`。你可以使用`call()`来调用函数。例如，您可以按如下方式编写前面的示例:

>>>

```py
>>> operator.call(sorted, [-3, -2, -1, 0, 1, 2, 3], key=operator.abs)
[0, -1, 1, -2, 2, -3, 3]
```

一般来说，像这样使用`call()`是没有用的。你应该坚持直接调用函数。一个可能的例外是当你调用被变量引用的函数时，因为添加`call()`可以使你的代码更加明确。

下一个例子展示了`call()`的一个更好的用例。你实现了一个可以用[挪威语](https://en.wikipedia.org/wiki/Norwegian_language)进行基本计算的计算器。它使用 [`parse`](https://realpython.com/python-packages/#parse-for-matching-strings) 库解析文本字符串，然后使用`call()`执行正确的算术运算:

```py
import operator
import parse

OPERATIONS = {
    "pluss": operator.add,        # Addition
    "minus": operator.sub,        # Subtraction
    "ganger": operator.mul,       # Multiplication
    "delt på": operator.truediv,  # Division
}
EXPRESSION = parse.compile("{operand1:g}  {operation}  {operand2:g}")

def calculate(text):
    if (ops := EXPRESSION.parse(text)) and ops["operation"] in OPERATIONS:
        operation = OPERATIONS[ops["operation"]]
 return operator.call(operation, ops["operand1"], ops["operand2"])
```

`OPERATIONS`是一个映射，指定您的计算器理解哪些命令，并定义它们对应的功能。`EXPRESSION`是一个模板，定义了将要解析的文本字符串的种类。`calculate()`解析你的字符串，如果可以识别，就调用相关的操作。

**注意:**你可以返回`operation(ops["operand1"], ops["operand2"])`而不是使用`operator.call()`。

您可以使用`calculate()`进行挪威算术运算，如下所示:

>>>

```py
>>> calculate("3 pluss 11")
14.0

>>> calculate("3 delt på 11")
0.2727272727272727
```

你的计算器算出 3 加 11 等于 14，而 3 除以 11 大约是 0.27。

`operator.call()`类似于 [`apply()`](https://docs.python.org/2.7/library/functions.html#apply) ，在 Python 2 中可用，随着参数[解包](https://realpython.com/python-kwargs-and-args/)的引入而失宠。`call()`让你在调用函数时更加灵活。然而，正如这些例子所示，您通常最好直接调用函数。

## 结论

现在你已经看到了 Python 3.11 在 2022 年 10 月发布时将会带来什么。您已经了解了它的一些新特性，并探索了如何利用这些改进。

**特别是，你已经:**

*   **在你的电脑上安装了** Python 3.11 Alpha
*   在 Python 3.11 中使用了增强的错误回溯功能，并用它们来更有效地调试你的代码
*   了解 Python 3.11 如何构建在 Python 3.10 的 **PEG 解析器**和**之上，更好的错误消息**
*   探索了第三方库如何让你的调试更加高效
*   尝试了 Python 3.11 中一些较小的改进，包括**新的数学函数**和更多的**可读分数**

试试 Python 3.11 中更好的错误消息！你怎么看待这些增强的回溯？在下面评论分享你的经验。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。*******************
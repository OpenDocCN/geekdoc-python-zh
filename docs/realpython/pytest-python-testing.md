# 使用 Pytest 进行有效的 Python 测试

> 原文：<https://realpython.com/pytest-python-testing/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 pytest**](/courses/testing-your-code-with-pytest/) 测试你的代码

测试你的代码会带来各种各样的好处。它增加了您对代码如您所期望的那样运行的信心，并确保对代码的更改不会导致回归。编写和维护测试是一项艰苦的工作，所以您应该利用您可以使用的所有工具，尽可能地使它变得轻松。 [`pytest`](https://docs.pytest.org/) 是你可以用来提高测试效率的最好工具之一。

在本教程中，您将学习:

*   **福利** `pytest`提供什么
*   如何确保你的测试是无状态的
*   如何让重复测试更容易理解
*   如何按名称或自定义组运行测试的子集
*   如何创建和维护**可重用的**测试工具

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 如何安装`pytest`

为了跟随本教程中的一些例子，您需要安装`pytest`。和大多数 [Python 包](https://realpython.com/python-modules-packages/)一样，`pytest`在 [PyPI](https://realpython.com/pypi-publish-python-package/) 上可用。你可以使用 [`pip`](https://realpython.com/what-is-pip/) 在[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中安装它:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python -m venv venv
PS> .\venv\Scripts\activate
(venv) PS> python -m pip install pytest
```

```py
$ python -m venv venv
$ source venv/bin/activate
(venv) $ python -m pip install pytest
```

`pytest`命令现在可以在您的安装环境中使用。

[*Remove ads*](/account/join/)

## 是什么让`pytest`如此有用？

如果你以前为你的 Python 代码编写过单元[测试](https://realpython.com/python-testing/)，那么你可能用过 Python 内置的 **`unittest`** 模块。`unittest`提供了构建测试套件的坚实基础，但是它有一些缺点。

许多第三方测试框架试图解决一些关于`unittest`的问题，而`pytest`已经被证明是最受欢迎的一个。`pytest`是一个功能丰富、基于插件的生态系统，用于测试你的 Python 代码。

如果你还没有享受过使用`pytest`的乐趣，那你就等着享受吧！它的理念和特性将使您的测试体验更加高效和愉快。使用`pytest`，普通任务需要更少的代码，高级任务可以通过各种节省时间的命令和插件来完成。它甚至可以运行你现有的测试，包括那些用`unittest`编写的测试。

与大多数框架一样，当您第一次开始使用`pytest`时，一些有意义的开发模式可能会随着您的测试套件的增长而开始带来痛苦。本教程将帮助你理解`pytest`提供的一些工具，这些工具可以在测试扩展时保持测试的效率和效果。

### 更少的样板文件

大多数功能测试遵循排列-动作-断言模型:

1.  安排测试的条件
2.  通过调用一些函数或方法来执行
3.  断言某个结束条件为真

测试框架通常与测试的[断言](https://realpython.com/python-assert-statement/)挂钩，这样它们就可以在断言失败时提供信息。例如，`unittest`提供了许多现成的有用的断言实用程序。然而，即使是一小组测试也需要相当数量的[样板代码](https://en.wikipedia.org/wiki/Boilerplate_code)。

想象一下，你想写一个测试套件来确保`unittest`在你的项目中正常工作。您可能希望编写一个总是通过的测试和一个总是失败的测试:

```py
# test_with_unittest.py

from unittest import TestCase

class TryTesting(TestCase):
    def test_always_passes(self):
        self.assertTrue(True)

    def test_always_fails(self):
        self.assertTrue(False)
```

然后，您可以使用`unittest`的`discover`选项从命令行运行这些测试:

```py
(venv) $ python -m unittest discover
F.
======================================================================
FAIL: test_always_fails (test_with_unittest.TryTesting)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "...\effective-python-testing-with-pytest\test_with_unittest.py",
 line 10, in test_always_fails
 self.assertTrue(False)
AssertionError: False is not true

----------------------------------------------------------------------

Ran 2 tests in 0.006s

FAILED (failures=1)
```

不出所料，一个测试通过，一个失败。你已经证明了`unittest`是有效的，但是看看你必须做什么:

1.  从`unittest`导入`TestCase`类
2.  创建`TryTesting`，一个`TestCase`的[子类](https://realpython.com/python3-object-oriented-programming/)
3.  在`TryTesting`中为每个测试编写一个方法
4.  使用来自`unittest.TestCase`的`self.assert*`方法之一进行断言

这需要编写大量的代码，因为这是任何*测试所需的最少代码，所以你最终会一遍又一遍地编写相同的代码。`pytest`通过允许您直接使用普通函数和 Python 的`assert`关键字，简化了工作流程:*

```py
# test_with_pytest.py

def test_always_passes():
    assert True

def test_always_fails():
    assert False
```

就是这样。您不必处理任何导入或类。您所需要做的就是包含一个带有前缀`test_`的函数。因为您可以使用`assert`关键字，所以您也不需要学习或记住`unittest`中所有不同的`self.assert*`方法。如果你能写出一个你期望评价给`True`的表达式，然后`pytest`会为你测试。

`pytest`不仅消除了许多样板文件，还为您提供了更详细、更易读的输出。

### 漂亮的输出

您可以使用项目顶层文件夹中的`pytest`命令来运行您的测试套件:

```py
(venv) $ pytest
============================= test session starts =============================
platform win32 -- Python 3.10.5, pytest-7.1.2, pluggy-1.0.0
rootdir: ...\effective-python-testing-with-pytest
collected 4 items

test_with_pytest.py .F                                                   [ 50%]
test_with_unittest.py F.                                                 [100%]

================================== FAILURES ===================================
______________________________ test_always_fails ______________________________

 def test_always_fails():
>       assert False
E       assert False

test_with_pytest.py:7: AssertionError
________________________ TryTesting.test_always_fails _________________________

self = <test_with_unittest.TryTesting testMethod=test_always_fails>

 def test_always_fails(self):
>       self.assertTrue(False)
E       AssertionError: False is not true

test_with_unittest.py:10: AssertionError
=========================== short test summary info ===========================
FAILED test_with_pytest.py::test_always_fails - assert False
FAILED test_with_unittest.py::TryTesting::test_always_fails - AssertionError:...

========================= 2 failed, 2 passed in 0.20s =========================
```

`pytest`显示的测试结果与`unittest`不同，`test_with_unittest.py`文件也被自动包含在内。报告显示:

1.  系统状态，包括 Python、`pytest`的版本，以及您已经安装的任何插件
2.  `rootdir`，或者在其中搜索配置和测试的目录
3.  跑步者发现的测试次数

这些项目出现在输出的第一部分:

```py
============================= test session starts =============================
platform win32 -- Python 3.10.5, pytest-7.1.2, pluggy-1.0.0
rootdir: ...\effective-python-testing-with-pytest
collected 4 items
```

然后，输出使用类似于`unittest`的语法指示每个测试的状态:

*   **一个圆点(`.` )** 表示测试通过。
*   **`F`**表示测试失败。
*   **An `E`** 表示测试引发了意外异常。

特殊字符显示在名称旁边，测试套件的整体进度显示在右侧:

```py
test_with_pytest.py .F                                                   [ 50%]
test_with_unittest.py F.                                                 [100%]
```

对于失败的测试，报告给出了失败的详细分类。在这个例子中，测试失败是因为`assert False`总是失败:

```py
================================== FAILURES ===================================
______________________________ test_always_fails ______________________________

 def test_always_fails():
>       assert False
E       assert False

test_with_pytest.py:7: AssertionError
________________________ TryTesting.test_always_fails _________________________

self = <test_with_unittest.TryTesting testMethod=test_always_fails>

 def test_always_fails(self):
>       self.assertTrue(False)
E       AssertionError: False is not true

test_with_unittest.py:10: AssertionError
```

这个额外的输出在调试时会非常方便。最后，报告给出了测试套件的总体状态报告:

```py
=========================== short test summary info ===========================
FAILED test_with_pytest.py::test_always_fails - assert False
FAILED test_with_unittest.py::TryTesting::test_always_fails - AssertionError:...

========================= 2 failed, 2 passed in 0.20s =========================
```

与 unittest 相比，`pytest`输出的信息量更大，可读性更强。

在下一节中，您将仔细看看`pytest`如何利用现有的`assert`关键字。

[*Remove ads*](/account/join/)

### 少学

能够使用 [`assert`](https://realpython.com/python-assert-statement/) 关键字也很强大。如果你以前用过，那就没什么新东西可学了。这里有几个断言示例，因此您可以了解可以进行的测试类型:

```py
# test_assert_examples.py

def test_uppercase():
    assert "loud noises".upper() == "LOUD NOISES"

def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]

def test_some_primes():
    assert 37 in {
        num
        for num in range(2, 50)
        if not any(num % div == 0 for div in range(2, num))
    }
```

它们看起来非常像普通的 Python 函数。所有这些都使得`pytest`的学习曲线比`unittest`要浅，因为你不需要学习新的构造来开始。

注意，每个测试都很小，并且是独立的。这很常见——你会看到很长的函数名，而在一个函数中不会发生很多事情。这主要是为了让你的测试相互隔离，这样如果有什么东西坏了，你就知道问题出在哪里了。一个很好的副作用是输出中的标签更好。

要查看与主项目一起创建测试套件的项目示例，请查看[使用 TDD 在 Python 中构建哈希表](https://realpython.com/python-hash-table/)教程。此外，当[为下一次面试](https://realpython.com/python-practice-problems/)或[解析 CSV 文件](https://realpython.com/python-interview-problem-parsing-csv-files/)做准备时，你可以解决 Python 实践问题，亲自尝试测试驱动开发。

在下一节中，您将检查 fixtures，这是一个很好的 pytest 特性，可以帮助您管理测试输入值。

### 更易于管理状态和依赖关系

你的测试将经常依赖于数据类型或者模拟你的代码可能遇到的对象的[测试加倍](https://en.wikipedia.org/wiki/Test_double)，比如[字典](https://realpython.com/python-dicts/)或者 [JSON](https://realpython.com/python-json/) 文件。

使用`unittest`，您可以将这些依赖项提取到`.setUp()`和`.tearDown()`方法中，这样类中的每个测试都可以使用它们。使用这些特殊的方法是好的，但是随着你的测试类变得越来越大，你可能会无意中使测试的依赖完全**隐含**。换句话说，通过孤立地看众多测试中的一个，你可能不会立即看到它依赖于其他东西。

随着时间的推移，隐式依赖可能会导致复杂的代码混乱，您必须展开代码才能理解您的测试。测试应该有助于让你的代码更容易理解。如果测试本身难以理解，那么你可能会有麻烦！

`pytest`采取不同的方法。它将您引向**显式**依赖声明，由于[夹具](https://docs.pytest.org/en/latest/fixture.html)的可用性，这些依赖声明仍然是可重用的。`pytest`fixture 是可以为测试套件创建数据、测试双精度或初始化系统状态的功能。任何想要使用 fixture 的测试都必须显式地使用这个 fixture 函数作为测试函数的参数，因此依赖关系总是在前面声明:

```py
# fixture_demo.py

import pytest

@pytest.fixture
def example_fixture():
    return 1

def test_with_fixture(example_fixture):
    assert example_fixture == 1
```

查看测试函数，您可以立即看出它依赖于一个 fixture，而不需要检查整个文件中的 fixture 定义。

**注意:**你通常想要把你的测试放在你的项目的根层的一个名为`tests`的文件夹中。

有关构建 Python 应用程序的更多信息，请查看关于该主题的视频课程。

固定设备也可以利用其他固定设备，同样是通过将它们显式声明为依赖关系。这意味着，随着时间的推移，你的设备会变得庞大和模块化。尽管将夹具插入到其他夹具中的能力提供了巨大的灵活性，但是随着测试套件的增长，这也使得管理依赖关系变得更加困难。

在本教程的后面，您将学习更多关于夹具的知识，并尝试一些应对这些挑战的技巧。

### 易于过滤的测试

随着您的测试套件的增长，您可能会发现您想要对一个特性只运行几个测试，并保存整个套件以备后用。`pytest`提供了几种方法:

*   **基于名称的过滤**:您可以将`pytest`限制为只运行那些完全限定名与特定表达式匹配的测试。您可以使用`-k`参数来实现这一点。
*   **目录范围**:默认情况下，`pytest`将只运行那些在当前目录下的测试。
*   **测试分类** : `pytest`可以包含或排除您定义的特定类别的测试。您可以使用`-m`参数来实现这一点。

特别是测试分类是一个微妙而强大的工具。`pytest`使您能够为您喜欢的任何测试创建**标记**，或自定义标签。一个测试可能有多个标签，您可以使用它们对要运行的测试进行粒度控制。在本教程的后面，你将看到一个关于[如何使用`pytest`标记](#marks-categorizing-tests)的例子，并学习如何在大型测试套件中使用它们。

[*Remove ads*](/account/join/)

### 允许测试参数化

当您测试处理数据或执行一般转换的函数时，您会发现自己编写了许多类似的测试。它们可能只是在被测试代码的[输入或输出](https://realpython.com/python-input-output/)上有所不同。这需要复制测试代码，这样做有时会掩盖您试图测试的行为。

提供了一种将几个测试集合成一个的方法，但是它们不会在结果报告中显示为单独的测试。如果一个测试失败了，而其余的通过了，那么整个组仍然会返回一个失败的结果。`pytest`提供自己的解决方案，每个测试都可以独立通过或失败。在本教程的后面，你会看到[如何用`pytest`参数化测试](#parametrization-combining-tests)。

### 拥有基于插件的架构

`pytest`最漂亮的特性之一是它对定制和新特性的开放性。几乎程序的每一部分都可以被破解和修改。结果，`pytest`用户开发了一个丰富的有用插件生态系统。

虽然有些`pytest`插件专注于特定的框架，比如 [Django](https://www.djangoproject.com/) ，但是其他的插件适用于大多数测试套件。在本教程的后面你会看到一些特定插件的细节。

## 夹具:管理状态和依赖关系

夹具是为你的测试提供数据、测试副本或状态设置的一种方式。Fixtures 是可以返回大量值的函数。每个依赖于 fixture 的测试必须明确地接受 fixture 作为参数。

### 何时创建夹具

在本节中，您将模拟一个典型的[测试驱动开发](https://realpython.com/courses/test-driven-development-pytest/) (TDD)工作流。

假设您正在编写一个函数`format_data_for_display()`，来处理 API 端点返回的数据。该数据表示一个人员列表，每个人都有一个名、姓和职务。该函数应该输出一个字符串列表，其中包括每个人的全名(他们的`given_name`后跟他们的`family_name`)、一个冒号和他们的`title`:

```py
# format_data.py

def format_data_for_display(people):
    ...  # Implement this!
```

在好的 TDD 方式中，您将希望首先为它编写一个测试。为此，您可以编写以下代码:

```py
# test_format_data.py

def test_format_data_for_display():
    people = [
        {
            "given_name": "Alfonsa",
            "family_name": "Ruiz",
            "title": "Senior Software Engineer",
        },
        {
            "given_name": "Sayid",
            "family_name": "Khan",
            "title": "Project Manager",
        },
    ]

    assert format_data_for_display(people) == [
        "Alfonsa Ruiz: Senior Software Engineer",
        "Sayid Khan: Project Manager",
    ]
```

在编写这个测试时，您会想到可能需要编写另一个函数来将数据转换成逗号分隔的值，以便在 [Excel](https://realpython.com/openpyxl-excel-spreadsheets-python/) 中使用:

```py
# format_data.py

def format_data_for_display(people):
    ...  # Implement this!

def format_data_for_excel(people):
    ... # Implement this!
```

你的待办事项越来越多！那就好！TDD 的优势之一是它帮助你提前计划好工作。对`format_data_for_excel()`函数的测试看起来与`format_data_for_display()`函数非常相似:

```py
# test_format_data.py

def test_format_data_for_display():
    # ...

def test_format_data_for_excel():
    people = [
        {
            "given_name": "Alfonsa",
            "family_name": "Ruiz",
            "title": "Senior Software Engineer",
        },
        {
            "given_name": "Sayid",
            "family_name": "Khan",
            "title": "Project Manager",
        },
    ]

    assert format_data_for_excel(people) == """given,family,title
Alfonsa,Ruiz,Senior Software Engineer
Sayid,Khan,Project Manager
"""
```

值得注意的是，两个测试都必须重复`people`变量的定义，这相当于几行代码。

如果你发现自己写了几个测试，都使用了相同的底层测试数据，那么你的未来可能会有一个夹具。您可以将重复的数据放入用`@pytest.fixture`修饰的单个函数中，以表明该函数是一个`pytest` fixture:

```py
# test_format_data.py

import pytest

@pytest.fixture
def example_people_data():
    return [
        {
            "given_name": "Alfonsa",
            "family_name": "Ruiz",
            "title": "Senior Software Engineer",
        },
        {
            "given_name": "Sayid",
            "family_name": "Khan",
            "title": "Project Manager",
        },
    ]

# ...
```

您可以通过将函数引用作为参数添加到测试中来使用 fixture。注意，您没有调用 fixture 函数。会处理好的。您将能够使用 fixture 函数的返回值作为 fixture 函数的名称:

```py
# test_format_data.py

# ...

def test_format_data_for_display(example_people_data):
 assert format_data_for_display(example_people_data) == [        "Alfonsa Ruiz: Senior Software Engineer",
        "Sayid Khan: Project Manager",
    ]

def test_format_data_for_excel(example_people_data):
 assert format_data_for_excel(example_people_data) == """given,family,title Alfonsa,Ruiz,Senior Software Engineer
Sayid,Khan,Project Manager
"""
```

每个测试现在都明显缩短了，但仍然有一条清晰的路径返回到它所依赖的数据。一定要给你的固定装置起一个具体的名字。这样，您可以在将来编写新测试时快速确定是否要使用它！

当你第一次发现固定物的力量时，总是使用它们是很诱人的，但是和所有事情一样，需要保持平衡。

[*Remove ads*](/account/join/)

### 何时避开夹具

Fixtures 对于提取您在多个测试中使用的数据或对象非常有用。然而，对于要求数据有细微变化的测试，它们并不总是那么好。在测试套件中乱放固定装置并不比乱放普通数据或对象好。由于增加了间接层，情况可能会更糟。

与大多数抽象一样，需要一些实践和思考来找到夹具使用的正确级别。

尽管如此，夹具很可能是测试套件中不可或缺的一部分。随着项目范围的扩大，规模的挑战开始显现。任何一种工具面临的挑战之一是如何处理大规模使用，幸运的是，`pytest`有一系列有用的功能可以帮助您管理增长带来的复杂性。

### 如何大规模使用夹具

随着您从测试中提取更多的装置，您可能会看到一些装置可以从进一步的抽象中受益。在`pytest`中，夹具是**模块化**。模块化意味着夹具可以[导入](https://realpython.com/python-import/)，可以导入其他模块，并且可以依赖和导入其他夹具。所有这些都允许您为您的用例构建一个合适的夹具抽象。

例如，您可能会发现两个独立文件中的装置，或[模块](https://realpython.com/python-modules-packages/)，共享一个公共的依赖关系。在这种情况下，您可以将 fixture 从测试模块移动到更一般的 fixture 相关模块中。这样，您可以将它们重新导入到任何需要它们的测试模块中。当您发现自己在整个项目中反复使用夹具时，这是一个很好的方法。

如果你想让一个 fixture 对你的整个项目可用而不需要导入它，一个叫做 [`conftest.py`](https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files) 的特殊配置模块将允许你这样做。

`pytest`在每个目录中寻找一个`conftest.py`模块。如果您将通用夹具添加到`conftest.py`模块，那么您将能够在整个模块的父目录和任何子目录中使用该夹具，而不必导入它。这是放置你最常用的灯具的好地方。

fixtures 和`conftest.py`的另一个有趣的用例是保护对资源的访问。假设您已经为处理 [API 调用](https://realpython.com/api-integration-in-python/)的代码编写了一个测试套件。您希望确保测试套件不会进行任何真正的网络调用，即使有人不小心编写了这样的测试。

`pytest`提供了一个 [`monkeypatch`](https://docs.pytest.org/en/latest/monkeypatch.html) 夹具来替换价值观和行为，你可以用它来产生很大的效果:

```py
# conftest.py

import pytest
import requests

@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    def stunted_get():
        raise RuntimeError("Network access not allowed during testing!")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_get())
```

通过将`disable_network_calls()`放在`conftest.py`中并添加`autouse=True`选项，您可以确保在套件的每个测试中禁用网络调用。任何执行代码调用`requests.get()`的测试都将引发一个`RuntimeError`，表明一个意外的网络调用将会发生。

您的测试套件在数量上正在增长，这给了您很大的信心来进行更改，而不是意外地破坏。也就是说，随着测试套件的增长，它可能会花费很长时间。即使不需要那么长时间，也许你正在关注一些核心行为，这些行为会慢慢渗透并破坏大多数测试。在这些情况下，您可能希望将测试运行程序限制在特定的测试类别中。

## 标记:分类测试

在任何大型测试套件中，当你试图快速迭代一个新特性时，最好避免运行所有的测试。除了默认行为`pytest`运行当前工作目录中的所有测试，或者[过滤](https://docs.pytest.org/en/7.1.x/example/markers.html#using-k-expr-to-select-tests-based-on-their-name)功能，您可以利用**标记**。

使您能够为您的测试定义类别，并在您运行套件时提供包括或排除类别的选项。您可以用任意数量的类别来标记测试。

标记测试对于按子系统或依赖项对测试进行分类很有用。例如，如果您的一些测试需要访问数据库，那么您可以为它们创建一个`@pytest.mark.database_access`标记。

**专家提示**:因为你可以给你的标记取任何你想要的名字，所以很容易打错或记错标记的名字。`pytest`将警告您测试输出中无法识别的标记。

您可以对`pytest`命令使用`--strict-markers`标志，以确保您的测试中的所有标记都注册到您的`pytest`配置文件`pytest.ini`中。它会阻止你运行你的测试，直到你注册任何未知的标记。

有关注册商标的更多信息，请查看 [`pytest`文档](https://docs.pytest.org/en/latest/mark.html#registering-marks)。

到了运行测试的时候，您仍然可以使用`pytest`命令默认运行所有的测试。如果您只想运行那些需要数据库访问的测试，那么您可以使用`pytest -m database_access`。要运行除了那些需要数据库访问的测试之外的所有测试*，您可以使用`pytest -m "not database_access"`。您甚至可以使用一个`autouse`夹具来限制对那些标有`database_access`的测试的数据库访问。*

一些插件通过添加自己的防护来扩展标记的功能。例如， [`pytest-django`](https://pytest-django.readthedocs.io/en/latest/) 插件提供了一个`django_db`标记。任何没有这个标记的试图访问数据库的测试都将失败。试图访问数据库的第一个测试将触发 Django 的测试数据库的创建。

添加`django_db`标记的要求促使你明确地陈述你的依赖关系。毕竟这就是`pytest`哲学！这也意味着你可以更快地运行不依赖于数据库的测试，因为`pytest -m "not django_db"`会阻止测试触发数据库创建。节省的时间真的越来越多，特别是如果你勤于频繁地运行测试。

`pytest`提供了一些现成的标志:

*   **`skip`** 无条件跳过一次考试。
*   **`skipif`** 如果传递给它的表达式计算结果为`True`，则跳过测试。
*   **`xfail`** 表示测试预计会失败，因此如果测试*和*失败，整个套件仍然会导致通过状态。
*   **`parametrize`** 用不同的值作为自变量创建测试的多个变量。你很快会了解到更多关于这个标记的信息。

您可以通过运行`pytest --markers`来查看`pytest`知道的所有标记的列表。

关于参数化的话题，这是接下来要讲的。

[*Remove ads*](/account/join/)

## 参数化:组合测试

在本教程的前面，您已经看到了如何使用`pytest`fixture 通过提取公共依赖来减少代码重复。当您有几个输入和预期输出略有不同的测试时，Fixtures 就没那么有用了。在这些情况下，您可以用 [**参数化**](http://doc.pytest.org/en/latest/example/parametrize.html) 一个单一的测试定义，然后`pytest`会用您指定的参数为您创建测试的变体。

假设你写了一个函数来判断一个字符串是否是回文。最初的一组测试可能如下所示:

```py
def test_is_palindrome_empty_string():
    assert is_palindrome("")

def test_is_palindrome_single_character():
    assert is_palindrome("a")

def test_is_palindrome_mixed_casing():
    assert is_palindrome("Bob")

def test_is_palindrome_with_spaces():
    assert is_palindrome("Never odd or even")

def test_is_palindrome_with_punctuation():
    assert is_palindrome("Do geese see God?")

def test_is_palindrome_not_palindrome():
    assert not is_palindrome("abc")

def test_is_palindrome_not_quite():
    assert not is_palindrome("abab")
```

除了最后两个测试，所有这些测试都具有相同的形状:

```py
def test_is_palindrome_<in some situation>():
    assert is_palindrome("<some string>")
```

这开始有点像样板文件了。到目前为止，它已经帮你摆脱了样板文件，现在也不会让你失望。您可以使用`@pytest.mark.parametrize()`用不同的值填充这个形状，从而大大减少您的测试代码:

```py
@pytest.mark.parametrize("palindrome", [
    "",
    "a",
    "Bob",
    "Never odd or even",
    "Do geese see God?",
])
def test_is_palindrome(palindrome):
    assert is_palindrome(palindrome)

@pytest.mark.parametrize("non_palindrome", [
    "abc",
    "abab",
])
def test_is_palindrome_not_palindrome(non_palindrome):
    assert not is_palindrome(non_palindrome)
```

`parametrize()`的第一个参数是一个逗号分隔的参数名称字符串。正如您在本例中看到的，您不必提供多个名称。第二个参数是代表参数值的[元组](https://realpython.com/python-lists-tuples/#python-tuples)或单个值的[列表](https://realpython.com/python-lists-tuples/#python-lists)。您可以进一步进行参数化，将所有测试合并成一个:

```py
@pytest.mark.parametrize("maybe_palindrome, expected_result", [
    ("", True),
    ("a", True),
    ("Bob", True),
    ("Never odd or even", True),
    ("Do geese see God?", True),
    ("abc", False),
    ("abab", False),
])
def test_is_palindrome(maybe_palindrome, expected_result):
    assert is_palindrome(maybe_palindrome) == expected_result
```

尽管这缩短了您的代码，但重要的是要注意，在这种情况下，您实际上丢失了原始函数的一些更具描述性的特性。确保你没有将你的测试套件参数化到不可理解的程度。您可以使用参数化将测试数据从测试行为中分离出来，这样就可以清楚地知道测试在测试什么，也可以使不同的测试用例更容易阅读和维护。

## 持续时间报告:对抗缓慢测试

每次您将上下文从实现代码切换到测试代码时，您都会招致一些[开销](https://en.wikipedia.org/wiki/Overhead_(computing))。如果你的测试一开始就很慢，那么开销会导致摩擦和挫折。

您在前面读到过在运行套件时使用标记来过滤掉缓慢的测试，但是在某些时候您将需要运行它们。如果你想提高测试的速度，那么知道哪些测试可能提供最大的改进是很有用的。`pytest`可以自动为您记录测试持续时间，并报告排名靠前的违规者。

使用`pytest`命令的`--durations`选项在您的测试结果中包含一个持续时间报告。`--durations`期望一个整数值`n`，并将报告最慢的`n`测试次数。您的测试报告中将包含一个新的部分:

```py
(venv) $ pytest --durations=5
...
============================= slowest 5 durations =============================
3.03s call     test_code.py::test_request_read_timeout
1.07s call     test_code.py::test_request_connection_timeout
0.57s call     test_code.py::test_database_read

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ===========================
...
```

durations 报告中显示的每个测试都是一个很好的加速对象，因为它花费了高于平均水平的总测试时间。请注意，默认情况下，短持续时间是隐藏的。如报告中所述，您可以增加报告的详细程度，并通过将`-vv`和`--durations`一起传递来显示这些内容。

请注意，一些测试可能会有不可见的设置开销。您在前面已经了解了标记有`django_db`的第一个测试将如何触发 Django 测试数据库的创建。`durations`报告反映了在触发数据库创建的测试中设置数据库所花费的时间，这可能会产生误导。

您正在朝着全面测试覆盖的方向前进。接下来，你将看到丰富的`pytest`插件生态系统中的一些插件。

## 有用的`pytest`插件

在本教程的前面，你已经了解了一些有价值的`pytest`插件。在这一节中，您将更深入地探索这些和其他一些插件——从像`pytest-randomly`这样的实用插件到像 Django 那样的特定于库的插件。

[*Remove ads*](/account/join/)

### `pytest-randomly`

通常，测试的顺序并不重要，但是随着代码库的增长，您可能会无意中引入一些副作用，如果这些副作用不按顺序运行，可能会导致一些测试失败。

[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly) 强制你的测试以随机的顺序运行。`pytest`总是在运行测试之前收集它能找到的所有测试。`pytest-randomly`只是在执行之前打乱测试列表。

这是发现依赖于以特定顺序运行的测试的好方法，这意味着它们对其他一些测试有一个**状态依赖**。如果您在`pytest`中从头开始构建您的测试套件，那么这是不太可能的。这更有可能发生在您迁移到`pytest`的测试套件中。

该插件将在配置描述中打印一个种子值。您可以使用该值按照尝试修复问题时的顺序运行测试。

### `pytest-cov`

如果你想测量你的测试覆盖你的实现代码有多好，那么你可以使用[覆盖率](https://coverage.readthedocs.io/)包。 [`pytest-cov`](https://pytest-cov.readthedocs.io/en/latest/) 集成了覆盖率，所以你可以运行`pytest --cov`来查看测试覆盖率报告，并在你的项目首页吹嘘它。

### `pytest-django`

[`pytest-django`](https://pytest-django.readthedocs.io/en/latest/) 提供了一些有用的夹具和标记来处理 Django 测试。您在本教程的前面看到了`django_db`标记。`rf` fixture 提供了对 Django 的 [`RequestFactory`](https://docs.djangoproject.com/en/3.0/topics/testing/advanced/#django.test.RequestFactory) 实例的直接访问。`settings`夹具提供了一种快速设置或覆盖 Django 设置的方法。这些插件极大地提高了 Django 测试的效率！

如果你有兴趣了解更多关于在 Django 中使用`pytest`的信息，那么看看[如何在 Pytest](https://realpython.com/django-pytest-fixtures/) 中为 Django 模型提供测试夹具。

### `pytest-bdd`

`pytest`可用于运行传统单元测试范围之外的测试。[行为驱动开发](https://en.wikipedia.org/wiki/Behavior-driven_development) (BDD)鼓励用简单的语言描述用户可能的行为和期望，然后你可以用它来决定是否实现一个给定的特性。pytest-bdd 帮助你使用[小黄瓜](http://docs.behat.org/en/v2.5/guides/1.gherkin.html)为你的代码编写特性测试。

你可以通过第三方插件的[列表来查看`pytest`还有哪些可用的插件。](http://plugincompat.herokuapp.com/)

## 结论

提供了一套核心的生产力特性来过滤和优化您的测试，以及一个灵活的插件系统来进一步扩展其价值。无论你是有一个庞大的遗产`unittest`套件，还是从头开始一个新项目，`pytest`都能为你提供一些东西。

在本教程中，您学习了如何使用:

*   用于处理测试依赖、状态和可重用功能的夹具
*   **标记**,用于对测试进行分类并限制对外部资源的访问
*   **参数化**用于减少测试之间的重复代码
*   **持续时间**来识别你最慢的测试
*   用于集成其他框架和测试工具的插件

安装`pytest`试试看。你会很高兴你做了。测试愉快！

如果你正在寻找一个用`pytest`构建的示例项目，那么看看关于[用 TDD](https://realpython.com/python-hash-table/) 构建哈希表的教程，它不仅能让你跟上`pytest`的速度，还能帮助你掌握哈希表！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**用 pytest**](/courses/testing-your-code-with-pytest/) 测试你的代码***********
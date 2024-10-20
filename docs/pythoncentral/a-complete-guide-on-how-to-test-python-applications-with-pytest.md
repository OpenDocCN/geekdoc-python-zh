# 关于如何用 Pytest 测试 Python 应用程序的完整指南

> 原文：<https://www.pythoncentral.io/a-complete-guide-on-how-to-test-python-applications-with-pytest/>

编写测试是 Python 开发的关键部分，因为测试使程序员能够检查他们代码的有效性。它们被认为是最有效的方法之一，可以证明编写的代码正在按需要运行，并减少将来的更改破坏功能的可能性。

这就是 Pytest 的用武之地。尽管 Pytest 主要用于为 API 编写测试，但测试框架使得程序员可以轻松地为 UI 和数据库编写可伸缩的测试用例。

在这本全面的指南中，我们将带您了解如何安装 Pytest，它的强大优势，以及如何使用它在您的机器上编写测试。

## **安装 Pytest**

与大多数 Python 包一样，您可以从 PyPI 安装 Pytest，并使用 pip 将其安装在虚拟环境中。如果你用的是 Windows 电脑，在 Windows PowerShell 上运行以下命令:

```py
PS> python -m venv venv
PS> .\venv\Scripts\activate
(venv) PS> python -m pip install Pytest
```

另一方面，如果你运行的是 macOS 或者 Linux 机器，在终端上运行这个:

```py
$ python -m venv venv
$ source venv/bin/activate
(venv) $ python -m pip install Pytest
```

## **Pytest 的优势**

如果你熟悉用 Python 编写单元测试，你可能使用过 Python 内置的 unittest Python 模块。该模块为程序员构建测试套件提供了良好的基础；但是，它也不是没有缺点。

许多第三方测试框架试图克服 unittest 的缺点，但 Pytest 是最有效的，因此也是最受欢迎的解决方案。Pytest 有几个特性，可以与各种各样的插件结合使用，使测试更加容易。

更具体地说，使用 Pytest，您可以执行普通任务而无需编写太多代码，并使用内置的省时命令更快地完成高级任务。

此外，Pytest 可以运行您现有的测试，而不需要额外的插件。再来详细讨论一下优势:

### **#1 少重码**

大部分功能测试使用排列-动作-断言模型，其中:

1.  安排测试的条件，
2.  一个函数或方法执行一些动作，
3.  代码断言特定的结束条件是否为真

测试框架通常在断言上工作，允许它们在断言失败时提供一些信息。例如，unittest 模块提供了几个断言工具，但是即使是最小的测试也需要大量重复的代码。

如果您要编写一个测试套件来检查 unittest 是否正常工作，理想情况下，套件应该有一个总是通过的测试和一个总是失败的测试。然而，要编写这个测试套件，您需要做以下事情:

1.  从 unittest 导入 TestCase 类。
2.  写一个 TestCase 的子类，姑且称之为“TryTest”
3.  为每个测试编写一个 TryTest 方法。
4.  使用 unittest 中的 self.assert*方法。断言的测试用例。

这四项任务是创建任何测试所需的最低要求。以这种方式编写测试是低效的，因为它涉及到多次编写相同的代码。

有了 Pytest，工作流程就简单多了，因为你可以自由地使用普通函数和内置的 assert 关键字。当使用 Pytest: 编写相同的测试时，它看起来像这样

```py
# test_with_Pytest.py

def test_always_passes():
    assert True

def test_always_fails():
    assert False
```

就这么简单——不需要导入任何模块或使用任何类。只需编写一个带有前缀“text_”的函数。因为 Pytest 使您能够使用 assert 关键字，所以没有必要阅读 unittest 中出现的所有 self.assert*方法。

Pytest 将测试您期望评估为真的任何表达式。除了你的测试套件有更少的重复代码，它也变得更加详细和易读。

### **#2 吸引输出**

Pytest 的一个优点是，您可以从项目的顶层文件夹中运行它的命令。现在您可能已经发现，Pytest 模块产生的测试结果与 unittest 不同。该报告将向您显示:

1.  系统状态，包括 Python 版本、Pytest 和其他插件的详细信息；
2.  您可以在其中搜索测试和配置的目录；和
3.  跑步者发现的测试数量。

报告的第一部分介绍了这些内容，在下一部分，它在测试名称旁边显示了每个测试的状态。如果出现圆点，则表示测试通过。如果出现 F，测试失败，如果出现 E，测试抛出意外异常。

失败的测试总是伴随着详细的分解，这种额外的输出使得调试更加易于管理。在报告的最后部分，有一个测试套件的总体状态。

### **#3 简单易学**

如果您熟悉 assert 关键字的使用，那么学习使用 Pytest 并不新鲜。这里有一些断言的例子来帮助你理解用 Pytest 测试的不同方法:

```py
# test_examples_of_assertion.py

def test_uppercase():
    assert "example text".upper() == "EXAMPLE TEST"

def test_reverseList():
    assert list(reverseList([5, 6, 7, 8])) == [8, 7, 6, 5]

def test_some_primes():
    assert 37 in {
        num
        for num in range(2, 50)
        if not any(num % div == 0 for div in range(2, num))
    }
```

上面套件中的测试看起来像常规的 Python 函数，这使得 Pytest 易于学习，并且消除了程序员学习任何新构造的需要。

注意测试是如何简短和独立的——这是 Pytest 测试的趋势。您可能会看到很长的函数名，但函数名中几乎没有什么内容。通过这种方式，Pytest 保持了测试的独立性；如果出了问题，程序员知道去哪里找问题。

### **#4 更易管理的状态和依赖关系**

使用 Pytest 编写的大多数测试将依赖于数据类型或模拟代码可能遇到的对象的测试类型，比如 JSON 文件和字典。

另一方面，当使用 unittest 时，程序员倾向于将依赖项提取到。设置()和。tearDown()方法。这样，类中的每个测试都可以使用依赖项。

虽然使用这些方法没有错，但是随着测试类变得越来越庞大，程序员可能最终会使测试的依赖性完全隐含。换句话说，当您查看孤立的测试时，您可能看不到该测试依赖于其他东西。

这些隐含的依赖会使测试变得混乱，并使理解它们变得困难。使用测试背后的想法是让代码更容易理解，从长远来看，使用这些方法会适得其反。

在使用 Pytest 时，你不必担心这一点，因为 fixtures 会将你引向可重用的显式依赖声明。

Pytest 中的 Fixtures 是使您能够为测试套件创建数据、测试 doubles 和初始化系统状态的函数。如果您决定使用一个 fixture，您必须显式地使用下面的 fixture 函数作为测试函数的参数:

```py
# fixture_example.py

import Pytest

@Pytest.fixture
def example_fixture():
    return 1

def testing_using_fixture(example_fixture):
    assert example_fixture == 1
```

以这种方式使用 fixture 函数可以保持依赖性。当你浏览测试函数时，很明显它依赖于 fixture。您不需要检查夹具定义的文件。

-

注意: 将你的测试放在一个单独的文件夹中，这个文件夹叫做 tests，位于你的项目文件夹的根目录下，这被认为是最佳实践。

-

Pytest 提供了很多灵活性，因为 fixture 可以通过简单、明确的依赖声明来使用其他 fixture。但是由于这个原因，当你继续使用它们的时候，固定装置可以变得模块化。换句话说，随着测试套件变得越来越大，您将需要小心地管理您的依赖项。

我们将在本帖的后面更详细地讨论夹具。

### **#5 过滤测试很简单**

测试套件的规模肯定会增长，程序员发现自己想要对一个特性运行少量的测试，并保存完整的套件以备后用。使用 Pytest，有几种方法可以做到这一点:

*   **目录范围:** Python 默认只运行当前目录下的测试。使用这个特性只运行所需的测试被称为目录范围。
*   **基于名字的过滤:** Pytest 允许程序员只运行那些完全限定名与特定表达式相同的测试。使用-k 参数很容易实现基于名称的过滤。
*   **测试分类:** 使用-m 参数可以很容易地从定义的类别中包含或排除测试。这是一个有效的方法，因为 Pytest 使程序员能够为测试创建称为“标记”的定制标签。单个测试可能有几个标签，允许程序员对将要运行的测试进行粒度控制。

### **#6 测试参数化**

程序员用 Pytest 测试处理数据的函数是很常见的；所以程序员经常会写很多类似的测试。这些测试可能只是在被测试代码的输入或输出上有所不同。

由于这个原因，程序员最终会重复测试代码，这有时会掩盖他们试图测试的代码的行为。

如果您熟悉 unittest，您可能知道有一种方法可以将许多测试收集为一个测试。但是这些测试的结果不会作为单独的测试出现在结果报告中。这意味着，如果除了一个测试之外的所有测试都通过了，那么整个测试组将返回一个失败的结果。

但是 Pytest 不是这种情况，因为它具有内置的参数化特性，允许每个测试独立地通过或失败。

### **#7 基于插件的架构**

Pytest 的可定制性使其成为想要测试代码的程序员的首选框架。添加新功能很容易，因为程序的几乎每个部分都可以改变。难怪 Pytest 有一个巨大的有用插件生态系统。

虽然有些插件只适用于 Django 这样的框架，但大多数可用的插件可以用于几乎所有的测试套件。

## **使用夹具管理状态和依赖关系**

如前所述，Pytest fixtures 允许你为测试提供数据，测试 doubles，或者描述测试的设置。fixture 函数能够返回大范围的值。每一个依赖于 fixture 的测试都需要将 fixture 作为一个参数显式地传递给它。

### **何时使用夹具**

理解何时使用夹具的最好方法之一是模拟测试驱动的开发工作流程。

假设您需要编写一个函数来处理从 API 端点接收的数据。该数据包括一个人员列表，每个条目都有姓名和工作职位。

这个函数需要输出一个带有全名的字符串列表，后跟一个冒号和它们的标题。你可以这样做:

```py
# format_data.py

def reformat_data(people):
    ...  # Instructions to implement
```

由于我们正在模拟一个测试驱动的开发工作流程，首要任务是为它编写一个测试。有一种方法可以做到这一点:

```py
# test_format_data.py

def test_reformat_data ():
    people = [
        {
           "given_name": "Mia",
            "family_name": "Alice",
            "title": "Software Developer",
        },
        {
            "given_name": "Arun",
            "family_name": "Niketa",
            "title": "HR Head",
        },
    ]

    assert reformat_data(people) == [
        "Mia Alice: Software Developer",
        "Arun Niketa: HR Head",
    ]
```

让我们更进一步，假设您需要编写另一个函数来处理数据，并以逗号分隔值的形式输出，以便在电子表格中使用:

```py
# format_data.py

def reformat_data(people):
    ...  # Instructions to implement 

def format_data_for_excel(people):
    ... # Instructions to implement
```

您的待办事项列表正在增长，通过测试驱动的开发，您可以轻松地提前计划事情。这个新函数看起来类似于 format_data()函数:

```py
# test_format_data.py

def test_reformat_data():
    # ...

def test_format_data_for_excel():
    people = [
        {
            "given_name": "Mia",
            "family_name": "Alice",
            "title": "Software Developer",
        },
        {
            "given_name": "Arun",
            "family_name": "Niketa",
            "title": "HR Head",
        },
    ]

    assert format_data_for_excel(people) == """given,family,title
Mia,Alice, Software Developer
Arun,Niketa,HR Head
"""
```

如您所见，两个测试都必须再次定义 people 变量，并且将这些代码行放在一起需要时间和精力。

如果几个测试使用相同的测试数据，夹具会有所帮助。使用 fixture，您可以将重复的数据放在一个函数中，并使用@Pytest.fixture 来表明该函数就是 fixture，就像这样:

```py
# test_format_data.py

import Pytest

@Pytest.fixture
def example_people_data():
    return [
        {
            "given_name": "Mia",
            "family_name": "Alice",
            "title": "Software Developer",
        },
        {
            "given_name": "Arun",
            "family_name": "Niketa",
            "title": "HR Head",
        },
    ]

# More code
```

使用 fixture 并不复杂——你只需要添加函数引用作为参数。请记住，程序员不是调用 fixture 函数的人，Pytest 会处理这些。fixture 函数的返回值可以作为 fixture 的名称:

```py
# test_format_data.py

# ...

def test_reformat_data(example_people_data):
    assert reformat_data(example_people_data) == [
        "Mia Alice: Software Developer",
        "Arun Niketa: HR Head",
    ]

def test_format_data_for_excel(example_people_data):
    assert format_data_for_excel(example_people_data) == """given,family,title
Mia,Alice, Software Developer
Arun,Niketa,HR Head
"""
```

现在，测试变得更小了，同时有了一条清晰的返回测试数据的路径。为您的 fixture 起一个突出的名字是在您以后添加更多测试时识别和使用它的最快方法。

### **何时不使用夹具**

夹具使得提取跨多个测试使用的对象和数据变得很方便。但是当测试要求数据发生变化时，使用 fixtures 并不总是正确的解决方案。

在测试套件中放置固定装置就像在其中放置对象和数据一样容易。使用 fixture 的结果有时会更糟，因为 fixture 引入了测试套件的一个额外的间接层。

学习使用任何抽象都需要练习，虽然使用 fixtures 很容易，但你需要一些时间来找到正确的数字。

不管怎样，你可以期待夹具成为你的测试套件的重要部分。随着项目变得越来越大，你会遇到扩展的挑战。幸运的是，Pytest 有一些特性可以帮助您克服这种伴随增长而来的复杂性挑战。

### **按比例使用夹具**

随着测试中夹具提取数量的增加，您可能会注意到一些夹具可能会因为更多的抽象而变得更加有效。

Pytest 框架夹具是模块化的，这意味着您可以导入它们和其他模块。此外，设备可以依赖于其他设备，也可以导入其他设备。

这种巨大的灵活性允许你根据用例创建合适的夹具抽象。

例如，不同模块中的设备可能有一个共同的依赖关系。在这种情况下，您可以将夹具从测试模块移动到通用模块。这将允许您将它们导入到需要它们的测试模块中。

程序员可能想在整个项目中使用一个夹具，而不需要导入它，这可以通过 conftest.py 配置模块来实现。

Pytest 自动在每个目录中寻找这个模块。如果您将通用夹具添加到这个模块中，那么您可以在整个父目录和子目录中使用这些夹具。

您还可以使用 conftest.py 和 fixtures 来保护对资源的访问。例如，如果您为处理 API 调用的代码编写一个测试套件，您将需要确保该套件不会进行任何调用，即使程序员无意中编写了一个测试来实现这一点。

Pytest 附带了一个 monkeypatch fixture，使您能够替换行为和值，就像这样:

```py
# conftest.py

import Pytest
import requests

@Pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    def stunted_get():
        raise RuntimeError("Network access unavailable in testing")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_get())
```

将 disable_network_calls()方法放入上述模块的配置中，并将 autouse 选项设置为 true，可以确保在整个套件中禁用网络调用。

当你的测试套件增长时，你会变得更有信心做出改变，因为你不会意外地破坏代码。然而，随着测试套件的增长，做出改变可能需要更长的时间。

如果做出改变不需要很长时间，一个核心行为可能会慢慢渗透并破坏许多测试。在这种情况下，最好将测试运行程序限制为只运行特定类别的测试。这是我们接下来要讨论的。

### **分类测试**

当你需要快速迭代特性时，大型测试套件可能是有害的，阻止所有测试运行可以节省时间。如前所述，Pytest 默认在当前工作目录下运行测试，但是使用标记也是一个很好的解决方案。

在 Pytest 中，您可以为您的测试定义类别，并在套件运行时提供包含或排除类别的选项。一个单独的测试可以分为几个类别。

标记测试是根据依赖关系和子系统对测试进行分类的一个很好的方法。例如，如果一些测试需要访问数据库，您可以做一个@Pytest.mark.database_access 标记并使用它。

此外，在 Pytest 命令中添加- strict-markers 标志可以确保在 Pytest.ini 配置文件中注册测试中的标记。这个文件将在你显式注册未知标记之前避免运行所有的测试。

如果您只想运行需要数据库访问的测试，您可以使用 Pytest -m database_access 命令。但是，如果您想要运行除了需要数据库访问的测试之外的所有测试，您所要做的就是运行命令 Pytest-m“not database _ access”。

您可以将其与 autouse fixture 配对，以限制对标记为可访问数据库的测试的访问。一些插件通过添加自定义防护为标记功能增加了更多功能。例如，Pytest-django 插件有一个 django_db 标记，没有该标记的测试无法访问数据库。

当一个测试试图访问数据库时，Django 将创建一个测试数据库。您在 django_db 标记中指定的需求导致您显式地陈述依赖关系。

您也可以运行不需要数据库的测试，因为您运行的命令会首先阻止数据库的创建。虽然在较小的测试套件中节省的时间可能不明显，但是在较大的测试套件中，它可以节省您几分钟的时间。

Pytest 默认包含的一些标记有 skip、skipif、xfail 和 parametrize。如果您想查看 Pytest 默认附带的所有标记，可以运行 Pytest - markers。

## **结论**

Pytest 拥有生产力特性，允许程序员优化编写和运行测试所需的时间和精力。此外，灵活的插件系统允许程序员扩展 Pytest 的基本功能之外的功能。

无论程序员是在处理一个庞大的遗留单元测试套件还是开始一个新的项目，Pytest 都能派上用场。有了这个指南，你就可以使用它了。
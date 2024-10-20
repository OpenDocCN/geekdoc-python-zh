# Python 201:模拟简介

> 原文：<https://www.blog.pythonlibrary.org/2016/07/19/python-201-an-intro-to-mock/>

从 Python 3.3 开始，unittest 模块现在包括一个**模拟**子模块。它将允许您用模拟对象替换正在测试的系统部分，并断言它们是如何被使用的。模拟对象用于模拟测试环境中不可用的系统资源。换句话说，你会发现有时候你想测试你的代码的一部分，而不测试它的其余部分，或者你需要测试一些代码，而不测试外部服务。

> 注意，如果您有 Python 3 之前的 Python 版本，您可以下载[模拟库](https://pypi.python.org/pypi/mock)并获得相同的功能。

让我们考虑一下为什么您可能想要使用 mock。一个很好的例子是，如果你的应用程序绑定到某种第三方服务，如 Twitter 或脸书。如果您的应用程序的测试套件在每次运行时转发一堆项目或“喜欢”一堆帖子，那么这可能是不可取的行为，因为每次运行测试时都会这样做。另一个例子可能是，如果您设计了一个工具，使更新您的数据库表更容易。每次测试运行时，它都会对相同的记录进行一些更新，可能会清除有价值的数据。

您可以使用 unittest 的 mock 来代替做这些事情。它会让你嘲笑和根除那些副作用，所以你不必担心它们。代替与第三方资源的交互，您将针对与那些资源匹配的虚拟 API 运行您的测试。您最关心的是您的应用程序正在调用它应该调用的函数。您可能不太关心 API 本身是否实际执行。当然，有时候你会想做一个真正执行 API 的端到端测试，但是这些测试不需要模拟！

* * *

### 简单的例子

Python 模拟类可以模拟任何其他 Python 类。这允许你检查在你模仿的类上调用了什么方法，甚至传递了什么参数给它们。让我们先看几个简单的例子，演示如何使用模拟模块:

```py
>>> from unittest.mock import Mock
>>> my_mock = Mock()
>>> my_mock.__str__ = Mock(return_value='Mocking')
>>> str(my_mock)
'Mocking'

```

在这个例子中，我们从 **unittest.mock** 模块中导入 **Mock** 类。然后我们创建一个模拟类的实例。最后，我们设置了模拟对象的 **__str__** 方法，这是一个神奇的方法，它控制着在对象上调用 Python 的 **str** 函数时会发生什么。在这种情况下，我们只返回字符串“Mocking”，这是我们最后实际执行 str()函数时所看到的。

模拟模块还支持五种断言。让我们来看看其中几个是如何运作的:

```py
>>> from unittest.mock import Mock
>>> class TestClass():
...     pass
... 
>>> cls = TestClass()
>>> cls.method = Mock(return_value='mocking is fun')
>>> cls.method(1, 2, 3)
'mocking is fun'
>>> cls.method.assert_called_once_with(1, 2, 3)
>>> cls.method(1, 2, 3)
'mocking is fun'
>>> cls.method.assert_called_once_with(1, 2, 3)
Traceback (most recent call last):
  Python Shell, prompt 9, line 1
  File "/usr/local/lib/python3.5/unittest/mock.py", line 802, in assert_called_once_with
    raise AssertionError(msg)
builtins.AssertionError: Expected 'mock' to be called once. Called 2 times.
>>> cls.other_method = Mock(return_value='Something else')
>>> cls.other_method.assert_not_called()
>>>

```

首先，我们导入并创建一个空类。然后，我们创建该类的一个实例，并添加一个使用模拟类返回字符串的方法。然后我们调用带有三个整数参数的方法。您会注意到，这返回了我们之前设置为返回值的字符串。现在我们可以测试一个断言了！所以我们调用* * assert _ called _ once _ with * * assert，如果我们用相同的参数调用我们的方法两次或更多次，它将断言。我们第一次调用 assert 时，它通过了。所以我们用同样的方法再次调用这个方法，并第二次运行 assert，看看会发生什么。

如您所见，我们得到了一个 **AssertionError** 。为了完善这个例子，我们继续创建第二个方法，我们根本不调用它，然后断言它不是通过 **assert_not_called** assert 调用的。

* * *

### 副作用

您还可以通过 **side_effect** 参数创建模拟对象的副作用。副作用是在运行函数时发生的事情。例如，一些视频游戏已经整合到社交媒体中。当你获得一定数量的分数，赢得一个奖杯，完成一个级别或其他一些预定的目标，它会记录下来，并发布到 Twitter，脸书或任何它集成的地方。运行一个函数的另一个副作用是，它可能与你的用户界面紧密相连，导致不必要的重绘。

因为我们预先知道这些副作用，所以我们可以在代码中模拟它们。让我们看一个简单的例子:

```py
from unittest.mock import Mock

def my_side_effect():
    print('Updating database!')

def main():
    mock = Mock(side_effect=my_side_effect)
    mock()

if __name__ == '__main__':
    main()

```

这里我们创建一个假装更新数据库的函数。然后在我们的 **main** 函数中，我们创建一个模拟对象，并给它一个副作用。最后，我们调用我们的模拟对象。如果这样做，您应该会看到一条关于数据库被更新的消息被打印到 stdout。

Python 文档还指出，如果您愿意，可以让副作用引发异常。一个相当常见的原因是，如果调用不正确，就会引发异常。一个例子可能是你没有传递足够的参数。您还可以创建一个 mock 来引发一个弃用警告。

* * *

### 自动筛选

模拟模块还支持**自动规范**的概念。autospec 允许您创建模拟对象，这些对象包含与您用模拟替换的对象相同的属性和方法。它们甚至会有与真实对象相同的调用签名！您可以使用 **create_autospec** 函数创建一个 autospec，或者通过将 **autospec** 参数传递给模拟库的 **patch** decorator 来创建一个 autospec，但是我们将把对 patch 的研究推迟到下一节。

现在，让我们看一个易于理解的 autospec 示例:

```py
>>> from unittest.mock import create_autospec
>>> def add(a, b):
...     return a + b
... 
>>> mocked_func = create_autospec(add, return_value=10)
>>> mocked_func(1, 2)
10
>>> mocked_func(1, 2, 3)
Traceback (most recent call last):
  Python Shell, prompt 5, line 1
  File "", line 2, in add
  File "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/mock.py", line 181, in checksig
    sig.bind(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/inspect.py", line 2921, in bind
    return args[0]._bind(args[1:], kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/inspect.py", line 2842, in _bind
    raise TypeError('too many positional arguments') from None
builtins.TypeError: too many positional arguments

```

在这个例子中，我们导入了 **create_autospec** 函数，然后创建了一个简单的加法函数。接下来，我们使用 create_autospec()，向它传递我们的 **add** 函数，并将其返回值设置为 10。只要你用两个参数传递这个新的模拟版本的 add，它将总是返回 10。但是，如果用不正确的参数数目调用它，将会收到一个异常。

* * *

### 补丁

mock 模块有一个叫做 **patch** 的简洁的小函数，可以用作函数装饰器、类装饰器甚至是上下文管理器。这将允许您轻松地在您想要测试的模块中创建模拟类或对象，因为它将被模拟替换。

让我们从创建一个阅读网页的简单函数开始。我们将称之为 **webreader.py** 。代码如下:

```py
import urllib.request

def read_webpage(url):
    response = urllib.request.urlopen(url)
    return response.read()

```

这段代码非常简单明了。它所做的就是获取一个 URL，打开页面，读取 HTML 并返回它。现在，在我们的测试环境中，我们不想陷入从网站读取数据的困境，尤其是我们的应用程序碰巧是一个每天下载千兆字节数据的网络爬虫。相反，我们想创建一个 Python 的 urllib 的模拟版本，这样我们就可以调用上面的函数，而不用真正下载任何东西。

让我们创建一个名为 **mock_webreader.py** 的文件，并将其保存在与上面代码相同的位置。然后将以下代码放入其中:

```py
import webreader

from unittest.mock import patch

@patch('urllib.request.urlopen')
def dummy_reader(mock_obj):
    result = webreader.read_webpage('https://www.google.com/')
    mock_obj.assert_called_with('https://www.google.com/')
    print(result)

if __name__ == '__main__':
    dummy_reader()

```

这里我们只是从模拟模块中导入我们之前创建的模块和**补丁**函数。然后我们创建一个装饰器来修补 **urllib.request.urlopen** 。在函数内部，我们用 Google 的 URL 调用 webreader 模块的**read _ 网页**函数并打印结果。如果您运行这段代码，您将看到我们的结果不是 HTML，而是 MagicMock 对象。这证明了补丁的威力。我们现在可以阻止下载数据，同时仍然正确调用原始函数。

文档指出，您可以像使用常规装饰器一样堆叠路径装饰器。因此，如果你有一个真正复杂的函数，访问数据库或写文件或几乎任何其他东西，你可以添加多个补丁，以防止副作用的发生。

* * *

### 包扎

模拟模块非常有用，也非常强大。也需要一些时间来学习如何正确有效地使用。Python 文档中有很多例子，尽管它们都是带有虚拟类的简单例子。我想你会发现这个模块对于创建健壮的测试很有用，它可以快速运行而不会产生意外的副作用。

* * *

### 相关阅读

*   关于[模拟库](https://docs.python.org/3/library/unittest.mock.html)的 Python 文档
*   top tal:[Python 中嘲讽的介绍](https://www.toptal.com/python/an-introduction-to-mocking-in-python)
*   StackOverflow: [Python 模拟单元测试和数据库](http://stackoverflow.com/questions/22963514/python-mock-unittest-and-database)
*   StackOverflow: [一个模仿/存根 Python 模块如何像 urllib](http://stackoverflow.com/questions/295438/how-can-one-mock-stub-python-module-like-urllib)
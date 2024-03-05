# 了解 Python 模拟对象库

> 原文：<https://realpython.com/python-mock-library/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，加深您的理解: [**使用 Python 模拟对象库**](/courses/python-mock-object-library/) 改进您的测试

当您编写健壮的代码时，测试对于验证您的应用程序逻辑是正确的、可靠的和有效的是必不可少的。然而，您的测试的价值取决于它们在多大程度上证明了这些标准。诸如复杂的逻辑和不可预测的依赖关系这样的障碍使得编写有价值的测试变得困难。Python 模拟对象库`unittest.mock`，可以帮助你克服这些障碍。

**本文结束时，你将能够:**

*   使用`Mock`创建 Python 模拟对象
*   断言你正在按照你的意图使用对象
*   检查存储在 Python 模拟中的使用数据
*   配置 Python 模拟对象的某些方面
*   使用`patch()`将你的模型替换成真实的物体
*   避免 Python 模仿中固有的常见问题

您将从了解什么是嘲讽以及它将如何改进您的测试开始。

**免费奖励:** [掌握 Python 的 5 个想法](https://realpython.com/bonus/python-mastery-course/)，这是一个面向 Python 开发者的免费课程，向您展示将 Python 技能提升到下一个水平所需的路线图和心态。

## 什么是嘲讽？

一个[模拟对象](https://en.wikipedia.org/wiki/Mock_object)在一个[测试环境](https://realpython.com/python-testing/)中替代并模仿一个真实对象。它是[提高测试质量](https://realpython.com/python-cli-testing/#mocks)的一个通用且强大的工具。

使用 Python 模拟对象的一个原因是为了在测试过程中控制代码的行为。

例如，如果您的代码向外部服务发出 [HTTP 请求](https://realpython.com/python-requests/)，那么您的测试只有在服务的行为符合您的预期时才会可预测地执行。有时，这些外部服务行为的临时变化会导致测试套件中的间歇性故障。

因此，在一个受控的环境中测试您的代码会更好。用模拟对象替换实际的请求将允许您以可预测的方式模拟外部服务中断和成功的响应。

有时候，测试代码库的某些部分是很困难的。这样的区域包括难以满足的`except`块和`if`语句。使用 Python 模拟对象可以帮助您控制代码的执行路径以到达这些区域，并提高您的[代码覆盖率](https://en.wikipedia.org/wiki/Code_coverage)。

使用模拟对象的另一个原因是为了更好地理解如何在代码中使用它们的真实对应物。Python 模拟对象包含关于其用法的数据，您可以检查这些数据，例如:

*   如果你调用了一个方法
*   您如何调用该方法
*   您调用该方法的频率

理解模拟对象的作用是学习如何使用它的第一步。

现在，您将看到如何使用 Python 模拟对象。

[*Remove ads*](/account/join/)

## Python 模拟库

Python [模拟对象库](https://docs.python.org/3/library/unittest.mock.html)是`unittest.mock`。它提供了一个简单的方法将模拟引入到你的测试中。

**注意:**标准库包括 Python 3.3 及以后版本中的`unittest.mock`。如果你使用的是旧版本的 Python，你需要安装库的官方后台。为此，从 [PyPI](https://pypi.org/project/mock/) 安装`mock`:

```py
$ pip install mock
```

`unittest.mock`提供了一个名为`Mock`的类，你可以用它来模仿代码库中的真实对象。`Mock`提供令人难以置信的灵活性和深刻的数据。这个及其子类将满足您在测试中面临的大多数 Python 模仿需求。

该库还提供了一个名为`patch()`的函数，它用`Mock`实例替换代码中的真实对象。您可以使用`patch()`作为装饰器或上下文管理器，让您控制对象被模仿的范围。一旦指定的作用域退出，`patch()`将通过用它们原来的对应物替换被模仿的对象来清理你的代码。

最后，`unittest.mock`为模仿对象中固有的一些问题提供了解决方案。

现在，您已经更好地理解了什么是嘲讽，以及您将用来做这件事的库。让我们深入探讨一下`unittest.mock`提供了哪些特性和功能。

## `Mock`对象

`unittest.mock`为模仿对象提供了一个基类，叫做`Mock`。因为`Mock`非常灵活，所以`Mock`的用例实际上是无限的。

首先实例化一个新的`Mock`实例:

>>>

```py
>>> from unittest.mock import Mock
>>> mock = Mock()
>>> mock
<Mock id='4561344720'>
```

现在，您可以用新的`Mock`替换代码中的对象。您可以通过将它作为参数传递给函数或重新定义另一个对象来实现这一点:

```py
# Pass mock as an argument to do_something()
do_something(mock)

# Patch the json library
json = mock
```

当你在代码中替换一个对象时，`Mock`必须看起来像它所替换的真实对象。否则，您的代码将无法使用`Mock`来代替原始对象。

例如，如果您正在模仿`json`库，并且您的程序调用了`dumps()`，那么您的 Python 模仿对象也必须包含`dumps()`。

接下来，您将看到`Mock`如何应对这一挑战。

### 惰性属性和方法

一个`Mock`必须模拟它替换的任何对象。为了实现这样的灵活性，当你访问属性时，它[会创建它的属性](https://docs.python.org/3/library/unittest.mock.html#quick-guide):

>>>

```py
>>> mock.some_attribute
<Mock name='mock.some_attribute' id='4394778696'>
>>> mock.do_something()
<Mock name='mock.do_something()' id='4394778920'>
```

由于`Mock`可以动态创建任意属性，因此适合替换任何对象。

使用前面的一个例子，如果您模仿`json`库并调用`dumps()`，Python 模仿对象将创建该方法，以便其接口可以匹配库的接口:

>>>

```py
>>> json = Mock()
>>> json.dumps()
<Mock name='mock.dumps()' id='4392249776'>
```

请注意这个`dumps()`模拟版本的两个关键特征:

1.  与真正的[`dumps()`](https://realpython.com/python-json/#serializing-json)不同，这种嘲弄的方法不需要争论。事实上，它会接受您传递给它的任何参数。

2.  `dumps()`的返回值也是一个`Mock`。`Mock`到[递归](https://realpython.com/python-thinking-recursively/)定义其他模拟的能力允许你在复杂的情况下使用模拟:

>>>

```py
>>> json = Mock()
>>> json.loads('{"k": "v"}').get('k')
<Mock name='mock.loads().get()' id='4379599424'>
```

因为每个被模仿的方法的返回值也是一个`Mock`，所以您可以以多种方式使用您的模仿。

模拟是灵活的，但它们也能提供信息。接下来，您将学习如何使用模拟来更好地理解您的代码。

[*Remove ads*](/account/join/)

### 断言和检验

实例存储你如何使用它们的数据。例如，您可以查看是否调用了一个方法，如何调用该方法，等等。使用这些信息有两种主要方式。

首先，您可以断言您的程序使用了您所期望的对象:

>>>

```py
>>> from unittest.mock import Mock

>>> # Create a mock object
... json = Mock()

>>> json.loads('{"key": "value"}')
<Mock name='mock.loads()' id='4550144184'>

>>> # You know that you called loads() so you can
>>> # make assertions to test that expectation
... json.loads.assert_called()
>>> json.loads.assert_called_once()
>>> json.loads.assert_called_with('{"key": "value"}')
>>> json.loads.assert_called_once_with('{"key": "value"}')

>>> json.loads('{"key": "value"}')
<Mock name='mock.loads()' id='4550144184'>

>>> # If an assertion fails, the mock will raise an AssertionError
... json.loads.assert_called_once()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 795, in assert_called_once
    raise AssertionError(msg)
AssertionError: Expected 'loads' to have been called once. Called 2 times.

>>> json.loads.assert_called_once_with('{"key": "value"}')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 824, in assert_called_once_with
    raise AssertionError(msg)
AssertionError: Expected 'loads' to be called once. Called 2 times.

>>> json.loads.assert_not_called()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 777, in assert_not_called
    raise AssertionError(msg)
AssertionError: Expected 'loads' to not have been called. Called 2 times.
```

`.assert_called()`确保您调用了被模仿的方法，而`.assert_called_once()`检查您只调用了该方法一次。

这两个断言函数都有变体，允许您检查传递给被模仿方法的参数:

*   `.assert_called_with(*args, **kwargs)`
*   `.assert_called_once_with(*args, **kwargs)`

要传递这些断言，您必须使用传递给实际方法的相同参数来调用模拟方法:

>>>

```py
>>> json = Mock()
>>> json.loads(s='{"key": "value"}')
>>> json.loads.assert_called_with('{"key": "value"}')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 814, in assert_called_with
    raise AssertionError(_error_message()) from cause
AssertionError: Expected call: loads('{"key": "value"}')
Actual call: loads(s='{"key": "value"}')
>>> json.loads.assert_called_with(s='{"key": "value"}')
```

`json.loads.assert_called_with('{"key": "value"}')`提出了一个`AssertionError`，因为它期望你用位置参数调用 [`loads()`](https://realpython.com/python-json/#deserializing-json) ，但你实际上用关键字参数调用了它。`json.loads.assert_called_with(s='{"key": "value"}')`这个断言是正确的。

其次，您可以查看特殊属性，以了解您的应用程序如何使用对象:

>>>

```py
>>> from unittest.mock import Mock

>>> # Create a mock object
... json = Mock()
>>> json.loads('{"key": "value"}')
<Mock name='mock.loads()' id='4391026640'>

>>> # Number of times you called loads():
... json.loads.call_count
1
>>> # The last loads() call:
... json.loads.call_args
call('{"key": "value"}')
>>> # List of loads() calls:
... json.loads.call_args_list
[call('{"key": "value"}')]
>>> # List of calls to json's methods (recursively):
... json.method_calls
[call.loads('{"key": "value"}')]
```

您可以使用这些属性编写测试，以确保您的对象如您所愿地运行。

现在，您可以创建模拟并检查它们的使用数据。接下来，您将看到如何定制模拟方法，以便它们在您的测试环境中变得更加有用。

### 管理模拟的返回值

使用模拟的一个原因是为了在测试过程中控制代码的行为。一种方法是指定函数的[返回值](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.return_value)。让我们用一个例子来看看这是如何工作的。

首先，创建一个名为`my_calendar.py`的文件。添加`is_weekday()`，这个函数使用 [Python 的`datetime`库](https://realpython.com/python-datetime/)来确定今天是否是工作日。最后，编写一个测试，断言该函数按预期工作:

```py
from datetime import datetime

def is_weekday():
    today = datetime.today()
    # Python's datetime library treats Monday as 0 and Sunday as 6
    return (0 <= today.weekday() < 5)

# Test if today is a weekday
assert is_weekday()
```

因为您正在测试今天是否是工作日，所以结果取决于您运行测试的日期:

```py
$ python my_calendar.py
```

如果该命令没有产生输出，则断言成功。不幸的是，如果您在周末运行该命令，您将得到一个`AssertionError`:

```py
$ python my_calendar.py
Traceback (most recent call last):
 File "test.py", line 9, in <module>
 assert is_weekday()
AssertionError
```

当编写测试时，确保结果是可预测的是很重要的。您可以使用`Mock`来消除测试过程中代码的不确定性。在这种情况下，您可以模仿`datetime`并将`.today()`的`.return_value`设置为您选择的日期:

```py
import datetime
from unittest.mock import Mock

# Save a couple of test days
tuesday = datetime.datetime(year=2019, month=1, day=1)
saturday = datetime.datetime(year=2019, month=1, day=5)

# Mock datetime to control today's date
datetime = Mock() 
def is_weekday():
    today = datetime.datetime.today()
    # Python's datetime library treats Monday as 0 and Sunday as 6
    return (0 <= today.weekday() < 5)

# Mock .today() to return Tuesday
datetime.datetime.today.return_value = tuesday # Test Tuesday is a weekday
assert is_weekday()
# Mock .today() to return Saturday
datetime.datetime.today.return_value = saturday # Test Saturday is not a weekday
assert not is_weekday()
```

在这个例子中，`.today()`是一个被模仿的方法。通过给模拟的`.return_value`指定一个特定的日期，您已经消除了不一致性。这样，当你调用`.today()`时，它会返回你指定的`datetime`。

在第一个测试中，您确保`tuesday`是工作日。在第二个测试中，您验证了`saturday`不是工作日。现在，哪一天运行测试并不重要，因为你已经模仿了`datetime`，并且控制了对象的行为。

**延伸阅读:**虽然这样嘲讽`datetime`是使用`Mock`的一个很好的实践例子，但是已经有一个很棒的嘲讽`datetime`的库叫做 [`freezegun`](https://github.com/spulec/freezegun) 。

在构建测试时，您可能会遇到这样的情况，仅仅模仿函数的返回值是不够的。这是因为函数通常比简单的单向逻辑流更复杂。

有时，当您不止一次调用函数或者甚至引发异常时，您会希望函数返回不同的值。您可以使用`.side_effect`来完成此操作。

[*Remove ads*](/account/join/)

### 管理模仿的副作用

您可以通过指定被模仿函数的[副作用](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect)来控制代码的行为。一个`.side_effect`定义了当你调用被模仿的函数时会发生什么。

为了测试这是如何工作的，向`my_calendar.py`添加一个新函数:

```py
import requests

def get_holidays():
    r = requests.get('http://localhost/api/holidays')
    if r.status_code == 200:
        return r.json()
    return None
```

`get_holidays()`向`localhost`服务器请求一组假期。如果服务器响应成功，`get_holidays()`将返回一个字典。否则，该方法将返回 [`None`](https://realpython.com/null-in-python/) 。

您可以通过设置`requests.get.side_effect`来测试`get_holidays()`将如何响应连接超时。

对于这个例子，您只会看到来自`my_calendar.py`的相关代码。您将使用 Python 的 [`unittest`](https://docs.python.org/3/library/unittest.html) 库构建一个测试用例:

```py
import unittest
from requests.exceptions import Timeout
from unittest.mock import Mock

# Mock requests to control its behavior
requests = Mock() 
def get_holidays():
    r = requests.get('http://localhost/api/holidays')
    if r.status_code == 200:
        return r.json()
    return None

class TestCalendar(unittest.TestCase):
    def test_get_holidays_timeout(self):
        # Test a connection timeout
 requests.get.side_effect = Timeout        with self.assertRaises(Timeout):
            get_holidays()

if __name__ == '__main__':
    unittest.main()
```

鉴于`get()`的新副作用，您使用`.assertRaises()`来验证`get_holidays()`是否引发了异常。

运行此测试以查看测试结果:

```py
$ python my_calendar.py
.
-------------------------------------------------------
Ran 1 test in 0.000s

OK
```

如果您想更动态一点，您可以将`.side_effect`设置为一个函数，当您调用您模仿的方法时，`Mock`将调用该函数。mock 共享`.side_effect`函数的参数和返回值:

```py
import requests
import unittest
from unittest.mock import Mock

# Mock requests to control its behavior
requests = Mock() 
def get_holidays():
    r = requests.get('http://localhost/api/holidays')
    if r.status_code == 200:
        return r.json()
    return None

class TestCalendar(unittest.TestCase):
    def log_request(self, url):
        # Log a fake request for test output purposes
        print(f'Making a request to {url}.')
        print('Request received!')

        # Create a new Mock to imitate a Response
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            '12/25': 'Christmas',
            '7/4': 'Independence Day',
        }
        return response_mock

    def test_get_holidays_logging(self):
        # Test a successful, logged request
 requests.get.side_effect = self.log_request        assert get_holidays()['12/25'] == 'Christmas'

if __name__ == '__main__':
    unittest.main()
```

首先，您创建了`.log_request()`，它接受一个 URL，使用 [`print()`](https://realpython.com/python-print/) 记录一些输出，然后返回一个`Mock`响应。接下来，您将`get()`的`.side_effect`设置为`.log_request()`，您将在调用`get_holidays()`时使用它。当您运行测试时，您会看到`get()`将其参数转发给`.log_request()`，然后接受返回值并返回它:

```py
$ python my_calendar.py
Making a request to http://localhost/api/holidays.
Request received!
.
-------------------------------------------------------
Ran 1 test in 0.000s

OK
```

太好了！ [`print()`语句](https://realpython.com/courses/python-print/)记录了正确的值。还有，`get_holidays()`返回了节假日字典。

`.side_effect`也可以是 iterable。iterable 必须由返回值、异常或两者的混合组成。每次调用被模仿的方法时，iterable 都会产生下一个值。例如，您可以测试在`Timeout`返回成功响应后的重试:

```py
import unittest
from requests.exceptions import Timeout
from unittest.mock import Mock

# Mock requests to control its behavior
requests = Mock() 
def get_holidays():
    r = requests.get('http://localhost/api/holidays')
    if r.status_code == 200:
        return r.json()
    return None

class TestCalendar(unittest.TestCase):
    def test_get_holidays_retry(self):
        # Create a new Mock to imitate a Response
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            '12/25': 'Christmas',
            '7/4': 'Independence Day',
        }
        # Set the side effect of .get()
 requests.get.side_effect = [Timeout, response_mock]        # Test that the first request raises a Timeout
        with self.assertRaises(Timeout):
            get_holidays()
        # Now retry, expecting a successful response
        assert get_holidays()['12/25'] == 'Christmas'
        # Finally, assert .get() was called twice
        assert requests.get.call_count == 2

if __name__ == '__main__':
    unittest.main()
```

第一次调用`get_holidays()`，`get()`引出一个`Timeout`。第二次，该方法返回一个有效的假日字典。这些副作用符合它们在传递给`.side_effect`的列表中出现的顺序。

您可以直接在`Mock`上设置`.return_value`和`.side_effect`。但是，因为 Python 模拟对象需要灵活地创建其属性，所以有一种更好的方法来配置这些和其他设置。

[*Remove ads*](/account/join/)

### 配置您的模拟

您可以配置一个`Mock`来设置对象的一些行为。一些可配置的成员包括`.side_effect`、`.return_value`和`.name`。当您[创建](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock)一个或者当您使用 [`.configure_mock()`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.configure_mock) 时，您配置一个`Mock`。

您可以在初始化对象时通过指定某些属性来配置`Mock`:

>>>

```py
>>> mock = Mock(side_effect=Exception)
>>> mock()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 939, in __call__
    return _mock_self._mock_call(*args, **kwargs)
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 995, in _mock_call
    raise effect
Exception

>>> mock = Mock(name='Real Python Mock')
>>> mock
<Mock name='Real Python Mock' id='4434041432'>

>>> mock = Mock(return_value=True)
>>> mock()
True
```

虽然`.side_effect`和`.return_value`可以在`Mock`实例本身上设置，但其他属性如`.name`只能通过`.__init__()`或`.configure_mock()`设置。如果您尝试在实例上设置`Mock`的`.name`，您将得到不同的结果:

>>>

```py
>>> mock = Mock(name='Real Python Mock')
>>> mock.name
<Mock name='Real Python Mock.name' id='4434041544'>

>>> mock = Mock()
>>> mock.name = 'Real Python Mock'
>>> mock.name
'Real Python Mock'
```

`.name`是对象使用的常用属性。因此，`Mock`不允许您像使用`.return_value`或`.side_effect`那样在实例上设置值。如果您访问`mock.name`，您将创建一个`.name`属性，而不是配置您的模拟。

您可以使用`.configure_mock()`配置现有的`Mock`:

>>>

```py
>>> mock = Mock()
>>> mock.configure_mock(return_value=True)
>>> mock()
True
```

通过将字典解包到`.configure_mock()`或`Mock.__init__()`，您甚至可以配置 Python 模拟对象的属性。使用`Mock`配置，您可以简化前面的例子:

```py
# Verbose, old Mock
response_mock = Mock()
response_mock.json.return_value = {
    '12/25': 'Christmas',
    '7/4': 'Independence Day',
}

# Shiny, new .configure_mock()
holidays = {'12/25': 'Christmas', '7/4': 'Independence Day'}
response_mock = Mock(**{'json.return_value': holidays})
```

现在，您可以创建和配置 Python 模拟对象。您还可以使用模拟来控制您的应用程序的行为。到目前为止，您已经使用 mocks 作为函数的参数，或者在测试的同一个模块中修补对象。

接下来，您将学习如何在其他模块中用模拟对象替换真实对象。

## `patch()`

`unittest.mock`提供了一个强大的模仿对象的机制，叫做 [`patch()`](https://docs.python.org/3/library/unittest.mock.html#patch) ，它在给定的模块中查找一个对象，并用一个`Mock`替换那个对象。

通常，您使用`patch()`作为装饰器或上下文管理器来提供一个模仿目标对象的范围。

### `patch()`当装潢师

如果你想在整个测试函数期间模仿一个对象，你可以使用`patch()`作为函数[的装饰者](https://realpython.com/primer-on-python-decorators/)。

要了解这是如何工作的，通过将逻辑和测试放入单独的文件来重新组织您的`my_calendar.py`文件:

```py
import requests
from datetime import datetime

def is_weekday():
    today = datetime.today()
    # Python's datetime library treats Monday as 0 and Sunday as 6
    return (0 <= today.weekday() < 5)

def get_holidays():
    r = requests.get('http://localhost/api/holidays')
    if r.status_code == 200:
        return r.json()
    return None
```

这些函数现在位于它们自己的文件中，与它们的测试分开。接下来，您将在名为`tests.py`的文件中重新创建您的测试。

到目前为止，您已经在对象所在的文件中对它们进行了猴子修补。[猴子补丁](https://en.wikipedia.org/wiki/Monkey_patch)是在运行时用一个对象替换另一个对象。现在，您将使用`patch()`来替换`my_calendar.py`中的对象:

```py
import unittest
from my_calendar import get_holidays
from requests.exceptions import Timeout
from unittest.mock import patch

class TestCalendar(unittest.TestCase):
 @patch('my_calendar.requests')    def test_get_holidays_timeout(self, mock_requests):
            mock_requests.get.side_effect = Timeout
            with self.assertRaises(Timeout):
                get_holidays()
                mock_requests.get.assert_called_once()

if __name__ == '__main__':
    unittest.main()
```

最初，您在本地范围内创建了一个`Mock`并修补了`requests`。现在，你需要从`tests.py`进入`my_calendar.py`的`requests`图书馆。

对于这种情况，您使用了`patch()`作为装饰器，并传递了目标对象的路径。目标路径是由模块名和对象组成的`'my_calendar.requests'`。

您还为测试函数定义了一个新参数。`patch()`使用此参数将被模仿的对象传递到您的测试中。从那里，您可以根据需要修改 mock 或做出断言。

您可以执行这个测试模块来确保它按预期工作:

```py
$ python tests.py
.
-------------------------------------------------------
Ran 1 test in 0.001s

OK
```

**技术细节:** `patch()`返回 [`MagicMock`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.MagicMock) 的一个实例，是`Mock`的子类。`MagicMock`很有用，因为它为你实现了大部分[魔法方法](https://dbader.org/blog/python-dunder-methods)，比如`.__len__()`、`.__str__()`和`.__iter__()`，并且有合理的默认值。

在这个例子中，使用`patch()`作为装饰器效果很好。在某些情况下，使用`patch()`作为上下文管理器更易读、更有效或更容易。

[*Remove ads*](/account/join/)

### `patch()`作为上下文管理器

有时，你会想要使用`patch()`作为[上下文管理器](https://dbader.org/blog/python-context-managers-and-with-statement)而不是装饰器。您可能更喜欢上下文管理器的一些原因包括:

*   您只想在测试范围的一部分模拟一个对象。
*   您已经使用了太多的装饰器或参数，这会损害测试的可读性。

要将`patch()`用作上下文管理器，可以使用 Python 的`with`语句:

```py
import unittest
from my_calendar import get_holidays
from requests.exceptions import Timeout
from unittest.mock import patch

class TestCalendar(unittest.TestCase):
    def test_get_holidays_timeout(self):
 with patch('my_calendar.requests') as mock_requests:            mock_requests.get.side_effect = Timeout
            with self.assertRaises(Timeout):
                get_holidays()
                mock_requests.get.assert_called_once()

if __name__ == '__main__':
    unittest.main()
```

当测试退出`with`语句时，`patch()`用原始对象替换被模仿的对象。

到目前为止，您已经模拟了完整的对象，但有时您只想模拟对象的一部分。

### 修补对象的属性

假设您只想模仿一个对象的一个方法，而不是整个对象。你可以使用 [`patch.object()`](https://docs.python.org/3/library/unittest.mock.html#patch-object) 来完成。

比如，`.test_get_holidays_timeout()`真的只需要模仿`requests.get()`，将其`.side_effect`设置为`Timeout`:

```py
import unittest
from my_calendar import requests, get_holidays
from unittest.mock import patch

class TestCalendar(unittest.TestCase):
 @patch.object(requests, 'get', side_effect=requests.exceptions.Timeout)    def test_get_holidays_timeout(self, mock_requests):
            with self.assertRaises(requests.exceptions.Timeout):
                get_holidays()

if __name__ == '__main__':
    unittest.main()
```

在这个例子中，你只模仿了`get()`，而不是所有的`requests`。其他所有属性保持不变。

`object()`采用与`patch()`相同的配置参数。但是不是传递目标的路径，而是提供目标对象本身作为第一个参数。第二个参数是您试图模仿的目标对象的属性。你也可以像使用`patch()`一样使用`object()`作为上下文管理器。

**延伸阅读:**除了对象和属性，还可以用 [`patch.dict()`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch.dict) 的`patch()`字典。

学习如何使用`patch()`对于模仿其他模块中的对象至关重要。然而，有时目标对象的路径并不明显。

### 哪里打补丁

知道在哪里告诉`patch()`寻找你想要嘲笑的对象是很重要的，因为如果你选择了错误的目标位置，`patch()`的结果可能是你意想不到的。

假设你在用`patch()`嘲讽`my_calendar.py`中的`is_weekday()`:

>>>

```py
>>> import my_calendar
>>> from unittest.mock import patch

>>> with patch('my_calendar.is_weekday'):
...     my_calendar.is_weekday()
...
<MagicMock name='is_weekday()' id='4336501256'>
```

首先，你导入`my_calendar.py`。然后你修补`is_weekday()`，用一个`Mock`替换它。太好了！这是预期的工作。

现在，让我们稍微修改一下这个例子，直接导入函数:

>>>

```py
>>> from my_calendar import is_weekday
>>> from unittest.mock import patch

>>> with patch('my_calendar.is_weekday'):
...     is_weekday()
...
False
```

**注意:**根据您阅读本教程的日期，您的控制台输出可能会显示`True`或`False`。重要的是，输出不是像以前一样的`Mock`。

注意，即使您传递给`patch()`的目标位置没有改变，调用`is_weekday()`的结果也是不同的。这种差异是由于导入函数的方式发生了变化。

将实函数绑定到局部范围。因此，即使您稍后`patch()`该函数，您也会忽略模仿，因为您已经有了对未模仿函数的本地引用。

一个[好的经验法则](https://docs.python.org/3/library/unittest.mock.html#where-to-patch)就是`patch()`被仰望的物体。

在第一个例子中，模仿`'my_calendar.is_weekday()'`是可行的，因为您在`my_calendar`模块中查找函数。在第二个例子中，您有一个对`is_weekday()`的本地引用。因为您使用了在局部范围内找到的函数，所以您应该模仿局部函数:

>>>

```py
>>> from unittest.mock import patch
>>> from my_calendar import is_weekday

>>> with patch('__main__.is_weekday'):
...     is_weekday()
...
<MagicMock name='is_weekday()' id='4502362992'>
```

现在，你牢牢掌握了`patch()`的力量。你已经看到了如何`patch()`对象和属性，以及在哪里修补它们。

接下来，您将看到对象模仿中固有的一些常见问题以及`unittest.mock`提供的解决方案。

[*Remove ads*](/account/join/)

## 常见嘲讽问题

模仿对象会给你的测试带来几个问题。有些问题是嘲讽固有的，有些问题是`unittest.mock`特有的。请记住，本教程中没有提到嘲讽的其他问题。

这里讨论的问题彼此相似，因为它们引起的问题基本上是相同的。在每种情况下，测试断言都是不相关的。虽然每个模仿的意图是有效的，但模仿本身却是无效的。

### 对象接口的变化和拼写错误

类和函数定义一直在变化。当一个对象的接口改变时，任何依赖于该对象的`Mock`的测试都可能变得无关紧要。

例如，您重命名了一个方法，但是忘记了一个测试模拟了这个方法并调用了`.assert_not_called()`。变化之后，`.assert_not_called()`依然是`True`。但是这个断言没有用，因为这个方法已经不存在了。

不相关的测试听起来可能不重要，但是如果它们是您唯一的测试，并且您认为它们工作正常，那么这种情况对您的应用程序来说可能是灾难性的。

一个特定于`Mock`的问题是拼写错误会破坏测试。回想一下，当您访问一个`Mock`的成员时，它会创建自己的接口。因此，如果您拼错了属性的名称，就会无意中创建新属性。

如果你调用`.asert_called()`而不是`.assert_called()`，你的测试将不会产生`AssertionError`。这是因为您已经在 Python 模拟对象上创建了一个名为`.asert_called()`的新方法，而不是评估一个实际的断言。

**技术细节:**有趣的是，`assret`是`assert`的特殊拼错。如果您试图访问一个以`assret`(或`assert`)开头的属性，`Mock`将自动引发一个`AttributeError`。

当您在自己的代码库中模仿对象时，会出现这些问题。当您模仿与外部代码库交互的对象时，会出现一个不同的问题。

### 外部依赖关系的变化

再想象一下，您的代码向外部 API 发出请求。在这种情况下，外部依赖是 API，它容易在未经您同意的情况下被更改。

一方面，单元测试测试代码的独立组件。因此，模拟发出请求的代码有助于您在受控条件下测试隔离的组件。然而，这也带来了一个潜在的问题。

如果一个外部依赖改变了它的接口，你的 Python 模拟对象将变得无效。如果发生这种情况(并且接口变化是破坏性的)，您的测试将会通过，因为您的模拟对象已经屏蔽了这种变化，但是您的生产代码将会失败。

不幸的是，这不是一个`unittest.mock`提供解决方案的问题。嘲笑外部依赖时，你必须运用判断力。

所有这三个问题都可能导致测试无关性和潜在的代价高昂的问题，因为它们威胁到您的模拟的完整性。给你一些处理这些问题的工具。

## 使用规范避免常见问题

如前所述，如果您更改了一个类或函数定义，或者拼错了 Python 模拟对象的属性，那么您的测试就会出现问题。

出现这些问题是因为当您访问属性和方法时,`Mock`会创建它们。这些问题的答案是防止`Mock`创建与您试图模仿的对象不一致的属性。

当配置一个`Mock`时，您可以将一个对象规范传递给`spec`参数。`spec`参数接受一个名称列表或另一个对象，并定义 mock 的接口。如果您试图访问一个不属于规范的属性，`Mock`将引发一个`AttributeError`:

>>>

```py
>>> from unittest.mock import Mock
>>> calendar = Mock(spec=['is_weekday', 'get_holidays'])

>>> calendar.is_weekday()
<Mock name='mock.is_weekday()' id='4569015856'>
>>> calendar.create_event()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 582, in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
AttributeError: Mock object has no attribute 'create_event'
```

这里，您已经指定了`calendar`具有名为`.is_weekday()`和`.get_holidays()`的方法。当你访问`.is_weekday()`时，它返回一个`Mock`。当您访问`.create_event()`时，一个与规范不匹配的方法`Mock`会引发一个`AttributeError`。

如果用对象配置`Mock`,规格的工作方式相同:

>>>

```py
>>> import my_calendar
>>> from unittest.mock import Mock

>>> calendar = Mock(spec=my_calendar)
>>> calendar.is_weekday()
<Mock name='mock.is_weekday()' id='4569435216'>
>>> calendar.create_event()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 582, in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
AttributeError: Mock object has no attribute 'create_event'
```

`.is_weekday()`对`calendar`可用，因为您配置了`calendar`来匹配`my_calendar`模块的接口。

此外，`unittest.mock`提供了自动指定`Mock`实例的接口的便利方法。

实现自动规格的一种方法是`create_autospec`:

>>>

```py
>>> import my_calendar
>>> from unittest.mock import create_autospec

>>> calendar = create_autospec(my_calendar)
>>> calendar.is_weekday()
<MagicMock name='mock.is_weekday()' id='4579049424'>
>>> calendar.create_event()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 582, in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
AttributeError: Mock object has no attribute 'create_event'
```

像以前一样，`calendar`是一个`Mock`实例，它的接口匹配`my_calendar`。如果您正在使用`patch()`，您可以向`autospec`参数发送一个参数来获得相同的结果:

>>>

```py
>>> import my_calendar
>>> from unittest.mock import patch

>>> with patch('__main__.my_calendar', autospec=True) as calendar:
...     calendar.is_weekday()
...     calendar.create_event()
...
<MagicMock name='my_calendar.is_weekday()' id='4579094312'>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py", line 582, in __getattr__
    raise AttributeError("Mock object has no attribute %r" % name)
AttributeError: Mock object has no attribute 'create_event'
```

[*Remove ads*](/account/join/)

## 结论

你已经学到了很多关于使用`unittest.mock`模仿物体的知识！

现在，您能够:

*   在你的测试中使用`Mock`来模仿物体
*   检查使用数据以了解如何使用对象
*   定制模拟对象的返回值和副作用
*   整个代码库中的对象
*   查看和避免使用 Python 模拟对象的问题

您已经建立了理解的基础，这将帮助您构建更好的测试。您可以使用模拟来深入了解您的代码，否则您将无法获得这些信息。

我留给你最后一个免责声明。当心过度使用模仿对象！

很容易利用 Python 模拟对象的强大功能，并且模拟得如此之多，以至于实际上降低了测试的价值。

如果你有兴趣了解更多关于`unittest.mock`的信息，我鼓励你阅读它优秀的[文档](https://docs.python.org/3/library/unittest.mock.html)。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，加深您的理解: [**使用 Python 模拟对象库**](/courses/python-mock-object-library/) 改进您的测试*********
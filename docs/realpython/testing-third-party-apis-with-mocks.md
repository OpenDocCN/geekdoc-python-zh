# 在 Python 中模仿外部 API

> 原文：<https://realpython.com/testing-third-party-apis-with-mocks/>

下面的教程演示了如何使用 [Python 模拟对象](https://realpython.com/python-mock-library/)来测试外部 API 的使用。

与第三方应用程序集成是扩展产品功能的好方法。

然而，附加值也伴随着障碍。您不拥有外部库，这意味着您无法控制托管它的服务器、组成其逻辑的代码或在它和您的应用程序之间传输的数据。在这些问题之上，用户通过与库的交互不断地操纵数据。

如果你想用第三方 API 增强你的应用程序的实用性，那么你需要确信这两个系统会很好的运行。您需要测试这两个应用程序以可预测的方式接口，并且您需要您的测试在一个受控的环境中执行。

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

乍一看，您似乎对第三方应用程序没有任何控制权。他们中的许多人不提供测试服务器。您不能测试实时数据，即使您可以，测试也会返回不可靠的结果，因为数据是通过使用更新的。此外，您永远不希望您的自动化测试连接到外部服务器。如果发布代码依赖于您的测试是否通过，那么他们那边的错误可能会导致您的开发停止。幸运的是，有一种方法可以在受控环境中测试第三方 API 的实现，而不需要实际连接到外部数据源。解决方案是使用所谓的模仿来伪造外部代码的功能。

mock 是一个假对象，您构建它的目的是让它看起来和行为起来像真实的数据。您将它与实际对象交换，并欺骗系统，使其认为模拟是真实的交易。使用模拟让我想起了一个经典的电影比喻，主人公抓住一个亲信，穿上制服，步入行进中的敌人行列。没有人注意到这个冒名顶替者，每个人都继续前进——一切照旧。

第三方认证，比如 OAuth，是在你的应用中模仿的一个很好的选择。OAuth 要求您的应用程序与外部服务器通信，它涉及真实的用户数据，并且您的应用程序依赖于它的成功来访问它的 API。模拟认证允许您作为授权用户[测试您的系统](https://realpython.com/python-testing/)，而不必经历实际的凭证交换过程。在这种情况下，您不想测试您的系统是否成功地对用户进行了身份验证；你想测试你的应用程序的功能在通过认证后*的表现。*

> 注意:本教程使用 Python 3 . 5 . 1 版。

## 第一步

首先建立一个新的开发环境来保存您的项目代码。创建一个新的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)，然后安装以下库:

```py
$ pip install nose requests
```

以下是您正在安装的每个库的快速概要，以防您从未遇到过它们:

*   [模拟](https://docs.python.org/3.5/library/unittest.mock.html)库通过用模拟对象替换部分系统来测试 Python 代码。*注意:如果您使用的是 Python 3.3 或更高版本，`mock`库是`unittest`的一部分。如果您使用的是旧版本，请安装 [backport mock](https://github.com/testing-cabal/mock) 库。*
*   [nose](http://nose.readthedocs.org/en/latest/) 库扩展了内置的 Python `unittest`模块，使得测试更加容易。你可以使用`unittest`或者其他第三方库比如 [*pytest*](https://realpython.com/pytest-python-testing/) 来达到同样的效果，但是我更喜欢 *nose* 的断言方法。
*   [请求](https://realpython.com/python-requests/)库极大地简化了 Python 中的 HTTP 调用。

在本教程中，您将与一个为测试而构建的假在线 API 进行通信- [JSON 占位符](http://jsonplaceholder.typicode.com/)。在您编写任何测试之前，您需要知道 API 会带来什么。

首先，您应该期望当您向您的目标 API 发送请求时，它实际上会返回一个响应。通过用 *cURL* 调用端点来确认这个假设:

```py
$ curl -X GET 'http://jsonplaceholder.typicode.com/todos'
```

这个调用应该返回一个 JSON 序列化的 todo 项目列表。注意响应中 todo 数据的结构。您应该会看到带有关键字`userId`、`id`、`title`和`completed`的对象列表。现在，您准备进行第二个假设——您知道预期的数据是什么样的。API 端点处于活动状态并正常工作。您可以通过从命令行调用它来证明这一点。现在，编写一个 *nose* 测试，这样你就可以确认未来服务器的寿命。保持简单。您应该只关心服务器是否返回 OK 响应。

**项目/测试/测试 _todos.py**

```py
# Third-party imports...
from nose.tools import assert_true
import requests

def test_request_response():
    # Send a request to the API server and store the response.
    response = requests.get('http://jsonplaceholder.typicode.com/todos')

    # Confirm that the request-response cycle completed successfully.
    assert_true(response.ok)
```

运行测试并观察其通过:

```py
$ nosetests --verbosity=2 project
test_todos.test_request_response ... ok

----------------------------------------------------------------------
Ran 1 test in 9.270s

OK
```

[*Remove ads*](/account/join/)

## 将代码重构为服务

在整个应用程序中，您很可能会多次调用外部 API。此外，这些 API 调用可能会涉及比简单的 HTTP 请求更多的逻辑，比如数据处理、错误处理和过滤。您应该从您的测试中提取代码，并[将它重构为一个封装了所有预期逻辑的服务函数。](https://realpython.com/python-refactoring/)

重写您的测试以引用服务函数并测试新的逻辑。

**项目/测试/测试 _todos.py**

```py
# Third-party imports...
from nose.tools import assert_is_not_none

# Local imports...
from project.services import get_todos

def test_request_response():
    # Call the service, which will send a request to the server.
    response = get_todos()

    # If the request is sent successfully, then I expect a response to be returned.
    assert_is_not_none(response)
```

运行测试并观察其失败，然后编写最少的代码使其通过:

**project/services.py**

```py
# Standard library imports...
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

# Third-party imports...
import requests

# Local imports...
from project.constants import BASE_URL

TODOS_URL = urljoin(BASE_URL, 'todos')

def get_todos():
    response = requests.get(TODOS_URL)
    if response.ok:
        return response
    else:
        return None
```

**project/constants.py**

```py
BASE_URL = 'http://jsonplaceholder.typicode.com'
```

您编写的第一个测试期望返回一个状态为 OK 的响应。您将编程逻辑重构为一个服务函数，当对服务器的请求成功时，该服务函数会返回响应本身。如果请求失败，则返回一个 [`None`](https://realpython.com/null-in-python/) 值。测试现在包括一个断言来确认函数不返回`None`。

注意我是如何指示您创建一个`constants.py`文件，然后用一个`BASE_URL`填充它的。服务函数扩展了`BASE_URL`来创建`TODOS_URL`，由于所有的 API 端点都使用相同的基础，您可以继续创建新的端点，而不必重写代码。将`BASE_URL`放在一个单独的文件中允许您在一个地方编辑它，如果多个模块引用该代码，这将很方便。

运行测试，看着它通过。

```py
$ nosetests --verbosity=2 project
test_todos.test_request_response ... ok

----------------------------------------------------------------------
Ran 1 test in 1.475s

OK
```

## 你的第一次嘲弄

代码按预期运行。你知道这一点，因为你有一个通过测试。不幸的是，您有一个问题-您的服务功能仍然是直接访问外部服务器。当您调用`get_todos()`时，您的代码正在向 API 端点发出请求，并返回一个依赖于该服务器活动的结果。在这里，我将演示如何通过用一个返回相同数据的假请求来交换真实请求，从而将您的编程逻辑与实际的外部库分离。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest.mock import Mock, patch

# Third-party imports...
from nose.tools import assert_is_not_none

# Local imports...
from project.services import get_todos

@patch('project.services.requests.get')
def test_getting_todos(mock_get):
    # Configure the mock to return a response with an OK status code.
    mock_get.return_value.ok = True

    # Call the service, which will send a request to the server.
    response = get_todos()

    # If the request is sent successfully, then I expect a response to be returned.
    assert_is_not_none(response)
```

请注意，我根本没有更改服务函数。我编辑的唯一一部分代码是测试本身。首先，我从`mock`库中导入了`patch()`函数。接下来，我用作为装饰器的`patch()`函数修改了测试函数，传入了对`project.services.requests.get`的引用。在函数本身中，我传入了一个参数`mock_get`，然后在测试函数体中，我添加了一行来设置`mock_get.return_value.ok = True`。

太好了。那么当测试运行时，实际上会发生什么呢？在我深入研究之前，您需要了解一些关于`requests`库的工作方式。当您调用`requests.get()`函数时，它在后台发出一个 HTTP 请求，然后以`Response`对象的形式返回一个 HTTP 响应。`get()`函数本身与外部服务器通信，这就是为什么您需要将它作为目标。还记得主人公穿着制服与敌人交换位置的画面吗？你需要让这个模拟看起来和行为起来像`requests.get()`函数。

当测试函数运行时，它找到声明了`requests`库的模块`project.services`，并用一个 mock 替换目标函数`requests.get()`。测试还告诉 mock 按照服务功能期望的方式进行操作。如果你看一下`get_todos()`，你会发现函数的成功取决于`if response.ok:`返回`True`。这就是生产线`mock_get.return_value.ok = True`正在做的事情。当在 mock 上调用`ok`属性时，它将像实际对象一样返回`True`。`get_todos()`函数将返回响应，也就是 mock，测试将通过，因为 mock 不是`None`。

运行测试以查看它是否通过。

```py
$ nosetests --verbosity=2 project
```

[*Remove ads*](/account/join/)

## 其他方式打补丁

使用装饰器只是用模仿来修补函数的几种方法之一。在下一个例子中，我使用上下文管理器在代码块中显式地修补一个函数。 [`with`语句](https://realpython.com/python-with-statement/)为代码块中任何代码使用的函数打补丁。当代码块结束时，原始函数被恢复。`with`语句和装饰器实现了相同的目标:两种方法都修补了`project.services.request.get`。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest.mock import patch

# Third-party imports...
from nose.tools import assert_is_not_none

# Local imports...
from project.services import get_todos

def test_getting_todos():
    with patch('project.services.requests.get') as mock_get:
        # Configure the mock to return a response with an OK status code.
        mock_get.return_value.ok = True

        # Call the service, which will send a request to the server.
        response = get_todos()

    # If the request is sent successfully, then I expect a response to be returned.
    assert_is_not_none(response)
```

运行测试以查看它们是否仍然通过。

修补函数的另一种方法是使用修补程序。在这里，我确定要修补的源代码，然后显式地开始使用 mock。直到我明确地告诉系统停止使用 mock，修补才会停止。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest.mock import patch

# Third-party imports...
from nose.tools import assert_is_not_none

# Local imports...
from project.services import get_todos

def test_getting_todos():
    mock_get_patcher = patch('project.services.requests.get')

    # Start patching `requests.get`.
    mock_get = mock_get_patcher.start()

    # Configure the mock to return a response with an OK status code.
    mock_get.return_value.ok = True

    # Call the service, which will send a request to the server.
    response = get_todos()

    # Stop patching `requests.get`.
    mock_get_patcher.stop()

    # If the request is sent successfully, then I expect a response to be returned.
    assert_is_not_none(response)
```

再次运行测试以获得相同的成功结果。

既然您已经看到了用 mock 修补函数的三种方法，那么您应该在什么时候使用每种方法呢？简短的回答是:这完全取决于你。每种打补丁的方法都是完全有效的。也就是说，我发现特定的编码模式与以下修补方法配合得特别好。

1.  当你的测试函数体中的所有代码都使用模仿时，使用装饰器。
2.  当你的测试函数中的一些代码使用了模拟代码，而其他代码引用了实际的函数时，使用上下文管理器。
3.  **当你需要在多个测试中明确地开始和停止模仿一个函数时，使用一个补丁**(例如，一个测试类中的`setUp()`和`tearDown()`函数)。

我在本教程中使用了这些方法中的每一种，并且我将在第一次介绍它们时重点介绍每一种方法。

## 嘲讽完整服务行为

在前面的例子中，您实现了一个基本的模拟，并测试了一个简单的断言——函数`get_todos()`是否返回了`None`。`get_todos()`函数调用外部 API 并接收响应。如果调用成功，该函数将返回一个响应对象，其中包含一个 JSON 序列化的 todos 列表。如果请求失败，`get_todos()`返回`None`。在下面的例子中，我演示了如何模拟`get_todos()`的全部功能。在本教程的开始，您使用 *cURL* 对服务器进行的第一次调用返回了一个 JSON 序列化的字典列表，它表示 todo 项。这个例子将向您展示如何模拟数据。

记住`@patch()`是如何工作的:你给它提供一个到你想要模仿的函数的路径。找到了函数，`patch()`创建了一个`Mock`对象，真正的函数被暂时替换成了 mock。当测试调用`get_todos()`时，该函数使用`mock_get`的方式与使用真实的`get()`方法的方式相同。这意味着它像函数一样调用`mock_get`,并期望它返回一个响应对象。

在这种情况下，响应对象是一个`requests`库`Response`对象，它有几个属性和方法。在前面的例子中，您伪造了其中一个属性`ok`。`Response`对象还有一个`json()`函数，它将其 JSON 序列化的字符串内容转换成 Python 数据类型(例如`list`或`dict`)。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest.mock import Mock, patch

# Third-party imports...
from nose.tools import assert_is_none, assert_list_equal

# Local imports...
from project.services import get_todos

@patch('project.services.requests.get')
def test_getting_todos_when_response_is_ok(mock_get):
    todos = [{
        'userId': 1,
        'id': 1,
        'title': 'Make the bed',
        'completed': False
    }]

    # Configure the mock to return a response with an OK status code. Also, the mock should have
    # a `json()` method that returns a list of todos.
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.json.return_value = todos

    # Call the service, which will send a request to the server.
    response = get_todos()

    # If the request is sent successfully, then I expect a response to be returned.
    assert_list_equal(response.json(), todos)

@patch('project.services.requests.get')
def test_getting_todos_when_response_is_not_ok(mock_get):
    # Configure the mock to not return a response with an OK status code.
    mock_get.return_value.ok = False

    # Call the service, which will send a request to the server.
    response = get_todos()

    # If the response contains an error, I should get no todos.
    assert_is_none(response)
```

我在前面的例子中提到，当您运行用 mock 修补的`get_todos()`函数时，该函数返回一个 mock 对象“response”。您可能已经注意到了一个模式:每当`return_value`被添加到一个 mock 时，该 mock 被修改为作为一个函数运行，并且默认情况下它返回另一个 mock 对象。在这个例子中，我通过显式声明`Mock`对象`mock_get.return_value = Mock(ok=True)`使这一点更加清楚。`mock_get()`镜像`requests.get()`，而`requests.get()`返回`Response`，而`mock_get()`返回`Mock`。`Response`对象有一个`ok`属性，所以您向`Mock`添加了一个`ok`属性。

`Response`对象也有一个`json()`函数，所以我给`Mock`添加了`json`，并给它附加了一个`return_value`，因为它会像函数一样被调用。`json()`函数返回一个 todo 对象列表。注意，测试现在包括了一个检查`response.json()`值的断言。您希望确保`get_todos()`函数返回 todos 列表，就像实际的服务器一样。最后，为了完善对`get_todos()`的测试，我添加了一个失败测试。

运行测试并观察它们是否通过。

```py
$ nosetests --verbosity=2 project
test_todos.test_getting_todos_when_response_is_not_ok ... ok
test_todos.test_getting_todos_when_response_is_ok ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.285s

OK
```

[*Remove ads*](/account/join/)

## 模拟集成功能

我给你看的例子相当简单，在下一个例子中，我会增加复杂性。想象一个场景，您创建一个新的服务函数，调用`get_todos()`，然后过滤这些结果，只返回已经完成的待办事项。非要再嘲讽一下`requests.get()`吗？不，在这种情况下你可以直接模仿`get_todos()`函数！请记住，当您模仿一个函数时，您是在用模仿对象替换实际对象，您只需要担心服务函数如何与模仿对象交互。在`get_todos()`的例子中，您知道它没有参数，它返回一个带有`json()`函数的响应，该函数返回一个 todo 对象列表。你不关心引擎盖下发生了什么；您只需要关心`get_todos()` mock 返回您期望真正的`get_todos()`函数返回的内容。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest.mock import Mock, patch

# Third-party imports...
from nose.tools import assert_list_equal, assert_true

# Local imports...
from project.services import get_uncompleted_todos

@patch('project.services.get_todos')
def test_getting_uncompleted_todos_when_todos_is_not_none(mock_get_todos):
    todo1 = {
        'userId': 1,
        'id': 1,
        'title': 'Make the bed',
        'completed': False
    }
    todo2 = {
        'userId': 1,
        'id': 2,
        'title': 'Walk the dog',
        'completed': True
    }

    # Configure mock to return a response with a JSON-serialized list of todos.
    mock_get_todos.return_value = Mock()
    mock_get_todos.return_value.json.return_value = [todo1, todo2]

    # Call the service, which will get a list of todos filtered on completed.
    uncompleted_todos = get_uncompleted_todos()

    # Confirm that the mock was called.
    assert_true(mock_get_todos.called)

    # Confirm that the expected filtered list of todos was returned.
    assert_list_equal(uncompleted_todos, [todo1])

@patch('project.services.get_todos')
def test_getting_uncompleted_todos_when_todos_is_none(mock_get_todos):
    # Configure mock to return None.
    mock_get_todos.return_value = None

    # Call the service, which will return an empty list.
    uncompleted_todos = get_uncompleted_todos()

    # Confirm that the mock was called.
    assert_true(mock_get_todos.called)

    # Confirm that an empty list was returned.
    assert_list_equal(uncompleted_todos, [])
```

请注意，现在我正在修补测试函数，以查找并使用模拟替换`project.services.get_todos`。mock 函数应该返回一个具有`json()`函数的对象。当被调用时，`json()`函数应该返回一个 todo 对象列表。我还添加了一个断言来确认`get_todos()`函数确实被调用了。这有助于确定当服务函数访问实际的 API 时，真正的`get_todos()`函数将会执行。这里，我还包含了一个测试来验证如果`get_todos()`返回`None`，那么`get_uncompleted_todos()`函数将返回一个空列表。我再次确认调用了`get_todos()`函数。

编写测试，运行它们以查看它们是否失败，然后编写必要的代码使它们通过。

**project/services.py**

```py
def get_uncompleted_todos():
    response = get_todos()
    if response is None:
        return []
    else:
        todos = response.json()
        return [todo for todo in todos if todo.get('completed') == False]
```

测试现在通过了。

## 重构测试以使用类

您可能已经注意到，有些测试似乎属于同一个组。您有两个测试碰到了`get_todos()`函数。你的另外两个测试重点是`get_uncompleted_todos()`。每当我开始注意到测试之间的趋势和相似之处，我就将它们重构为一个测试类。这种重构实现了几个目标:

1.  将常见的测试函数移动到一个类中，可以让您更容易地将它们作为一个组一起测试。您可以告诉 *nose* 以一系列函数为目标，但是以单个类为目标更容易。
2.  常见的测试功能通常需要相似的步骤来创建和销毁每个测试所使用的数据。这些步骤可以分别封装在`setup_class()`和`teardown_class()`函数中，以便在适当的阶段执行代码。
3.  您可以在类上创建实用函数，以重用测试函数中重复的逻辑。想象一下，必须在每个函数中单独调用相同的数据创建逻辑。多痛苦啊！

注意，我使用了**补丁**技术来模拟测试类中的目标函数。正如我前面提到的，这种修补方法非常适合创建跨越多个函数的模拟。当测试完成时，`teardown_class()`方法中的代码显式地恢复原始代码。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest.mock import Mock, patch

# Third-party imports...
from nose.tools import assert_is_none, assert_list_equal, assert_true

# Local imports...
from project.services import get_todos, get_uncompleted_todos

class TestTodos(object):
    @classmethod
    def setup_class(cls):
        cls.mock_get_patcher = patch('project.services.requests.get')
        cls.mock_get = cls.mock_get_patcher.start()

    @classmethod
    def teardown_class(cls):
        cls.mock_get_patcher.stop()

    def test_getting_todos_when_response_is_ok(self):
        # Configure the mock to return a response with an OK status code.
        self.mock_get.return_value.ok = True

        todos = [{
            'userId': 1,
            'id': 1,
            'title': 'Make the bed',
            'completed': False
        }]

        self.mock_get.return_value = Mock()
        self.mock_get.return_value.json.return_value = todos

        # Call the service, which will send a request to the server.
        response = get_todos()

        # If the request is sent successfully, then I expect a response to be returned.
        assert_list_equal(response.json(), todos)

    def test_getting_todos_when_response_is_not_ok(self):
        # Configure the mock to not return a response with an OK status code.
        self.mock_get.return_value.ok = False

        # Call the service, which will send a request to the server.
        response = get_todos()

        # If the response contains an error, I should get no todos.
        assert_is_none(response)

class TestUncompletedTodos(object):
    @classmethod
    def setup_class(cls):
        cls.mock_get_todos_patcher = patch('project.services.get_todos')
        cls.mock_get_todos = cls.mock_get_todos_patcher.start()

    @classmethod
    def teardown_class(cls):
        cls.mock_get_todos_patcher.stop()

    def test_getting_uncompleted_todos_when_todos_is_not_none(self):
        todo1 = {
            'userId': 1,
            'id': 1,
            'title': 'Make the bed',
            'completed': False
        }
        todo2 = {
            'userId': 2,
            'id': 2,
            'title': 'Walk the dog',
            'completed': True
        }

        # Configure mock to return a response with a JSON-serialized list of todos.
        self.mock_get_todos.return_value = Mock()
        self.mock_get_todos.return_value.json.return_value = [todo1, todo2]

        # Call the service, which will get a list of todos filtered on completed.
        uncompleted_todos = get_uncompleted_todos()

        # Confirm that the mock was called.
        assert_true(self.mock_get_todos.called)

        # Confirm that the expected filtered list of todos was returned.
        assert_list_equal(uncompleted_todos, [todo1])

    def test_getting_uncompleted_todos_when_todos_is_none(self):
        # Configure mock to return None.
        self.mock_get_todos.return_value = None

        # Call the service, which will return an empty list.
        uncompleted_todos = get_uncompleted_todos()

        # Confirm that the mock was called.
        assert_true(self.mock_get_todos.called)

        # Confirm that an empty list was returned.
        assert_list_equal(uncompleted_todos, [])
```

进行测试。一切都会过去的，因为你没有引入任何新的逻辑。你只是移动了代码。

```py
$ nosetests --verbosity=2 project
test_todos.TestTodos.test_getting_todos_when_response_is_not_ok ... ok
test_todos.TestTodos.test_getting_todos_when_response_is_ok ... ok
test_todos.TestUncompletedTodos.test_getting_uncompleted_todos_when_todos_is_none ... ok
test_todos.TestUncompletedTodos.test_getting_uncompleted_todos_when_todos_is_not_none ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.300s

OK
```

## 测试 API 数据的更新

在本教程中，我一直在演示如何模拟第三方 API 返回的数据。这个模拟数据基于一个假设，即真实数据和你伪造的数据使用相同的数据契约。您的第一步是调用实际的 API 并记录返回的数据。您可以相当自信地认为，在您研究这些示例的短时间内，数据的结构没有发生变化，但是，您不应该确信数据会永远保持不变。任何好的外部库都会定期更新。虽然开发人员的目标是使新代码向后兼容，但最终会有代码被弃用的时候。

可以想象，完全依赖假数据是很危险的。因为您是在不与实际服务器通信的情况下测试代码，所以您很容易对测试的强度过于自信。当需要将您的应用程序与真实数据结合使用时，一切都会崩溃。应该使用以下策略来确认您期望从服务器获得的数据与您正在测试的数据相匹配。*这里的目标是比较数据结构(如对象中的键)而不是实际数据。*

注意我是如何使用**上下文管理器**修补技术的。在这里，你需要调用真正的服务器*和*你需要分别模拟它。

**项目/测试/测试 _todos.py**

```py
def test_integration_contract():
    # Call the service to hit the actual API.
    actual = get_todos()
    actual_keys = actual.json().pop().keys()

    # Call the service to hit the mocked API.
    with patch('project.services.requests.get') as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = [{
            'userId': 1,
            'id': 1,
            'title': 'Make the bed',
            'completed': False
        }]

        mocked = get_todos()
        mocked_keys = mocked.json().pop().keys()

    # An object from the actual API and an object from the mocked API should have
    # the same data structure.
    assert_list_equal(list(actual_keys), list(mocked_keys))
```

你的测试应该会通过。您模拟的数据结构与实际 API 中的数据结构相匹配。

[*Remove ads*](/account/join/)

## 有条件测试场景

现在您有了一个测试来比较实际的数据契约和模拟的数据契约，您需要知道何时运行它。击中真实服务器的测试不应该是自动化的，因为失败不一定意味着您的代码是坏的。由于十几个您无法控制的原因，您可能无法在测试套件执行时连接到真实的服务器。与自动化测试分开运行这个测试，但是也要相当频繁地运行它。有选择地跳过测试的一种方法是使用环境变量作为开关。在下面的例子中，所有的测试都会运行，除非环境变量`SKIP_REAL`被设置为`True`。当`SKIP_REAL`变量被打开时，任何带有`@skipIf(SKIP_REAL)`装饰器的测试都将被跳过。

**项目/测试/测试 _todos.py**

```py
# Standard library imports...
from unittest import skipIf

# Local imports...
from project.constants import SKIP_REAL

@skipIf(SKIP_REAL, 'Skipping tests that hit the real API server.')
def test_integration_contract():
    # Call the service to hit the actual API.
    actual = get_todos()
    actual_keys = actual.json().pop().keys()

    # Call the service to hit the mocked API.
    with patch('project.services.requests.get') as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = [{
            'userId': 1,
            'id': 1,
            'title': 'Make the bed',
            'completed': False
        }]

        mocked = get_todos()
        mocked_keys = mocked.json().pop().keys()

    # An object from the actual API and an object from the mocked API should have
    # the same data structure.
    assert_list_equal(list(actual_keys), list(mocked_keys))
```

**project/constants.py**

```py
# Standard-library imports...
import os

BASE_URL = 'http://jsonplaceholder.typicode.com'
SKIP_REAL = os.getenv('SKIP_REAL', False)
```

```py
$ export SKIP_REAL=True
```

运行测试并注意输出。一个测试被忽略，控制台显示消息“跳过命中实际 API 服务器的测试”太棒了。

```py
$ nosetests --verbosity=2 project
test_todos.TestTodos.test_getting_todos_when_response_is_not_ok ... ok
test_todos.TestTodos.test_getting_todos_when_response_is_ok ... ok
test_todos.TestUncompletedTodos.test_getting_uncompleted_todos_when_todos_is_none ... ok
test_todos.TestUncompletedTodos.test_getting_uncompleted_todos_when_todos_is_not_none ... ok
test_todos.test_integration_contract ... SKIP: Skipping tests that hit the real API server.

----------------------------------------------------------------------
Ran 5 tests in 0.240s

OK (SKIP=1)
```

## 接下来的步骤

至此，您已经看到了如何使用 mocks 测试您的应用程序与第三方 API 的集成。既然您已经知道了如何解决这个问题，那么您可以继续练习为 JSON 占位符中的其他 API 端点编写服务函数(例如，帖子、评论、用户)。

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

通过将你的应用程序连接到一个真实的外部库，如谷歌、脸书或 Evernote，进一步提高你的技能，看看你是否能编写使用模拟的测试。继续编写干净可靠的代码，并关注下一篇教程，它将描述如何使用[模拟服务器](https://realpython.com/testing-third-party-apis-with-mock-servers/)将测试提升到一个新的水平！

从[回购](https://github.com/realpython/python-mocks)中抓取代码。****
# 用模拟服务器测试外部 API

> 原文：<https://realpython.com/testing-third-party-apis-with-mock-servers/>

尽管如此有用，外部 API 测试起来还是很痛苦。当您遇到一个实际的 API 时，您的测试会受到外部服务器的支配，这会导致以下棘手问题:

*   请求-响应周期可能需要几秒钟。起初，这可能看起来不多，但随着每次测试的进行，时间会越来越长。想象一下，在测试整个应用程序时，调用一个 API 10 次、50 次甚至 100 次。
*   API 可以设置速率限制。
*   API 服务器可能无法访问。可能服务器停机维护了？也许它因为一个错误而失败了，开发团队正在努力让它再次正常工作>您真的希望测试的成功依赖于您无法控制的服务器的健康吗？

你的测试不应该评估一个 API 服务器是否正在运行；他们应该测试你的代码是否按预期运行。

在[之前的教程](https://realpython.com/testing-third-party-apis-with-mocks/)中，我们介绍了[模拟对象](https://realpython.com/python-mock-library/)的概念，演示了如何使用它们来测试与外部 API 交互的代码。**本教程建立在相同的主题上，但是在这里我们将带你实际构建一个模拟服务器，而不是模拟 API。有了模拟服务器，您可以执行端到端的测试。您可以使用您的应用程序，并从模拟服务器实时获得实际的反馈。**

当您完成下面的示例时，您将已经编写了一个基本的模拟服务器和两个测试——一个使用真正的 API 服务器，另一个使用模拟服务器。这两个测试将访问相同的服务，一个检索用户列表的 API。

> 注意:本教程使用 Python 3 . 5 . 1 版。

## 开始使用

从上一篇文章的[第一步](https://realpython.com/testing-third-party-apis-with-mocks/#first-steps)开始。或者从[库](https://github.com/realpython/python-mock-server/releases/tag/v1)中抓取代码。在继续之前，确保测试通过:

```py
$ nosetests --verbosity=2 project
test_todos.test_request_response ... ok

----------------------------------------------------------------------
Ran 1 test in 1.029s

OK
```

[*Remove ads*](/account/join/)

## 测试模拟 API

设置完成后，您就可以对模拟服务器进行编程了。编写描述该行为的测试:

**project/tests/test _ mock _ server . py**

```py
# Third-party imports...
from nose.tools import assert_true
import requests

def test_request_response():
    url = 'http://localhost:{port}/users'.format(port=mock_server_port)

    # Send a request to the mock API server and store the response.
    response = requests.get(url)

    # Confirm that the request-response cycle completed successfully.
    assert_true(response.ok)
```

注意，它开始看起来几乎与真正的 API 测试一样。URL 已经更改，现在指向模拟服务器将运行的本地主机上的 API 端点。

下面是如何用 Python 创建一个模拟服务器:

**project/tests/test _ mock _ server . py**

```py
# Standard library imports...
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
from threading import Thread

# Third-party imports...
from nose.tools import assert_true
import requests

class MockServerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Process an HTTP GET request and return a response with an HTTP 200 status.
        self.send_response(requests.codes.ok)
        self.end_headers()
        return

def get_free_port():
    s = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    address, port = s.getsockname()
    s.close()
    return port

class TestMockServer(object):
    @classmethod
    def setup_class(cls):
        # Configure mock server.
        cls.mock_server_port = get_free_port()
        cls.mock_server = HTTPServer(('localhost', cls.mock_server_port), MockServerRequestHandler)

        # Start running mock server in a separate thread.
        # Daemon threads automatically shut down when the main process exits.
        cls.mock_server_thread = Thread(target=cls.mock_server.serve_forever)
        cls.mock_server_thread.setDaemon(True)
        cls.mock_server_thread.start()

    def test_request_response(self):
        url = 'http://localhost:{port}/users'.format(port=self.mock_server_port)

        # Send a request to the mock API server and store the response.
        response = requests.get(url)

        # Confirm that the request-response cycle completed successfully.
        print(response)
        assert_true(response.ok)
```

首先创建一个`BaseHTTPRequestHandler`的子类。这个类捕获请求并构造返回的响应。覆盖`do_GET()`函数，为 HTTP GET 请求创建响应。在这种情况下，只需返回一个 OK 状态。接下来，编写一个函数来获取可供模拟服务器使用的端口号。

下一个代码块实际上配置了服务器。注意代码是如何实例化一个`HTTPServer`实例并传递给它一个端口号和一个处理程序的。接下来创建一个[线程](https://realpython.com/intro-to-python-threading/)，这样服务器就可以异步运行，并且你的主程序线程可以与之通信。让线程成为守护进程，当主程序退出时，守护进程告诉线程停止。最后，启动线程永远为模拟服务器服务(直到测试完成)。

创建一个测试类，并将测试函数移动到其中。您必须添加一个额外的方法来确保在任何测试运行之前启动模拟服务器。注意，这个新代码位于一个特殊的类级函数`setup_class()`中。

运行测试并观察它们是否通过:

```py
$ nosetests --verbosity=2 project
```

## 测试符合 API 的服务

您可能希望在代码中调用多个 API 端点。在设计应用程序时，您可能会创建服务函数来向 API 发送请求，然后以某种方式处理响应。也许您会将响应数据存储在数据库中。或者将数据传递给用户界面。

重构您的代码，将硬编码的 API 基 URL 提取到一个常量中。将该变量添加到一个 *constants.py* 文件中:

**project/constants.py**

```py
BASE_URL = 'http://jsonplaceholder.typicode.com'
```

接下来，将从 API 检索用户的逻辑封装到一个函数中。请注意如何通过将 URL 路径连接到基础来创建新的 URL。

**project/services.py**

```py
# Standard library imports...
from urllib.parse import urljoin

# Third-party imports...
import requests

# Local imports...
from project.constants import BASE_URL

USERS_URL = urljoin(BASE_URL, 'users')

def get_users():
    response = requests.get(USERS_URL)
    if response.ok:
        return response
    else:
        return None
```

将模拟服务器代码从特性文件移动到一个新的 Python 文件中，这样就可以很容易地重用它。向请求处理程序添加条件逻辑，以检查 HTTP 请求的目标是哪个 API 端点。通过添加一些简单的头信息和基本的响应负载来增强响应。服务器创建和启动代码可以封装在一个方便的方法`start_mock_server()`中。

**项目/测试/模拟. py**

```py
# Standard library imports...
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
import socket
from threading import Thread

# Third-party imports...
import requests

class MockServerRequestHandler(BaseHTTPRequestHandler):
    USERS_PATTERN = re.compile(r'/users')

    def do_GET(self):
        if re.search(self.USERS_PATTERN, self.path):
            # Add response status code.
            self.send_response(requests.codes.ok)

            # Add response headers.
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()

            # Add response content.
            response_content = json.dumps([])
            self.wfile.write(response_content.encode('utf-8'))
            return

def get_free_port():
    s = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    address, port = s.getsockname()
    s.close()
    return port

def start_mock_server(port):
    mock_server = HTTPServer(('localhost', port), MockServerRequestHandler)
    mock_server_thread = Thread(target=mock_server.serve_forever)
    mock_server_thread.setDaemon(True)
    mock_server_thread.start()
```

完成对逻辑的更改后，更改测试以使用新的服务函数。更新测试以检查从服务器传回的增加的信息。

**project/tests/test _ real _ server . py**

```py
# Third-party imports...
from nose.tools import assert_dict_contains_subset, assert_is_instance, assert_true

# Local imports...
from project.services import get_users

def test_request_response():
    response = get_users()

    assert_dict_contains_subset({'Content-Type': 'application/json; charset=utf-8'}, response.headers)
    assert_true(response.ok)
    assert_is_instance(response.json(), list)
```

**project/tests/test _ mock _ server . py**

```py
# Third-party imports...
from unittest.mock import patch
from nose.tools import assert_dict_contains_subset, assert_list_equal, assert_true

# Local imports...
from project.services import get_users
from project.tests.mocks import get_free_port, start_mock_server

class TestMockServer(object):
    @classmethod
    def setup_class(cls):
        cls.mock_server_port = get_free_port()
        start_mock_server(cls.mock_server_port)

    def test_request_response(self):
        mock_users_url = 'http://localhost:{port}/users'.format(port=self.mock_server_port)

        # Patch USERS_URL so that the service uses the mock server URL instead of the real URL.
        with patch.dict('project.services.__dict__', {'USERS_URL': mock_users_url}):
            response = get_users()

        assert_dict_contains_subset({'Content-Type': 'application/json; charset=utf-8'}, response.headers)
        assert_true(response.ok)
        assert_list_equal(response.json(), [])
```

注意在 *test_mock_server.py* 代码中使用了一项新技术。`response = get_users()`行用来自*模拟库*的`patch.dict()`函数包装。

这个语句是做什么的？

记住，您将`requests.get()`功能从特性逻辑移到了`get_users()`服务功能。在内部，`get_users()`使用`USERS_URL`变量调用`requests.get()`。`patch.dict()`功能临时替换`USERS_URL`变量的值。事实上，它只在`with`语句的范围内这样做。在代码运行之后，`USERS_URL`变量被恢复到它的原始值。这段代码*修补了*URL 以使用模拟服务器地址。

运行测试并观察它们是否通过。

```py
$ nosetests --verbosity=2
test_mock_server.TestMockServer.test_request_response ... 127.0.0.1 - - [05/Jul/2016 20:45:30] "GET /users HTTP/1.1" 200 -
ok
test_real_server.test_request_response ... ok
test_todos.test_request_response ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.871s

OK
```

[*Remove ads*](/account/join/)

## 跳过触及真正 API 的测试

我们在本教程开始时描述了测试模拟服务器而不是真实服务器的优点，然而，您的代码目前测试两者。如何配置测试来忽略真实的服务器？Python 的“unittest”库提供了几个允许你跳过测试的函数。您可以使用条件跳过函数' skipIf '和一个环境变量来打开和关闭真正的服务器测试。在下面的例子中，我们传递了一个应该被忽略的标记名:

```py
$ export SKIP_TAGS=real
```

**project/constants.py**

```py
# Standard-library imports...
import os

BASE_URL = 'http://jsonplaceholder.typicode.com'
SKIP_TAGS = os.getenv('SKIP_TAGS', '').split()
```

**project/tests/test _ real _ server . py**

```py
# Standard library imports...
from unittest import skipIf

# Third-party imports...
from nose.tools import assert_dict_contains_subset, assert_is_instance, assert_true

# Local imports...
from project.constants import SKIP_TAGS
from project.services import get_users

@skipIf('real' in SKIP_TAGS, 'Skipping tests that hit the real API server.')
def test_request_response():
    response = get_users()

    assert_dict_contains_subset({'Content-Type': 'application/json; charset=utf-8'}, response.headers)
    assert_true(response.ok)
    assert_is_instance(response.json(), list)
```

运行测试并注意真实的服务器测试是如何被忽略的:

```py
$ nosetests --verbosity=2 project
test_mock_server.TestMockServer.test_request_response ... 127.0.0.1 - - [05/Jul/2016 20:52:18] "GET /users HTTP/1.1" 200 -
ok
test_real_server.test_request_response ... SKIP: Skipping tests that hit the real API server.
test_todos.test_request_response ... ok

----------------------------------------------------------------------
Ran 3 tests in 1.196s

OK (SKIP=1)
```

## 接下来的步骤

既然您已经创建了一个模拟服务器来测试您的外部 API 调用，那么您可以将这些知识应用到您自己的项目中。以这里创建的简单测试为基础。扩展处理程序的功能，以更接近地模拟真实 API 的行为。

尝试以下练习来升级:

*   如果以未知路径发送请求，则返回状态为 HTTP 404(未找到)的响应。
*   如果使用不允许的方法(POST、DELETE、UPDATE)发送请求，则返回状态为 HTTP 405(不允许的方法)的响应。
*   将有效请求的实际用户数据返回给`/users`。
*   编写测试来捕捉这些场景。

从[回购](https://github.com/realpython/python-mock-server)中抓取代码。**
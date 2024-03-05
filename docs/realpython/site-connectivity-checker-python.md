# 用 Python 构建站点连通性检查器

> 原文：<https://realpython.com/site-connectivity-checker-python/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，加深您的理解: [**构建站点连通性检查器**](/courses/python-site-connectivity-checker/)

用 Python 构建站点连通性检查器是一个提高技能的有趣项目。在这个项目中，您将整合与处理 **HTTP 请求**，创建**命令行界面(CLI)** ，以及使用通用 Python **项目布局**实践组织应用程序代码相关的知识。

通过构建这个项目，您将了解 Python 的**异步特性**如何帮助您高效地处理多个 HTTP 请求。

**在本教程中，您将学习如何:**

*   使用 Python 的 **`argparse`** 创建命令行界面(CLI)
*   使用标准库中 Python 的 **`http.client`** 检查网站是否在线
*   对多个网站实施**同步检查**
*   使用第三方库 **`aiohttp`** 检查网站是否在线
*   对多个网站实施**异步检查**

为了充分利用这个项目，您需要了解处理 [HTTP 请求](https://realpython.com/urllib-request/)和使用 [`argparse`](https://realpython.com/command-line-interfaces-python-argparse/) 创建 CLI 的基本知识。你还应该熟悉 [`asyncio`](https://realpython.com/async-io-python/) 模块以及 [`async`和`await`](https://realpython.com/python-keywords/#asynchronous-programming-keywords-async-await) 关键字。

但是不用担心！整个教程中的主题将以循序渐进的方式进行介绍，以便您可以在学习过程中掌握它们。此外，您可以通过单击下面的链接下载该项目的完整源代码和其他资源:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

## 演示:站点连接检查器

在这个循序渐进的项目中，您将构建一个应用程序来检查一个或多个网站在给定时刻是否在线。该应用程序将在命令行中获取一个目标 URL 列表，并检查它们的连通性，同步**或异步**。以下视频展示了该应用程序的工作原理:***

***[https://player.vimeo.com/video/688288790?background=1](https://player.vimeo.com/video/688288790?background=1)

您的站点连接检查器可以在命令行中接受一个或多个 URL。然后，它创建一个目标 URL 的内部列表，并通过发出 HTTP 请求和处理相应的响应来检查它们的连通性。

使用`-a`或`--asynchronous`选项使应用程序异步执行连接性检查，这可能会降低执行时间，尤其是当您处理一长串网站时。

[*Remove ads*](/account/join/)

## 项目概述

您的网站连接检查器应用程序将通过最小的[命令行界面(CLI)](https://en.wikipedia.org/wiki/Command-line_interface) 提供一些选项。以下是这些选项的摘要:

*   **`-u`** 或 **`--urls`** 允许您在评论行提供一个或多个目标 URL。
*   **`-f`** 或 **`--input-file`** 允许您提供一个包含要检查的 URL 列表的文件。
*   **`-a`** 或 **`--asynchronous`** 允许您异步运行连通性检查。

默认情况下，您的应用程序将同步运行连通性检查。换句话说，应用程序将一个接一个地执行检查。

使用`-a`或`--asynchronous`选项，您可以修改这种行为，并让应用程序同时运行连通性检查。为此，您将利用 Python 的[异步特性](https://realpython.com/python-async-features/)和第三方库 [`aiohttp`](https://docs.aiohttp.org/en/stable/) 。

运行异步检查可以使您的网站连通性检查更快、更有效，尤其是当您有一长串 URL 要检查时。

在内部，您的应用程序将使用标准库 [`http.client`](https://docs.python.org/dev/library/http.client.html#module-http.client) 模块来创建到目标网站的连接。一旦建立了连接，就可以向网站发出 HTTP 请求，希望网站能够给出适当的响应。如果请求成功，那么您将知道该站点在线。否则，你会知道该网站是离线的。

为了在屏幕上显示每次连接检查的结果，您将为应用程序提供一个格式良好的输出，这将使应用程序对您的用户有吸引力。

## 先决条件

您将在本教程中构建的项目需要熟悉一般 Python 编程。此外，还需要具备以下主题的基本知识:

*   在 Python 中处理[异常](https://realpython.com/python-exceptions/)
*   使用[文件](https://realpython.com/working-with-files-in-python/)、 [`with`语句](https://realpython.com/python-with-statement/)和 [`pathlib`](https://realpython.com/python-pathlib/) 模块
*   用标准库或第三方工具处理 HTTP 请求
*   使用 [`argparse`](https://realpython.com/command-line-interfaces-python-argparse/) 模块创建 CLI 应用程序
*   使用 Python 的[异步特性](https://realpython.com/python-async-features/)

了解第三方库 [`aiohttp`](https://docs.aiohttp.org/en/stable/) 的基础知识也是有利条件，但不是必要条件。但是，如果你还没有掌握所有这些知识，那也没关系！你可以通过尝试这个项目学到更多东西。如果遇到困难，您可以随时停下来查看此处链接的资源。

有了这个网站连通性检查项目和先决条件的简短概述，您就差不多准备好开始 Pythoning 化并在编码时享受乐趣了。但是首先，您需要创建一个合适的工作环境，并设置您的项目布局。

## 步骤 1:在 Python 中设置站点连通性检查器项目

在本节中，您将准备开始编写站点连通性检查器应用程序。您将从为项目创建一个 Python 虚拟环境开始。该环境将允许您将项目及其依赖项与其他项目和您的系统 Python 安装隔离开来。

下一步是通过创建所有需要的文件和目录结构来设置项目的[布局](https://realpython.com/python-application-layouts/)。

要下载第一步的代码，请点击以下链接并导航至`source_code_step_1/`文件夹:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

### 设置开发环境

在你开始为一个新项目编码之前，你应该做一些准备。在 Python 中，通常从为项目创建一个**虚拟环境**开始。虚拟环境提供了一个独立的 Python 解释器和一个安装项目依赖项的空间。

首先，创建项目的根目录，名为`rpchecker_project/`。然后转到该目录，在系统的命令行或终端上运行以下命令:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python -m venv venv
PS> venv\Scripts\activate
(venv) PS>
```

```py
$ python -m venv venv
$ source venv/bin/activate
(venv) $
```

第一个命令在项目的根目录中创建一个名为`venv`的全功能 Python 虚拟环境，而第二个命令激活该环境。现在运行下面的命令来安装项目对标准 Python 包管理器 [`pip`](https://realpython.com/what-is-pip/) 的依赖项:

```py
(venv) $ python -m pip install aiohttp
```

使用这个命令，您可以将`aiohttp`安装到您的虚拟环境中。您将使用这个第三方库和 Python 的`async`特性来处理站点连通性检查器应用程序中的异步 HTTP 请求。

酷！您有了一个工作的 Python 虚拟环境，其中包含了开始构建项目所需的所有依赖项。现在，您可以创建项目布局，按照 Python 最佳实践来组织代码。

[*Remove ads*](/account/join/)

### 组织您的站点连通性检查项目

Python 在构建应用程序时具有惊人的灵活性，因此您可能会发现不同项目的结构大相径庭。然而，小型的[可安装 Python 项目](https://realpython.com/python-application-layouts/#installable-single-package)通常有一个单独的[包](https://realpython.com/python-modules-packages/)，它通常以项目本身命名。

遵循本练习，您可以使用以下目录结构组织站点连通性检查器应用程序:

```py
rpchecker_project/
│
├── rpchecker/
│   ├── __init__.py
│   ├── __main__.py
│   ├── checker.py
│   └── cli.py
│
├── README.md
└── requirements.txt
```

您可以为此项目及其主包使用任何名称。在本教程中，该项目将被命名为`rpchecker`，作为真实 Python ( `rp`)和`checker`的组合，指出了 app 的主要功能。

`README.md`文件将包含项目的描述以及安装和运行应用程序的说明。向您的项目添加一个`README.md`文件是编程中的最佳实践，尤其是如果您计划将项目作为开源解决方案发布的话。要了解更多关于编写好的`README.md`文件的信息，请查看[如何为你的 GitHub 项目](https://dbader.org/blog/write-a-great-readme-for-your-github-project)编写一个好的自述文件。

`requirements.txt`文件将保存项目的外部依赖列表。在这种情况下，您只需要`aiohttp`库，因为您将使用的其他工具和模块都可以在 Python [标准库](https://docs.python.org/3/library/index.html)中找到。您可以使用这个文件，通过使用标准的包管理器 [`pip`](https://realpython.com/what-is-pip/) ，为您的应用程序自动重现合适的 Python 虚拟环境。

**注意:**在本教程中，您不会向`README.md`和`requirements.txt`文件添加内容。要体验它们的内容，请下载本教程中提供的额外资料，并查看相应的文件。

在`rpchecker/`目录中，您将拥有以下文件:

*   **`__init__.py`** 启用`rpchecker/`作为 Python 包。
*   **`__main__.py`** 作为应用程序的入口脚本。
*   **`checker.py`** 提供了应用的核心功能。
*   **`cli.py`** 包含了应用程序的命令行界面。

现在，继续将所有这些文件创建为空文件。你可以通过使用你最喜欢的[代码编辑器或者 IDE](https://realpython.com/python-ides-code-editors-guide/) 来实现。一旦你完成创建项目的布局，然后你就可以开始编写应用程序的主要功能:*检查一个网站是否在线*。

## 第二步:用 Python 检查网站的连通性

此时，您应该有一个合适的 Python 虚拟环境，其中安装了项目的依赖项。您还应该有一个项目目录，其中包含您将在本教程中使用的所有文件。是时候开始编码了！

在进入真正有趣的内容之前，继续将应用程序的版本号添加到您的`rpchecker`包中的`__init__.py`模块:

```py
# __init__.py

__version__ = "0.1.0"
```

`__version__`模块级常量保存项目的当前版本号。因为您正在创建一个全新的应用程序，所以初始版本被设置为`0.1.0`。有了这个最小的设置，您就可以开始实现连通性检查功能了。

要下载此步骤的代码，请单击以下链接并查看`source_code_step_2/`文件夹:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

### 实施连通性检查功能

有几个 Python 工具和库可用于检查网站在给定时间是否在线。例如，一个流行的选项是 [`requests`](https://realpython.com/python-requests/) 第三方库，它允许您使用人类可读的 [API](https://en.wikipedia.org/wiki/API) 来执行 HTTP 请求。

然而，使用`requests`的缺点是安装一个外部库只是为了使用它的一小部分功能。在 Python 标准库中找到合适的工具会更有效。

快速浏览一下标准库，您会发现 [`urllib`](https://docs.python.org/3/library/urllib.html#module-urllib) 包，它提供了几个处理 HTTP 请求的模块。例如，要检查网站是否在线，您可以使用 [`urllib.request`](https://docs.python.org/3/library/urllib.request.html#module-urllib.request) 模块中的 [`urlopen()`](https://docs.python.org/3/library/urllib.request.html#urllib.request.urlopen) 功能:

>>>

```py
>>> from urllib.request import urlopen

>>> response = urlopen("https://python.org")
>>> response.read()
b'<!doctype html>\n<!--[if lt IE 7]>
 ...
```

`urlopen()`函数获取一个 URL 并打开它，以字符串或 [`Request`](https://docs.python.org/3/library/urllib.request.html#urllib.request.Request) 对象的形式返回其内容。但是你只需要检查网站是否在线，所以下载整个页面是一种浪费。你需要更有效的方法。

有没有一种工具可以让您对 HTTP 请求进行低级别的控制？这就是 [`http.client`](https://docs.python.org/3/library/http.client.html#module-http.client) 模块的用武之地。这个模块提供了 [`HTTPConnection`](https://docs.python.org/3/library/http.client.html#http.client.HTTPConnection) 类，代表一个到给定 HTTP 服务器的连接。

`HTTPConnection`有一个 [`.request()`](https://docs.python.org/3/library/http.client.html#http.client.HTTPConnection.request) 方法，允许您使用不同的 [HTTP 方法](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)执行 HTTP 请求。对于这个项目，您可以使用 [`HEAD`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/HEAD) HTTP 方法来请求只包含目标网站的[头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)的响应。此选项将减少要下载的数据量，从而提高连通性检查应用程序的效率。

至此，您对要使用的工具有了一个清晰的概念。现在你可以做一些快速测试。继续在 Python 交互式会话中运行以下代码:

>>>

```py
>>> from http.client import HTTPConnection

>>> connection = HTTPConnection("pypi.org", port=80, timeout=10)
>>> connection.request("HEAD", "/")

>>> response = connection.getresponse()
>>> response.getheaders()
[('Server', 'Varnish'), ..., ('X-Permitted-Cross-Domain-Policies', 'none')]
```

在本例中，首先创建一个针对 [`pypi.org`](https://pypi.org/) 网站的`HTTPConnection`实例。该连接使用端口`80`，这是默认的 HTTP 端口。最后，`timeout`参数提供了连接尝试超时前等待的秒数。

然后使用`.request()`在站点的根路径`"/"`上执行一个`HEAD`请求。为了从服务器获得实际的响应，您在`connection`对象上调用`.getresponse()`。最后，通过调用`.getheaders()`来检查响应的头部。

您的网站连接检查器只需要创建一个连接并发出一个`HEAD`请求。如果请求成功，则目标网站在线。否则，该网站处于脱机状态。在后一种情况下，向用户显示一条错误消息是合适的。

现在在代码编辑器中打开`checker.py`文件。然后向其中添加以下代码:

```py
 1# checker.py
 2
 3from http.client import HTTPConnection
 4from urllib.parse import urlparse
 5
 6def site_is_online(url, timeout=2):
 7    """Return True if the target URL is online.
 8
 9 Raise an exception otherwise.
10 """
11    error = Exception("unknown error")
12    parser = urlparse(url)
13    host = parser.netloc or parser.path.split("/")[0]
14    for port in (80, 443):
15        connection = HTTPConnection(host=host, port=port, timeout=timeout)
16        try:
17            connection.request("HEAD", "/")
18            return True
19        except Exception as e:
20            error = e
21        finally:
22            connection.close()
23    raise error
```

下面是这段代码的逐行分解:

*   **三号线** [从`http.client`进口](https://realpython.com/python-import/) `HTTPConnection`。您将使用该类建立与目标网站的连接并处理 HTTP 请求。

*   **4 号线**从 [`urllib.parse`](https://docs.python.org/3/library/urllib.parse.html#module-urllib.parse) 进口 [`urlparse()`](https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse) 。这个函数将帮助您解析目标 URL。

*   **第 6 行**定义了`site_is_online()`，它带有一个`url`和一个`timeout`参数。`url`参数将保存一个表示网站 URL 的字符串。同时，`timeout`将保持连接尝试超时前等待的秒数。

*   第 11 行定义了一个通用的`Exception`实例作为占位符。

*   **第 12 行**定义了一个`parser`变量，包含使用`urlparse()`解析目标 URL 的结果。

*   **第 13 行**使用 [`or`操作符](https://realpython.com/python-or-operator/)从目标 URL 中提取主机名。

*   **第 14 行**通过 HTTP 和 HTTPS 端口开始一个 [`for`循环](https://realpython.com/python-for-loop/)。这样，您可以检查网站是否在任一端口上可用。

*   **第 15 行**使用`host`、`port`和`timeout`作为参数创建一个`HTTPConnection`实例。

*   **第 16 到 22 行**定义了一个`try` … `except` … `finally`语句。`try`块试图通过调用`.request()`向目标网站发出`HEAD`请求。如果请求成功，那么函数[返回](https://realpython.com/python-return-statement/)T6。如果出现异常，那么`except`模块在`error`中保存一个对该异常的引用。`finally`模块关闭连接以释放获取的资源。

*   **第 23 行**如果循环结束而没有成功的请求，则引发存储在`error`中的异常。

如果目标网站在线，您的`site_is_online()`函数将返回`True`。否则，它会引发一个异常，指出遇到的问题。后一种行为很方便，因为当站点不在线时，您需要显示一条信息性的错误消息。现在是时候尝试你的新功能了。

[*Remove ads*](/account/join/)

### 运行第一次连通性检查

要尝试您的`site_is_online()`功能，请继续并返回您的互动会话。然后运行以下代码:

>>>

```py
>>> from rpchecker.checker import site_is_online

>>> site_is_online("python.org")
True

>>> site_is_online("non-existing-site.org")
Traceback (most recent call last):
    ...
socket.gaierror: [Errno -2] Name or service not known
```

在这个代码片段中，首先从`checker`模块导入`site_is_online()`。然后你调用带有`"python.org"`作为参数的函数。因为函数返回`True`，所以你知道目标站点在线。

在最后一个例子中，您使用一个不存在的网站作为目标 URL 来调用`site_is_online()`。在这种情况下，该函数会引发一个异常，您可以稍后捕获并处理该异常，以便向用户显示一条错误消息。

太好了！您已经实现了应用程序检查网站连通性的主要功能。现在，您可以通过设置项目的 CLI 来继续您的项目。

## 步骤 3:创建您的网站连接检查器的 CLI

到目前为止，您已经有了一个工作函数，它允许您通过使用标准库中的`http.client`模块执行 HTTP 请求来检查给定网站是否在线。在这一步结束时，您将拥有一个最小的 CLI，允许您从命令行运行网站连通性检查器应用程序。

CLI 将包括在命令行获取 URL 列表和从文本文件加载 URL 列表的选项。该应用程序还将显示连通性检查结果，并显示一条用户友好的消息。

要创建应用程序的 CLI，您将使用 Python 标准库中的`argparse`。该模块允许您构建用户友好的 CLI，而无需安装任何外部依赖项，如 [Click](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/) 或 [Typer](https://realpython.com/python-typer-cli/) 。

首先，您将编写使用`argparse`所需的样板代码。您还将编写从命令行读取 URL 的选项。

单击下面的链接下载此步骤的代码，以便您可以跟随项目。您将在`source_code_step_3/`文件夹中找到您需要的内容:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

### 在命令行解析网站 URLs】

要用`argparse`构建应用程序的 CLI，您需要创建一个 [`ArgumentParser`](https://docs.python.org/3/library/argparse.html?highlight=argparser#argparse.ArgumentParser) 实例，这样您就可以解析命令行中提供的[参数](https://en.wikipedia.org/wiki/Command-line_interface#Arguments)。一旦你有了一个参数解析器，你就可以开始向你的应用程序的命令行界面添加参数和选项。

现在在代码编辑器中打开`cli.py`文件。然后添加以下代码:

```py
# cli.py

import argparse

def read_user_cli_args():
    """Handle the CLI arguments and options."""
    parser = argparse.ArgumentParser(
        prog="rpchecker", description="check the availability of websites"
    )
    parser.add_argument(
        "-u",
        "--urls",
        metavar="URLs",
        nargs="+",
        type=str,
        default=[],
        help="enter one or more website URLs",
    )
    return parser.parse_args()
```

在这个代码片段中，您创建了`read_user_cli_args()`来将与参数解析器相关的功能保存在一个地方。要构建解析器对象，需要使用两个参数:

*   **`prog`** 定义了程序的名称。
*   **`description`** 为应用提供了合适的描述。当您使用`--help`选项调用应用程序时，将显示此描述。

创建参数解析器后，使用 [`.add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) 添加第一个命令行参数。该参数将允许用户在命令行输入一个或多个 URL。它将使用`-u`和`--urls`开关。

`.add_argument()`的其余参数如下:

*   **`metavar`** 为用法或帮助消息中的参数设置名称。
*   **`nargs`** 告诉`argparse`在`-u`或`--urls`开关后接受一系列命令行参数。
*   **`type`** 设置命令行参数的数据类型，即本参数中的`str`。
*   **`default`** 默认情况下将命令行参数设置为空列表。
*   **`help`** 为用户提供了帮助信息。

最后，您的函数返回对解析器对象调用 [`.parse_args()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args) 的结果。该方法返回一个包含已解析参数的 [`Namespace`](https://docs.python.org/3/library/argparse.html#argparse.Namespace) 对象。

[*Remove ads*](/account/join/)

### 从文件中加载网址

在站点连通性检查器中实现的另一个有价值的选项是从本地机器上的文本文件中加载 URL 列表的能力。为此，您可以添加带有`-f`和`--input-file`标志的第二个命令行参数。

继续使用以下代码更新`read_user_cli_args()`:

```py
# cli.py
# ...

def read_user_cli_args():
    # ...
 parser.add_argument(        "-f",
        "--input-file",
        metavar="FILE",
        type=str,
        default="",
        help="read URLs from a file",
    )
    return parser.parse_args()
```

要创建这个新的命令行参数，可以使用`.add_argument()`和上一节中几乎相同的参数。在这种情况下，您没有使用`nargs`参数，因为您希望应用程序在命令行只接受一个输入文件。

### 显示检查结果

通过命令行与用户交互的每个应用程序的基本组件是应用程序的输出。您的应用程序需要向用户显示其操作的结果。该功能对于确保愉快的用户体验至关重要。

您的站点连通性检查器不需要非常复杂的输出。它只需要通知用户关于被检查网站的当前状态。为了实现这个功能，您将编写一个名为`display_check_result()`的函数。

现在回到`cli.py`文件，在末尾添加函数:

```
# cli.py
# ...

def display_check_result(result, url, error=""):
    """Display the result of a connectivity check."""
    print(f'The status of "{url}" is:', end=" ")
    if result:
        print('"Online!" 👍')
    else:
        print(f'"Offline?" 👎 \n Error: "{error}"')
```py

这个函数接受连通性检查结果、检查的 URL 和一个可选的错误消息。[条件语句](https://realpython.com/python-conditional-statements/)测试查看`result`是否为真，在这种情况下，一条`"Online!"`消息被[打印](https://realpython.com/python-print/)到屏幕上。如果`result`为假，那么`else`子句打印`"Offline?"`以及一个关于刚刚发生的实际问题的错误报告。

就是这样！您的网站连接检查器有一个命令行界面，允许用户与应用程序进行交互。现在是时候将所有东西放在应用程序的入口点脚本中了。

## 第四步:把所有东西放在应用程序的主脚本中

到目前为止，您的站点连通性检查器项目具有检查给定网站是否在线的功能。它还有一个 CLI，您可以使用 Python 标准库中的`argparse`模块快速构建。在这一步中，您将编写[粘合代码](https://en.wikipedia.org/wiki/Glue_code)——这些代码将把所有这些组件结合在一起，使您的应用程序作为一个成熟的命令行应用程序工作。

首先，您将从设置应用程序的主脚本或[入口点](https://en.wikipedia.org/wiki/Entry_point)脚本开始。该脚本将包含 [`main()`](https://realpython.com/python-main-function/) 函数和一些高级代码，这些代码将帮助您将[前端的 CLI](https://en.wikipedia.org/wiki/Frontend_and_backend)与后端的连通性检查功能连接起来。

要下载该步骤的代码，请单击下面的链接，然后查看`source_code_step_4/`文件夹:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

### 创建应用程序的入口点脚本

构建网站连通性检查器应用程序的下一步是用合适的`main()`函数定义入口点脚本。为此，您将使用位于`rpchecker`包中的 [`__main__.py`](https://docs.python.org/3/library/__main__.html#module-__main__) 文件。在 Python 包中包含一个`__main__.py`文件使您能够使用命令`python -m <package_name>`将包作为可执行程序运行。

要开始用代码填充`__main__.py`，请在代码编辑器中打开文件。然后添加以下内容:

```
 1# __main__.py
 2
 3import sys
 4
 5from rpchecker.cli import read_user_cli_args
 6
 7def main():
 8    """Run RP Checker."""
 9    user_args = read_user_cli_args()
10    urls = _get_websites_urls(user_args)
11    if not urls:
12        print("Error: no URLs to check", file=sys.stderr)
13        sys.exit(1)
14    _synchronous_check(urls)
```py

从`cli`模块导入`read_user_cli_args()`后，定义应用的`main()`功能。在`main()`中，您会发现几行代码还没有运行。在您提供了缺少的功能后，下面是这段代码应该做的事情:

*   **第 9 行**调用`read_user_cli_args()`来解析命令行参数。产生的`Namespace`对象然后被存储在`user_args`局部[变量](https://realpython.com/python-variables/)中。

*   **第 10 行**通过调用一个名为`_get_websites_urls()`的**辅助函数**来组合一个目标 URL 列表。一会儿你就要编写这个函数了。

*   **第 11 行**定义了一个`if`语句来检查 URL 列表是否为空。如果是这种情况，那么`if`块向用户打印一条错误消息并退出应用程序。

*   **第 14 行**调用一个名为`_synchronous_check()`的函数，该函数将目标 URL 列表作为参数，并对每个 URL 进行连通性检查。顾名思义，这个函数将同步运行连通性检查，或者一个接一个地运行。同样，您将在一会儿编写这个函数。

有了`main()`,你可以开始编码缺失的部分，使其正确工作。在接下来的部分中，您将实现`_get_websites_urls()`和`_synchronous_check()`。一旦它们准备就绪，您就可以首次运行您的网站连通性检查器应用程序了。

[*Remove ads*](/account/join/)

### 建立目标网站 URL 列表

你的网站连通性检查应用程序将能够在每次执行时检查多个 URL。用户将通过在命令行列出 URL、在文本文件中提供 URL 或者两者都提供来将 URL 输入应用程序。为了创建目标 URL 的内部列表，应用程序将首先处理命令行中提供的 URL。然后它会从一个文件中添加额外的 URL，如果有的话。

下面是完成这些任务并返回目标 URL 列表的代码，该列表结合了两个源、命令行和一个可选的文本文件:

```
 1# __main__.py
 2
 3import pathlib 4import sys
 5
 6from rpchecker.cli import read_user_cli_args
 7
 8def main():
 9    # ...
10
11def _get_websites_urls(user_args): 12    urls = user_args.urls
13    if user_args.input_file:
14        urls += _read_urls_from_file(user_args.input_file)
15    return urls
16
17def _read_urls_from_file(file): 18    file_path = pathlib.Path(file)
19    if file_path.is_file():
20        with file_path.open() as urls_file:
21            urls = [url.strip() for url in urls_file]
22            if urls:
23                return urls
24            print(f"Error: empty input file, {file}", file=sys.stderr)
25    else:
26        print("Error: input file not found", file=sys.stderr)
27    return []
```py

该代码片段中的第一个更新是导入`pathlib`来管理可选 URL 文件的路径。第二个更新是添加了`_get_websites_urls()`辅助函数，它执行以下操作:

*   **第 12 行**定义了`urls`，它最初存储命令行提供的 URL 列表。注意，如果用户没有提供任何 URL，那么`urls`将存储一个空列表。

*   **第 13 行**定义了一个条件，检查用户是否提供了一个 URL 文件。如果是这样的话，`if`块会用`user_args.input_file`命令行参数中提供的文件来增加调用`_read_urls_from_file()`所得到的目标 URL 列表。

*   第 15 行返回 URL 的结果列表。

同时，`_read_urls_from_file()`运行以下动作:

*   **第 18 行**将`file`实参变为 [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) 对象，以便于进一步处理。

*   **第 19 行**定义了一个条件语句，检查当前文件是否是本地文件系统中的一个实际文件。为了执行这个检查，条件调用`Path`对象上的 [`.is_file()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_file) 。然后`if`块打开文件，并使用 list comprehension 读取其内容。这种理解去除了文件中每一行任何可能的前导和结尾空格，以防止以后出现处理错误。

*   第 22 行定义了一个嵌套的条件来检查是否已经收集了任何 URL。如果是，那么第 23 行返回 URL 的结果列表。否则，第 24 行打印一条错误消息，通知读者输入文件是空的。

第 25 到 26 行的`else`子句打印一条错误消息，指出输入文件不存在。如果函数运行时没有返回有效的 URL 列表，它将返回一个空列表。

哇！太多了，但是你坚持到了最后！现在你可以继续进行`__main__.py`的最后部分了。换句话说，您可以实现`_synchronous_check()`功能，以便应用程序可以在多个网站上执行连通性检查。

### 检查多个网站的连接性

要对多个网站运行连通性检查，您需要遍历目标 URL 列表，进行检查，并显示相应的结果。这就是下面的`_synchronous_check()`函数的作用:

```
 1# __main__.py
 2
 3import pathlib
 4import sys
 5
 6from rpchecker.checker import site_is_online 7from rpchecker.cli import display_check_result, read_user_cli_args 8
 9# ...
10
11def _synchronous_check(urls): 12    for url in urls:
13        error = ""
14        try:
15            result = site_is_online(url)
16        except Exception as e:
17            result = False
18            error = str(e)
19        display_check_result(result, url, error)
20
21if __name__ == "__main__": 22    main()
```py

在这段代码中，首先通过添加`site_is_online()`和`display_check_result()`来更新您的导入。然后定义`_synchronous_check()`，它接受一个 URL 列表作为参数。该函数的主体是这样工作的:

*   **第 12 行**开始一个`for`循环，遍历目标 URL。

*   **第 13 行**定义并初始化`error`，它将保存应用程序没有从目标网站得到响应时显示的消息。

*   **第 14 行到第 18 行**定义了一个`try` … `except`语句，该语句捕捉连通性检查期间可能发生的任何异常。这些检查运行在第 15 行，该行使用目标 URL 作为参数调用`site_is_online()`。然后，如果出现连接问题，第 17 行和第 18 行更新`result`和`error`变量。

*   **第 19 行**最后用适当的参数调用`display_check_result()`，将连通性检查结果显示到屏幕上。

为了包装`__main__.py`文件，您添加典型的 Python [`if __name__ == "__main__":`](https://realpython.com/if-name-main-python/) 样板代码。当模块[作为脚本](https://realpython.com/run-python-scripts/)或可执行程序运行时，这个片段调用`main()`。有了这些更新，您的应用程序现在可以进行测试飞行了！

### 从命令行运行连通性检查

您已经编写了大量代码，却没有机会看到它们的运行。您已经编写了站点连通性检查器的 CLI 及其入口点脚本。现在是时候尝试一下您的应用程序了。在此之前，请确保您已经下载了本步骤开始时提到的奖励材料，尤其是`sample-urls.txt`文件。

现在回到命令行，执行以下命令:

```
$ python -m rpchecker -h
python -m rpchecker -h
usage: rpchecker [-h] [-u URLs [URLs ...]] [-f FILE] [-a]

check the availability of web sites

options:
 -h, --help            show this help message and exit
 -u URLs [URLs ...], --urls URLs [URLs ...]
 enter one or more website URLs
 -f FILE, --input-file FILE
 read URLs from a file

$ python -m rpchecker -u python.org pypi.org peps.python.org
The status of "python.org" is: "Online!" 👍
The status of "pypi.org" is: "Online!" 👍
The status of "peps.python.org" is: "Online!" 👍

$ python -m rpchecker --urls non-existing-site.org
The status of "non-existing-site.org" is: "Offline?" 👎
 Error: "[Errno -2] Name or service not known"

$ cat sample-urls.txt
python.org
pypi.org
docs.python.org
peps.python.org

$ python -m rpchecker -f sample-urls.txt
The status of "python.org" is: "Online!" 👍
The status of "pypi.org" is: "Online!" 👍
The status of "docs.python.org" is: "Online!" 👍
The status of "peps.python.org" is: "Online!" 👍
```py

你的网站连接检查工作很棒！当你用`-h`或`--help`选项运行`rpchecker`时，你会得到一条解释如何使用该应用的使用信息。

应用程序可以从命令行或文本文件中获取几个 URL，并检查它们的连通性。如果在检查过程中出现错误，屏幕上会显示一条消息，告诉您是什么导致了错误。

继续尝试一些其他的网址和功能。例如，尝试使用`-u`和`-f`开关将命令行中的 URL 与文件中的 URL 结合起来。此外，检查当您提供一个空的或不存在的 URL 文件时会发生什么。

酷！你的网站连通性检查应用程序工作得很好，很流畅，不是吗？然而，它有一个隐藏的问题。当您使用一个很长的目标 URL 列表运行应用程序时，执行时间可能会很长，因为所有的连接性检查都是同步运行的。

要解决这个问题并提高应用程序的性能，您可以实现异步连接检查。这就是你在下一节要做的。

[*Remove ads*](/account/join/)

## 步骤 5:异步检查网站的连通性

通过异步编程对多个网站同时执行连接性检查，可以提高应用程序的整体性能。为此，您可以利用 Python 的异步特性和第三方库`aiohttp`,它们已经安装在您项目的虚拟环境中。

Python 支持使用`asyncio`模块、 [`async`和`await`](https://realpython.com/python-keywords/#asynchronous-programming-keywords-async-await) 关键字进行异步编程。在下面几节中，您将编写所需的代码，使您的应用程序使用这些工具异步运行连通性检查。

要下载这最后一步的代码，请单击以下链接并查看`source_code_step_5/`文件夹:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

### 实施异步连接检查功能

让您的网站连通性检查器同时工作的第一步是编写一个`async`函数，允许您在给定的网站上执行一次连通性检查。这将是您的`site_is_online()`函数的异步等价物。

回到`checker.py`文件，添加以下代码:

```
 1# checker.py
 2
 3import asyncio 4from http.client import HTTPConnection
 5from urllib.parse import urlparse
 6
 7import aiohttp 8# ...
 9
10async def site_is_online_async(url, timeout=2): 11    """Return True if the target URL is online.
12
13 Raise an exception otherwise.
14 """
15    error = Exception("unknown error")
16    parser = urlparse(url)
17    host = parser.netloc or parser.path.split("/")[0]
18    for scheme in ("http", "https"):
19        target_url = scheme + "://" + host
20        async with aiohttp.ClientSession() as session:
21            try:
22                await session.head(target_url, timeout=timeout)
23                return True
24            except asyncio.exceptions.TimeoutError:
25                error = Exception("timed out")
26            except Exception as e:
27                error = e
28    raise error
```py

在这个更新中，首先添加所需的导入，`asyncio`和`aiohttp`。然后在第 10 行定义`site_is_online_async()`。这是一个`async`函数，它有两个参数:要检查的 URL 和请求超时前的秒数。该函数的主体执行以下操作:

*   第 15 行定义了一个通用的`Exception`实例作为占位符。

*   **第 16 行**定义了一个`parser`变量，包含使用`urlparse()`解析目标 URL 的结果。

*   **第 17 行**使用 [`or`操作符](https://realpython.com/python-or-operator/)从目标 URL 中提取主机名。

*   **第 18 行**定义了一个通过 HTTP 和 HTTPS 方案的`for`循环。这将允许您检查网站是否可用。

*   **第 19 行**使用当前方案和主机名构建一个 URL。

*   **第 20 行**定义了一个 [`async with`语句](https://realpython.com/async-io-python/)来处理一个 [`aiohttp.ClientSession`](https://docs.aiohttp.org/en/stable/client_reference.html) 实例。这个类是使用`aiohttp`进行 HTTP 请求的推荐接口。

*   **第 21 到 27 行**定义了一个`try` … `except`语句。`try`块通过调用`session`对象上的`.head()`来执行并等待对目标网站的`HEAD`请求。如果请求成功，那么函数返回`True`。第一个`except`子句捕获`TimeoutError`异常并将`error`设置为一个新的`Exception`实例。第二个`except`子句捕捉任何其他异常，并相应地更新`error`变量。

*   **第 28 行**如果循环没有成功请求就结束，则引发存储在`error`中的异常。

`site_is_online_async()`的实现与`site_is_online()`的实现类似。如果目标网站在线，它返回`True`。否则，它会引发一个异常，指出遇到的问题。

这些函数之间的主要区别在于，`site_is_online_async()`使用第三方库`aiohttp`异步执行 HTTP 请求。当你有一长串网站需要查看时，这种方法可以帮助你优化应用程序的性能。

有了这个函数，您就可以使用一个新选项来更新您的应用程序的 CLI，该选项允许您异步运行连接性检查。

### 将异步选项添加到应用程序的 CLI

现在，您需要向站点连通性检查器应用程序的 CLI 添加一个选项。这个新选项将告诉应用程序异步运行检查。该选项可以只是一个[布尔](https://realpython.com/python-boolean/)标志。要实现这种类型的选项，可以使用`.add_argument()`的`action`参数。

现在继续用下面的代码更新`cli.py`文件上的`read_user_cli_args()`:

```
# cli.py
# ...

def read_user_cli_args():
    """Handles the CLI user interactions."""
    # ...
 parser.add_argument(        "-a",
        "--asynchronous",
 action="store_true",        help="run the connectivity check asynchronously",
    )
    return parser.parse_args()

# ...
```py

对`parser`对象上的`.add_argument()`的调用向应用程序的 CLI 添加了一个新的`-a`或`--asynchronous`选项。`action`参数被设置为`"store_true"`，这告诉`argparse``-a`和`--asynchronous`是布尔标志，当在命令行提供时将存储`True`。

有了这个新选项，是时候编写异步检查多个网站连通性的逻辑了。

[*Remove ads*](/account/join/)

### 异步检查多个网站的连通性

为了异步检查多个网站的连通性，您将编写一个`async`函数，该函数调用并等待来自`checker`模块的`site_is_online_async()`。回到`__main__.py`文件，向其中添加以下代码:

```
 1# __main__.py
 2import asyncio
 3import pathlib
 4import sys
 5
 6from rpchecker.checker import site_is_online, site_is_online_async 7from rpchecker.cli import display_check_result, read_user_cli_args
 8# ...
 9
10async def _asynchronous_check(urls): 11    async def _check(url):
12        error = ""
13        try:
14            result = await site_is_online_async(url)
15        except Exception as e:
16            result = False
17            error = str(e)
18        display_check_result(result, url, error)
19
20    await asyncio.gather(*(_check(url) for url in urls))
21
22def _synchronous_check(urls):
23    # ...
```py

在这段代码中，首先更新您的导入来访问`site_is_online_async()`。然后使用`async`关键字将第 10 行的`_asynchronous_check()`定义为异步函数。这个函数获取一个 URL 列表，并异步检查它们的连通性。它是这样做的:

*   **第 11 行**定义了一个名为`_check()`的[内部](https://realpython.com/inner-functions-what-are-they-good-for/) `async`函数。该函数允许您重用检查单个 URL 连通性的代码。

*   **第 12 行**定义并初始化一个占位符`error`变量，该变量将在稍后对`display_check_result()`的调用中使用。

*   **第 13 行到第 17 行**定义了一个`try` … `except`语句来完成连通性检查。`try`块使用目标 URL 作为参数调用并等待`site_is_online_async()`。如果呼叫成功，那么`result`会变成`True`。如果调用引发异常，那么`result`将会是`False`，而`error`将会保存产生的错误消息。

*   **第 18 行**使用`result`、`url`和`error`作为参数调用`display_check_result()`。该呼叫显示关于网站可用性的信息。

*   **第 20 行**调用并等待来自`asyncio`模块的 [`gather()`](https://docs.python.org/3/library/operator.html#operator.itemgetter) 函数。该函数同时运行一个由[个可奖励对象](https://docs.python.org/3/library/asyncio-task.html#asyncio-awaitables)组成的列表，如果所有可奖励对象都成功完成，则返回一个结果值的汇总列表。为了提供一个合适的对象列表，您使用一个[生成器表达式](https://realpython.com/introduction-to-python-generators/#building-generators-with-generator-expressions)，它为每个目标 URL 调用`_check()`。

好吧！您几乎已经准备好尝试站点连通性检查器应用程序的异步功能了。在此之前，您需要注意最后一个细节:更新`main()`函数来集成这个新特性。

### 向应用程序的主代码添加异步检查

为了给应用程序的`main()`函数添加异步功能，您将使用一个条件语句来检查用户是否在命令行提供了`-a`或`--asynchronous`标志。该条件将允许您根据用户的输入使用正确的工具运行连通性检查。

继续并再次打开`__main__.py`文件。然后更新`main()`，如下面的代码片段所示:

```
 1# __main__.py
 2# ...
 3
 4def main():
 5    """Run RP Checker."""
 6    user_args = read_user_cli_args()
 7    urls = _get_websites_urls(user_args)
 8    if not urls:
 9        print("Error: no URLs to check", file=sys.stderr)
10        sys.exit(1)
11
12    if user_args.asynchronous: 13        asyncio.run(_asynchronous_check(urls))
14    else:
15        _synchronous_check(urls)
16
17# ...
```py

第 12 到 15 行的条件语句检查用户是否在命令行提供了`-a`或`--asynchronous`标志。如果是这种情况，`main()`使用`asyncio.run()`异步运行连通性检查。否则，它使用`_synchronous_check()`同步运行检查。

就是这样！您现在可以在实践中测试网站连接检查器的这一新功能。回到命令行，运行以下命令:

```
$ python -m rpchecker -h
usage: rpchecker [-h] [-u URLs [URLs ...]] [-f FILE] [-a]

check the availability of web sites

options:
 -h, --help            show this help message and exit
 -u URLs [URLs ...], --urls URLs [URLs ...]
 enter one or more website URLs
 -f FILE, --input-file FILE
 read URLs from a file
 -a, --asynchronous    run the connectivity check asynchronously 
$ # Synchronous execution $ python -m rpchecker -u python.org pypi.org docs.python.org
The status of "python.org" is: "Online!" 👍
The status of "pypi.org" is: "Online!" 👍
The status of "docs.python.org" is: "Online!" 👍

$ # Asynchronous execution $ python -m rpchecker -u python.org pypi.org docs.python.org -a
The status of "pypi.org" is: "Online!" 👍
The status of "docs.python.org" is: "Online!" 👍
The status of "python.org" is: "Online!" 👍
```

第一个命令显示您的应用程序现在有了一个新的`-a`或`--asynchronous`选项，它将异步运行连通性检查。

第二个命令让`rpchecker`同步运行连通性检查，就像您在上一节中所做的那样。这是因为您没有提供`-a`或`--asynchronous`标志。请注意，URL 的检查顺序与它们在命令行中输入的顺序相同。

最后，在第三个命令中，您在行尾使用了`-a`标志。该标志使`rpchecker`同时运行连通性检查。现在，检查结果的显示顺序不同于 URL 的输入顺序，而是按照目标网站的响应顺序。

作为练习，您可以尝试使用一长串目标 URL 运行站点连通性检查应用程序，并比较应用程序同步和异步运行检查时的执行时间。

## 结论

您已经用 Python 构建了一个功能站点连通性检查器应用程序。现在你知道了处理给定网站的 HTTP 请求的基本知识。您还学习了如何为您的应用程序创建一个最小但功能强大的**命令行界面(CLI)** ，以及如何组织一个真实的 Python 项目。此外，您已经尝试了 Python 的异步特性。

**在本教程中，您学习了如何:**

*   用 **`argparse`** 在 Python 中创建命令行界面(CLI)
*   使用 Python 的 **`http.client`** 检查网站是否在线
*   在多个网站上运行**同步**检查
*   使用 **`aiohttp`** 检查网站是否在线
*   异步检查多个网站**的连通性**

有了这个基础，您就可以通过创建更复杂的命令行应用程序来继续提升您的技能。您还可以更好地准备继续学习 Python 中的 HTTP 请求。

要查看您为构建应用所做的工作，您可以下载下面的完整源代码:

**获取源代码:** [点击此处获取源代码，您将使用](https://realpython.com/bonus/site-connectivity-checker-python-project-code/)来构建您的站点连接检查器应用程序。

[*Remove ads*](/account/join/)

## 接下来的步骤

现在，您已经完成了站点连通性检查器应用程序的构建，您可以进一步实现一些附加功能。自己添加新特性会推动您学习新的、令人兴奋的编码概念和主题。

以下是一些关于新功能的想法:

*   **定时支持:**测量每个目标网站的响应时间。
*   **检查安排支持:**安排多轮连接检查，以防某些网站离线。

要实现这些特性，您可以利用 Python 的 [`time`](https://realpython.com/python-timer/) 模块，它将允许您测量代码的执行时间。

一旦你实现了这些新特性，你就可以换个方式，投入到其他更酷、更复杂的项目中。以下是您继续学习 Python 和构建项目的一些很好的后续步骤:

*   [用 Python 构建掷骰子应用程序](https://realpython.com/python-dice-roll/):在这个循序渐进的项目中，您将使用 Python 构建一个最小化的基于文本的用户界面的掷骰子模拟器应用程序。该应用程序将模拟多达六个骰子的滚动。每个独立的骰子将有六个面。

*   [外面下雨了？使用 Python](https://realpython.com/build-a-python-weather-app-cli/) 构建天气 CLI 应用程序:在本教程中，您将编写一个格式良好的 Python CLI 应用程序，显示您提供名称的任何城市的当前天气信息。

*   [为命令行构建一个 Python 目录树生成器](https://realpython.com/directory-tree-generator-python/):在这个分步项目中，您将为您的命令行创建一个 Python 目录树生成器应用程序。您将使用`argparse`编写命令行界面，并使用 [`pathlib`](https://realpython.com/python-pathlib/) 遍历文件系统。

*   [用 Python 和 Typer](https://realpython.com/python-typer-cli/) 构建命令行待办事项应用:在这个循序渐进的项目中，你将使用 Python 和 [Typer](https://typer.tiangolo.com/) 为你的命令行创建一个待办事项应用。当您构建这个应用程序时，您将学习 Typer 的基础知识，这是一个用于构建命令行界面(CLI)的现代通用库。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。与书面教程一起观看，加深您的理解: [**构建站点连通性检查器**](/courses/python-site-connectivity-checker/)*****************
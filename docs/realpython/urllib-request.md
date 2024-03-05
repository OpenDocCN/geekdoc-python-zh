# Python 的 urllib.request 用于 HTTP 请求

> 原文：<https://realpython.com/urllib-request/>

如果你需要用 Python 发出 HTTP 请求，那么你可能会发现自己被引向了辉煌的 [`requests`](https://docs.python-requests.org/en/latest/) 库。虽然它是一个很棒的库，但是您可能已经注意到它不是 Python 的内置部分。无论出于什么原因，如果您喜欢限制依赖项并坚持使用标准库 Python，那么您可以使用`urllib.request`！

**在本教程中，您将:**

*   学习如何用`urllib.request`制作基本的 [**HTTP 请求**](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol#Request_and_response_messages_through_connections)
*   深入了解 [**HTTP 消息**](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages) 的具体细节以及`urllib.request`如何表示它
*   了解如何处理 HTTP 消息的**字符编码**
*   探索使用`urllib.request`时的一些**常见错误**，并学习如何解决它们
*   用`urllib.request`体验一下**认证请求**的世界
*   理解为什么`urllib`和`requests`库都存在，以及**何时使用其中一个**

如果你听说过 HTTP 请求，包括 [GET](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/GET) 和 [POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) ，那么你可能已经准备好学习本教程了。此外，您应该已经使用 Python 对文件进行了[读写，最好是使用](https://realpython.com/read-write-files-python/)[上下文管理器](https://realpython.com/python-with-statement/)，至少一次。

最终，你会发现提出请求并不一定是一次令人沮丧的经历，尽管它确实有这样的名声。你可能遇到的许多问题都是由于互联网这个神奇的东西固有的复杂性。好消息是,`urllib.request`模块可以帮助揭开这种复杂性的神秘面纱。

**了解更多:** ，获取新的 Python 教程和新闻，让您成为更有效的 Python 爱好者。

## 带有`urllib.request` 的基本 HTTP GET 请求

在深入了解什么是 HTTP 请求以及它是如何工作的之前，您将通过向一个[示例 URL](https://www.iana.org/domains/reserved) 发出一个基本的 get 请求来尝试一下。您还将向 mock [REST API](https://realpython.com/api-integration-in-python/) 请求一些 [JSON](https://realpython.com/python-json/) 数据。如果你对 POST 请求感到疑惑，一旦你对`urllib.request`有了更多的了解，你将在教程的后面[中涉及到它们。](#post-requests-with-urllibrequest)

**小心:**根据你的具体设置，你可能会发现其中一些例子并不适用。如果是，请跳到常见 [`urllib.request`错误](#common-urllibrequest-troubles)部分进行故障排除。

如果你遇到了这里没有涉及的问题，一定要在下面用一个精确的可重复的例子来评论。

首先，您将向`www.example.com`发出请求，服务器将返回一条 HTTP 消息。确保您使用的是 Python 3 或更高版本，然后使用来自`urllib.request`的`urlopen()`函数:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://www.example.com") as response:
...     body = response.read()
...
>>> body[:15]
b'<!doctype html>'
```

在这个例子中，您从`urllib.request`导入`urlopen()`。使用[上下文管理器](https://realpython.com/python-with-statement/) `with`，你发出一个请求，然后用`urlopen()`接收一个响应。然后，读取响应的主体并关闭响应对象。这样，您显示了正文的前 15 个位置，注意到它看起来像一个 HTML 文档。

原来你在这里！您已成功提出请求，并收到了回复。通过检查内容，您可以知道这可能是一个 HTML 文档。注意正文的打印输出前面有`b`。这表示一个[字节的文字](https://docs.python.org/3/reference/lexical_analysis.html#strings)，你可能需要解码。在本教程的后面，您将学习如何将字节转换成一个[字符串](#going-from-bytes-to-strings)，将它们写入一个[文件](#going-from-bytes-to-file)，或者将它们解析成一个[字典](#going-from-bytes-to-dictionary)。

如果您想调用 REST APIs 来获取 JSON 数据，这个过程只是略有不同。在以下示例中，您将向[{ JSON }占位符](https://jsonplaceholder.typicode.com)请求一些虚假的待办事项数据:

>>>

```py
>>> from urllib.request import urlopen
>>> import json
>>> url = "https://jsonplaceholder.typicode.com/todos/1"
>>> with urlopen(url) as response:
...     body = response.read()
...
>>> todo_item = json.loads(body)
>>> todo_item
{'userId': 1, 'id': 1, 'title': 'delectus aut autem', 'completed': False}
```

在这个例子中，您所做的与上一个例子非常相似。但是在这个例子中，您导入了`urllib.request` *和* `json`，使用带有`body`的`json.loads()`函数将返回的 JSON 字节解码并解析到一个 [Python 字典](https://realpython.com/python-dicts/)。瞧啊。

如果你足够幸运地使用无错误的[端点](https://en.wikipedia.org/wiki/Web_API#Endpoints)，比如这些例子中的那些，那么也许以上就是你从`urllib.request`开始所需要的全部。话说回来，你可能会发现这还不够。

现在，在进行一些`urllib.request`故障排除之前，您将首先了解 HTTP 消息的底层结构，并学习`urllib.request`如何处理它们。这种理解将为解决许多不同类型的问题提供坚实的基础。

[*Remove ads*](/account/join/)

## HTTP 消息的基本要素

为了理解使用`urllib.request`时可能遇到的一些问题，您需要研究一下`urllib.request`是如何表示响应的。要做到这一点，您将从什么是 **HTTP 消息**的高层次概述中受益，这也是您将在本节中得到的内容。

在高级概述之前，先简要说明一下参考源。如果你想进入技术领域，[互联网工程任务组(IETF)](https://www.ietf.org/) 有一套广泛的[征求意见稿(RFC)](https://www.ietf.org/standards/rfcs/) 文档。这些文档最终成为诸如 HTTP 消息之类的实际规范。[例如，RFC 7230，第 1 部分:消息语法和路由](https://datatracker.ietf.org/doc/html/rfc7230)，都是关于 HTTP 消息的。

如果你正在寻找一些比 RFC 更容易理解的参考资料，那么[Mozilla Developer Network(MDN)](https://developer.mozilla.org/)有大量的参考文章。例如，他们关于 [HTTP 消息](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages)的文章，虽然仍然是技术性的，但是更容易理解。

现在您已经了解了这些参考信息的基本来源，在下一节中，您将获得一个对 HTTP 消息的初学者友好的概述。

### 了解什么是 HTTP 消息

简而言之，HTTP 消息可以理解为文本，作为一个由[字节](https://en.wikipedia.org/wiki/Byte)组成的流传输，其结构遵循 RFC 7230 规定的指导原则。解码后的 HTTP 消息可能只有两行:

```py
GET / HTTP/1.1
Host: www.google.com
```

这使用`HTTP/1.1`协议在根(`/`)指定了一个 [GET](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/GET) 请求。唯一需要的[头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)是主机`www.google.com`。目标服务器有足够的信息来用这些信息作出响应。

响应在结构上类似于请求。HTTP 消息有两个主要部分，元数据[](https://en.wikipedia.org/wiki/Metadata)**和主体**。在上面的请求示例中，消息都是元数据，没有正文。另一方面，响应确实有两个部分:****

```py
HTTP/1.1 200 OK
Content-Type: text/html; charset=ISO-8859-1
Server: gws
(... other headers ...)

<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage"
...
```

响应以指定 HTTP 协议`HTTP/1.1`和状态`200 OK`的**状态行**开始。在状态行之后，您会得到许多键值对，比如`Server: gws`，代表所有的响应[头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)。这是响应的元数据。

在元数据之后，有一个空行，作为标题和正文之间的分隔符。空行之后的一切组成了正文。当你使用`urllib.request`时，这是被读取的部分。

**注**:空行在技术上通常被称为[换行符](https://en.wikipedia.org/wiki/Newline)。HTTP 消息中的换行符必须是一个 Windows 风格的[回车符](https://en.wikipedia.org/wiki/Carriage_return) ( `\r`)和一个[行结束符](https://en.wikipedia.org/wiki/Newline) ( `\n`)。在类似 Unix 的系统上，换行符通常只是一个行尾(`\n`)。

您可以假设所有的 HTTP 消息都遵循这些规范，但是也有可能有些消息违反了这些规则或者遵循了一个旧的规范。不过，这很少会引起任何问题。所以，把它放在你的脑海里，以防你遇到一个奇怪的 bug！

在下一节中，您将看到`urllib.request`如何处理原始 HTTP 消息。

### 理解`urllib.request`如何表示 HTTP 消息

使用`urllib.request`时，您将与之交互的 HTTP 消息的主要表示是 [`HTTPResponse`](https://docs.python.org/3/library/http.client.html#http.client.HTTPResponse) 对象。`urllib.request`模块本身依赖于底层的`http`模块，你不需要直接与之交互。不过，你最终还是会使用一些`http`提供的数据结构，比如`HTTPResponse`和`HTTPMessage`。

**注意**:Python 中表示 HTTP 响应和消息的对象的内部命名可能有点混乱。你通常只与`HTTPResponse`的实例交互，而*请求*的事情在内部处理。

你可能认为`HTTPMessage`是一种基类，它是从`HTTPResponse`继承而来的，但事实并非如此。`HTTPResponse`直接继承`io.BufferedIOBase`，而`HTTPMessage`类继承 [`email.message.EmailMessage`](https://docs.python.org/3/library/email.message.html#email.message.EmailMessage) 。

在源代码中,`EmailMessage`被定义为一个包含一堆头和一个有效载荷的对象，所以它不一定是一封电子邮件。`HTTPResponse`仅仅使用`HTTPMessage`作为其头部的容器。

然而，如果您谈论的是 HTTP 本身而不是它的 Python 实现，那么您将 HTTP 响应视为一种 HTTP 消息是正确的。

当你用`urllib.request.urlopen()`发出请求时，你得到一个`HTTPResponse`对象作为回报。花些时间探索带有 [`pprint()`](https://realpython.com/python-pretty-print/) 和 [`dir()`](https://realpython.com/python-modules-packages/#the-dir-function) 的`HTTPResponse`对象，看看属于它的所有不同方法和属性:

>>>

```py
>>> from urllib.request import urlopen
>>> from pprint import pprint
>>> with urlopen("https://www.example.com") as response:
...     pprint(dir(response))
...
```

要显示此代码片段的输出，请单击展开下面的可折叠部分:



```py
['__abstractmethods__',
 '__class__',
 '__del__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__enter__',
 '__eq__',
 '__exit__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__iter__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__next__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '_abc_impl',
 '_checkClosed',
 '_checkReadable',
 '_checkSeekable',
 '_checkWritable',
 '_check_close',
 '_close_conn',
 '_get_chunk_left',
 '_method',
 '_peek_chunked',
 '_read1_chunked',
 '_read_and_discard_trailer',
 '_read_next_chunk_size',
 '_read_status',
 '_readall_chunked',
 '_readinto_chunked',
 '_safe_read',
 '_safe_readinto',
 'begin',
 'chunk_left',
 'chunked',
 'close',
 'closed',
 'code',
 'debuglevel',
 'detach',
 'fileno',
 'flush',
 'fp',
 'getcode',
 'getheader',
 'getheaders',
 'geturl',
 'headers',
 'info',
 'isatty',
 'isclosed',
 'length',
 'msg',
 'peek',
 'read',
 'read1',
 'readable',
 'readinto',
 'readinto1',
 'readline',
 'readlines',
 'reason',
 'seek',
 'seekable',
 'status',
 'tell',
 'truncate',
 'url',
 'version',
 'will_close',
 'writable',
 'write',
 'writelines']
```

这有很多方法和属性，但是您最终只会使用其中的一小部分。除了`.read()`之外，重要的通常包括获得关于**报头**的信息。

检查所有标题的一种方法是访问`HTTPResponse`对象的 [`.headers`](https://docs.python.org/3/library/http.client.html#http.client.HTTPResponse.headers) 属性。这将返回一个`HTTPMessage`对象。方便的是，您可以像对待字典一样对待一个`HTTPMessage`,方法是对它调用`.items()`,以元组的形式获取所有的头:

>>>

```py
>>> with urlopen("https://www.example.com") as response:
...     pass
...
>>> response.headers
<http.client.HTTPMessage object at 0x000001E029D9F4F0>
>>> pprint(response.headers.items())
[('Accept-Ranges', 'bytes'),
 ('Age', '398424'),
 ('Cache-Control', 'max-age=604800'),
 ('Content-Type', 'text/html; charset=UTF-8'),
 ('Date', 'Tue, 25 Jan 2022 12:18:53 GMT'),
 ('Etag', '"3147526947"'),
 ('Expires', 'Tue, 01 Feb 2022 12:18:53 GMT'),
 ('Last-Modified', 'Thu, 17 Oct 2019 07:18:26 GMT'),
 ('Server', 'ECS (nyb/1D16)'),
 ('Vary', 'Accept-Encoding'),
 ('X-Cache', 'HIT'),
 ('Content-Length', '1256'),
 ('Connection', 'close')]
```

现在您可以访问所有的响应头了！您可能不需要这些信息中的大部分，但是请放心，有些应用程序确实会用到它们。例如，您的浏览器可能会使用标题来读取响应、设置 cookies 并确定适当的[缓存](https://en.wikipedia.org/wiki/Cache_(computing))生命周期。

有一些方便的方法可以从一个`HTTPResponse`对象中获取标题，因为这是一个非常常见的操作。您可以直接在`HTTPResponse`对象上调用`.getheaders()`，这将返回与上面完全相同的元组列表。如果您只对一个头感兴趣，比如说`Server`头，那么您可以在`HTTPResponse`上使用单数`.getheader("Server")`，或者在`.headers`上使用方括号(`[]`)语法，从`HTTPMessage`:

>>>

```py
>>> response.getheader("Server")
'ECS (nyb/1D16)'
>>> response.headers["Server"]
'ECS (nyb/1D16)'
```

说实话，您可能不需要像这样直接与标题交互。您最可能需要的信息可能已经有了一些内置的帮助器方法，但是现在您知道了，以防您需要更深入地挖掘！

[*Remove ads*](/account/join/)

### 关闭`HTTPResponse`

`HTTPResponse`对象与[文件对象](https://docs.python.org/3/glossary.html#term-file-object)有很多共同之处。像文件对象一样，`HTTPResponse`类继承了 [`IOBase`类](https://docs.python.org/3/library/io.html#i-o-base-classes)，这意味着你必须注意打开和关闭。

在简单的程序中，如果你忘记关闭`HTTPResponse`对象，你不太可能注意到任何问题。然而，对于更复杂的项目，这可能会显著降低执行速度，并导致难以查明的错误。

出现问题是因为[输入/输出](https://en.wikipedia.org/wiki/Input/output) (I/O)流受到限制。每个`HTTPResponse`都要求一个流在被读取时保持清晰。如果您从不关闭您的流，这将最终阻止任何其他流被打开，并且它可能会干扰其他程序甚至您的操作系统。

所以，一定要关闭你的`HTTPResponse`对象！为了方便起见，您可以使用上下文管理器，正如您在示例中看到的那样。您也可以通过在响应对象上显式调用`.close()`来获得相同的结果:

>>>

```py
>>> from urllib.request import urlopen
>>> response = urlopen("https://www.example.com")
>>> body = response.read()
>>> response.close()
```

在这个例子中，您没有使用上下文管理器，而是显式地关闭了响应流。不过，上面的例子仍然有一个问题，因为在调用`.close()`之前可能会引发一个异常，阻止正确的拆卸。要使这个调用无条件，正如它应该的那样，您可以使用一个带有一个`else`和一个`finally`子句的 [`try` … `except`](https://realpython.com/python-exceptions/#the-try-and-except-block-handling-exceptions) 块:

>>>

```py
>>> from urllib.request import urlopen
>>> response = None
>>> try:
...     response = urlopen("https://www.example.com")
... except Exception as ex:
...     print(ex)
... else:
...     body = response.read()
... finally:
...     if response is not None:
...         response.close()
```

在本例中，您通过使用`finally`块实现了对`.close()`的无条件调用，无论出现什么异常，该块都将一直运行。`finally`块中的代码首先检查`response`对象是否与`is not None`一起存在，然后关闭它。

也就是说，这正是上下文管理器所做的，并且通常首选`with`语法。`with`语法不仅不那么冗长、可读性更好，而且还能防止令人讨厌的遗漏错误。换句话说，这是防止意外忘记关闭对象的更好的方法:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://www.example.com") as response:
...     response.read(50)
...     response.read(50)
...
b'<!doctype html>\n<html>\n<head>\n    <title>Example D'
b'omain</title>\n\n    <meta charset="utf-8" />\n    <m'
```

在这个例子中，您从`urllib.request`模块导入`urlopen()`。您使用带有`.urlopen()`的`with`关键字将`HTTPResponse`对象赋给变量`response`。然后，读取响应的前 50 个字节，然后读取随后的 50 个字节，所有这些都在`with`块中。最后，关闭`with`块，它执行请求并运行其块中的代码行。

使用这段代码，可以显示两组各 50 个字节的内容。一旦退出`with`块范围，`HTTPResponse`对象将关闭，这意味着`.read()`方法将只返回空字节对象:

>>>

```py
>>> import urllib.request
>>> with urllib.request.urlopen("https://www.example.com") as response:
...     response.read(50)
...
b'<!doctype html>\n<html>\n<head>\n    <title>Example D'
>>> response.read(50)
b''
```

在这个例子中，读取 50 个字节的第二个调用在`with`范围之外。在`with`块之外意味着`HTTPResponse`被关闭，即使你仍然可以访问这个变量。如果你试图在`HTTPResponse`关闭时读取它，它将返回一个空字节对象。

另一点需要注意的是，一旦你从头到尾阅读了一遍，你就不能重读一遍:

>>>

```py
>>> import urllib.request
>>> with urllib.request.urlopen("https://www.example.com") as response:
...     first_read = response.read()
...     second_read = response.read()
...
>>> len(first_read)
1256
>>> len(second_read)
0
```

这个例子表明，一旦你读了一个回复，你就不能再读了。如果您已经完整地读取了响应，那么即使响应没有关闭，后续的尝试也只是返回一个空字节对象。你必须再次提出请求。

在这方面，响应与文件对象不同，因为对于文件对象，可以使用 [`.seek()`](https://docs.python.org/3/library/io.html#io.IOBase.seek) 方法多次读取，而`HTTPResponse`不支持。即使在关闭响应之后，您仍然可以访问头和其他元数据。

### 探索文本、八位字节和位

在迄今为止的大多数例子中，您从`HTTPResponse`读取响应体，立即显示结果数据，并注意到它显示为一个[字节对象](https://docs.python.org/3/library/stdtypes.html#bytes)。这是因为计算机中的文本信息不是以字母的形式存储或传输的，而是以字节的形式！

通过网络发送的原始 HTTP 消息被分解成一系列的[字节](https://en.wikipedia.org/wiki/Byte)，有时被称为[八位字节](https://en.wikipedia.org/wiki/Octet_(computing))。字节是 8- [位](https://en.wikipedia.org/wiki/Bit)块。例如，`01010101`是一个字节。要了解关于二进制、位和字节的更多信息，请查看 Python 中的[位运算符。](https://realpython.com/python-bitwise-operators/)

那么如何用字节表示字母呢？一个字节有 256 种可能的组合，你可以给每种组合分配一个字母。您可以将`00000001`分配给`A`，将`00000010`分配给`B`，以此类推。 [ASCII](https://en.wikipedia.org/wiki/ASCII) 字符编码，相当普遍，使用这种类型的系统编码 128 个字符，对于英语这样的语言来说足够了。这非常方便，因为只要一个字节就可以表示所有的字符，还留有空间。

所有标准的英语字符，包括大写字母、标点符号和数字，都适合 ASCII。另一方面，日语被认为有大约五万个标识字符，所以 128 个字符是不够的！即使一个字节理论上有 256 个字符，对日语来说也远远不够。因此，为了适应世界上所有的语言，有许多不同的字符编码系统。

即使有许多系统，你可以依赖的一件事是它们总是被分成**字节**的事实。大多数服务器，如果不能解析 [MIME 类型](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types)和字符编码，默认为`application/octet-stream`，字面意思是字节流。然后，接收消息的人可以计算出字符编码。

[*Remove ads*](/account/join/)

### 处理字符编码

正如您可能已经猜到的那样，问题经常出现，因为有许多不同的潜在字符编码。今天占主导地位的字符编码是 [UTF-8](https://en.wikipedia.org/wiki/UTF-8) ，它是 [Unicode](https://en.wikipedia.org/wiki/Unicode) 的一个实现。幸运的是，[今天百分之九十八的网页](https://w3techs.com/technologies/cross/character_encoding/ranking)都是用 UTF-8 编码的！

UTF 8 占优势，因为它可以有效地处理数量惊人的字符。它处理 Unicode 定义的所有 1，112，064 个潜在字符，包括中文、日文、阿拉伯文(从右到左书写)、俄文和更多字符集，包括[表情符号](https://home.unicode.org/emoji/about-emoji/)！

UTF-8 仍然有效，因为它使用可变数量的字节来编码字符，这意味着对于许多字符，它只需要一个字节，而对于其他字符，它可能需要多达四个字节。

**注意**:要了解更多关于 Python 中的编码，请查看 Python 中的 [Unicode &字符编码:无痛指南](https://realpython.com/python-encodings-guide/)。

虽然 UTF-8 占主导地位，并且假设 UTF-8 编码通常不会出错，但您仍然会一直遇到不同的编码。好消息是，在使用`urllib.request`时，你不需要成为编码专家来处理它们。

### 从字节到字符串

当您使用`urllib.request.urlopen()`时，响应的主体是一个 bytes 对象。您可能要做的第一件事是将 bytes 对象转换为字符串。也许你想做一些[网络搜集](https://realpython.com/beautiful-soup-web-scraper-python/)。为此，你需要**解码**字节。要用 Python 解码字节，你只需要找出使用的**字符编码**。编码，尤其是当提到字符编码时，通常被称为**字符集**。

如上所述，在 98%的情况下，您可能会安全地默认使用 UTF-8:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://www.example.com") as response:
...     body = response.read()
...
>>> decoded_body = body.decode("utf-8")
>>> print(decoded_body[:30])
<!doctype html>
<html>
<head>
```

在本例中，您获取从`response.read()`返回的 bytes 对象，并使用 bytes 对象的`.decode()`方法对其进行解码，将`utf-8`作为参数传入。当你[打印](https://realpython.com/python-print/) `decoded_body`时，你可以看到它现在是一个字符串。

也就是说，听天由命很少是一个好策略。幸运的是，标题是获取字符集信息的好地方:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://www.example.com") as response:
...     body = response.read()
...
>>> character_set = response.headers.get_content_charset()
>>> character_set
'utf-8'
>>> decoded_body = body.decode(character_set)
>>> print(decoded_body[:30])
<!doctype html>
<html>
<head>
```

在这个例子中，你在`response`的`.headers`对象上调用`.get_content_charset()`，并使用它来解码。这是一个方便的方法，它解析`Content-Type`头，这样您就可以轻松地将字节解码成文本。

### 从字节到文件

如果你想把字节解码成文本，现在你可以开始了。但是，如果您想将响应的主体写入文件，该怎么办呢？好吧，你有两个选择:

1.  将字节直接写入文件
2.  将字节解码成 Python 字符串，然后将字符串编码回文件

第一种方法最简单，但是第二种方法允许您根据需要更改编码。要更详细地了解文件操作，请看一下 Real Python 的[用 Python (Guide)](https://realpython.com/read-write-files-python/) 读写文件。

要将字节直接写入文件而无需解码，您需要内置的 [`open()`](https://docs.python.org/3/library/functions.html#open) 函数，并且您需要确保使用写二进制模式:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://www.example.com") as response:
...     body = response.read()
...
>>> with open("example.html", mode="wb") as html_file: ...     html_file.write(body)
...
1256
```

在`wb`模式下使用`open()`绕过了解码或编码的需要，将 HTTP 消息体的字节转储到`example.html`文件中。写操作后输出的数字表示已经写入的字节数。就是这样！您已经将字节直接写入文件，没有进行任何编码或解码。

现在假设您有一个不使用 UTF 8 的 URL，但是您想将内容写入一个使用 UTF 8 的文件。为此，首先将字节解码成字符串，然后将字符串编码成文件，指定字符编码。

谷歌的主页似乎根据你的位置使用不同的编码。在欧洲和美国的大部分地区，它使用 [ISO-8859-1](https://en.wikipedia.org/wiki/ISO/IEC_8859-1) 编码:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://www.google.com") as response:
...     body = response.read()
...
>>> character_set = response.headers.get_content_charset()
>>> character_set
'ISO-8859-1'
>>> content = body.decode(character_set)
>>> with open("google.html", encoding="utf-8", mode="w") as file:
...     file.write(content)
...
14066
```

在这段代码中，您获得了响应字符集，并使用它将 bytes 对象解码成一个字符串。然后，您将字符串写入一个文件，使用 UTF-8 编码它。

**注意**:有趣的是，谷歌似乎有各种各样的检查层，用来决定网页使用何种语言和编码。这意味着你可以指定一个 [`Accept-Language`头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language)，它似乎覆盖了你的 IP 地址。尝试使用不同的[区域标识符](https://www.w3.org/TR/ltli/)来看看你能得到什么样的编码！

写入文件后，您应该能够在浏览器或文本编辑器中打开结果文件。大多数现代文本处理器可以自动检测字符编码。

如果存在编码错误，并且您正在使用 Python 读取文件，那么您可能会得到一个错误:

>>>

```py
>>> with open("encoding-error.html", mode="r", encoding="utf-8") as file:
...     lines = file.readlines()
...
UnicodeDecodeError:
 'utf-8' codec can't decode byte
```

Python 显式地停止了这个过程并引发了一个异常，但是在一个显示文本的程序中，比如你正在查看这个页面的浏览器，你可能会发现臭名昭著的[替换字符](https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character):

[![Unicode Replacement Character](img/897d331c7cbea35299f79f279074be84.png)](https://files.realpython.com/media/220px-Replacement_character.65a078b17af7.png)

<figcaption class="figure-caption text-center">A Replacement Character</figcaption>

带白色问号的黑色菱形(ᦅ)、正方形(□)和矩形(▯)通常用于替换无法解码的字符。

有时，解码看似可行，但会产生难以理解的序列，如*？到-到-到。*，这也暗示使用了错误的字符集。在日本，他们甚至有一个词来形容由于字符编码问题而产生的乱码，即 [Mojibake](https://en.wikipedia.org/wiki/Mojibake) ，因为这些问题在互联网时代开始时就困扰着他们。

这样，您现在应该可以用从`urlopen()`返回的原始字节编写文件了。在下一节中，您将学习如何使用 [`json`](https://realpython.com/python-json/) 模块将字节解析到 Python 字典中。

[*Remove ads*](/account/join/)

### 从字节到字典

对于`application/json`响应，您通常会发现它们不包含任何编码信息:

>>>

```py
>>> from urllib.request import urlopen
>>> with urlopen("https://httpbin.org/json") as response:
...     body = response.read()
...
>>> character_set = response.headers.get_content_charset()
>>> print(character_set)
None
```

在这个例子中，您使用了 [httpbin](https://httpbin.org/) 的`json`端点，这个服务允许您试验不同类型的请求和响应。`json`端点模拟一个返回 JSON 数据的典型 API。请注意，`.get_content_charset()`方法在其响应中不返回任何内容。

即使没有字符编码信息，也不会全部丢失。根据 [RFC 4627](https://datatracker.ietf.org/doc/html/rfc4627) ，UTF-8 的默认编码是`application/json`规范的*绝对要求*。这并不是说每一台服务器都遵守规则，但是一般来说，如果 JSON 被传输，它几乎总是使用 UTF-8 编码。

幸运的是，`json.loads()`在幕后解码字节对象，甚至在它可以处理的不同[编码](https://docs.python.org/3/library/json.html#character-encodings)方面有一些余地。因此，`json.loads()`应该能够处理您扔给它的大多数字节对象，只要它们是有效的 JSON:

>>>

```py
>>> import json
>>> json.loads(body)
{'slideshow': {'author': 'Yours Truly', 'date': 'date of publication', 'slides'
: [{'title': 'Wake up to WonderWidgets!', 'type': 'all'}, {'items': ['Why <em>W
onderWidgets</em> are great', 'Who <em>buys</em> WonderWidgets'], 'title': 'Ove
rview', 'type': 'all'}], 'title': 'Sample Slide Show'}}
```

如您所见，`json`模块自动处理解码并生成一个 Python 字典。几乎所有的 API 都以 JSON 的形式返回键值信息，尽管您可能会遇到一些使用 [XML](https://en.wikipedia.org/wiki/XML) 的旧 API。为此，您可能想看看 Python 中 XML 解析器的[路线图。](https://realpython.com/python-xml-parser/)

有了这些，你应该对字节和编码有足够的了解。在下一节中，您将学习如何对使用`urllib.request`时可能遇到的一些常见错误进行故障诊断和修复。

## 常见`urllib.request`故障

不管你有没有使用`urllib.request`，在这个世界*狂野*的网络上你都会遇到各种各样的问题。在本节中，您将学习如何在开始时处理几个最常见的错误: **`403`错误**和 **TLS/SSL 证书错误**。不过，在查看这些特定的错误之前，您将首先学习如何在使用`urllib.request`时更普遍地实现**错误处理**。

### 实施错误处理

在您将注意力转向特定的错误之前，提高您的代码优雅地处理各种错误的能力将会得到回报。Web 开发受到错误的困扰，您可以投入大量时间明智地处理错误。在这里，您将学习在使用`urllib.request`时处理 HTTP、URL 和超时错误。

HTTP 状态代码伴随着状态行中的每个响应。如果您可以在响应中读取状态代码，那么请求就到达了它的目标。虽然这很好，但是只有当响应代码以`2`开头时，您才能认为请求完全成功。例如，`200`和`201`代表成功的请求。例如，如果状态码是`404`或`500`，则出错了，`urllib.request`会抛出一个 [`HTTPError`](https://docs.python.org/3/library/urllib.error.html#urllib.error.HTTPError) 。

有时会发生错误，提供的 URL 不正确，或者由于其他原因无法建立连接。在这些情况下，`urllib.request`会养出一个 [`URLError`](https://docs.python.org/3/library/urllib.error.html#urllib.error.URLError) 。

最后，有时服务器就是不响应。也许您的网络连接速度慢，服务器关闭，或者服务器被编程为忽略特定的请求。为了处理这个问题，您可以将一个`timeout`参数传递给`urlopen()`以在一定时间后引发一个 [`TimeoutError`](https://docs.python.org/3/library/exceptions.html#TimeoutError) 。

处理这些异常的第一步是捕捉它们。您可以利用`HTTPError`、`URLError`和`TimeoutError`类，用`try` … `except`块捕获`urlopen()`内产生的错误:

```py
# request.py

from urllib.error import HTTPError, URLError from urllib.request import urlopen

def make_request(url):
    try:
 with urlopen(url, timeout=10) as response:            print(response.status)
            return response.read(), response
 except HTTPError as error:        print(error.status, error.reason)
 except URLError as error:        print(error.reason)
 except TimeoutError:        print("Request timed out")
```

函数`make_request()`将一个 URL 字符串作为参数，尝试用`urllib.request`从该 URL 获得响应，并捕捉在出错时引发的`HTTPError`对象。如果 URL 是坏的，它将捕获一个`URLError`。如果没有任何错误，它将打印状态并返回一个包含主体和响应的元组。回应将在`return`后关闭。

该函数还使用一个`timeout`参数调用`urlopen()`，这将导致在指定的秒数后引发一个`TimeoutError`。十秒钟通常是等待响应的合适时间，不过和往常一样，这在很大程度上取决于您需要向哪个服务器发出请求。

现在，您已经准备好优雅地处理各种错误，包括但不限于您将在接下来讨论的错误。

[*Remove ads*](/account/join/)

### 处理`403`错误

现在您将使用`make_request()`函数向 [httpstat.us](https://httpstat.us/) 发出一些请求，这是一个用于测试的模拟服务器。这个模拟服务器将返回具有您所请求的状态代码的响应。例如，如果你向`https://httpstat.us/200`发出请求，你应该期待一个`200`的响应。

像 httpstat.us 这样的 API 用于确保您的应用程序能够处理它可能遇到的所有不同的状态代码。httpbin 也有这个功能，但是 httpstat.us 有更全面的状态代码选择。它甚至还有[臭名昭著的半官方](https://www.ietf.org/rfc/rfc2324.txt) `418`状态码，返回消息*我是茶壶*！

要与您在上一节中编写的`make_request()`函数进行交互，请在交互模式下运行该脚本:

```py
$ python3 -i request.py
```

使用`-i`标志，该命令将在[交互模式](https://docs.python.org/3/using/cmdline.html#cmdoption-i)下运行脚本。这意味着它将执行脚本，然后打开 [Python REPL](https://realpython.com/interacting-with-python/#using-the-python-interpreter-interactively) ，因此您现在可以调用您刚刚定义的函数:

>>>

```py
>>> make_request("https://httpstat.us/200")
200
(b'200 OK', <http.client.HTTPResponse object at 0x0000023D612660B0>)
>>> make_request("https://httpstat.us/403")
403 Forbidden
```

在这里，您尝试了 httpstat.us 的`200`和`403`端点。`200`端点按照预期通过并返回响应体和响应对象。`403`端点只是打印了错误消息，没有返回任何东西，这也是意料之中的。

[`403`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403) 状态意味着服务器理解了请求，但不会执行它。这是一个你会遇到的常见错误，尤其是在抓取网页的时候。在许多情况下，您可以通过传递一个[用户代理](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent)头来解决这个问题。

**注意**:有两个密切相关的 4xx 代码有时会引起混淆:

1.  [`401`未经授权](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401)
2.  [`403`禁](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403)

如果用户没有被识别或登录，服务器应该返回`401`，并且必须做一些事情来获得访问权，比如登录或注册。

如果用户被充分识别，但没有访问资源的权限，则应该返回`403`状态。例如，如果你登录了一个社交媒体账户，并试图查看一个人的个人资料页面，那么你很可能会获得一个`403`状态。

也就是说，不要完全信任状态代码。在复杂的分布式服务中，bug 是存在的，也是常见的。有些服务器根本就不是模范公民！

服务器识别谁或什么发出请求的主要方法之一是检查`User-Agent`头。由`urllib.request`发送的原始默认请求如下:

```py
GET https://httpstat.us/403 HTTP/1.1
Accept-Encoding: identity
Host: httpstat.us
User-Agent: Python-urllib/3.10
Connection: close
```

请注意，`User-Agent`被列为`Python-urllib/3.10`。你可能会发现一些网站会试图屏蔽网页抓取器，这个`User-Agent`就是一个很好的例子。也就是说，你可以用`urllib.request`来设置你自己的`User-Agent`，尽管你需要稍微修改一下你的函数:

```py
 # request.py from urllib.error import HTTPError, URLError -from urllib.request import urlopen +from urllib.request import urlopen, Request -def make_request(url): +def make_request(url, headers=None): +    request = Request(url, headers=headers or {}) try: -        with urlopen(url, timeout=10) as response: +        with urlopen(request, timeout=10) as response: print(response.status) return response.read(), response except HTTPError as error: print(error.status, error.reason) except URLError as error: print(error.reason) except TimeoutError: print("Request timed out")
```

要定制随请求发出的标题，首先必须用 URL 实例化一个 [`Request`](https://docs.python.org/3/library/urllib.request.html#urllib.request.Request) 对象。此外，您可以传入一个`headers`的[关键字参数](https://realpython.com/defining-your-own-python-function/#keyword-arguments)，它接受一个标准字典来表示您希望包含的任何头。因此，不是将 URL 字符串直接传递给`urlopen()`，而是传递这个已经用 URL 和头实例化的`Request`对象。

**注意**:在上面的例子中，当`Request`被实例化时，你需要给它传递头文件，如果它们已经被定义的话。否则，传递一个空白对象，比如`{}`。你不能通过`None`，因为这会导致错误。

要使用这个修改过的函数，重新启动交互会话，然后调用`make_request()`,用一个字典将头表示为一个参数:

>>>

```py
>>> body, response = make_request(
...     "https://www.httpbin.org/user-agent",
...     {"User-Agent": "Real Python"}
... )
200
>>> body
b'{\n  "user-agent": "Real Python"\n}\n'
```

在这个例子中，您向 httpbin 发出一个请求。这里您使用`user-agent`端点来返回请求的`User-Agent`值。因为您是通过定制用户代理`Real Python`发出请求的，所以这是返回的内容。

不过，有些服务器很严格，只接受来自特定浏览器的请求。幸运的是，可以在网上找到标准的`User-Agent`字符串，包括通过[用户代理数据库](https://developers.whatismybrowser.com/useragents/explore/)。它们只是字符串，所以您需要做的就是复制您想要模拟的浏览器的用户代理字符串，并将其用作`User-Agent`头的值。

[*Remove ads*](/account/join/)

### 修复 SSL `CERTIFICATE_VERIFY_FAILED`错误

另一个常见错误是由于 Python 无法访问所需的安全证书。为了模拟这个错误，你可以使用一些已知不良 SSL 证书的模拟网站，由[badssl.com](https://badssl.com/)提供。您可以向其中一个(如`superfish.badssl.com`)发出请求，并直接体验错误:

>>>

```py
>>> from urllib.request import urlopen
>>> urlopen("https://superfish.badssl.com/")
Traceback (most recent call last):
 (...)
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate (_ssl.c:997)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
 (...)
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>
```

这里，用已知的坏 SSL 证书向一个地址发出请求将导致`CERTIFICATE_VERIFY_FAILED`，它是`URLError`的一种类型。

SSL 代表安全套接字层。这有点用词不当，因为 SSL 已被弃用，取而代之的是 TLS、[传输层安全性](https://en.wikipedia.org/wiki/Transport_Layer_Security)。有时旧的术语只是坚持！这是一种加密网络流量的方法，这样一个假想的监听者就无法窃听到通过网络传输的信息。

如今，大多数网站的地址不是以`http://`开头，而是以`https://`开头，其中*代表*安全*。 [HTTPS](https://en.wikipedia.org/wiki/HTTPS) 连接必须通过 TLS 加密。`urllib.request`可以处理 HTTP 和 HTTPS 连接。*

HTTPS 的细节远远超出了本教程的范围，但是您可以将 HTTPS 连接想象成包含两个阶段，握手和信息传输。握手确保连接是安全的。有关 Python 和 HTTPS 的更多信息，请查看使用 Python 探索 HTTPS 的。

为了确定特定的服务器是安全的，发出请求的程序依赖于存储的可信证书。握手阶段会验证服务器的证书。Python 使用[操作系统的证书库](https://github.com/python/cpython/blob/9bf2cbc4c498812e14f20d86acb61c53928a5a57/Lib/ssl.py#L771)。如果 Python 找不到系统的证书存储库，或者存储库过期，那么您就会遇到这个错误。

**注意**:在之前的 Python 版本中，`urllib.request`的默认行为是**而不是**验证证书，这导致 [PEP 476](https://www.python.org/dev/peps/pep-0476/) 默认启用证书验证。在 [Python 3.4.3](https://docs.python.org/3/whatsnew/3.4.html#changed-in-3-4-3) 中默认改变。

有时，Python 可以访问的证书存储区已经过期，或者 Python 无法访问它，不管是什么原因。这是令人沮丧的，因为你有时可以从你的浏览器访问 URL，它认为它是安全的，然而`urllib.request`仍然引发这个错误。

你可能很想选择不验证证书，但这会使你的连接*不安全*，并且绝对*不推荐*:

>>>

```py
>>> import ssl
>>> from urllib.request import urlopen
>>> unverified_context = ssl._create_unverified_context()
>>> urlopen("https://superfish.badssl.com/", context=unverified_context)
<http.client.HTTPResponse object at 0x00000209CBE8F220>
```

这里您导入了 [`ssl`](https://docs.python.org/3/library/ssl.html) 模块，它允许您创建一个[未验证的上下文](https://docs.python.org/3/whatsnew/3.4.html#changed-in-3-4-3)。然后，您可以将这个上下文传递给`urlopen()`并访问一个已知的坏 SSL 证书。因为没有检查 SSL 证书，所以连接成功通过。

在采取这些孤注一掷的措施之前，请尝试更新您的操作系统或 Python 版本。如果失败了，那么您可以从`requests`库中取出一个页面并安装`certifi`:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python -m venv venv
PS> .\venv\Scripts\activate
(venv) PS> python -m pip install certifi
```

```py
$ python3 -m venv venv
$ source venv/bin/activate.sh
(venv) $ python3 -m pip install certifi
```

[`certifi`](https://github.com/certifi/python-certifi) 是一个证书集合，你可以用它来代替你系统的集合。您可以通过使用`certifi`证书包而不是操作系统的证书包来创建 SSL 上下文:

>>>

```py
>>> import ssl
>>> from urllib.request import urlopen
>>> import certifi
>>> certifi_context = ssl.create_default_context(cafile=certifi.where())
>>> urlopen("https://sha384.badssl.com/", context=certifi_context)
<http.client.HTTPResponse object at 0x000001C7407C3490>
```

在这个例子中，您使用了`certifi`作为您的 SSL 证书库，并使用它成功地连接到一个具有已知良好 SSL 证书的站点。请注意，您使用的是`.create_default_context()`，而不是`._create_unverified_context()`。

这样，您可以保持安全，而不会有太多的麻烦！在下一节中，您将尝试身份验证。

[*Remove ads*](/account/join/)

## 认证请求

认证是一个庞大的主题，如果您正在处理的认证比这里讨论的要复杂得多，这可能是进入`requests`包的一个很好的起点。

在本教程中，您将只讨论一种身份验证方法，它作为对您的请求进行身份验证所必须进行的调整类型的示例。确实有很多其他功能有助于身份验证，但这不会在本教程中讨论。

最常见的认证工具之一是不记名令牌，由 [RFC 6750](https://datatracker.ietf.org/doc/html/rfc6750) 指定。它经常被用作 [OAuth](https://en.wikipedia.org/wiki/OAuth) 的一部分，但也可以单独使用。它也是最常见的标题，您可以在当前的`make_request()`函数中使用它:

>>>

```py
>>> token = "abcdefghijklmnopqrstuvwxyz"
>>> headers = {
...     "Authorization": f"Bearer {token}"
... }
>>> make_request("https://httpbin.org/bearer", headers)
200
(b'{\n  "authenticated": true, \n  "token": "abcdefghijklmnopqrstuvwxyz"\n}\n',
<http.client.HTTPResponse object at 0x0000023D612642E0>)
```

在这个例子中，您向 httpbin `/bearer`端点发出一个请求，它模拟了载体认证。它将接受任何字符串作为令牌。它只需要 RFC 6750 指定的正确格式。名字*的*是`Authorization`，或者有时是小写的`authorization`，值*的*是`Bearer`，在值和令牌之间有一个空格。

**注意**:如果您使用任何形式的令牌或秘密信息，请确保妥善保护这些令牌。例如，不要将它们提交给 GitHub 库，而是将它们存储为临时的[环境变量](https://en.wikipedia.org/wiki/Environment_variable)。

恭喜您，您已经使用不记名令牌成功认证！

另一种形式的认证称为 [**基本访问认证**](https://en.wikipedia.org/wiki/Basic_access_authentication) ，这是一种非常简单的认证方法，仅比在报头中发送用户名和密码稍微好一点。很没有安全感！

当今最常用的协议之一是 [**【开放授权】**](https://en.wikipedia.org/wiki/OAuth) 。如果你曾经使用过谷歌、GitHub 或脸书登录另一个网站，那么你就使用过 OAuth。OAuth 流通常涉及您希望与之交互的服务和身份服务器之间的一些请求，从而产生一个短期的承载令牌。然后，该承载令牌可以与承载认证一起使用一段时间。

大部分身份验证归结于理解目标服务器使用的特定协议，并仔细阅读文档以使其工作。

## 用`urllib.request` 发布请求

您已经发出了许多 GET 请求，但是有时您想要*发送*信息。这就是发布请求的来源。要使用`urllib.request`进行 POST 请求，您不必显式地更改方法。你可以将一个`data`对象传递给一个新的`Request`对象或者直接传递给`urlopen()`。然而，`data`对象必须是一种特殊的格式。您将通过添加`data`参数来稍微修改您的`make_request()`函数以支持 POST 请求:

```py
 # request.py from urllib.error import HTTPError, URLError from urllib.request import urlopen, Request -def make_request(url, headers=None): +def make_request(url, headers=None, data=None): -    request = Request(url, headers=headers or {}) +    request = Request(url, headers=headers or {}, data=data) try: with urlopen(request, timeout=10) as response: print(response.status) return response.read(), response except HTTPError as error: print(error.status, error.reason) except URLError as error: print(error.reason) except TimeoutError: print("Request timed out")
```

在这里，您只是修改了函数来接受一个默认值为`None`的`data`参数，并将该参数传递给了`Request`实例化。然而，这并不是所有需要做的事情。您可以使用两种不同的格式之一来执行 POST 请求:

1.  **表格数据** : `application/x-www-form-urlencoded`
2.  **JSON** : `application/json`

第一种格式是 POST 请求最古老的格式，它涉及到用百分比编码对数据进行编码，也称为 URL 编码。您可能已经注意到键值对 URL 编码为一个[查询字符串](https://en.wikipedia.org/wiki/Query_string)。键用等号(`=`)与值分开，键-值对用&符号(`&`)分开，空格通常被取消，但可以用加号(`+`)代替。

如果您从 Python 字典开始，要将表单数据格式用于您的`make_request()`函数，您需要编码两次:

1.  一次对字典进行 URL 编码
2.  然后再次将结果字符串编码成字节

对于 URL 编码的第一阶段，您将使用另一个`urllib`模块`urllib.parse`。记得在交互模式下启动你的脚本，这样你就可以使用`make_request()`功能并在 REPL 上玩它:

>>>

```py
>>> from urllib.parse import urlencode
>>> post_dict = {"Title": "Hello World", "Name": "Real Python"}
>>> url_encoded_data = urlencode(post_dict)
>>> url_encoded_data
'Title=Hello+World&Name=Real+Python'
>>> post_data = url_encoded_data.encode("utf-8")
>>> body, response = make_request(
...     "https://httpbin.org/anything", data=post_data
... )
200
>>> print(body.decode("utf-8"))
{
 "args": {},
 "data": "",
 "files": {},
 "form": { "Name": "Real Python", "Title": "Hello World" }, "headers": {
 "Accept-Encoding": "identity",
 "Content-Length": "34",
 "Content-Type": "application/x-www-form-urlencoded",
 "Host": "httpbin.org",
 "User-Agent": "Python-urllib/3.10",
 "X-Amzn-Trace-Id": "Root=1-61f25a81-03d2d4377f0abae95ff34096"
 },
 "json": null,
 "method": "POST", "origin": "86.159.145.119",
 "url": "https://httpbin.org/anything"
}
```

在本例中，您可以:

1.  从`urllib.parse`模块导入`urlencode()`
2.  初始化你的文章数据，从字典开始
3.  使用`urlencode()`功能对字典进行编码
4.  使用 UTF-8 编码将结果字符串编码成字节
5.  向`httpbin.org`的`anything`端点发出请求
6.  打印 UTF-8 解码响应正文

UTF-8 编码是`application/x-www-form-urlencoded`类型的[规范](https://url.spec.whatwg.org/#urlencoded-parsing)的一部分。UTF-8 被优先用于解码身体，因为你已经知道`httpbin.org`可靠地使用 UTF-8。

来自 httpbin 的`anything`端点充当一种 echo，返回它接收到的所有信息，以便您可以检查您所做请求的细节。在这种情况下，你可以确认`method`确实是`POST`，你可以看到你发送的数据列在`form`下面。

要使用 JSON 发出同样的请求，您将使用`json.dumps()`将 Python 字典转换成 JSON 字符串，使用 UTF-8 对其进行编码，将其作为`data`参数传递，最后添加一个特殊的头来指示数据类型是 JSON:

>>>

```py
>>> post_dict = {"Title": "Hello World", "Name": "Real Python"}
>>> import json
>>> json_string = json.dumps(post_dict) >>> json_string
'{"Title": "Hello World", "Name": "Real Python"}'
>>> post_data = json_string.encode("utf-8")
>>> body, response = make_request(
...     "https://httpbin.org/anything",
...     data=post_data,
...     headers={"Content-Type": "application/json"}, ... )
200
>>> print(body.decode("utf-8"))
{
 "args": {},
 "data": "{\"Title\": \"Hello World\", \"Name\": \"Real Python\"}",
 "files": {},
 "form": {},
 "headers": {
 "Accept-Encoding": "identity",
 "Content-Length": "47",
 "Content-Type": "application/json",
 "Host": "httpbin.org",
 "User-Agent": "Python-urllib/3.10",
 "X-Amzn-Trace-Id": "Root=1-61f25a81-3e35d1c219c6b5944e2d8a52"
 },
 "json": { "Name": "Real Python", "Title": "Hello World" }, "method": "POST", "origin": "86.159.145.119",
 "url": "https://httpbin.org/anything"
}
```

这次为了[序列化](https://en.wikipedia.org/wiki/Serialization)字典，您使用`json.dumps()`而不是`urlencode()`。您还显式添加了值为`application/json`的 [`Content-Type`头](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type)。有了这些信息，httpbin 服务器可以在接收端反序列化 JSON。在它的回复中，你可以看到列在`json`键下的数据。

**注意**:有时候需要以纯文本的形式发送 JSON 数据，这种情况下步骤同上，只是你把`Content-Type`设置为`text/plain; charset=UTF-8`。很多这些必需品依赖于你发送数据的服务器或 API，所以一定要阅读文档并进行实验！

这样，你现在就可以开始发布请求了。本教程不会详细介绍其他请求方法，比如 [PUT](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/PUT) 。我只想说，您也可以通过向 [`Request`对象](https://docs.python.org/3/library/urllib.request.html#urllib.request.Request)的实例化传递一个`method`关键字参数来显式设置该方法。

[*Remove ads*](/account/join/)

## 请求包生态系统

为了使事情更加完整，本教程的最后一部分将致力于阐明 Python 中 HTTP 请求的包生态系统。因为有很多套餐，没有明确的标准，会比较混乱。也就是说，每个包都有用例，这意味着您有更多的选择！

### 什么是`urllib2`和`urllib3`？

要回答这个问题，你需要追溯到早期的 Python，一直追溯到 1.2 版本，最初的 [`urllib`](https://docs.python.org/2/library/urllib.html) 推出的时候。在 1.6 版本左右，增加了一个改版的 [`urllib2`](https://docs.python.org/2/library/urllib2.html) ，它与原来的`urllib`并存。当 Python 3 出现时，最初的`urllib`被弃用，`urllib2`放弃了`2`，取了最初的`urllib`名称。它也分裂成几部分:

*   [T2`urllib.error`](https://docs.python.org/3/library/urllib.error.html#module-urllib.error)
*   [T2`urllib.parse`](https://docs.python.org/3/library/urllib.parse.html#module-urllib.parse)
*   [T2`urllib.request`](https://docs.python.org/3/library/urllib.request.html#module-urllib.request)
*   [T2`urllib.response`](https://docs.python.org/3/library/urllib.request.html#module-urllib.response)
*   [T2`urllib.robotparser`](https://docs.python.org/3/library/urllib.robotparser.html#module-urllib.robotparser)

那么 [`urllib3`](https://github.com/urllib3/urllib3) 呢？那是在`urllib2`还在的时候开发的第三方库。它与标准库无关，因为它是一个独立维护的库。有意思的是，`requests`库居然在遮光罩下使用`urllib3`， [`pip`](https://realpython.com/what-is-pip/) 也是如此！

### 什么时候应该用`requests`而不用`urllib.request`？

主要答案是易用性和安全性。`urllib.request`被认为是一个低级的库，它公开了许多关于 HTTP 请求工作的细节。针对`urllib.request`的 Python [文档](https://docs.python.org/3/library/urllib.request.html)毫不犹豫地推荐`requests`作为更高级的 HTTP 客户端接口。

如果您日复一日地与许多不同的 REST APIs 交互，那么强烈推荐使用`requests`。`requests`库标榜自己为“为人类而建”,并成功地围绕 HTTP 创建了一个直观、安全和简单的 API。它通常被认为是最重要的图书馆！如果你想了解更多关于`requests`库的信息，请查看真正的 Python[`requests`](https://realpython.com/python-requests/)指南。

关于`requests`如何让事情变得更简单的一个例子是字符编码。你会记得使用`urllib.request`时，你必须了解编码，并采取一些步骤来确保没有错误的体验。`requests`包将它抽象出来，并通过使用 [`chardet`](https://github.com/chardet/chardet) (一种通用的字符编码检测器)来解析编码，以防有什么有趣的事情发生。

如果你的目标是学习更多关于标准 Python 和它如何处理 HTTP 请求的细节，那么`urllib.request`是一个很好的方法。你甚至可以更进一步，使用非常低级的 [`http`模块](https://docs.python.org/3/library/http.html)。另一方面，你可能只是想将依赖性保持在最低限度，而`urllib.request`完全有能力做到这一点。

### 为什么`requests`不是标准库的一部分？

也许你想知道为什么`requests`现在还不是核心 Python 的一部分。

这是一个复杂的问题，没有简单快速的答案。关于原因有许多猜测，但有两个原因似乎很突出:

1.  `requests`有其他需要集成的第三方依赖关系。
2.  需要保持敏捷，并且能够在标准库之外做得更好。

`requests`库具有第三方依赖性。将`requests`集成到标准库中也意味着集成`chardet`、`certifi`和`urllib3`等等。另一种选择是从根本上改变`requests`,只使用 Python 现有的标准库。这不是一项简单的任务！

整合`requests`也意味着开发这个库的现有团队将不得不放弃对设计和实现的完全控制，让位于 [PEP](https://www.python.org/dev/peps/pep-0001/#what-is-a-pep) 决策过程。

HTTP 规范和建议一直在变化，一个高水平的库必须足够敏捷才能跟上。如果有一个安全漏洞需要修补，或者有一个新的工作流需要添加，`requests`团队可以比作为 Python 发布过程的一部分更快地构建和发布。据说曾经有过这样的情况，他们在漏洞被发现 12 小时后发布了一个安全补丁！

关于这些问题的有趣概述，请查看[向标准库](https://lwn.net/Articles/640838/)添加请求，其中总结了在 Python 语言峰会上与请求的创建者和维护者 [Kenneth Reitz](https://kennethreitz.org/) 的讨论。

因为这种敏捷性对于`requests`和它的底层`urllib3`来说是如此的必要，所以经常使用矛盾的说法，即`requests`对于标准库来说太重要了。这是因为 Python 社区如此依赖于`requests`及其灵活性，以至于将它集成到核心 Python 中可能会损害它和 Python 社区。

在`requests`的 GitHub 库问题板上，发布了一个问题，要求将 [`requests`包含在标准库](https://github.com/psf/requests/issues/2424)中。`requests`和`urllib3`的开发者插话说，他们很可能会对自己维护它失去兴趣。一些人甚至说他们将分叉存储库并继续为他们自己的用例开发它们。

也就是说，注意`requests`库 GitHub 存储库是托管在 Python 软件基金会的账户下的。仅仅因为某些东西不是 Python 标准库的一部分，并不意味着它不是生态系统不可分割的一部分！

似乎目前的情况对 Python 核心团队和`requests`的维护者都有效。虽然对于新手来说可能有点困惑，但是现有的结构为 HTTP 请求提供了最稳定的体验。

同样重要的是要注意 HTTP 请求本质上是复杂的。不要试图掩饰得太多。它公开了 HTTP 请求的许多内部工作方式，这就是它被称为低级模块的原因。您选择`requests`还是`urllib.request`实际上取决于您的特定用例、安全考虑和偏好。

[*Remove ads*](/account/join/)

## 结论

现在，您可以使用`urllib.request`来发出 HTTP 请求了。现在，您可以在您的项目中使用这个内置模块，让它们在更长时间内保持无依赖性。您还通过使用较低级别的模块，如`urllib.request`，对 HTTP 有了深入的了解。

**在本教程中，您已经:**

*   学会了如何用`urllib.request`制作基本的 [**HTTP 请求**](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol#Request_and_response_messages_through_connections)
*   探究了一条 **HTTP 消息**的具体细节，并研究了它是如何被`urllib.request`表示的
*   弄清楚如何处理 HTTP 消息的字符编码
*   探究使用`urllib.request`时的一些**常见错误**，并学习如何解决它们
*   用`urllib.request`尝试一下**认证请求**的世界
*   理解了为什么`urllib`和`requests`库都存在，以及**何时使用其中一个**

现在，您已经能够使用`urllib.request`发出基本的 HTTP 请求，并且您还拥有使用标准库深入底层 HTTP 领域的工具。最后，你可以选择是使用`requests`还是`urllib.request`，这取决于你想要什么或者需要什么。尽情探索网络吧！

**了解更多:** ，获取新的 Python 教程和新闻，让您成为更有效的 Python 爱好者。****************
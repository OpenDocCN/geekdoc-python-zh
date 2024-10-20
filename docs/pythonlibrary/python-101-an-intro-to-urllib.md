# python 101:URL lib 简介

> 原文：<https://www.blog.pythonlibrary.org/2016/06/28/python-101-an-intro-to-urllib/>

Python 3 中的 **urllib** 模块是一个模块集合，可以用来处理 URL。如果你来自 Python 2 背景，你会注意到在 Python 2 中你有 urllib 和 urllib2。这些现在是 Python 3 中 urllib 包的一部分。urllib 的当前版本由以下模块组成:

*   urllib.request
*   urllib.error
*   urllib.parse
*   urllib.rebotparser

除了 **urllib.error** 之外，我们将分别讨论每个部分。官方文档实际上建议您可能想要查看第三方库，**请求**，以获得更高级别的 HTTP 客户端接口。然而，我相信知道如何打开 URL 并与它们交互而不使用第三方是有用的，它也可以帮助你理解为什么请求包如此受欢迎。

* * *

### urllib.request

**urllib.request** 模块主要用于打开和获取 URL。让我们来看看您可以使用 **urlopen** 函数做的一些事情:

```py

>>> import urllib.request
>>> url = urllib.request.urlopen('https://www.google.com/')
>>> url.geturl()
'https://www.google.com/'
>>> url.info()
 >>> header = url.info()
>>> header.as_string()
('Date: Fri, 24 Jun 2016 18:21:19 GMT\n'
 'Expires: -1\n'
 'Cache-Control: private, max-age=0\n'
 'Content-Type: text/html; charset=ISO-8859-1\n'
 'P3P: CP="This is not a P3P policy! See '
 'https://www.google.com/support/accounts/answer/151657?hl=en for more info."\n'
 'Server: gws\n'
 'X-XSS-Protection: 1; mode=block\n'
 'X-Frame-Options: SAMEORIGIN\n'
 'Set-Cookie: '
 'NID=80=tYjmy0JY6flsSVj7DPSSZNOuqdvqKfKHDcHsPIGu3xFv41LvH_Jg6LrUsDgkPrtM2hmZ3j9V76pS4K_cBg7pdwueMQfr0DFzw33SwpGex5qzLkXUvUVPfe9g699Qz4cx9ipcbU3HKwrRYA; '
 'expires=Sat, 24-Dec-2016 18:21:19 GMT; path=/; domain=.google.com; HttpOnly\n'
 'Alternate-Protocol: 443:quic\n'
 'Alt-Svc: quic=":443"; ma=2592000; v="34,33,32,31,30,29,28,27,26,25"\n'
 'Accept-Ranges: none\n'
 'Vary: Accept-Encoding\n'
 'Connection: close\n'
 '\n')
>>> url.getcode()
200 
```

在这里，我们导入我们的模块，并要求它打开谷歌的网址。现在我们有了一个可以与之交互的 **HTTPResponse** 对象。我们做的第一件事是调用 **geturl** 方法，该方法将返回检索到的资源的 url。这有助于发现我们是否遵循了重定向。

接下来我们调用 **info** ，它将返回关于页面的元数据，比如标题。因此，我们将结果赋给我们的 **headers** 变量，然后调用它的 **as_string** 方法。这将打印出我们从 Google 收到的标题。您还可以通过调用 **getcode** 来获取 HTTP 响应代码，在本例中是 200，这意味着它成功地工作了。

如果您想查看页面的 HTML，您可以在我们创建的 url 变量上调用 **read** 方法。我不会在这里重复，因为输出会很长。

请注意，请求对象默认为 GET 请求，除非您指定了**数据**参数。如果您传入数据参数，那么请求对象将发出 POST 请求。

* * *

#### 下载文件

urllib 包的一个典型用例是下载文件。让我们找出完成这项任务的几种方法:

```py

>>> import urllib.request
>>> url = 'https://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'
>>> response = urllib.request.urlopen(url)
>>> data = response.read()
>>> with open('/home/mike/Desktop/test.zip', 'wb') as fobj:
...     fobj.write(data)
... 

```

在这里，我们只需打开一个 URL，它会将我们带到存储在我的博客上的一个 zip 文件。然后我们读取数据并把它写到磁盘上。实现这一点的另一种方法是使用 **urlretrieve** :

```py

>>> import urllib.request
>>> url = 'https://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'
>>> tmp_file, header = urllib.request.urlretrieve(url)
>>> with open('/home/mike/Desktop/test.zip', 'wb') as fobj:
...     with open(tmp_file, 'rb') as tmp:
...         fobj.write(tmp.read())

```

urlretrieve 方法将网络对象复制到本地文件。它复制到的文件是随机命名的，并放在 temp 目录中，除非您使用 urlretrieve 的第二个参数，在那里您可以实际指定您想要保存文件的位置。这将为您节省一个步骤，并使您的代码更加简单:

```py

>>> import urllib.request
>>> url = 'https://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'
>>> urllib.request.urlretrieve(url, '/home/mike/Desktop/blog.zip')
('/home/mike/Desktop/blog.zip',
 ) 
```

如您所见，它从请求中返回保存文件的位置和头信息。

#### 指定您的用户代理

当你用浏览器访问一个网站时，浏览器会告诉网站它是谁。这被称为**用户代理**字符串。Python 的 urllib 将自己标识为 **Python-urllib/x.y** ，其中 x 和 y 是 Python 的主版本号和次版本号。一些网站不会识别这个用户代理字符串，并且会以奇怪的方式运行或根本不工作。幸运的是，您很容易设置自己的自定义用户代理字符串:

```py

>>> import urllib.request
>>> user_agent = ' Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0'
>>> url = 'http://www.whatsmyua.com/'
>>> headers = {'User-Agent': user_agent}
>>> request = urllib.request.Request(url, headers=headers)
>>> with urllib.request.urlopen(request) as response:
...     with open('/home/mdriscoll/Desktop/user_agent.html', 'wb') as out:
...         out.write(response.read())

```

这里我们设置了 Mozilla FireFox 的用户代理，我们设置了 http://www.whatsmyua.com/的 URL，它会告诉我们它认为我们的用户代理字符串是什么。然后，我们使用我们的 url 和头创建一个**请求**实例，并将其传递给 **urlopen** 。最后我们保存结果。如果您打开结果文件，您将看到我们成功地更改了用户代理字符串。您可以用这段代码尝试一些不同的字符串，看看它会有什么变化。

* * *

### urllib.parse

**urllib.parse** 库是分解 URL 字符串并将它们组合在一起的标准接口。例如，您可以使用它将相对 URL 转换为绝对 URL。让我们尝试使用它来解析包含查询的 URL:

```py

>>> from urllib.parse import urlparse
>>> result = urlparse('https://duckduckgo.com/?q=python+stubbing&t=canonical&ia=qa')
>>> result
ParseResult(scheme='https', netloc='duckduckgo.com', path='/', params='', query='q=python+stubbing&t=canonical&ia=qa', fragment='')
>>> result.netloc
'duckduckgo.com'
>>> result.geturl()
'https://duckduckgo.com/?q=python+stubbing&t=canonical&ia=qa'
>>> result.port
None

```

这里，我们导入了 **urlparse** 函数，并向它传递一个包含对 duckduckgo 网站的搜索查询的 URL。我的查询是查找关于“python stubbing”的文章。如您所见，它返回了一个 **ParseResult** 对象，您可以用它来了解更多关于 URL 的信息。例如，您可以获得端口信息(本例中没有)、网络位置、路径等等。

#### 提交 Web 表单

这个模块还保存了 **urlencode** 方法，这对于向 URL 传递数据非常有用。urllib.parse 库的一个典型用例是提交 web 表单。让我们通过让 duckduckgo 搜索引擎查找 Python 来了解如何做到这一点:

```py

>>> import urllib.request
>>> import urllib.parse
>>> data = urllib.parse.urlencode({'q': 'Python'})
>>> data
'q=Python'
>>> url = 'http://duckduckgo.com/html/'
>>> full_url = url + '?' + data
>>> response = urllib.request.urlopen(full_url)
>>> with open('/home/mike/Desktop/results.html', 'wb') as f:
...     f.write(response.read())

```

这很简单。基本上，我们希望自己使用 Python 而不是浏览器向 duckduckgo 提交一个查询。为此，我们需要使用 **urlencode** 来构造查询字符串。然后我们将它们放在一起创建一个完全合格的 URL，并使用 urllib.request 提交表单。然后我们获取结果并保存到磁盘上。

* * *

### URL lib robotparser

**robotparser** 模块由一个类 **RobotFileParser** 组成。这个类将回答关于一个特定的用户代理是否可以获取一个已经发布了 **robot.txt** 文件的 URL 的问题。robots.txt 文件将告诉 web scraper 或 robot 服务器的哪些部分不应该被访问。让我们用 ArsTechnica 的网站看一个简单的例子:

```py

>>> import urllib.robotparser
>>> robot = urllib.robotparser.RobotFileParser()
>>> robot.set_url('http://arstechnica.com/robots.txt')
None
>>> robot.read()
None
>>> robot.can_fetch('*', 'http://arstechnica.com/')
True
>>> robot.can_fetch('*', 'http://arstechnica.com/cgi-bin/')
False

```

这里我们导入 robot parser 类并创建它的一个实例。然后我们传递给它一个 URL，指定网站的 robots.txt 文件所在的位置。接下来，我们告诉解析器读取文件。现在已经完成了，我们给它几个不同的 URL 来找出哪些我们可以抓取，哪些不可以。我们很快发现我们可以访问主站点，但不能访问 cgi-bin。

* * *

### 包扎

您已经达到了应该能够胜任使用 Python 的 urllib 包的程度。在本章中，我们学习了如何下载文件、提交 web 表单、更改用户代理以及访问 robots.txt 文件。urllib 有很多额外的功能没有在这里讨论，比如网站认证。然而，在尝试使用 urllib 进行身份验证之前，您可能需要考虑切换到 **requests** 库，因为 requests 实现更容易理解和调试。我还想指出的是，Python 通过其 http.Cookiess 模块支持 cookie，尽管这也很好地包装在 requests 包中。你也许应该考虑两者都尝试一下，看看哪一个对你最有意义。
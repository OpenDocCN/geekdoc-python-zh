# Python 中的请求–使用 Python 请求网页

> 原文：<https://www.askpython.com/python-modules/requests-in-python>

Python 中的 Requests 是一个优雅的库，允许您通过 Python 向网页发送 HTTP/1.1 请求。

Python 2.7 和 3.5+都正式支持它。诸如保持活动、连接池、具有持久 cookies 的会话、浏览器风格的 SSL 验证等高级功能使其成为开发人员的首选。

在本文中，我们将学习更多关于这些特性的知识，以及如何开始使用 Python 请求模块创建 web 请求。

## 如何在 Python 中安装请求？

在 Python 中安装**请求**很容易，也很容易理解。在 Python 中安装模块有几种方法。但是在本文中，我们将向您展示如何使用 [pip 模块](https://www.askpython.com/python-modules/python-pip)。

打开您的终端或命令提示符(如果您是 windows 用户)并键入以下命令。

```py
pip install requests 
#Or (if the first command doesn't work) use:
pip3 install requests

```

它应该成功地在您的设备中安装请求模块。

## 在 Python 中使用请求

为了理解请求模块是如何工作的，我们需要知道当我们浏览网页时发生了什么的基本原理，以及它如何立即向你显示你希望看到的内容。

每当你点击一个链接，我们就向请求页面的服务器发送一个 HTTP(超文本传输协议)请求。

收到请求后，服务器会向我们发回我们请求的正确内容。我们将要学习的两个最有用的 HTTP 请求是 GET 和 POST 请求。

在下一节中，我们将学习如何在请求库中使用这些方法。但是首先，我们需要将它导入到您的脚本或解释器中。

```py
import requests

```

### 1.获取请求

这个方法用于指示我们正在从服务器请求我们选择的 URL 的内容。所以，假设我们想使用 HTTP 请求获得 google 的主页。

**输入下面一行。**

```py
r = requests.get("http://google.com")

```

下面是这一行代码的作用:它通过 GET()方法向 Google 主页发送一个 HTTP GET 请求，其中 URL 作为参数提供。响应对象存储在我们的“r”变量中。

我们的响应对象的实例进一步对保留的数据进行分类，并将它们存储在适当的属性中。下面是一个例子

```py
print(r.status_code) 
#The output provides the status code for the url. For a success full attempt, the result is 200

print(r.headers)
#The following attribute returns a python dictionary containing key-value pair of the headers

print(r.text)
#This command prints out the response content from the server or Static Source Code. 

print(r.encoding)
r.encoding = 'utf-8' 
#Requests library also allows you to see or change the encoding of the response content. 

```

### 2.使用 GET 方法传递参数

通常一个 get 方法不能让我们找到我们需要访问的所有信息，所以我们需要在最初的 GET 请求中传递额外的参数。

参数主要是封装在一个[元组](https://www.askpython.com/python/tuple/python-tuple)或[列表](https://www.askpython.com/python/difference-between-python-list-vs-array)中的数据的键值对。我们可以使用 get()方法的 params 参数来发送它。

**见语法跟随。**

```py
import requests 
payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.get('http://httpbin.org/get', params=payload)
print(r.text)

```

输出:

```py
{
  "args": {
    "key1": "value1",
    "key2": "value2"
  },
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Host": "httpbin.org",
    "User-Agent": "python-requests/2.22.0",
    "X-Amzn-Trace-Id": "Root=1-5f9a64d1-2abfc74b2725386140a897e3"
  },
  "origin": 0.0.0.0, 
  "url": "http://httpbin.org/get?key1=value1&key2=value2"
}

```

### 3.发布请求

与 Python 中的 GET 请求不同，HTTP **中的 POST 方法需要**用它来发布有效负载。此方法用于向服务器发送数据，而不是直接检索数据。在我们的请求库中，我们可以使用 post()方法访问 POST。

快速浏览一下语法:

```py
import requests 
payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.post("https://httpbin.org/post", data=payload)
print(r.text)

```

输出:

```py
{
  "args": {},
  "data": "",
  "files": {},
  "form": {
    "key1": "value1",
    "key2": "value2"
  },
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Content-Length": "23",
    "Content-Type": "application/x-www-form-urlencoded",
    "Host": "httpbin.org",
    "User-Agent": "python-requests/2.22.0",
    "X-Amzn-Trace-Id": "Root=1-5f9a6726-276da087230912e01dd5dcd7"
  },
  "json": null,
  "origin": [REDACTED],
  "url": "https://httpbin.org/post"
}

```

## Python 中请求的一些高级特性

我们的文章主要关注两个最基本但非常重要的 HTTP 方法。但是请求模块支持许多这样的方法，比如 PUT、PATCH、DELETE 等。

“请求”模块在开发人员中如此出名的一个主要原因是它的高级功能，如:

1.  **Sessions 对象:**它主要用于在不同的请求之间存储相同的 cookies，从而提供更快的响应。
2.  **对 SOCKS 代理的支持:**虽然你需要安装一个独立的依赖项(称为“requests[socks]”)，但它可以极大地帮助你处理多个请求的性能，尤其是当服务器速率限制了你的 IP 时。
3.  **SSL 验证:**您可以通过在 get()方法中提供一个额外的参数“verify=True”来强制检查网站是否正确支持 SSL 使用请求。如果网站未能显示对 SSL 的适当支持，脚本将抛出一个错误。

## 结论

无论是 web 抓取还是其他与 HTTP 相关的工作，请求模块都是最受欢迎的选项。

requests 模块唯一不足的地方是处理页面源代码中的动态变化，因为该模块不是为执行 javascript 命令而设计的。

希望这篇文章能让你对这个模块有一个基本的了解。

## 资源

你可以在他们的官方文档网站上了解更多关于这个模块的信息:[https://requests.readthedocs.io/en/latest/](https://requests.readthedocs.io/en/latest/)
他们的官方 Github 回购:[https://github.com/psf/requests](https://github.com/psf/requests)
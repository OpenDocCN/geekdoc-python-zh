# Python HTTP 模块–您需要知道的一切！

> 原文：<https://www.askpython.com/python-modules/http-module>

读者朋友们，你们好！在本文中，我们将详细关注 **Python HTTP 模块**。所以，让我们开始吧！！🙂

***推荐阅读:[Python 中的 ReLU 函数](https://www.askpython.com/python/examples/relu-function)***

* * *

## 简明概述–Python HTTP 模块

Python 是一种多用途的编程语言，它帮助我们轻松地在不同层次上执行各种操作。python 模块提供了大量的模块和内置函数来执行经典和定制/用户定义的操作。

当涉及到[数据抓取](https://www.askpython.com/python/beautiful-soup-web-scraping)，或者通过 API 或 JSON 数据路径获取信息时，我们需要能够打开到 web URL 的连接，然后在其上执行响应操作的函数。

Python 为我们提供了 HTTP 模块。借助 HTTP 模块，我们可以轻松处理 web URL 连接并执行各种操作，例如:

1.  **获取请求**
2.  **发布请求**
3.  **上传请求**
4.  **从响应头**中获取头，等等

我们将通过 HTTP 模块来看看上面提到的每一个函数。HTTP 模块通常与 urllib 模块一起处理最新的 HTTP 请求。

让我们开始吧！！

* * *

## 1.建立 HTTP 连接

在使用 web URL 执行任何请求操作之前，与 URL 建立连接非常重要。在 HTTP 模块中，HTTPConnection()函数使我们能够在特定的端口(最好是 80)上打开到 URL 的连接，并且有一个超时期限。

**语法**:

```py
http.client.HTTPConnection('URL', port, timeout=)

```

*   URL:用来建立连接的 web URL。
*   port:需要建立连接的端口号。
*   超时:连接中止的宽限期。

**举例**:

```py
import http.client
request = http.client.HTTPConnection('www.google.com', 80, timeout=10)
print(request)

```

**输出**:

```py
<http.client.HTTPConnection object at 0x00000223BAD2DDD8>

```

* * *

## 2.Python HTTP GET 请求

使用 HTTP 模块，我们可以对 web URL 执行 GET 请求，我们可以使用它从 web URL 获得响应。使用 GET response，我们建立一个与 web URL 的 give-away 连接，获取由 URL 提供的响应数据，并分配一个对象来表示它。

此外，还可以使用 request()函数的**原因**和**状态**属性来验证响应数据。

**语法**:

```py
request("GET")

```

**举例**:

```py
import http.client

data = http.client.HTTPSConnection("www.askpython.com")
data.request("GET", "/")
response = data.getresponse()
print(response.reason)
print(response.status)
data.close()

```

**输出**:

```py
OK
200

```

## 3.Python HTTP Post & Put 请求

除了 HTTP GET 请求，我们还可以使用 POST 请求来注入数据，即将数据发送到 URL，然后使用 GET 请求从 URL 获得响应。

此外，如果我们希望修改某些数据并将其添加到 URL/API 的 [JSON 数据](https://www.askpython.com/python/examples/serialize-deserialize-json)中，我们可以使用 PUT 请求来完成。使用 PUT 请求，我们可以将数据添加到 URL 的现有 JSON 中，并使用 GET 请求检查它的连接。

**语法**–**发布请求**:

```py
request('POST', '/post', json_data, headers)

```

**语法–上传请求**:

```py
request("PUT", "/put", json_data)

```

## 4.从响应中检索标题列表

一旦您建立了与 web URL 的连接并请求 GET 响应，我们现在就可以使用 getheaders()函数从可用的响应中提取和检索标题数据。getheaders()函数表示来自 GET 响应的标题数据列表。

**语法**:

```py
request.getheaders()

```

**举例**:

```py
import http.client

data = http.client.HTTPSConnection("www.askpython.com")
data.request("GET", "/")
response = data.getresponse()
header = response.getheaders()

print(header)
print(response.reason)
print(response.status)
data.close()

```

**输出—**

```py
[('Connection', 'Keep-Alive'), ('Content-Type', 'text/html; charset=UTF-8'), ('Link', '<https://www.askpython.com/wp-json/>; rel="https://api.w.org/"'), ('Link', '</wp-content/themes/astra/assets/css/minified/style.min.css>; rel=preload; as=style,</wp-content/themes/astra/assets/css/minified/menu-animation.min.css>; rel=preload; as=style,</wp-includes/css/dist/block-library/style.min.css>; rel=preload; as=style,</wp-content/plugins/wp-to-twitter/css/twitter-feed.css>; rel=preload; as=style,</wp-content/plugins/easy-table-of-contents/vendor/icomoon/style.min.css>; rel=preload; as=style,</wp-content/plugins/easy-table-of-contents/assets/css/screen.min.css>; rel=preload; as=style,</wp-content/themes/obsidian/style.css>; rel=preload; as=style'), ('Etag', '"294191-1623490484;;;"'), ('X-LiteSpeed-Cache', 'hit'), ('Transfer-Encoding', 'chunked'), ('Date', 'Sun, 13 Jun 2021 07:30:37 GMT'), ('Server', 'LiteSpeed')]
OK 
200

```

* * *

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂
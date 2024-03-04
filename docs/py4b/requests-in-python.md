# Python 中的请求

> 原文：<https://www.pythonforbeginners.com/requests/requests-in-python>

### 什么是请求

Requests 模块是 Python 的一个优雅而简单的 HTTP 库。

### 我能对请求做什么？

请求允许您发送 HTTP/1.1 请求。

您可以使用简单的 Python 字典添加头、表单数据、多部分文件和参数，并以相同的方式访问响应数据。

注意，本帖笔记摘自 Python-Requests.org:[http://docs.python-requests.org/en/latest/](http://docs.python-requests.org/en/latest/ "requests")

### 请求安装

要安装请求，只需:
$ pip 安装请求

或者，如果你绝对必须:
$ easy_install 请求

### 请求

首先通过导入请求模块:
> > >导入请求

现在，让我们试着得到一个网页。

对于这个例子，让我们获取 GitHub 的公开时间表

```py
>>> r = requests.get('https://github.com/timeline.json') 
```

#现在，我们有一个名为 r 的响应对象，我们可以从这个对象中获得我们需要的所有信息。

### 要发出 HTTP POST 请求

```py
>>> r = requests.post("http://httpbin.org/post")

You can also use other HTTP request types, like PUT, DELETE, HEAD and OPTIONS

>>> r = requests.put("http://httpbin.org/put")

>>> r = requests.delete("http://httpbin.org/delete")

>>> r = requests.head("http://httpbin.org/get")

>>> r = requests.options("http://httpbin.org/get") 
```

### 响应内容

我们可以读取服务器响应的内容。

再次考虑 GitHub 时间轴:

```py
>>> import requests
>>> r = requests.get('https://github.com/timeline.json')
>>> r.text
'[{"repository":{"open_issues":0,"url":"https://github.com/... 
```

请求将自动解码来自服务器的内容。

大多数 unicode 字符集都可以无缝解码。

当您发出请求时，Requests 会根据 HTTP 头对响应的编码进行有根据的猜测。

当您访问 r.text 时，将使用请求猜测的文本编码。

您可以使用
r.encoding 属性找出正在使用的编码请求，并对其进行更改:

```py
>>> r.encoding
'utf-8'
>>> r.encoding = 'ISO-8859-1' 
```

如果您更改编码，每当您调用 r.text 时，请求将使用 r.encoding 的新值
。

### 二元响应内容

对于非文本请求，还可以以字节形式访问响应正文:

```py
>>> r.content
b'[{"repository":{"open_issues":0,"url":"https://github.com/... 
```

### JSON 响应内容

如果您正在处理 JSON 数据，还有一个内置的 JSON 解码器:

```py
>>> import requests
>>> r = requests.get('https://github.com/timeline.json')
>>> r.json
[{u'repository': {u'open_issues': 0, u'url': 'https://github.com/... 
```

如果 JSON 解码失败，r.json 只返回 None。

### 自定义标题

如果您想在请求中添加 HTTP 头，只需向 headers 参数传递一个 dict。

例如，在前面的示例中，我们没有指定内容类型:

```py
>>> import json
>>> url = 'https://api.github.com/some/endpoint'
>>> payload = {'some': 'data'}
>>> headers = {'content-type': 'application/json'}

>>> r = requests.post(url, data=json.dumps(payload), headers=headers) 
```

### 响应状态代码

我们可以检查响应状态代码:

```py
>>> r = requests.get('http://httpbin.org/get')
>>> r.status_code
200 
```

```py
# Requests also comes with a built-in status code lookup object for easy reference:
>>> r.status_code == requests.codes.ok
True 
```

```py
# If we made a bad request (non-200 response), 
# we can raise it with Response.raise_for_status():
>>> bad_r = requests.get('http://httpbin.org/status/404')
>>> bad_r.status_code
404 
```

### 响应标题

我们可以使用 Python 字典查看服务器的响应头:

```py
>>> r.headers
{
    'status': '200 OK',
    'content-encoding': 'gzip',
    'transfer-encoding': 'chunked',
    'connection': 'close',
    'server': 'nginx/1.0.4',
    'x-runtime': '148ms',
    'etag': '"e1ca502697e5c9317743dc078f67693f"',
    'content-type': 'application/json; charset=utf-8'
} 
```

HTTP 头是不区分大小写的，所以我们可以使用我们想要的任何
大写来访问头:

```py
>>> r.headers['Content-Type']
'application/json; charset=utf-8'

>>> r.headers.get('content-type')
'application/json; charset=utf-8'

# If a header doesn’t exist in the Response, its value defaults to None:
>>> r.headers['X-Random']
None 
```

### 饼干

如果响应包含一些 Cookies，您可以快速访问它们:

```py
>>> url = 'http://httpbin.org/cookies/set/requests-is/awesome'
>>> r = requests.get(url)

>>> r.cookies['requests-is']
'awesome' 
```

```py
# To send your own cookies to the server, you can use the cookies parameter:

>>> url = 'http://httpbin.org/cookies'
>>> cookies = dict(cookies_are='working')

>>> r = requests.get(url, cookies=cookies)

>>> r.text
'{"cookies": {"cookies_are": "working"}}' 
```

### 基本认证

许多 web 服务需要身份验证。

有许多不同类型的认证，但最常见的是 HTTP
基本认证。

使用基本身份验证发出请求非常简单:

```py
from requests.auth import HTTPBasicAuth
requests.get('https://api.github.com/user', auth=HTTPBasicAuth('user', 'pass'))

# Due to the prevalence of HTTP Basic Auth, 
# requests provides a shorthand for this authentication method:

requests.get('https://api.github.com/user', auth=('user', 'pass')) 
```

以这种方式提供元组形式的凭证在功能上等同于上面的
HTTPBasicAuth 示例。

### 摘要认证

```py
# Another popular form of web service protection is Digest Authentication:

>>> from requests.auth import HTTPDigestAuth

>>> url = 'http://httpbin.org/digest-auth/auth/user/pass'

>>> requests.get(url, auth=HTTPDigestAuth('user', 'pass')) 
```

### 重定向和历史记录

当使用 GET
和 OPTIONS 动词时，请求将自动执行位置重定向。

GitHub 将所有 HTTP 请求重定向到 HTTPS。

我们可以使用响应对象的历史方法来跟踪重定向。

让我们看看 Github 做了什么:

```py
>>> import requests
>>> r = requests.get("http://github.com")
>>> r.url
u'https://github.com/'
>>> r.status_code
200
>>> r.history
[]
>>> 
```

Response.history 列表包含为完成请求而创建的请求对象列表。

该列表从最早的请求到最近的请求进行排序。

如果使用 GET 或 OPTIONS，可以用
allow_redirects 参数禁用重定向处理:

```py
>>> r = requests.get('http://github.com', allow_redirects=False)
>>> r.status_code
301
>>> r.history
[] 
```

### 超时设定

您可以使用 timeout 参数告诉请求在给定的
秒后停止等待响应:

```py
>>> requests.get('http://github.com', timeout=0.001)
Traceback (most recent call last):
  File "", line 1, in 
requests.exceptions.Timeout: Request timed out. 
```

### 错误和异常

如果出现网络问题(例如 DNS 故障、拒绝连接等)，
请求将引发 ConnectionError 异常。

在罕见的无效 HTTP 响应事件中，
请求将引发一个 HTTPError 异常。

如果请求超时，将引发超时异常。

如果请求超过了配置的最大重定向数，就会引发 TooManyRedirects 异常。

请求显式引发的所有异常都继承自
Requests . exceptions . request exception。

您可以参考配置 API 文档，通过 danger_mode 选项立即引发
HTTPError 异常，或者让请求
捕获大多数 requests.exceptions

带有 safe_mode 选项的 RequestException 异常。
# 在 Python 中使用请求库

> 原文：<https://www.pythonforbeginners.com/requests/using-requests-in-python>

首先，让我们向您介绍请求。

### 什么是请求资源？

Requests 是一个 Apache2 许可的 HTTP 库，用 Python 编写。它被设计成供人类用来与语言互动。这意味着您不必手动向 URL 添加查询字符串，或者对您的帖子数据进行格式编码。如果这对你来说没有意义，不要担心。它会在适当的时候。

请求能做什么？

请求将允许您使用 Python 发送 HTTP/1.1 请求。有了它，您可以通过简单的 Python 库添加内容，如标题、表单数据、多部分文件和参数。它还允许您以同样的方式访问 Python 的响应数据。

在编程中，库是程序可以使用的例程、函数和操作的集合或预先配置的选择。这些元素通常被称为模块，并以对象格式存储。

库很重要，因为你加载一个模块并利用它提供的一切，而不用显式地链接到依赖它们的每个程序。它们是真正独立的，所以你可以用它们来构建你自己的程序，但它们仍然是独立于其他程序的。

把模块想象成一种代码模板。

重申一下，Requests 是一个 Python 库。

### 如何安装请求

好消息是有几种方法可以安装请求库。要查看可供您选择的选项的完整列表，您可以[在此](http://docs.python-requests.org/en/latest/user/install/)查看请求的官方安装文档。

您可以使用 pip、easy_install 或 tarball。

如果你更喜欢使用源代码，你也可以在 GitHub 上找到。

出于本指南的目的，我们将使用 pip 来安装库。

在 Python 解释器中，输入以下内容:

```py
*pip install requests* 
```

### 导入请求模块

要使用 Python 中的请求库，必须导入适当的模块。只需在脚本的开头添加以下代码，就可以做到这一点:

```py
*import requests* 
```

当然，要做到这些——包括安装库——您需要首先下载必要的包，并让解释器可以访问它。

### 提出请求

当您 ping 一个网站或门户网站以获取信息时，这被称为发出请求。这正是请求库的设计目的。

要获得一个网页，你需要做如下事情:

```py
*r =* *requests.get**(‘https://github.com/**timeline.json**’)*
```

### 使用响应代码

在使用 Python 对网站或 URL 进行任何操作之前，最好检查一下所述门户的当前状态代码。您可以使用字典查找对象来实现这一点。

```py
*r =* *requests.get**('https://github.com/**timeline.json**')*
*r.status**_code*
*>>200*

*r.status**_code* *==* *requests.codes.ok*
*>>> True*

*requests.codes**['**temporary_redirect**']*
*>>> 307*

*requests.codes**.teapot*
*>>> 418*

*requests.codes**['o/']*
*>>> 200*
```

### 获取内容

在 web 服务器返回响应后，您可以收集您需要的内容。这也是使用 get requests 函数完成的。

```py
*import requests*
*r =* *requests.get**('https://github.com/**timeline.json**')*
*print* *r.text*

*#* *The* *Requests* *library* *also comes with a built**-**in JSON decoder,*
*#* *just* *in case* *you have to deal* *with JSON data*

*import requests*
*r =* *requests.get**('https://github.com/**timeline.json**')*
*print* *r.json*
```

### 使用标题

通过使用 Python 字典，您可以访问和查看服务器的响应头。由于 Requests 的工作方式，您可以使用任何大小写来访问标题。

如果您执行了这个函数，但是响应中不存在标题，那么这个值将默认为 None。

```py
*r.headers*
*{*
*    'status': '200 OK',*
*    'content-encoding': '**gzip**',*
*    'transfer-encoding': 'chunked',*
*    'connection': 'close',*
*    'server': '**nginx**/1.0.4',*
*    'x-runtime': '148ms',*
*    '**etag**': '"e1ca502697e5c9317743dc078f67693f"',*
*    'content-type': 'application/**json**; charset=utf-8'*
*}*

*r.headers**['Content-Type']*
*>>>'application/**json**; charset=utf-8'*

*r.headers.get**('content-type')*
*>>>'application/**json**; charset=utf-8'*

*r.headers**['X-Random']*
*>>>None*

*# Get the headers of a given URL*
*resp* *=* *requests.head**("http://www.google.com")*
*print* *resp.status**_code**,* *resp.text**,* *resp.headers*
```

### 编码

请求将自动十年从服务器拉任何内容。但是大多数 Unicode 字符集无论如何都是无缝解码的。

当您向服务器发出请求时，请求库会根据 HTTP 头对响应的编码进行有根据的猜测。当您访问 r.text 文件时，将使用猜测的编码。

通过这个文件，您可以辨别请求库使用的是什么编码，并在需要时进行更改。这是可能的，因为您将在文件中找到的 *r.encoding* 属性。

如果您更改编码值，只要您在代码中调用 r.text，请求就会使用新的类型。

```py
*print* *r.encoding*
*>> utf-8*

*>>>* *r.encoding* *= ‘ISO-8859-1’*
```

### 自定义标题

如果要向请求添加自定义 HTTP 头，必须通过字典将它们传递给 headers 参数。

```py
*import* *json*
*url* *= 'https://api.github.com/some/endpoint'*
*payload = {'some': 'data'}*
*headers = {'content-type': 'application/**json**'}*

*r =* *requests.post**(**url**, data=**json.dumps**(payload), headers=headers)*
```

### 重定向和历史记录

当您在 Python 中使用 GET 和 OPTIONS 谓词时，请求将自动执行位置重定向。

GitHub 会自动将所有 HTTP 请求重定向到 HTTPS。这保证了东西的安全和加密。

您可以使用 response 对象的 history 方法来跟踪重定向状态。

```py
*r =* *requests.get**('http://github.com')*
*r.url*
*>>> 'https://github.com/'*

*r.status**_code*
*>>> 200*

*r.history* 
*>>> []*
```

### 发出 HTTP Post 请求

您还可以使用请求库来处理 post 请求。

```py
*r =* *requests.post**(http://httpbin.org/post)*
```

但是你也可以依赖其他 HTTP 请求，比如 ***PUT*** ， ***DELETE*** ， ***HEAD*** ，以及 ***OPTIONS*** 。

```py
*r =* *requests.put**("http://httpbin.org/put")*
*r =* *requests.delete**("http://httpbin.org/delete")*
*r =* *requests.head**("http://httpbin.org/get")*
*r =* *requests.options**("http://httpbin.org/get")*
```

你可以用这些方法来完成很多事情。例如，使用 Python 脚本创建 GitHub repo。

```py
*import requests,* *json*

*github_url* *= "https://api.github.com/user/repos"*
*data =* *json.dumps**({'**name':'test**', '**description':'some* *test repo'})*
*r =* *requests.post**(**github_url**, data,* *auth**=('user', '*****'))*

*print* *r.json*
```

### 错误和异常

在 Python 中使用请求库时，您需要熟悉许多异常和错误代码。

*   如果出现网络问题，如 DNS 故障或拒绝连接，请求库将引发 ConnectionError 异常。
*   对于无效的 HTTP 响应，请求也会引发 HTTPError 异常，但这种情况很少见。
*   如果请求超时，将引发超时异常。
*   如果请求超过预配置的最大重定向数，则会引发 TooManyRedirects 异常。

请求引发的任何异常都将从 Requests . exceptions . request exception 对象继承。

你可以通过下面的链接了解更多关于请求库的信息。

[http://docs.python-requests.org/en/latest/api/](http://docs.python-requests.org/en/latest/api/)

[http://pipi . python . org/pipi/request】t1](https://pypi.python.org/pypi/requests)

[http://docs.python-requests.org/en/latest/user/quickstart/](http://docs.python-requests.org/en/latest/user/quickstart/)

[http://isbullsh.it/2012/06/Rest-api-in-python/#requests](http://isbullsh.it/2012/06/Rest-api-in-python/#requests)
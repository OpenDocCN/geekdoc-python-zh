# 如何在 Python 中使用 urllib2

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/how-to-use-urllib2-in-python>

## 概观

虽然这篇文章的标题是“Urllib2”，但我们将展示一些使用 Urllib 的例子，因为它们经常一起使用。

这将是 urllib2 的一篇介绍文章，在这里我们将
关注获取 URL、请求、文章、用户代理和错误处理。

有关更多信息，请参见官方文档。

**此外，本文是为 Python 版本编写的**

HTTP 基于请求和响应——客户端发出请求，而
服务器发送响应。

互联网上的程序可以作为客户端(访问资源)或服务器(提供服务)。

URL 标识互联网上的资源。

## 什么是 Urllib2？

**urllib2** 是一个 Python 模块，可以用来获取 URL。

它定义了函数和类来帮助 URL 操作(基本和摘要
认证、重定向、cookies 等)

神奇之处始于导入 urllib2 模块。

## urllib 和 urllib2 有什么区别？

虽然两个模块都做与 URL 请求相关的事情，但是它们有不同的
功能

**urllib2** 可以接受一个请求对象来设置 URL 请求的头，
urllib 只接受一个 URL。

**urllib** 提供了 urlencode 方法，用于生成
GET 查询字符串， **urllib2** 没有这个功能。

正因为如此，urllib 和 **urllib2** 经常一起使用。

有关更多信息，请参见文档。

[Urllib](https://docs.python.org/2/library/urllib.html "urllib")
Urllib 2

## 什么是 urlopen？

urllib2 以 urlopen 函数的形式提供了一个非常简单的接口。

这个函数能够使用各种不同的协议
(HTTP、FTP、…)获取 URL

只需将 URL 传递给 **urlopen()** 就可以获得一个针对远程数据的**“类文件”**句柄。

另外， **urllib2** 提供了一个接口来处理常见的情况—
,比如基本认证、cookies、代理等等。

这些是由称为处理程序和打开程序的对象提供的。

## 获取 URL

这是图书馆最基本的使用方法。

下面你可以看到如何用 urllib2 做一个简单的请求。

首先导入 urllib2 模块。

将响应放入变量(response)

响应现在是一个类似文件的对象。

将响应中的数据读入一个字符串(html)

用那根绳子做点什么。

**注意**如果 URL 中有空格，您将需要使用 urlencode 解析它。

让我们来看一个例子。

```py
import urllib2
response = urllib2.urlopen('https://www.pythonforbeginners.com/')
print response.info()
html = response.read()
# do something
response.close()  # best practice to close the file

Note: you can also use an URL starting with "ftp:", "file:", etc.). 
```

远程服务器接受传入的值，并格式化一个纯文本响应
发回。

来自 **urlopen()** 的返回值通过 **info()** 方法访问来自 HTTP 服务器
的头，并通过
**read()** 和 **readlines()** 等方法访问远程资源的数据。

此外，由 **urlopen()** 返回的文件对象是可迭代的。

## 简单的 urllib2 脚本

让我们展示另一个简单的 urllib2 脚本示例

```py
import urllib2
response = urllib2.urlopen('http://python.org/')
print "Response:", response

# Get the URL. This gets the real URL. 
print "The URL is: ", response.geturl()

# Getting the code
print "This gets the code: ", response.code

# Get the Headers. 
# This returns a dictionary-like object that describes the page fetched, 
# particularly the headers sent by the server
print "The Headers are: ", response.info()

# Get the date part of the header
print "The Date is: ", response.info()['date']

# Get the server part of the header
print "The Server is: ", response.info()['server']

# Get all data
html = response.read()
print "Get all data: ", html

# Get only the length
print "Get the length :", len(html)

# Showing that the file object is iterable
for line in response:
 print line.rstrip()

# Note that the rstrip strips the trailing newlines and carriage returns before
# printing the output. 
```

## 使用 Urllib2 下载文件

这个小脚本将从 pythonforbeginners.com 网站下载一个文件

```py
import urllib2

# file to be written to
file = "downloaded_file.html"

url = "https://www.pythonforbeginners.com/"
response = urllib2.urlopen(url)

#open the [file for writing](https://www.pythonforbeginners.com/files/reading-and-writing-files-in-python)
fh = open(file, "w")

# read from request while writing to file
fh.write(response.read())
fh.close()

# You can also use the with statement:
with open(file, 'w') as f: f.write(response.read()) 
```

这个脚本的不同之处在于我们使用了‘WB’，这意味着我们打开了
文件二进制文件。

```py
import urllib2

mp3file = urllib2.urlopen("http://www.example.com/songs/mp3.mp3")

output = open('test.mp3','wb')

output.write(mp3file.read())

output.close() 
```

## Urllib2 请求

Request 对象代表您正在发出的 HTTP 请求。

最简单的形式是创建一个请求对象，它指定了您希望
获取的 URL。

用这个请求对象调用 urlopen 将返回所请求的 URL
的响应对象。

urllib2 类下的请求函数接受 url 和参数。

当您不包含数据(只传递 url)时，发出的请求
实际上是一个 GET 请求

当您包含数据时，发出的请求是 POST 请求，其中
url 将是您的 post url，参数将是 http post content。

让我们看看下面的例子

```py
import urllib2
import urllib

# Specify the url
url = 'https://www.pythonforbeginners.com'

# This packages the request (it doesn't make it) 
request = urllib2.Request(url)

# Sends the request and catches the response
response = urllib2.urlopen(request)

# Extracts the response
html = response.read()

# Print it out
print html 
```

您可以在请求上设置传出数据，以将其发送到服务器。

此外，您可以将关于数据的数据额外信息(“元数据”)或
about 请求本身传递给服务器——这些信息作为 HTTP
“headers”发送。

如果您想要发布数据，您必须首先将数据创建到字典中。

确保您理解代码的作用。

```py
# Prepare the data
query_args = { 'q':'query string', 'foo':'bar' }

# This urlencodes your data (that's why we need to import urllib at the top)
data = urllib.urlencode(query_args)

# Send HTTP POST request
request = urllib2.Request(url, data)

response = urllib2.urlopen(request)

html = response.read()

# Print the result
print html 
```

## 用户代理

浏览器通过用户代理头标识自己。

默认情况下，urllib2 将自己标识为 **Python-urllib/x.y**
，其中 x 和 y 是 Python 版本的主版本号和次版本号。

这可能会混淆网站，或者干脆不工作。

使用 urllib2，您可以使用 urllib2 添加自己的标题。

你想这么做的原因是一些网站不喜欢被程序浏览。

如果你正在创建一个访问其他人的网络资源的应用程序，
在你的请求中包含真实的用户代理信息是礼貌的，
这样他们可以更容易地识别点击的来源。

当您创建请求对象时，您可以将您的头添加到字典
中，并在打开请求之前使用 add_header()设置用户代理值。

看起来大概是这样的:

```py
# Importing the module
import urllib2

# Define the url
url = 'http://www.google.com/#q=my_search'

# Add your headers
headers = {'User-Agent' : 'Mozilla 5.10'}

# Create the Request. 
request = urllib2.Request(url, None, headers)

# Getting the response
response = urllib2.urlopen(request)

# Print the headers
print response.headers 
```

也可以用“add_header()”添加头

语法:Request.add_header(key，val)

[urllib2。Request.add_header](https://docs.python.org/2/library/urllib2.html#urllib2.Request.add_header "add-header")

下面的例子，使用 Mozilla 5.10 作为用户代理，这也是
将在 web 服务器日志文件中显示的内容。

```py
import urllib2

req = urllib2.Request('http://192.168.1.2/')

req.add_header('User-agent', 'Mozilla 5.10')

res = urllib2.urlopen(req)

html = res.read()

print html 
```

这是将在日志文件中显示的内容。
“GET/HTTP/1.1？200 151”——“Mozilla 5.10？

## urllib.urlparse

urlparse 模块提供了分析 URL 字符串的函数。

它定义了一个标准接口，将统一资源定位符(URL)
字符串分成几个可选部分，称为组件，称为
(方案、位置、路径、查询和片段)

假设你有一个网址:【http://www.python.org:80/index.html 

**方案**应该是 http

地点将会是 www.python.org:80

**路径**是 index.html

我们没有任何**查询**和**片段**

最常见的函数是 urljoin 和 urlsplit

```py
import urlparse

url = "http://python.org"

domain = urlparse.urlsplit(url)[1].split(':')[0]

print "The domain name of the url is: ", domain 
```

有关 urlparse 的更多信息，请参见官方[文档](https://docs.python.org/2/library/urlparse.html "urlparse")。

## urllib，urlencode

当您通过 URL 传递信息时，您需要确保它只使用特定的允许字符。

允许的字符是任何字母字符、数字和一些在 URL 字符串中有意义的特殊
字符。

最常见的编码字符是**空格**字符。

每当您在 URL 中看到加号(+)时，就会看到此字符。

这代表空格字符。

加号充当代表 URL 中空格的特殊字符

可以通过对参数进行编码并将它们附加到 URL 上来将参数传递给服务器。

让我们看看下面的例子。

```py
import urllib
import urllib2

query_args = { 'q':'query string', 'foo':'bar' } # you have to pass in a dictionary  

encoded_args = urllib.urlencode(query_args)

print 'Encoded:', encoded_args

url = 'http://python.org/?' + encoded_args

print urllib2.urlopen(url).read() 
```

如果我现在打印它，我会得到这样一个编码字符串:
**q = query+string&foo = bar**

Python 的 urlencode 接受变量/值对，并创建一个正确转义的
querystring:

```py
from urllib import urlencode

artist = "Kruder & Dorfmeister"

artist = urlencode({'ArtistSearch':artist}) 
```

这将变量 artist 设置为等于:

输出:ArtistSearch = Kruder+% 26+dorf meister

## 错误处理

这一段错误处理是基于 Voidspace.org.uk 大文章中的信息:
[urllib 2——缺失的手册](http://www.voidspace.org.uk/python/articles/urllib2.shtml#handling-exceptions "voidspace.org.uk")

当 urlopen 无法处理响应时，它会引发 **URLError** 。

**HTTPError** 是在 HTTP URLs 的具体情况下引发的 **URLError** 的子类。

##### URLError

通常，引发 URLError 的原因是没有网络连接，
或者指定的服务器不存在。

在这种情况下，引发的异常将有一个“原因”属性，
，它是一个包含错误代码和文本错误消息的元组。

URLError 示例

```py
req = urllib2.Request('http://www.pretend_server.org')

try: 
    urllib2.urlopen(req)

except URLError, e:
    print e.reason

(4, 'getaddrinfo failed') 
```

##### HTTPError

来自服务器的每个 HTTP 响应都包含一个数字“状态代码”。

有时，状态代码表示服务器无法满足请求。

默认处理程序将为您处理其中的一些响应(例如，
如果响应是一个“重定向”,请求客户端从不同的 URL 获取文档
, URL lib 2 将为您处理它)。

对于那些它不能处理的，urlopen 会抛出一个 HTTPError。

典型的错误包括“404”(找不到页面)、“403”(禁止请求)、
和“401”(需要验证)。

当出现错误时，服务器通过返回 HTTP 错误代码
和错误页面进行响应。

您可以在返回的页面上使用 HTTPError 实例作为响应。

这意味着除了 code 属性之外，它还有 read、geturl、
和 info 等方法。

```py
req = urllib2.Request('http://www.python.org/fish.html')

try:
    urllib2.urlopen(req)

except URLError, e:
    print e.code
    print e.read() 
```

```py
from urllib2 import Request, urlopen, URLError

req = Request(someurl)

try:
    response = urlopen(req)

except URLError, e:

    if hasattr(e, 'reason'):
        print 'We failed to reach a server.'
        print 'Reason: ', e.reason

    elif hasattr(e, 'code'):
        print 'The server could not fulfill the request.'
        print 'Error code: ', e.code
else:
    # everything is fine 
```

请看看下面的链接，以获得对 Urllib2
库的更多了解。

##### 资料来源和进一步阅读

[http://pymotw . com/2/urllib 2/](http://pymotw.com/2/urllib2/ "pymotw-urllib2")
[【http://www . kensjauson . com/](http://www.kentsjohnson.com/ "kentsjohnson")[【http://www . voidspace . org . uk/python/articles/urllib 2 . shtml】](http://www.voidspace.org.uk/python/articles/urllib2.shtml "voidspace")
# 第一章：引言

> 本书出处：[`demo.pythoner.com/itt2zh/`](http://demo.pythoner.com/itt2zh/)
> 中文翻译：[你像从前一样](http://www.pythoner.com/)

在过去的五年里，Web 开发人员的可用工具实现了跨越式地增长。当技术专家不断推动极限，使 Web 应用无处不在时，我们也不得不升级我们的工具、创建框架以保证构建更好的应用。我们希望能够使用新的工具，方便我们写出更加整洁、可维护的代码，使部署到世界各地的用户时拥有高效的可扩展性。

这就让我们谈论到 Tornado，一个编写易创建、扩展和部署的强力 Web 应用的梦幻选择。我们三个都因为 Tornado 的速度、简单和可扩展性而深深地爱上了它，在一些个人项目中尝试之后，我们将其运用到日常工作中。我们已经看到，Tornado 在很多大型或小型的项目中提升了开发者的速度（和乐趣！），同时，其鲁棒性和轻量级也给开发者一次又一次留下了深刻的印象。

本书的目的是对 Tornado Web 服务器进行一个概述，通过框架基础、一些示例应用和真实世界使用的最佳实践来引导读者。我们将使用示例来详细讲解 Tornado 如何工作，你可以用它做什么，以及在构建自己第一个应用时要避免什么。

在本书中，我们假定你对 Python 已经有了粗略的了解，知道 Web 服务如何运作，对数据库有一定的熟悉。有一些不错的书籍可以为你深入了解这些提供参考（比如 Learning Python，Restful Web Service 和 MongoDB: The Definitive Guide）。

你可以在[Github](https://github.com/Introduction-to-Tornado)上获得本书中示例的代码。如果你有关于这些示例或其他方面的任何思想，欢迎在那里告诉我们。

所以，事不宜迟，让我们开始深入了解吧！

*   1.1 Tornado 是什么？
    *   1.1.1 Tornado 入门
    *   1.1.2 社区和支持
    *   1.2 简单的 Web 服务
        *   1.2.1 Hello Tornado
        *   1.2.2 字符串服务
        *   1.2.3 关于 RequestHandler 的更多知识
        *   1.2.4 下一步

## 1.1 Tornado 是什么？

Tornado 是使用 Python 编写的一个强大的、可扩展的 Web 服务器。它在处理严峻的网络流量时表现得足够强健，但却在创建和编写时有着足够的轻量级，并能够被用在大量的应用和工具中。

我们现在所知道的 Tornado 是基于 Bret Taylor 和其他人员为 FriendFeed 所开发的网络服务框架，当 FriendFeed 被 Facebook 收购后得以开源。不同于那些最多只能达到 10,000 个并发连接的传统网络服务器，Tornado 在设计之初就考虑到了性能因素，旨在解决 C10K 问题，这样的设计使得其成为一个拥有非常高性能的框架。此外，它还拥有处理安全性、用户验证、社交网络以及与外部服务（如数据库和网站 API）进行异步交互的工具。

> 延伸阅读：C10K 问题
> 
> 基于线程的服务器，如 Apache，为了传入的连接，维护了一个操作系统的线程池。Apache 会为每个 HTTP 连接分配线程池中的一个线程，如果所有的线程都处于被占用的状态并且尚有内存可用时，则生成一个新的线程。尽管不同的操作系统会有不同的设置，大多数 Linux 发布版中都是默认线程堆大小为 8MB。Apache 的架构在大负载下变得不可预测，为每个打开的连接维护一个大的线程池等待数据极易迅速耗光服务器的内存资源。

大多数社交网络应用都会展示实时更新来提醒新消息、状态变化以及用户通知，这就要求客户端需要保持一个打开的连接来等待服务器端的任何响应。这些长连接或推送请求使得 Apache 的最大线程池迅速饱和。一旦线程池的资源耗尽，服务器将不能再响应新的请求。

异步服务器在这一场景中的应用相对较新，但他们正是被设计用来减轻基于线程的服务器的限制的。当负载增加时，诸如 Node.js，lighttpd 和 Tornodo 这样的服务器使用协作的多任务的方式进行优雅的扩展。也就是说，如果当前请求正在等待来自其他资源的数据（比如数据库查询或 HTTP 请求）时，一个异步服务器可以明确地控制以挂起请求。异步服务器用来恢复暂停的操作的一个常见模式是当合适的数据准备好时调用回调函数。我们将会在第五章讲解回调函数模式以及一系列 Tornado 异步功能的应用。

自从 2009 年 9 月 10 日发布以来，TornadoTornado 已经获得了很多社区的支持，并且在一系列不同的场合得到应用。除 FriendFeed 和 Facebook 外，还有很多公司在生产上转向 Tornado，包括 Quora、Turntable.fm、Bit.ly、Hipmunk 以及 MyYearbook 等。

总之，如果你在寻找你那庞大的 CMS 或一体化开发框架的替代品，Tornado 可能并不是一个好的选择。Tornado 并不需要你拥有庞大的模型建立特殊的方式，或以某种确定的形式处理表单，或其他类似的事情。它所做的是让你能够快速简单地编写高速的 Web 应用。如果你想编写一个可扩展的社交应用、实时分析引擎，或 RESTful API，那么简单而强大的 Python，以及 Tornado（和这本书）正是为你准备的！

### 1.1.1 Tornado 入门

在大部分*nix 系统中安装 Tornado 非常容易--你既可以从 PyPI 获取（并使用`easy_install`或`pip`安装），也可以从 Github 上下载源码编译安装，如下所示[1]：

```py
$ curl -L -O https://github.com/facebook/tornado/archive/v3.1.0.tar.gz
$ tar xvzf v3.1.0.tar.gz
$ cd tornado-3.1.0
$ python setup.py build
$ sudo python setup.py install

```

Tornado 官方并不支持 Windows，但你可以通过 ActivePython 的 PyPM 包管理器进行安装，类似如下所示：

```py
C:\> pypm install tornado

```

一旦 Tornado 在你的机器上安装好，你就可以很好的开始了！压缩包中包含很多 demo，比如建立博客、整合 Facebook、运行聊天服务等的示例代码。我们稍后会在本书中通过一些示例应用逐步讲解，不过你也应该看看这些官方 demo。

本书中的代码假定你使用的是基于 Unix 的系统，并且使用的是 Python2.6 或 2.7 版本。如果是这样，你就不需要任何除了 Python 标准库之外的东西。如果你的 Python 版本是 2.5 或更低，在安装 pycURL、simpleJSON 和 Python 开发头文件后可以运行 Tornado。[2]

### 1.1.2 社区和支持

对于问题、示例和一般的指南，Tornado 官方文档是个不错的选择。在[tornadoweb.org](http://tornadoweb.org/)上有大量的例子和功能缺陷，更多细节和变更可以在[Tornado 在 Github 上的版本库](http://github.com/facebook/tornado)中看到。而对于更具体的问题，可以到[Tornado 的 Google Group](http://groups.google.com/group/python-tornado)中咨询，那里有很多活跃的日常使用 Tornado 的开发者。

## 1.2 简单的 Web 服务

既然我们已经知道了 Tornado 是什么了，现在让我们看看它能做什么吧。我们首先从使用 Tornado 编写一个简单的 Web 应用开始。

### 1.2.1 Hello Tornado

Tornado 是一个编写对 HTTP 请求响应的框架。作为程序员，你的工作是编写响应特定条件 HTTP 请求的响应的 handler。下面是一个全功能的 Tornado 应用的基础示例：

代码清单 1-1 基础：hello.py

```py
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        self.write(greeting + ', friendly user!')

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

```

编写一个 Tornado 应用中最多的工作是定义类继承 Tornado 的 RequestHandler 类。在这个例子中，我们创建了一个简单的应用，在给定的端口监听请求，并在根目录（"/"）响应请求。

你可以在命令行里尝试运行这个程序以测试输出：

```py
$ python hello.py --port=8000

```

现在你可以在浏览器中打开[`localhost:8000`](http://localhost:8000)，或者打开另一个终端窗口使用 curl 测试我们的应用：

```py
$ curl http://localhost:8000/
Hello, friendly user!
$ curl http://localhost:8000/?greeting=Salutations
Salutations, friendly user!

```

让我们把这个例子分成小块，逐步分析它们：

```py
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

```

在程序的最顶部，我们导入了一些 Tornado 模块。虽然 Tornado 还有另外一些有用的模块，但在这个例子中我们必须至少包含这四个模块。

```py
from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

```

Tornado 包括了一个有用的模块（tornado.options）来从命令行中读取设置。我们在这里使用这个模块指定我们的应用监听 HTTP 请求的端口。它的工作流程如下：如果一个与 define 语句中同名的设置在命令行中被给出，那么它将成为全局 options 的一个属性。如果用户运行程序时使用了`--help`选项，程序将打印出所有你定义的选项以及你在 define 函数的 help 参数中指定的文本。如果用户没有为这个选项指定值，则使用 default 的值进行代替。Tornado 使用 type 参数进行基本的参数类型验证，当不合适的类型被给出时抛出一个异常。因此，我们允许一个整数的 port 参数作为 options.port 来访问程序。如果用户没有指定值，则默认为 8000。

```py
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        self.write(greeting + ', friendly user!')

```

这是 Tornado 的请求处理函数类。当处理一个请求时，Tornado 将这个类实例化，并调用与 HTTP 请求方法所对应的方法。在这个例子中，我们只定义了一个 get 方法，也就是说这个处理函数将对 HTTP 的 GET 请求作出响应。我们稍后将看到实现不止一个 HTTP 方法的处理函数。

```py
greeting = self.get_argument('greeting', 'Hello')

```

Tornado 的 RequestHandler 类有一系列有用的内建方法，包括 get_argument，我们在这里从一个查询字符串中取得参数 greeting 的值。（如果这个参数没有出现在查询字符串中，Tornado 将使用 get_argument 的第二个参数作为默认值。）

```py
self.write(greeting + ', friendly user!')

```

RequestHandler 的另一个有用的方法是 write，它以一个字符串作为函数的参数，并将其写入到 HTTP 响应中。在这里，我们使用请求中 greeting 参数提供的值插入到 greeting 中，并写回到响应中。

```py
if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])

```

这是真正使得 Tornado 运转起来的语句。首先，我们使用 Tornado 的 options 模块来解析命令行。然后我们创建了一个 Tornado 的 Application 类的实例。传递给 Application 类**init**方法的最重要的参数是 handlers。它告诉 Tornado 应该用哪个类来响应请求。马上我们讲解更多相关知识。

```py
http_server = tornado.httpserver.HTTPServer(app)
http_server.listen(options.port)
tornado.ioloop.IOLoop.instance().start()

```

从这里开始的代码将会被反复使用：一旦 Application 对象被创建，我们可以将其传递给 Tornado 的 HTTPServer 对象，然后使用我们在命令行指定的端口进行监听（通过 options 对象取出。）最后，在程序准备好接收 HTTP 请求后，我们创建一个 Tornado 的 IOLoop 的实例。

#### 1.2.1.1 参数 handlers

让我们再看一眼 hello.py 示例中的这一行：

```py
app = tornado.web.Application(handlers=[(r"/", IndexHandler)])

```

这里的参数 handlers 非常重要，值得我们更加深入的研究。它应该是一个元组组成的列表，其中每个元组的第一个元素是一个用于匹配的正则表达式，第二个元素是一个 RequestHanlder 类。在 hello.py 中，我们只指定了一个正则表达式-RequestHanlder 对，但你可以按你的需要指定任意多个。

#### 1.2.1.2 使用正则表达式指定路径

Tornado 在元组中使用正则表达式来匹配 HTTP 请求的路径。（这个路径是 URL 中主机名后面的部分，不包括查询字符串和碎片。）Tornado 把这些正则表达式看作已经包含了行开始和结束锚点（即，字符串"/"被看作为"^/$"）。

如果一个正则表达式包含一个捕获分组（即，正则表达式中的部分被括号括起来），匹配的内容将作为相应 HTTP 请求的参数传到 RequestHandler 对象中。我们将在下个例子中看到它的用法。

### 1.2.2 字符串服务

例 1-2 是一个我们目前为止看到的更复杂的例子，它将介绍更多 Tornado 的基本概念。

代码清单 1-2 处理输入：string_service.py

```py
import textwrap

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class ReverseHandler(tornado.web.RequestHandler):
    def get(self, input):
        self.write(input[::-1])

class WrapHandler(tornado.web.RequestHandler):
    def post(self):
        text = self.get_argument('text')
        width = self.get_argument('width', 40)
        self.write(textwrap.fill(text, int(width)))

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[
            (r"/reverse/(\w+)", ReverseHandler),
            (r"/wrap", WrapHandler)
        ]
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

```

如同运行第一个例子，你可以在命令行中运行这个例子使用如下的命令：

```py
$ python string_service.py --port=8000

```

这个程序是一个通用的字符串操作的 Web 服务端基本框架。到目前为止，你可以用它做两件事情。其一，到`/reverse/string`的 GET 请求将会返回 URL 路径中指定字符串的反转形式。

```py
$ curl http://localhost:8000/reverse/stressed
desserts

$ curl http://localhost:8000/reverse/slipup
pupils

```

其二，到`/wrap`的 POST 请求将从参数 text 中取得指定的文本，并返回按照参数 width 指定宽度装饰的文本。下面的请求指定一个没有宽度的字符串，所以它的输出宽度被指定为程序中的 get_argument 的默认值 40 个字符。

```py
$ http://localhost:8000/wrap -d text=Lorem+ipsum+dolor+sit+amet,+consectetuer+adipiscing+elit.
Lorem ipsum dolor sit amet, consectetuer
adipiscing elit.

```

字符串服务示例和上一节示例代码中大部分是一样的。让我们关注那些新的代码。首先，让我们看看传递给 Application 构造函数的 handlers 参数的值：

```py
app = tornado.web.Application(handlers=[
    (r"/reverse/(\w+)", ReverseHandler),
    (r"/wrap", WrapHandler)
])

```

在上面的代码中，Application 类在"handlers"参数中实例化了两个 RequestHandler 类对象。第一个引导 Tornado 传递路径匹配下面的正则表达式的请求：

```py
/reverse/(\w+)

```

正则表达式告诉 Tornado 匹配任何以字符串/reverse/开始并紧跟着一个或多个字母的路径。括号的含义是让 Tornado 保存匹配括号里面表达式的字符串，并将其作为请求方法的一个参数传递给 RequestHandler 类。让我们检查 ReverseHandler 的定义来看看它是如何工作的：

```py
class ReverseHandler(tornado.web.RequestHandler):
    def get(self, input):
        self.write(input[::-1])

```

你可以看到这里的 get 方法有一个额外的参数 input。这个参数将包含匹配处理函数正则表达式第一个括号里的字符串。（如果正则表达式中有一系列额外的括号，匹配的字符串将被按照在正则表达式中出现的顺序作为额外的参数传递进来。）

现在，让我们看一下 WrapHandler 的定义：

```py
class WrapHandler(tornado.web.RequestHandler):
    def post(self):
        text = self.get_argument('text')
        width = self.get_argument('width', 40)
        self.write(textwrap.fill(text, int(width)))

```

WrapHandler 类处理匹配路径为`/wrap`的请求。这个处理函数定义了一个 post 方法，也就是说它接收 HTTP 的 POST 方法的请求。

我们之前使用 RequestHandler 对象的 get_argument 方法来捕获请求查询字符串的的参数。同样，我们也可以使用相同的方法来获得 POST 请求传递的参数。（Tornado 可以解析 URLencoded 和 multipart 结构的 POST 请求）。一旦我们从 POST 中获得了文本和宽度的参数，我们使用 Python 内建的 textwrap 模块来以指定的宽度装饰文本，并将结果字符串写回到 HTTP 响应中。

### 1.2.3 关于 RequestHandler 的更多知识

到目前为止，我们已经了解了 RequestHandler 对象的基础：如何从一个传入的 HTTP 请求中获得信息（使用 get_argument 和传入到 get 和 post 的参数）以及写 HTTP 响应（使用 write 方法）。除此之外，还有很多需要学习的，我们将在接下来的章节中进行讲解。同时，还有一些关于 RequestHandler 和 Tornado 如何使用它的只是需要记住。

#### 1.2.3.1 HTTP 方法

截止到目前讨论的例子，每个 RequestHandler 类都只定义了一个 HTTP 方法的行为。但是，在同一个处理函数中定义多个方法是可能的，并且是有用的。把概念相关的功能绑定到同一个类是一个很好的方法。比如，你可能会编写一个处理函数来处理数据库中某个特定 ID 的对象，既使用 GET 方法，也使用 POST 方法。想象 GET 方法来返回这个部件的信息，而 POST 方法在数据库中对这个 ID 的部件进行改变：

```py
# matched with (r"/widget/(\d+)", WidgetHandler)
class WidgetHandler(tornado.web.RequestHandler):
    def get(self, widget_id):
        widget = retrieve_from_db(widget_id)
        self.write(widget.serialize())

    def post(self, widget_id):
        widget = retrieve_from_db(widget_id)
        widget['foo'] = self.get_argument('foo')
        save_to_db(widget)

```

我们到目前为止只是用了 GET 和 POST 方法，但 Tornado 支持任何合法的 HTTP 请求（GET、POST、PUT、DELETE、HEAD、OPTIONS）。你可以非常容易地定义上述任一种方法的行为，只需要在 RequestHandler 类中使用同名的方法。下面是另一个想象的例子，在这个例子中针对特定 frob ID 的 HEAD 请求只根据 frob 是否存在给出信息，而 GET 方法返回整个对象：

```py
# matched with (r"/frob/(\d+)", FrobHandler)
class FrobHandler(tornado.web.RequestHandler):
    def head(self, frob_id):
        frob = retrieve_from_db(frob_id)
        if frob is not None:
            self.set_status(200)
        else:
            self.set_status(404)
    def get(self, frob_id):
        frob = retrieve_from_db(frob_id)
        self.write(frob.serialize())

```

#### 1.2.3.2 HTTP 状态码

从上面的代码可以看出，你可以使用 RequestHandler 类的 ser_status()方法显式地设置 HTTP 状态码。然而，你需要记住在某些情况下，Tornado 会自动地设置 HTTP 状态码。下面是一个常用情况的纲要：

##### 404 Not Found

Tornado 会在 HTTP 请求的路径无法匹配任何 RequestHandler 类相对应的模式时返回 404（Not Found）响应码。

##### 400 Bad Request

如果你调用了一个没有默认值的 get_argument 函数，并且没有发现给定名称的参数，Tornado 将自动返回一个 400（Bad Request）响应码。

##### 405 Method Not Allowed

如果传入的请求使用了 RequestHandler 中没有定义的 HTTP 方法（比如，一个 POST 请求，但是处理函数中只有定义了 get 方法），Tornado 将返回一个 405（Methos Not Allowed）响应码。

##### 500 Internal Server Error

当程序遇到任何不能让其退出的错误时，Tornado 将返回 500（Internal Server Error）响应码。你代码中任何没有捕获的异常也会导致 500 响应码。

##### 200 OK

如果响应成功，并且没有其他返回码被设置，Tornado 将默认返回一个 200（OK）响应码。

当上述任何一种错误发生时，Tornado 将默认向客户端发送一个包含状态码和错误信息的简短片段。如果你想使用自己的方法代替默认的错误响应，你可以重写 write_error 方法在你的 RequestHandler 类中。比如，代码清单 1-3 是 hello.py 示例添加了常规的错误消息的版本。

代码清单 1-3 常规错误响应：hello-errors.py

```py
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        self.write(greeting + ', friendly user!')
    def write_error(self, status_code, **kwargs):
        self.write("Gosh darnit, user! You caused a %d error." % status_code)

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

```

当我们尝试一个 POST 请求时，会得到下面的响应。一般来说，我们应该得到 Tornado 默认的错误响应，但因为我们覆写了 write_error，我们会得到不一样的东西：

```py
$ curl -d foo=bar http://localhost:8000/
Gosh darnit, user! You caused a 405 error.

```

### 1.2.4 下一步

现在你已经明白了最基本的东西，我们渴望你想了解更多。在接下来的章节，我们将向你展示能够帮助你使用 Tornado 创建成熟的 Web 服务和应用的功能和技术。首先是：Tornado 的模板系统。

[1] 压缩包地址已更新到 Tornado 的最新版本 3.1.0。
[2] 书中原文中关于 Python3.X 版本的兼容性问题目前已不存在，因此省略该部分。
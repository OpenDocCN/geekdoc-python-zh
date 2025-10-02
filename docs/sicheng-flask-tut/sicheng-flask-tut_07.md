# Flask 进阶系列(二)–信号

信号(Signal)就是两个独立的模块用来传递消息的方式，它有一个消息的发送者 Sender，还有一个消息的订阅者 Subscriber。信号的存在使得模块之间可以摆脱互相调用的模式，也就是解耦合。发送者无需知道谁会接收消息，接收者也可自由选择订阅何种消息。这一篇，让我们来了解下 Flask 中的信号。

### 系列文章

*   Flask 进阶系列(一)–上下文环境
*   Flask 进阶系列(二)–信号
*   Flask 进阶系列(三)–Jinja2 模板引擎
*   Flask 进阶系列(四)–视图
*   Flask 进阶系列(五)–文件和流
*   Flask 进阶系列(六)–蓝图(Blueprint)
*   Flask 进阶系列(七)–应用最佳实践
*   Flask 进阶系列(八)–部署和分发
*   Flask 进阶系列(九)–测试

Flask 的信号功能是基于 Python 消息分发组件[Blinder](https://pypi.python.org/pypi/blinker)之上的，在开始此篇之前，你先要安装 blinder 包”pip install blinder”。

### 订阅一个 Flask 信号

还是老习惯，从例子开始，让我们找回入门系列第三篇中的代码和模板，并加入下面的代码：

```py
from flask import template_rendered, request

def print_template_rendered(sender, template, context, **extra):
    print 'Using template: %s with context: %s' % (template.name, context)
    print request.url

template_rendered.connect(print_template_rendered, app)

```

访问”http://localhost:5000/hello”时，你会在控制台看到：

```py
Using template: hello.html with context: {'session': , 'request': , 'name': None, 'g': }
http://localhost:5000/hello
```

而访问”http://localhost:5000/”时，却没有这些信息。简单解释下，”flask.template_rendered”是一个信号，更确切的说是 Flask 的核心信号。当任意一个模板被渲染成功后，这个信号就会被发出。信号的”connect()”方法用来连接订阅者，它的第一个参数就是订阅者回调函数，当信号发出时，这个回调函数就会被调用；第二个参数是指定消息的发送者，也就是指明只有”app”作为发送者发出的”template_rendered”消息才会被此订阅者接收。你可以不指定发送者，这样，任何发送者发出的”template_rendered”都会被接收。一般使用中我们建议指定发送者，以避免接收所有消息。”connect()”方法可以多次调用，来连接多个订阅者。

再来看看这个回调函数，它有四个参数，前三个参数是必须的。

1.  sender: 获取消息的发送者
2.  template: 被渲染的模板对象
3.  context: 当前请求的上下文环境
4.  **extra: 匹配任意额外的参数。如果上面三个存在，这个参数不加也没问题。但是如果你的参数少于三个，就一定要有这个参数。一般习惯上加上此参数。

我们在回调函数中，可以获取请求上下文，也就是它在一个请求的生命周期和线程内。所以，我们可以在函数中访问 request 对象。如果忘了请求上下文是个什么鬼，可以参阅上一篇。

Flask 同时提供了信号装饰器来简化代码，上面的信号订阅也可以写成：

```py
from flask import template_rendered, request

@template_rendered.connect_via(app)
def with_template_rendered(sender, template, context, **extra):
    print 'Using template: %s with context: %s' % (template.name, context)
    print request.url

```

是不是简洁不少？注，”connect_via()”方法中的参数指定了发送者，不加的话就指所有发送者。

### Flask 核心信号(Core Signals)

上例中的”flask.template_rendered”就是一个 Flask 核心信号，定义在 flask 包下，由 Flask 核心代码提供，消息发送者都是 Flask App 对象。除了”template_rendered”外，这里列举一些常见的核心信号。

*   **request_started**

请求开始时发送。回调函数参数:

1.  sender: 消息的发送者

*   **request_finished**

请求结束后发送。回调函数参数:

1.  sender: 消息的发送者
2.  response: 待返回的响应对象

*   **got_request_exception**

请求发生异常时发送。回调函数参数:

1.  sender: 消息的发送者
2.  exception: 被抛出的异常对象

*   **request_tearing_down**

请求被销毁时发送，不管有无异常都会被发送。回调函数参数:

1.  sender: 消息的发送者
2.  exc: 有异常时，抛出的异常对象

*   **appcontext_tearing_down**

应用上下文被销毁时发送。回调函数参数:

1.  sender: 消息的发送者

*   **appcontext_pushed**

应用上下文被压入”_app_ctx_stack”栈后发送。回调函数参数:

1.  sender: 消息的发送者

*   **appcontext_popped**

应用上下文从”_app_ctx_stack”栈中弹出后发送。回调函数参数:

1.  sender: 消息的发送者

*   **message_flashed**

消息闪现时发送。回调函数参数:

1.  sender: 消息的发送者
2.  message: 被闪现的消息内容
3.  category: 被闪现的消息类别

注，所有回调函数都建议加上”**extra”作为最后的参数。关于应用上下文可以参阅本系列上一篇，关于消息闪现可以参阅入门系列第五篇。

### 同上下文 Hook 函数的区别

朋友们有没发现，部分信号回调函数同上一篇讲到的上下文 Hook 函数功能基本上一样？是的，拿”request_started”信号举例，它同”before_request”装饰的 Hook 函数都是在请求开始时被调用。那它们有什么区别呢？首先，实现原理不一样（废话！！）。然后，信号的目的只是为了通知订阅者某件事情发生了，但它不鼓励订阅者去修改数据。比如”request_finished”信号回调函数无需返回 response 对象，而”after_request”修饰的 Hook 函数必须返回 response 对象。

对于各函数的调用顺序，我用下面的代码测试了下：

```py
########## Capture flask core signals ##########
@request_started.connect_via(app)
def print_request_started(sender, **extra):
    print 'Signal: request_started'

@request_finished.connect_via(app)
def print_request_finished(sender, response, **extra):
    print 'Signal: request_finished'

@request_tearing_down.connect_via(app)
def print_request_tearingdown(sender, exc, **extra):
    print 'Signal: request_tearing_down'

########## Request Context Hook ##########
@app.before_request
def before_request():
    print 'Hook: before_request'

@app.after_request
def after_request(response):
    print 'Hook: after_request'
    return response

@app.teardown_request
def teardown_request(exception):
    print 'Hook: teardown_request'

```

运行结果如下：

```py
Signal: request_started
Hook: before_request
Hook: after_request
Signal: request_finished
Hook: teardown_request
Signal: request_tearing_down

```

朋友们也可以自己试一下。

### 自定义信号

除了 Flask 的核心信号，我们也可以自定义信号。这是 Hook 函数无法做到的。这里，我们要引入 Blinder 的库了：

```py
from blinker import Namespace

signals = Namespace()
index_called = signals.signal('index-called')

```

我们在全局定义了一个”index_called”信号对象，表示根路径被访问了。然后我们在根路径的请求处理中发出这个信号：

```py
@app.route('/')
def index():
    index_called.send(current_app._get_current_object(), method=request.method)
    return 'Hello Flask!'

```

发送信号消息的方法是”send()”，它必须包含一个参数指向信号的发送者。这里我们使用了”current_app._get_current_object()”来获取应用上下文中的 app 应用对象。怎么，忘了这个”current_app”是什么了？回顾下上一篇吧。这样每次客户端访问根路径时，都会发送”index_called”信号。”send()”方法可以有多个参数，从第二个参数开始是可选的，如果你要提供，就必须是 key=value 形式。而这个 key 就可以在订阅回调函数中接收。这个例子中，我们传递了请求的方法。

现在我们来定义订阅回调函数：

```py
def log_index_called(sender, method, **extra):
    print 'URL "%s" is called with method "%s"' % (request.url, method)

index_called.connect(log_index_called, app)

```

函数很简单，就是将请求地址和方法打印在控制台上，大家可以运行下试试。另外，同核心信号一样，自定义信号的回调函数也可以用装饰器来修饰，上面的代码等同于：

```py
@index_called.connect_via(app)
def log_index_called(sender, method, **extra):
    print 'URL "%s" is called with method "%s"' % (request.url, method)

```

对于信号更详细的使用，可以参考 Blinder 的[官方文档](http://pythonhosted.org/blinker/)。

本例中的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad2.html)
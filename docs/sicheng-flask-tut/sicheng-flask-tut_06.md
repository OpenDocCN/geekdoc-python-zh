# Flask 进阶系列(一)–上下文环境

Flask 目前最新的版本是 0.10.1，在其版本更新过程中，Flask 也在不断增加新的、炫酷的功能。我们在入门系列中介绍一些的基本功能，现在让我们开始更深入地了解 Flask。

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

Flask 提供了两种上下文环境，一个是应用上下文(Application Context)，另一个是请求上下文(Request Context)。从名字上就可以知道一个是应用级别的，另一个是单个请求级别的。不过 Flask 的实现有些令人混淆，下面我们先来看下请求上下文。

### 请求上下文环境

#### 请求上下文的生命周期

在入门系列第六篇中，出现了上下文装饰器”@app.before_request”和”@app.teardown_request”，用其修饰的函数也可以称为上下文 Hook 函数。此外，Flask 还提供了装饰器”@app.after_request”。看名字就能猜到，被”before_request”修饰的函数会在请求处理之前被调用，”after_request”和”teardown_request”会在请求处理完成后被调用。区别是”after_request”只会在请求正常退出时才会被调用，它必须传入一个参数来接受响应对象，并返回一个响应对象，一般用来统一修改响应的内容。而”teardown_request”在任何情况下都会被调用，它必须传入一个参数来接受异常对象，一般用来统一释放请求所占有的资源。同一种类型的 Hook 函数可以存在多个，程序会按代码中的顺序执行。我们开始看例子吧：

```py
from flask import Flask, g, request

app = Flask(__name__)

@app.before_request
def before_request():
    print 'before request started'
    print request.url

@app.before_request
def before_request2():
    print 'before request started 2'
    print request.url
    g.name="SampleApp"

@app.after_request
def after_request(response):
    print 'after request finished'
    print request.url
    response.headers['key'] = 'value'
    return response

@app.teardown_request
def teardown_request(exception):
    print 'teardown request'
    print request.url

@app.route('/')
def index():
    return 'Hello, %s!' % g.name

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

访问”http://localhost:5000/”后，会在控制台输出：

```py
before request started
http://localhost:5000/
before request started 2
http://localhost:5000/
after request finished
http://localhost:5000/
teardown request
http://localhost:5000/

```

由此我们可以看出各函数的调用顺序。如果一个”before_request”函数中有返回 response，则后面的”before_request”以及该请求的处理函数将不再被执行。直接进入”after_request”。我们可以修改上面的”before_request()”函数试试：

```py
@app.before_request
def before_request():
    print 'before request started'
    print request.url
    return 'hello'

```

另外，朋友们有没有注意到，在每个请求上下文 Hook 函数中，我们都可以访问”request”对象，然而，没有任何地方传入这个对象。难道它是全局的？那么我们随便声明个函数，并调用 request 对象会怎样？

```py
def handle_request():
    print 'handle request'
    print request.url

handle_request()

```

你会收到运行时错误：

```py
RuntimeError: working outside of request context
```

可见，request 对象只有在请求上下文的生命周期内才可以访问。离开了请求的生命周期，其上下文环境也就不存在了，自然也无法获取 request 对象。而上面介绍的几个由上下文装饰器修饰的 Hook 函数，会挂载在请求生命周期内的不同阶段，所以其内部可以访问 request 对象。

#### 构建请求上下文环境

一个请求一般是由客户端发起的，那么我们是否可以在服务器端手动构建请求上下文呢？答案是可以，也正因为如此，Flask 提供了在没有客户端的情况下实现自动测试，可通过”test_request_context()”来模拟客户端请求。关于 Flask 测试，我们会在本系列第九篇中介绍。这里，我们使用 Flask 的内部方法”request_context()”来构建一个请求上下文。

```py
from werkzeug.test import EnvironBuilder

ctx = app.request_context(EnvironBuilder('/','http://localhost/').get_environ())
ctx.push()
try:
    print request.url
finally:
    ctx.pop()

```

“request_context()”会创建一个请求上下文”RequestContext”类型的对象，其需接收”werkzeug”中的”environ”对象为参数。”werkzeug”是 Flask 所依赖的 WSGI 函数库，这里就不详述了，感兴趣的朋友可以查阅其[官网](http://werkzeug.pocoo.org/)。

上例中，我们可以在客户端的请求之外访问 request 对象，其实此时的 request 对象即是刚创建的请求上下文中的一个属性”request == ctx.request”。启动 Flask 时，控制台仍然可以打印出访问地址”http://localhost/”。上面的代码可以用 with 语句来简化：

```py
from werkzeug.test import EnvironBuilder

with app.request_context(EnvironBuilder('/','http://localhost/').get_environ()):
    print request.url

```

Flask 源码中的请求上下文构建方式也同此类似。

#### 请求上下文的实现方式

看到上一节的例子，好奇的朋友们不禁要问，既然”request_context”方法已经创建了请求上下文，为什么还要调用 push 和 pop 方法呢？这就是 Flask 关于上下文实现的关键了。

对于 Flask Web 应用来说，每个请求就是一个独立的线程。请求之间的信息要完全隔离，避免冲突，这就需要使用本地线程环境(ThreadLocal)，这个概念在其他语言如 Java 中也有。”ctx.push()”方法，会将当前请求上下文，压入”flask._request_ctx_stack”的栈中，这个”_request_ctx_stack”是内部对象，我们在应用开发时最好不要使用它，一般在 Flask 扩展开发中才会使用。同时这个”_request_ctx_stack”栈是个 ThreadLocal 对象。也就是”flask._request_ctx_stack”看似全局对象，其实每个线程的都不一样。请求上下文压入栈后，再次访问其都会从这个栈的顶端通过”_request_ctx_stack.top”来获取，所以取到的永远是只属于本线程中的对象，这样不同请求之间的上下文就做到了完全隔离。请求结束后，线程退出，ThreadLocal 线程本地变量也随即销毁，”ctx.pop()”用来将请求上下文从栈里弹出，避免内存无法回收。

这里涉及到了 ThreadLocal 的概念，还有 Python 垃圾回收机制。鉴于篇幅关系就不多说了。感兴趣的朋友可以自己去查查。

### 应用上下文环境

#### current_app 代理

介绍完请求级别的上下文环境，我们再来了解应用级别的上下文环境。先来看一段代码：

```py
from flask import Flask, current_app

app = Flask('SampleApp')

@app.route('/')
def index():
    return 'Hello, %s!' % current_app.name

```

我们可以通过”current_app.name”来获取当前应用的名称，也就是”SampleApp”。”current_app”是一个本地代理，它的类型是”werkzeug.local. LocalProxy”，它所代理的即是我们的 app 对象，也就是说”current_app == LocalProxy(app)”。使用”current_app”是因为它也是一个 ThreadLocal 变量，对它的改动不会影响到其他线程。你可以通过”current_app._get_current_object()”方法来获取 app 对象。

既然是 ThreadLocal 对象，那它就只在请求线程内存在，它的生命周期就是在应用上下文里。离开了应用上下文，”current_app”一样无法使用。

```py
app = Flask('SampleApp')
print current_app.name

```

```py
RuntimeError: working outside of application context
```

#### 构建应用上下文环境

同请求上下文一样，我们也可以手动构建应用上下文环境：

```py
with app.app_context():
    print current_app.name

```

“app_context()”方法会创建一个”AppContext”类型对象，即应用上下文对象，此后我们就可以在应用上下文中，访问”current_app”对象了。

#### 应用上下文的实现方式

上例中我们使用了”with”语句，其实应用上下文也有压栈和出栈的操作。在请求线程创建时，Flask 会创建应用上下文对象，并将其压入”flask._app_ctx_stack”的栈中，然后在线程退出前将其从栈里弹出。这个”_app_ctx_stack”同上一节请求中介绍的”_request_ctx_stack”一样，都是 ThreadLocal 变量。也就是说应用上下文的生命周期，也只在一个请求线程内，我们无法通过应用上下文在请求之间传递信息。这个很多人容易混淆，以为像 JSP 中的 application 对象一样，可以跨请求。

“_app_ctx_stack”一样是给 Flask 扩展开发用，应用开发不要去访问它。如果想在应用上下文中保存信息，可以用”flask.g”对象，我们在入门系列第四篇中介绍过它。

#### 应用上下文 Hook 函数

应用上下文也提供了装饰器来修饰 Hook 函数，不过只有一个”@app.teardown_appcontext”。它会在应用上下文生命周期结束前，也就是从”_app_ctx_stack”出栈时被调用。我们可以加入下面的代码，顺便也验证下，是否应用上下文在每个请求结束时会被销毁。

```py
@app.teardown_appcontext
def teardown_db(exception):
    print 'teardown application'

```

### 上下文设计思想

当我了解这两个上下文后，不禁要问：

*   既然请求上下文和应用上下文生命周期都在线程内，其实他们的作用域基本一样，为什么还要两个级别的上下文存在呢？
*   既然上下文环境只能在一个请求中，而一个请求中似乎也不会创建两个以上的请求或应用上下文。那用 ThreadLocal 本地变量就行，什么要用栈呢？

查了些相关资料。对于第一个问题，设计初衷是为了能让两个以上的 Flask 应用共存在一个 WSGI 应用中，这样在请求中，你需要通过应用上下文来获取当前请求的应用信息。

对于第二问题，Web 客户端下，的确是不需要。不过 Flask 支持在离线环境中跑自动测试，这时候，代码可以实现上下文环境的嵌套。比如下例：

```py
app = Flask('MainApp')
sub_app = Flask('SubApp')

with app.app_context():
    print current_app.name
    with sub_app.app_context():
        print sub_app.name

```

如同函数内调用函数一样，使用栈就可以支持上述代码。

上下文环境是 Flask 中一个比较复杂的地方，如果还有不解的地方，建议朋友们读一下[Flask 源码](https://github.com/mitsuhiko/flask)。

本例中的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad1.html)
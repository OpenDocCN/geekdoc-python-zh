# Flask 进阶系列(四)–视图

终于结束了 Jinja2 的模板引擎之旅，让我们回到 Flask 中来。在一开始介绍 Flask 模板时，我们曾说过它是处在 MVC 模型中的 View 层，其实更确切的说，应该是模板渲染后的返回内容，才是真正的 View，也就是视图。可以理解为，视图就是最终会显示在浏览器上的内容，将其同控制器，也就是路由规则绑定后，用户就可以通过 URL 地址来访问它。即便不使用模板，直接返回字符串，返回的结果也是视图。Flask 提供了很多针对视图强化的功能，比可插拔视图 Pluggable View，基于方法的视图，延迟加载视图，你还可以针对视图写自己的装饰器。本篇就会详细介绍这些功能。

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

### 视图装饰器

我们先创建个最简单的 Flask 应用，想必大家都信手拈来了。

```py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Hello World!</h1>'

@app.route('/hello')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello-view.html', name=name)

@app.route('/admin')
def admin():
    return '<h1>Admin Dashboard</h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

模板代码如下：

```py
<!DOCTYPE html>
<title>Hello View</title>
{% if name %}
  <h1>Welcome {{ name }}</h1>
{% else %}
  <h1>Welcome Guest</h1>
{% endif %}

```

现在，我希望当用户访问 admin 页面时，必须先登录。大家马上会想到在”admin()”方法里判断会话 session。这样的确可以达成目的，不过当我们有 n 多页面都要进行用户验证的话，判断用户登录的代码就会到处都是，即便我们封装在一个函数里，至少调此函数的代码也会重复出现。有没有什么办法，可以像 Java Sevlet 中的 Filter 一样，能够在请求入口统一处理呢？Flask 没有提供特别的功能来实现这个，因为 Python 本身有，那就是装饰器。对于 Python 装饰器不太熟悉的朋友，可以参考下我之前的一篇介绍。

我们现在就来写个验证用户登录的装饰器：

```py
from functools import wraps
from flask import session, abort

def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if not 'user' in session:
            abort(401)
        return func(*args, **kwargs)
    return decorated_function

app.secret_key = '12345678'

```

代码很简单，就是在调用函数前，先检查 session 里有没有用户。此后，我们只需将此装饰器加在每个需要验证登录的请求方法上即可：

```py
@app.route('/admin')
@login_required
def admin():
    return '<h1>Admin Dashboard</h1>'

```

这个装饰器就被称为视图装饰器(View Decorator)。为了减少篇幅，我这里省略了登录部分的代码，大家可以参阅入门系列的第四篇中。朋友们想想，还有什么功能可以写在视图装饰器里？对，页面缓存。我们可以把页面的路径作为键，页面内容作为值，放在缓存里。每次进入请求函数前，先判断缓存里有没有该页面，有就直接将缓存里的值返回，没有则执行请求函数，将结果存在缓存后再返回。

[官方文档](http://flask.pocoo.org/docs/0.10/patterns/viewdecorators/)中有更多视图装饰器的示例，大家可以借鉴下。

### URL 集中映射

为了引出后面的内容，我们先来介绍下 Flask 中的 URL 集中映射功能。之前所有的例子中，URL 路由都是作为装饰器写在请求函数，也就是视图函数之上的。这样做的优点就是程序一目了然，方便修改，就像 Java 领域的 Hibernate，从 3.0 开始支持将数据库的字段映射直接写在 JavaBean 的属性上，代码维护起来方便多了。熟悉 Django 的朋友们知道，Django 的 URL 路由是统一写在一个专门的文件里，习惯上是”urls.py”。Django 为什么不学 Flask 呢？其实，对于规模较大的应用，URL 路径相当多，这时所有的路由规则放在一起其实更容易管理，批量修改起来也方便。Django 面向的应用一般规模较 Flask 要大，所以这种情况下，统一路由管理比程序一目了然可能更重要。

说了这么多，就是要告诉大家，其实 Flask 也支持像 Django 一样，把 URL 路由规则统一管理，而不是写在视图函数上。怎么做呢？我们先来写个视图函数，将它放在一个”views.py”文件中：

```py
def foo():
    return '<h1>Hello Foo!</h1>'

```

然后在 Flask 主程序上调用”app.add_url_rule”方法：

```py
app.add_url_rule('/foo', view_func=views.foo)

```

这样，路由”/foo”就绑定在”views.foo()”函数上了，效果等同于在”views.foo()”函数上加上”@app.route(‘/foo’)”装饰器。通过”app.add_url_rule”方法，我们就可以将路由同视图分开，将路由统一管理，实现完全的 MVC。

那么在这种情况下，上一节定义的装饰器怎么用？大家想想，装饰器本质上是一个闭包函数，所以我们当然可以把它当函数使用：

```py
app.add_url_rule('/foo', view_func=login_required(views.foo))

```

### 可插拔视图 Pluggable View

#### 视图类

上一节的 URL 集中映射，就是视图可插拔的基础，因为它可以支持在程序中动态的绑定路由和视图。Flask 提供了视图类，使其可以变得更灵活，我们先看个例子：

```py
from flask.views import View

class HelloView(View):
    def dispatch_request(self, name=None):
        return render_template('hello-view.html', name=name)

view = HelloView.as_view('helloview')
app.add_url_rule('/helloview', view_func=view)
app.add_url_rule('/helloview/<name>', view_func=view)

```

我们创建了一个”flask.views.View”的子类，并覆盖了其”dispatch_request()”函数，渲染视图的主要代码必须写在这个函数里。然后我们通过”as_view()”方法把类转换为实际的视图函数，”as_view()”必须传入一个唯一的视图名。此后，这个视图就可以由”app.add_url_rule”方法绑定到路由上了。上例的效果，同本篇第一节中”/hello”路径的效果，完全一样。

这个例子比较简单，只是为了介绍怎么用视图类，体现不出它的灵活性，我们再看个例子：

```py
class RenderTemplateView(View):
    def __init__(self, template):
        self.template = template

    def dispatch_request(self):
        return render_template(self.template)

app.add_url_rule('/hello', view_func=RenderTemplateView.as_view('hello', template='hello-view.html'))
app.add_url_rule('/login', view_func=RenderTemplateView.as_view('login', template='login-view.html'))

```

很多时候，渲染视图的代码都类似，只是模板不一样罢了。我们完全可以把渲染视图的代码重用，上例中，我们就省去了分别定义”hello”和”login”视图函数的工作了。

#### 视图装饰器支持

在使用视图类的情况下，视图装饰器要怎么用呢？Flask 在 0.8 版本后支持这样的写法：

```py
class HelloView(View):
    decorators = [login_required]

    def dispatch_request(self, name=None):
        return render_template('hello-view.html', name=name)

```

我们只需将装饰器函数加入到视图类变量”decorators”中即可。它是一个列表，所以能够支持多个装饰器，并按列表中的顺序执行。

#### 请求方法的支持

当我们的视图要同时支持 GET 和 POST 请求时，视图类可以这么定义：

```py
class MyMethodView(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        if request.method == 'GET':
            return '<h1>Hello World!</h1>This is GET method.'
        elif request.method == 'POST':
            return '<h1>Hello World!</h1>This is POST method.'

app.add_url_rule('/mmview', view_func=MyMethodView.as_view('mmview'))

```

我们只需将需要支持的 HTTP 请求方法加入到视图类变量”methods”中即可。没加的话，默认只支持 GET 请求。

#### 基于方法的视图

上节介绍的 HTTP 请求方法的支持，的确比较方便，但是对于 RESTFul 类型的应用来说，有没有更简单的方法，比如省去那些 if, else 判断语句呢？Flask 中的”flask.views.MethodView”就可以做到这点，它是”flask.views.View”的子类。我们写个 user API 的视图吧：

```py
from flask.views import MethodView

class UserAPI(MethodView):
    def get(self, user_id):
        if user_id is None:
            return 'Get User called, return all users'
        else:
            return 'Get User called with id %s' % user_id

    def post(self):
        return 'Post User called'

    def put(self, user_id):
        return 'Put User called with id %s' % user_id

    def delete(self, user_id):
        return 'Delete User called with id %s' % user_id

```

现在我们分别定义了 get, post, put, delete 方法来对应四种类型的 HTTP 请求，注意函数名必须这么写。怎么将它绑定到路由上呢？

```py
user_view = UserAPI.as_view('users')
# 将 GET /users/请求绑定到 UserAPI.get()方法上，并将 get()方法参数 user_id 默认为 None
app.add_url_rule('/users/', view_func=user_view, 
                            defaults={'user_id': None}, 
                            methods=['GET',])
# 将 POST /users/请求绑定到 UserAPI.post()方法上
app.add_url_rule('/users/', view_func=user_view, 
                            methods=['POST',])
# 将/users/<user_id>URL 路径的 GET，PUT，DELETE 请求，
# 绑定到 UserAPI 的 get(), put(), delete()方法上，并将参数 user_id 传入。
app.add_url_rule('/users/<user_id>', view_func=user_view, 
                                     methods=['GET', 'PUT', 'DELETE'])

```

为了方便阅读，我将注释放到了代码上，大家如果直接拷贝的话，记得在代码开头加上”#coding:utf8″来支持中文。上例中”app.add_url_rule()”可以传入参数 default，来设置默认值；参数 methods，来指定支持的请求方法。

如果 API 多，有人觉得每次都要加这么三个路由规则太麻烦，可以将其封装个函数：

```py
def register_api(view, endpoint, url, primary_id='id', id_type='int'):
    view_func = view.as_view(endpoint)
    app.add_url_rule(url, view_func=view_func,
                          defaults={primary_id: None},
                          methods=['GET',])
    app.add_url_rule(url, view_func=view_func,
                          methods=['POST',])
    app.add_url_rule('%s<%s:%s>' % (url, id_type, primary_id),
                          view_func=view_func,
                          methods=['GET', 'PUT', 'DELETE'])

register_api(UserAPI, 'users', '/users/', primary_id='user_id')

```

现在，一个”register_api()”就可以绑定一个 API 了，还是挺 easy 的吧。

### 延迟加载视图

当某一视图很占内存，而且很少会被使用，我们会希望它在应用启动时不要被加载，只有当它被使用时才会被加载。也就是接下来要介绍的延迟加载。Flask 原生并不支持视图延迟加载功能，但我们可以通过代码实现。这里，我引用了[官方文档](http://flask.pocoo.org/docs/0.10/patterns/lazyloading/)上的一个实现。

```py
from werkzeug import import_string, cached_property

class LazyView(object):
    def __init__(self, import_name):
        self.__module__, self.__name__ = import_name.rsplit('.', 1)
        self.import_name = import_name

    @cached_property
    def view(self):
        return import_string(self.import_name)

    def __call__(self, *args, **kwargs):
        return self.view(*args, **kwargs)

```

我们先写了一个 LazyView，然后在 views.py 中定义一个名为 bar 的视图函数：

```py
def bar():
    return '<h1>Hello Bar!</h1>'

```

现在让我们来绑定路由：

```py
app.add_url_rule('/lazy/bar', view_func=LazyView('views.bar'))

```

路由绑定在 LazyView 的对象上，因为实现了 __call__ 方法，所以这个对象可被调用，不过只有当’/lazy/bar’地址被请求时才会被调用。此时”werkzeug.import_string”方法会被调用，看了下 Werkzeug 的源码，它的本质就是调用”__import__”来动态地导入 Python 的模块和函数。所以，这个”view.bar”函数只会在’/lazy/bar’请求发生时才被导入到主程序中。不过要是每次请求发生都被导入一次的话，开销也很大，所以，代码使用了”werkzeug.cached_property”装饰器把导入后的函数缓存起来。

这个 LazyView 的实现还是挺有趣的吧。可能有一天，Flask 会把延迟加载视图的功能加入到它的原生代码中。同上一节的”register_api()”函数一样，你也可以把绑定延迟加载视图的代码封装在一个函数里。

```py
def add_url_for_lazy(url_rule, import_name, **options):
    view = LazyView(import_name)
    app.add_url_rule(url_rule, view_func=view, **options)

add_url_for_lazy('/lazy/bar', 'views.bar')

```

本篇中的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad4.html)
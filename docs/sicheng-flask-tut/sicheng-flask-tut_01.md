# Flask 入门系列(二)–路由

上一篇中，我们用 Flask 写了一个 Hello World 程序，让大家领略到了 Flask 的简洁轻便。从这篇开始我们将对 Flask 框架的各功能作更详细的介绍，我们首先从路由(Route)开始。

### 系列文章

*   Flask 入门系列(一)–Hello World
*   Flask 入门系列(二)–路由
*   Flask 入门系列(三)–模板
*   Flask 入门系列(四)–请求，响应及会话
*   Flask 入门系列(五)–错误处理及消息闪现
*   Flask 入门系列(六)–数据库集成

### 路由

从 Hello World 中，我们了解到 URL 的路由可以直接写在其要执行的函数上。有人会质疑，这样不是把 Model 和 Controller 绑在一起了吗？的确，如果你想灵活的配置 Model 和 Controller，这样是不方便，但是对于轻量级系统来说，灵活配置意义不大，反而写在一块更利于维护。Flask 路由规则都是基于 Werkzeug 的路由模块的，它还提供了很多强大的功能。

#### 带参数的路由

让我们在上一篇 Hello World 的基础上，加上下面的函数。并运行程序。

```py
@app.route('/hello/<name>')
def hello(name):
    return 'Hello %s' % name

```

当你在浏览器的地址栏中输入”http://localhost:5000/hello/man”，你将在页面上看到”Hello man”的字样。URL 路径中”/hello/”后面的参数被作为”hello()”函数的”name”参数传了进来。
你还可以在 URL 参数前添加转换器来转换参数类型，我们再来加个函数：

```py
@app.route('/user/<int:user_id>')
def get_user(user_id):
    return 'User ID: %d' % user_id

```

试下访问”http://localhost:5000/user/man”，你会看到 404 错误。但是试下”http://localhost:5000/user/123″，页面上就会有”User ID: 123″显示出来。参数类型转换器”int:”帮你控制好了传入参数的类型只能是整形。目前支持的参数类型转换器有：

| 类型转换器 | 作用 |
| 缺省 | 字符型，但不能有斜杠 |
| int: | 整型 |
| float: | 浮点型 |
| path: | 字符型，可有斜杠 |

另外，大家有没有注意到，Flask 自带的 Web 服务器支持热部署。当你修改好文件并保存后，Web 服务器自动部署完毕，你无需重新运行程序。

#### 多 URL 的路由

一个函数上可以设施多个 URL 路由规则

```py
@app.route('/')
@app.route('/hello')
@app.route('/hello/<name>')
def hello(name=None):
    if name is None:
        name = 'World'
    return 'Hello %s' % name

```

这个例子接受三种 URL 规则，”/”和”/hello”都不带参数，函数参数”name”值将为空，页面显示”Hello World”；”/hello/<name>“带参数，页面会显示参数”name”的值，效果与上面第一个例子相同。</name>

#### HTTP 请求方法设置

HTTP 请求方法常用的有 Get, Post, Put, Delete。不熟悉的朋友们可以去度娘查下。Flask 路由规则也可以设置请求方法。

```py
from flask import request

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return 'This is a POST request'
    else:
        return 'This is a GET request'

```

当你请求地址”http://localhost:5000/login”，”GET”和”POST”请求会返回不同的内容，其他请求方法则会返回 405 错误。有没有觉得用 Flask 来实现 Restful 风格很方便啊？

#### URL 构建方法

Flask 提供了”url_for()”方法来快速获取及构建 URL，方法的第一个参数指向函数名（加过”@app.route”注解的函数），后续的参数对应于要构建的 URL 变量。下面是几个例子：

```py
url_for('login')    # 返回/login
url_for('login', id='1')    # 将 id 作为 URL 参数，返回/login?id=1
url_for('hello', name='man')    # 适配 hello 函数的 name 参数，返回/hello/man
url_for('static', filename='style.css')    # 静态文件地址，返回/static/style.css

```

#### 静态文件位置

一个 Web 应用的静态文件包括了 JS, CSS, 图片等，Flask 的风格是将所有静态文件放在”static”子目录下。并且在代码或模板（下篇会介绍）中，使用”url_for(‘static’)”来获取静态文件目录。上小节中第四个的例子就是通过”url_for()”函数获取”static”目录下的指定文件。如果你想改变这个静态目录的位置，你可以在创建应用时，指定”static_folder”参数。

```py
app = Flask(__name__, static_folder='files')

```

本例中的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-2.html)
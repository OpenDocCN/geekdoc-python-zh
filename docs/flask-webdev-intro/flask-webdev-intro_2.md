# 快速入门

本章主要介绍 flask 的基础使用，主要包含以下几个方面：

*   路由和视图
*   静态文件
*   Jinja2 模板引擎
*   请求、重定向及会话
*   数据库集成
*   REST Web 服务
*   部署

# 一个最简单的应用

我们在第一章已经看到了一个简单的 Hello World 的例子，相信你已经成功地把它跑起来了，下面我们对这个程序进行讲解。回顾一下这个程序：

```py
$ cat hello.py

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run() 
```

*   先看程序的第 1 句：

```py
from flask import Flask 
```

该句从 flask 包导入了一个 Flask 类，这也是后面构建 Flask Web 程序的基础。

*   接着看程序的第 2 句：

```py
app = Flask(__name__) 
```

上面这一句通过将 `__name__` 参数传给 Flask 类的构造函数，创建了一个程序实例 `app`，也就创建了一个 Flask 集成的开发 Web 服务器。**Flask 用 `__name__` 这个参数决定程序的根目录，以便程序能够找到相对于程序根目录的资源文件位置，比如静态文件等。**

*   接着看程序的第 3，4，5 句：

```py
@app.route("/")
def hello():
    return "Hello World!" 
```

可能读者会对这三句感到很困惑：它们的作用是什么呢？我们知道，Web 浏览器把请求发送给 Web 服务器，Web 服务器再把请求发送给 Flask 程序实例，那么程序实例就需要知道对每个 URL 请求应该运行哪些代码。

上面这三句代码的意思就是：如果浏览器要访问服务器程序的根地址（"/"），那么 Flask 程序实例就会执行函数 `hello()` ，返回『Hello World!』。

比如，假设我们部署程序的服务器域名为 `www.hello.com`，当我们在浏览器访问 http:// www.hello.com（也就是根地址）时，会触发 Flask 程序执行 `hello()` 这个函数，返回『Hello World!』，**这个函数的返回值称为响应，是客户端接收到的内容。**

但是，如果我们在浏览器访问 [`www.hello.com/peter`](http://www.hello.com/peter) 时，程序会返回 `404` 错误，因为我们的 Flask 程序并没有对这个 URL 指定处理函数，所以会返回错误代码。

*   接着看程序的最后两句：

```py
if __name__ == "__main__":
    app.run() 
```

上面两句的意思，当我们运行该脚本的时候（第 1 句），启动 Flask 集成的开发 Web 服务器（第 2 句）。默认情况下，改服务器会监听本地的 5000 端口，如果你想改变端口的话，可以传入 "port=端口号"，另外，如果你想支持远程，需要传入 "host=0.0.0.0"，你还可以设置调试模式，如下：

```py
app.run(host='0.0.0.0', port=8234, debug=True) 
```

服务器启动后，程序会进入轮询，等待并处理请求。轮询会一直运行，直到程序被终止。需要注意的是，Flask 提供的 Web 服务器不适合在生产环境中使用，后面我们会介绍生产环境中的 Web 服务器。

OK，到此为止，我们基本明白一个简单的 Flask 程序是怎么运作的了，后面我们就一起慢慢揭开 Flask 的神秘面纱吧~~

# 路由和视图

我们在前面的一小节介绍了一个简单的 Flask 程序是怎么运行的。其中，有三行代码，我们并没有深入讲解。在这里，我们就对它们进行深入解析。回顾这三行代码：

```py
@app.route("/")
def hello():
    return "Hello World!" 
```

这三行代码的意思就是：如果浏览器要访问服务器程序的根地址（"/"），那么 Flask 程序实例就会执行函数 `hello()` ，返回『Hello World!』。

也就是说，**上面三行代码定义了一个 URL 到 Python 函数的映射关系**，我们将处理这种映射关系的程序称为『路由』，而 `hello()` 就是视图函数。

## 动态路由

假设服务器域名为 `https://hello.com`, 我们来看下面一个路由：

```py
@app.route("/ethan")
def hello():
    return '<h1>Hello, ethan!</h1>' 
```

再来看一个路由：

```py
@app.route("/peter")
def hello():
    return '<h1>Hello, peter!</h1>' 
```

可以看到，上面两个路由的功能是当用户访问 `https://hello.com/<user_name>` 时，网页显示对该用户的问候。按上面的写法，如果对每个用户都需要写一个路由，那么 100 个用户岂不是要写 100 个路由！这当然是不能忍受的，实际上一个路由就够了！且看下面：

```py
@app.route("/<user_name>")
def hello(user_name):
    return '<h1>Hello, %s!</h1>' % user_name 
```

现在，任何类似 `https://hello.com/<user_name>` 的 URL 都会映射到这个路由上，比如 `https://hello.com/ethan-funny`，`https://hello.com/torvalds`，访问这些 URL 都会执行上面的路由程序。

也就是说，Flask 支持这种动态形式的路由，路由中的动态部分默认是字符串，像上面这种情况。当然，除了字符串，Flask 也支持在路由中使用 int、float，比如路由 /articles/ <intu0003aid class="hljs-meta">只会匹配动态片段 id 为整数的 URL，例如匹配 [`hello.com/articles/100，https://hello.com/articles/101，但不匹配`](https://hello.com/articles/100，https://hello.com/articles/101，但不匹配) [`hello.com/articles/the-first-article`](https://hello.com/articles/the-first-article) 这种 URL。</intu0003aid>

# 静态文件

静态文件，顾名思义，就是那些不会被改变的文件，比如图片，CSS 文件和 JavaScript 源码文件。默认情况下，Flask 在程序根目录中名为 static 的子目录中寻找静态文件。因此，我们一般在应用的包中创建一个叫 static 的文件夹，并在里面放置我们的静态文件。比如，我们可以按下面的结构组织我们的 app：

```py
app/
    __init__.py
    static/
        css/
            style.css
            home.css
            admin.css
        js/
            home.js
            admin.js
        img/
            favicon.co
            logo.svg
    templates/
        index.html
        home.html
        admin.html
    views/
    models/
run.py 
```

但是，我们有时还会应用到第三方库，比如 jQuery, Bootstrap 等，这时我们为了不跟自己的 Javascript 和 CSS 文件混起来，我们可以将这些第三方库放到 lib 文件夹或者 vendor 文件夹，比如下面这种：

```py
static/
    css/
        lib/
            bootstrap.css
        style.css
        home.css
        admin.css
    js/
        lib/
            jquery.js
            chart.js
        home.js
        admin.js
    img/
        logo.svg
        favicon.ico 
```

## 提供一个 favicon 图标

favicon 是 favorites icon 的缩写，也被称为 website icon（网页图标）、page icon（页面图标）等。通常而言，定义一个 favicon 的方法是将一个名为『favicon.ico』的文件置于 Web 服务器的根目录下。但是，正如我们在上面指出，我们一般将图片等静态资源放在一个单独的 static 文件夹中。为了解决这种不一致，我们可以在站点模板的 部分添加两个 link 组件，比如我们可以在 template/base.html 中定义 favicon 图标：

```py
{% block head %}
{{ super() }}
<link rel="shortcut icon" href="{{ url_for('static', filename = 'favicon.ico') }}" type="image/x-icon">
<link rel="icon" href="{{ url_for('static', filename = 'favicon.ico') }}" type="image/x-icon">
{% endblock %} 
```

在上面的代码中，我们使用了 `super()` 来保留基模板中定义的块的原始内容，并添加了两个 link 组件声明图标位置，这两个 link 组件声明会插入到 head 块的末尾。

# 使用 Jinja2 模板引擎

## 什么是模板引擎

在 Web 开发中，我们经常会使用到模板引擎。简单点来说，我们可以**把模板看成是一个含有某些变量的字符串，它们的具体值需要在动态运行时（请求的上下文）才能知道**。比如，有下面一个模板：

```py
<h1>Hello, {{ name }}!</h1> 
```

其中，name 是一个变量名，我们用 `{{ }}` 包裹它表示它是一个变量。我们给 name 传不同的值，模板会返回不同的字符串。像这样，使用真实的值替换变量，再返回最终得到的响应字符串，这一过程称为**渲染**。**模板引擎就是渲染模板的程序。**

Flask 默认使用 [Jinja2](http://jinja.pocoo.org/) 模板引擎。

## 为什么要使用模板引擎

先来看一个简单的程序。

```py
$ cat hello.py

from flask import Flask

app = Flask(__name__)

@app.route("/<name>")
def hello(name):
    if name == 'ethan':
        return "<h1>Hello, world!</h1> <h2>Hello, %s!</h2>" % name
    else:
        return "<h1>Hello, world!</h1> <h2>Hello, world!</h2>"

if __name__ == "__main__":
    app.run() 
```

在终端运行上面的代码 `python hello.py`，终端输出：

```py
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit) 
```

我们在浏览器地址栏输入 `http://127.0.0.1:5000/ethan`，显示如下：

![helloworld2](img/helloworld2.png)

符合预期，没什么问题。但是，我们看到，上面的视图函数 `hello()` 夹杂了一些 `HTML` 代码，如果 HTML 代码多了，会使得我们的程序变得难以理解和维护。我们可以看到，其实视图函数主要有两部分逻辑：业务逻辑和表现逻辑。像上下文判断，数据库查询等后台处理都可以算是业务逻辑，而返回给前端的响应内容则算是表现逻辑，它们是需要在前端展现的。

在上面的代码中，我们将业务逻辑和表现逻辑混杂在一起了，代码很不优雅，而且当代码变多了后，程序将会变得难以理解和维护。因此，良好的做法应该是将业务逻辑和表现逻辑分开，模板引擎正好可以满足这种需求。

## Jinja 模板引擎入门

我们将上面的例子用 Jinja 模板进行改写。默认情况下，Flask 在程序文件夹中的 templates 子文件夹中寻找模板。改写后的文件结构如下：

```py
.
├── hello.py
└── templates
    └── index.html 
```

`hello.py` 文件内容如下：

```py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/<name>')
def hello(name):
    if name == 'ethan':
        return render_template('index.html', name=name)
    else:
        return render_template('index.html', name='world')

if __name__ == "__main__":
    app.run() 
```

`index.html` 文件内容如下：

```py
<h1>Hello, world!</h1> <h2>Hello, {{ name }}!</h2> 
```

在 `hello.py` 中，我们使用了 Flask 提供的 render_template 函数，该函数把 Jinja2 模板引擎集成到了程序中。render_template 函数的第一个参数是模板的文件名。随后的参数都是键值对，表示模板中变量对应的真实值。

### 变量

Jinja 模板使用 `{{ 变量名 }}` 表示一个变量，比如上面的 `{{ name }}`，它告诉模板引擎这个位置的值从渲染模板时使用的数据中获取。

在 Jinja 中，还能使用列表，字典和对象等复杂的类型，比如：

```py
<p> Hello, {{ mydict['key'] }}. Hello, {{ mylist[0] }}. </p> 
```

### 控制结构

Jinja 提供了多种控制结构，来改变模板的渲染流程，比如常见的判断结构，循环结构。示例如下：

```py
{% if user == 'ethan' or user =='peter' %}
    <p> Hello, {{ user }} </p>
{% else %}
    <p> Hello, world! </p>
{% endif %}

<ul>
    {% for user in user_list %}
        <li> {{ user }} </li>
    {% endfor %}
</ul> 
```

### 宏

当有一段代码我们经常要用到的时候，我们往往会写一个函数，在 Jinja 中，我们可以使用宏来实现。例如:

```py
{% macro render_user(user) %}
    {% if user == 'ethan' %}
        <p> Hello, {{ user }} </p>
    {% endif %}
{% endmacro %} 
```

为了重复使用宏，我们将其保存在单独的文件中，比如 'macros.html'，然后在需要使用的模板中导入：

```py
{% import 'macros.html' as macros %}

{{ macros.render_user(user) }} 
```

### 模板继承

另一种重复使用代码的强大方式是**模板继承**，就像类继承需要有一个**基类**一样，我们需要一个**基模板**。比如，我们可以创建一个名为 base.html 的基模板：

```py
<html>
<head>
    {% block head %}
    <title>{% block title %}{% endblock %} - My Application</title> 
    {% endblock %}
</head>

<body>
    {% block body %}
    {% endblock %} 
</body>
</html> 
```

我们可以看到上面的基模板含有三个 `block` 块：head、title 和 body。下面，我们通过这个基模板来派生新的模板：

```py
{% extends "base.html" %}
{% block title %}Index{% endblock %}
{% block head %}
    {{ super() }}
{% endblock %}
{% block body %}
<h1>Hello, World!</h1>
{% endblock %} 
```

注意到上面第一行代码使用了 `extends` 命令，表明该模板继承自 base.html。接着，我们重新定义了 title、head 和 body。另外，我们还使用了 super() 获取基模板原来的内容。

更多关于 Jinja 模板引擎的使用可以参考 [Jinja2 2.7 documentation](http://docs.jinkan.org/docs/jinja2/)。

# 请求、重定向及会话

Web 开发中经常需要处理 HTTP 请求、重定向和会话等诸多事务，相应地，Flask 也内建了一些常见的对象如 request, session, redirect 等对它们进行处理。

## 请求对象 request

HTTP 请求方法有 GET、POST、PUT 等，request 对象也相应地提供了支持。举个例子，假设现在我们开发一个功能：用户注册。如果 HTTP 请求方法是 POST，我们就注册该用户，如果是 GET 请求，我们就显示注册的字样。代码示例如下（注意，下面代码并不能直接运行，文末提供了完整的代码）：

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/register', methods=['POST', 'GET']):
def register():
    if request.method == 'GET':
        return 'please register!'
    elif request.method == 'POST':
        user = request.form['user']
        return 'hello', user 
```

## 重定向对象 redirect

当用户访问某些网页时，如果他还没登录，我们往往会把网页**重定向**到登录页面，Flask 提供了 redirect 对象对其进行处理，我们对上面的代码做一点简单的改造，如果用户注册了，我们将网页重定向到首页。代码示例如下：

```py
from flask import Flask, request, redirect

app = Flask(__name__)

@app.route('/home', methods=['GET']):
def index():
    return 'hello world!'

@app.route('/register', methods=['POST', 'GET']):
def register():
    if request.method == 'GET':
        return 'please register!'
    elif request.method == 'POST':
        user = request.form['user']
        return redirect('/home') 
```

## 会话对象 session

程序可以把数据存储在**用户会话**中，用户会话是一种私有存储，默认情况下，它会保存在客户端 cookie 中。Flask 提供了 session 对象来操作用户会话，下面看一个示例：

```py
from flask import Flask, request, session, redirect, url_for, render_template

app = Flask(__name__)

@app.route('/home', methods=['GET'])
def index():
    return 'hello world!'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_name = request.form['user']
        session['user'] = user_name
        return 'hello, ' + session['user']
    elif request.method == 'GET':
        if 'user' in session:
            return redirect(url_for('index'))
        else:
            return render_template('login.html')

app.secret_key = '123456'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5632, debug=True) 
```

操作 `session` 就像操作 python 中的字典一样，我们可以使用 `session['user']` 获取值，也可以使用 `session.get('user')` 获取值。注意到，我们使用了 `url_for` 生成 URL，比如 `/home` 写成了 `url_for('index')`。`url_for()` 函数的第一个且唯一必须指定的参数是端点名，即路由的内部名字。默认情况下，路由的端点是相应视图函数的名字，因此 `/home` 应该写成 `url_for('index')`。还有一点，使用`session` 时要设置一个密钥 `app.secret_key`。

## 附录

本节完整的代码如下：

```py
$ tree .
.
├── flask-session.py
└── templates
    ├── layout.html
    └── login.html

$ cat flask-session.py
from flask import Flask, request, session, redirect, url_for, render_template

app = Flask(__name__)

@app.route('/')
def head():
    return redirect(url_for('register'))

@app.route('/home', methods=['GET'])
def index():
    return 'hello world!'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_name = request.form['user']
        session['user'] = user_name
        return 'hello, ' + session['user']
    elif request.method == 'GET':
        if 'user' in session:
            return redirect(url_for('index'))
        else:
            return render_template('login.html')

app.secret_key = '123456'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5632, debug=True)

$ cat layout.html
<!doctype html>
<title>Hello Sample</title>
<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
<div class="page">
    {% block body %}
    {% endblock %}
</div>

$ cat login.html
{% extends "layout.html" %}
{% block body %}
<form name="register" action="{{ url_for('register') }}" method="post">
    Hello {{ title }}, please login by:
    <input type="text" name="user" />
</form>
{% endblock %} 
```

# 数据库

## ORM 框架

Web 开发中，一个重要的组成部分便是数据库了。Web 程序中最常用的莫过于关系型数据库了，也称 SQL 数据库。另外，文档数据库（如 mongodb）、键值对数据库（如 redis）近几年也逐渐在 web 开发中流行起来，我们习惯把这两种数据库称为 NoSQL 数据库。

本书中，我们使用的数据库仍是常见的 SQL 数据库。大多数的关系型数据库引擎（比如 MySQL、Postgres 和 SQLite）都有对应的 Python 包。在这里，我们不直接使用这些数据库引擎提供的 Python 包，而是使用对象关系映射（Object-Relational Mapper, ORM）框架，它将低层的数据库操作指令抽象成高层的面向对象操作。也就是说，如果我们直接使用数据库引擎，我们就要写 SQL 操作语句，但是，如果我们使用了 ORM 框架，我们对诸如表、文档此类的数据库实体就可以简化成对 Python 对象的操作。

Python 中最广泛使用的 ORM 框架是 [SQLAlchemy](http://www.sqlalchemy.org/)，它是一个很强大的关系型数据库框架，不仅支持高层的 ORM，也支持使用低层的 SQL 操作，另外，它也支持多种数据库引擎，如 MySQL、Postgres 和 SQLite 等。

## Flask-SQLAlchemy

在 Flask 中，为了简化配置和操作，我们使用的 ORM 框架是 [Flask-SQLAlchemy](http://flask-sqlalchemy.pocoo.org/)，这个 Flask 扩展封装了 [SQLAlchemy](http://www.sqlalchemy.org/) 框架。在 Flask-SQLAlchemy 中，数据库使用 URL 指定，下表列出了常见的数据库引擎和对应的 URL。

| 数据库引擎 | URL |
| --- | --- |
| MySQL | mysql://username:password@hostname/database |
| Postgres | postgresql://username:password@hostname/database |
| SQLite (Unix) | sqlite:////absolute/path/to/database |
| SQLite (Windows) | sqlite:///c:/absolute/path/to/database |

上面的表格中，username 和 password 表示登录数据库的用户名和密码，hostname 表示 SQL 服务所在的主机，可以是本地主机（localhost）也可以是远程服务器，database 表示要使用的数据库。有一点需要注意的是，SQLite 数据库不需要使用服务器，它使用硬盘上的文件名作为 database。

## 一个最小的应用

### 创建数据库

首先，我们使用 pip 安装 Flask-SQLAlchemy:

```py
$ pip install flask-sqlalchemy 
```

接下来，我们配置一个简单的 SQLite 数据库：

```py
$ cat app.py
# -*- coding: utf-8 -*-

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

class User(db.Model):
    """定义数据模型"""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)

    def __init__(self, username, email):
        self.username = username
        self.email = email

    def __repr__(self):
        return '<User %r>' % self.username 
```

这里有几点需要注意的是：

1.  app 应用配置项 "SQLALCHEMY_DATABASE_URI" 指定了 SQLAlchemy 所要操作的数据库，这里我们使用的是 SQLite，数据库 URL 以 "sqlite:///" 开头，后面的 "db/users.db" 表示数据库文件存放在当前目录的 "db" 子目录中的 "users.db" 文件。当然，你也可以使用绝对路径，如 "/tmp/users.db" 等。
2.  db 对象是 SQLAlchemy 类的实例，表示程序使用的数据库。
3.  我们定义的 "User" 模型必须继承自 "db.Model"，**这里的模型其实就对应着数据库中的表**。其中，类变量`__tablename__` 定义了在数据库中使用的表名，如果该变量没有被定义，Flask-SQLAlchemy 会使用一个默认名字。

接着，我们创建表和数据库。为此，我们先在当前目录创建 "db" 子目录和新建一个 "users.db" 文件，然后在交互式 Python shell 中导入 db 对象并调用 SQLAlchemy 类的 create_all() 方法：

```py
$ mkdir db && cd db && touch users.db
$ python
>>> from app import db
>>> db.create_all() 
```

我们验证一下，"users" 表是否创建成功：

```py
$ sqlite3 db/users.db    # 打开数据库文件
SQLite version 3.8.10.2 2015-05-20 18:17:19
Enter ".help" for usage hints.

sqlite> .schema users   # 查看 "user" 表的 schema
CREATE TABLE users (
        id INTEGER NOT NULL,
        username VARCHAR(80),
        email VARCHAR(120),
        PRIMARY KEY (id),
        UNIQUE (username),
        UNIQUE (email)
); 
```

### 插入数据

现在，我们创建一些用户：

```py
>>> from app import db
>>> from app import User
>>>
>>> admin = User('admin', 'admin@example.com')
>>> guest = User('guest', 'guest@example.com')
>>> 
>>> db.session.add(admin)
>>> db.session.add(guest)
>>> db.session.commit() 
```

这里有一点要注意的是，我们在将数据添加到会话后，在最后要记得调用 "db.session.commit()" 提交事务，这样，数据才会被写入到数据库。

### 查询数据

查询数据主要是用 "query" 接口，例如 `all()` 方法返回所有数据，`filter_by()` 方法对查询结果进行过滤，参数是键值对，更多方法可以查看[这里](http://flask-sqlalchemy.pocoo.org/2.1/api/)。

```py
>>> from app import User
>>> users = User.query.all()
>>> users
[<User u'admin'>, <User u'guest'>]
>>>
>>> admin = User.query.filter_by(username='admin').first()
>>> admin
<User u'admin'>
>>> admin.email
u'admin@example.com' 
```

如果我们想查看 SQLAlchemy 为查询生成的原生 SQL 语句，只需要把 query 对象转化成字符串：

```py
>>> str(User.query.filter_by(username='guest'))
'SELECT users.id AS users_id, users.username AS users_username, users.email AS users_email \nFROM users \nWHERE users.username = :username_1' 
```

### 更新数据

更新数据也用 "add()" 方法，如果存在要更新的对象，SQLAlchemy 就更新该对象而不是添加。

```py
>>> from app import db
>>> from app import User
>>>
>>> admin = User.query.filter_by(username='admin').first()
>>>
>>> admin.email = 'admin@hotmail.com'
>>> db.session.add(admin)
>>> db.session.commit()
>>>
>>> admin = User.query.filter_by(username='admin').first()
>>> admin.email
u'admin@hotmail.com' 
```

### 删除数据

删除数据用 "delete()" 方法，同样要记得 "delete" 数据后，要调用 "commit()" 提交事务：

```py
>>> from app import db
>>> from app import User
>>>
>>> admin = User.query.filter_by(username='admin').first()
>>> db.session.delete(admin)
>>> db.session.commit() 
```

# RESTful

# REST Web 服务

## 什么是 REST

REST 全称是 Representational State Transfer，翻译成中文是『表现层状态转移』，估计读者看到这个词也是云里雾里的，我当初也是！这里，我们先不纠结这个词到底是什么意思。事实上，REST 是一种 Web 架构风格，它有六条准则，满足下面六条准则的 Web 架构可以说是 Restuful 的。

1.  客户端-服务器（Client-Server）

    服务器和客户端之间有明确的界限。一方面，服务器端不再关注用户界面和用户状态。另一方面，客户端不再关注数据的存储问题。这样，服务器端跟客户端可以独立开发，只要它们共同遵守约定。

2.  无状态（Stateless）

    来自客户端的每个请求必须包含服务器所需要的所有信息，也就是说，服务器端不存储来自客户端的某个请求的信息，这些信息应由客户端负责维护。

3.  可缓存（Cachable）

    服务器的返回内容可以在通信链的某处被缓存，以减少交互次数，提高网络效率。

4.  分层系统（Layered System）

    允许在服务器和客户端之间通过引入中间层（比如代理，网关等）代替服务器对客户端的请求进行回应，而且这些对客户端来说不需要特别支持。

5.  统一接口（Uniform Interface）

    客户端和服务器之间通过统一的接口（比如 GET, POST, PUT, DELETE 等）相互通信。

6.  支持按需代码（Code-On-Demand，可选）

    服务器可以提供一些代码（比如 Javascript）并在客户端中执行，以扩展客户端的某些功能。

## 使用 Flask 提供 REST Web 服务

REST Web 服务的核心概念是资源（resources）。资源被 URI（Uniform Resource Identifier, 统一资源标识符）定位，客户端使用 HTTP 协议操作这些资源，我们用一句不是很全面的话来概括就是：URI 定位资源，用 HTTP 动词（GET, POST, PUT, DELETE 等）描述操作。下面列出了 REST 架构 API 中常用的请求方法及其含义：

| HTTP Method | Action | Example |
| --- | --- | --- |
| GET | 从某种资源获取信息 | [`example.com/api/articles`](http://example.com/api/articles) (获取所有文章) |
| GET | 从某个资源获取信息 | [`example.com/api/articles/1`](http://example.com/api/articles/1) (获取某篇文章) |
| POST | 创建新资源 | [`example.com/api/articles`](http://example.com/api/articles) (创建新文章) |
| PUT | 更新资源 | [`example.com/api/articles/1`](http://example.com/api/articles/1) (更新文章) |
| DELETE | 删除资源 | [`example.com/api/articels/1`](http://example.com/api/articels/1) (删除文章) |

### 设计一个简单的 Web Service

现在假设我们要为一个 blog 应用设计一个 Web Service。

首先，我们先明确访问该 Service 的根地址是什么。这里，我们可以这样定义：

```py
http://[hostname]/blog/api/ 
```

然后，我们明确有哪些资源是要公开的。可以知道，我们这个 blog 应用的资源就是 articles。

下一步，我们要明确怎么去操作这些资源，如下所示：

| HTTP Method | URI | Action |
| --- | --- | --- |
| GET | [http://[hostname]/blog/api/articles](http://[hostname]/blog/api/articles) | 获取所有文章列表 |
| GET | [http://[hostname]/blog/api/articles/[article_id](http://[hostname]/blog/api/articles/[article_id)] | 获取某篇文章内容 |
| POST | [http://[hostname]/blog/api/articles](http://[hostname]/blog/api/articles) | 创建一篇新的文章 |
| PUT | [http://[hostname]/blog/api/articles/[article_id](http://[hostname]/blog/api/articles/[article_id)] | 更新某篇文章 |
| DELETE | [http://[hostname]/blog/api/articles/[article_id](http://[hostname]/blog/api/articles/[article_id)] | 删除某篇文章 |

为了简便，我们定义一篇 article 的属性如下：

*   id：文章的 id，Numeric 类型
*   title: 文章的标题，String 类型
*   content: 文章的内容，TEXT 类型

至此，我们基本完成了这个 Web Service 的设计，下面我们就来实现它。

### 使用 Flask 提供 RESTful api

在实现这个 Web 服务之前，我们还有一个问题没有考虑到：我们应该怎么存储我们的数据。毫无疑问，我们应该使用数据库，比如 MySql、MongoDB 等。但是，数据库的存储不是我们这里要讨论的重点，所以我们采用一种偷懒的做法：使用一个内存中的数据结构来代替数据库。

#### GET 方法

下面我们使用 GET 方法获取资源。

```py
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, abort, make_response

app = Flask(__name__)

articles = [
    {
        'id': 1,
        'title': 'the way to python',
        'content': 'tuple, list, dict'
    },
    {
        'id': 2,
        'title': 'the way to REST',
        'content': 'GET, POST, PUT'
    }
]

@app.route('/blog/api/articles', methods=['GET'])
def get_articles():
    """ 获取所有文章列表 """
    return jsonify({'articles': articles})

@app.route('/blog/api/articles/<int:article_id>', methods=['GET'])
def get_article(article_id):
    """ 获取某篇文章 """
    article = filter(lambda a: a['id'] == article_id, articles)
    if len(article) == 0:
        abort(404)

    return jsonify({'article': article[0]})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5632, debug=True) 
```

将上面的代码保存为文件 `app.py`，通过 `python app.py` 启动这个 Web Service。

接下来，我们进行测试。这里，我们采用命令行语句 [curl](https://curl.haxx.se/) 进行测试。

开启终端，敲入如下命令进行测试：

```py
$ curl -i http://localhost:5632/blog/api/articles
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 224
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Tue, 16 Aug 2016 15:21:45 GMT

{
  "articles": [
    {
      "content": "tuple, list, dict",
      "id": 1,
      "title": "the way to python"
    },
    {
      "content": "GET, POST, PUT",
      "id": 2,
      "title": "the way to REST"
    }
  ]
}

$ curl -i http://localhost:5632/blog/api/articles/2
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 101
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Wed, 17 Aug 2016 02:37:48 GMT

{
  "article": {
    "content": "GET, POST, PUT",
    "id": 2,
    "title": "the way to REST"
  }
}

$ curl -i http://localhost:5632/blog/api/articles/3
HTTP/1.0 404 NOT FOUND
Content-Type: application/json
Content-Length: 26
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Wed, 17 Aug 2016 02:32:10 GMT

{
  "error": "Not found"
} 
```

上面，我们分别测试了『获取所有文章列表』、『获取某篇文章』和『获取不存在的文章』这三个功能，结果也正是我们所预料的。

#### POST 方法

下面我们使用 POST 方法创建一个新的资源。在上面的代码中添加以下代码：

```py
from flask import request

@app.route('/blog/api/articles', methods=['POST'])
def create_article():
    if not request.json or not 'title' in request.json:
        abort(400)
    article = {
        'id': articles[-1]['id'] + 1,
        'title': request.json['title'],
        'content': request.json.get('content', '')
    }
    articles.append(article)
    return jsonify({'article': article}), 201 
```

测试如下：

```py
$ curl -i -H "Content-Type: application/json" -X POST -d '{"title":"the way to java"}' http://localhost:5632/blog/api/articles
HTTP/1.0 201 CREATED
Content-Type: application/json
Content-Length: 87
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Wed, 17 Aug 2016 03:07:14 GMT

{
  "article": {
    "content": "",
    "id": 3,
    "title": "the way to java"
  }
} 
```

可以看到，创建一篇新的文章也是很简单的。request.json 保存了请求中的 JSON 格式的数据。如果请求中没有数据，或者数据中没有 title 的内容，我们将会返回一个 "Bad Request" 的 400 错误。如果数据合法（必须要有 title 的字段），我们就会创建一篇新的文章。

#### PUT 方法

下面我们使用 PUT 方法更新文章，继续添加代码：

```py
@app.route('/blog/api/articles/<int:article_id>', methods=['PUT'])
def update_article(article_id):
    article = filter(lambda a: a['id'] == article_id, articles)
    if len(article) == 0:
        abort(404)
    if not request.json:
        abort(400)

    article[0]['title'] = request.json.get('title', article[0]['title'])
    article[0]['content'] = request.json.get('content', article[0]['content'])

    return jsonify({'article': article[0]}) 
```

测试如下：

```py
$ curl -i -H "Content-Type: application/json" -X PUT -d '{"content": "hello, rest"}' http://localhost:5632/blog/api/articles/2
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 98
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Wed, 17 Aug 2016 03:44:09 GMT

{
  "article": {
    "content": "hello, rest",
    "id": 2,
    "title": "the way to REST"
  }
} 
```

可以看到，更新文章也是很简单的，上面我们更新了第 2 篇文章的内容。

#### DELETE 方法

下面我们使用 DELETE 方法删除文章，继续添加代码：

```py
@app.route('/blog/api/articles/<int:article_id>', methods=['DELETE'])
def delete_article(article_id):
    article = filter(lambda t: t['id'] == article_id, articles)
    if len(article) == 0:
        abort(404)
    articles.remove(article[0])
    return jsonify({'result': True}) 
```

测试如下：

```py
$ curl -i -H "Content-Type: application/json" -X DELETE http://localhost:5632/blog/api/articles/2
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 20
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Wed, 17 Aug 2016 03:46:04 GMT

{
  "result": true
}

$ curl -i http://localhost:5632/blog/api/articles
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 125
Server: Werkzeug/0.11.4 Python/2.7.11
Date: Wed, 17 Aug 2016 03:46:09 GMT

{
  "articles": [
    {
      "content": "tuple, list, dict",
      "id": 1,
      "title": "the way to python"
    }
  ]
} 
```

## 附录

### 常见 HTTP 状态码

HTTP 状态码主要有以下几类：

*   1xx —— 元数据

*   2xx —— 正确的响应

*   3xx —— 重定向

*   4xx —— 客户端错误

*   5xx —— 服务端错误

常见的 HTTP 状态码可见以下表格：

| 代码 | 说明 |
| --- | --- |
| 100 | Continue。客户端应当继续发送请求。 |
| 200 | OK。请求已成功，请求所希望的响应头或数据体将随此响应返回。 |
| 201 | Created。请求成功，并且服务器创建了新的资源。 |
| 301 | Moved Permanently。请求的网页已永久移动到新位置。 服务器返回此响应（对 GET 或 HEAD 请求的响应）时，会自动将请求者转到新位置。 |
| 302 | Found。服务器目前从不同位置的网页响应请求，但请求者应继续使用原有位置来进行以后的请求。 |
| 400 | Bad Request。服务器不理解请求的语法。 |
| 401 | Unauthorized。请求要求身份验证。 对于需要登录的网页，服务器可能返回此响应。 |
| 403 | Forbidden。服务器拒绝请求。 |
| 404 | Not Found。服务器找不到请求的网页。 |
| 500 | Internal Server Error。服务器遇到错误，无法完成请求。 |

### curl 命令参考

| 选项 | 作用 |
| --- | --- |
| -X | 指定 HTTP 请求方法，如 POST，GET, PUT |
| -H | 指定请求头，例如 Content-type:application/json |
| -d | 指定请求数据 |
| --data-binary | 指定发送的文件 |
| -i | 显示响应头部信息 |
| -u | 指定认证用户名与密码 |
| -v | 输出请求头部信息 |

### 完整代码

本文的完整代码可在 [Gist](https://gist.github.com/ethan-funny/cf19aa89175055de6517d851759ad743) 查看。

## 参考链接

*   [理解本真的 REST 架构风格](http://www.infoq.com/cn/articles/understanding-restful-style)
*   [基于 Flask 实现 RESTful API | Ross's Page](http://xhrwang.me/2014/12/13/restful-api-by-flask.html)
*   [(译)使用 Flask 实现 RESTful API - nummy 的专栏 - SegmentFault](https://segmentfault.com/a/1190000005642670)
*   [怎样用通俗的语言解释什么叫 REST，以及什么是 RESTful？ - 知乎](https://www.zhihu.com/question/28557115)
*   [What Does RESTful Really Mean? - DZone Integration](https://dzone.com/articles/what-does-restful-really-mean?utm_medium=feed&utm_source=feedpress.me&utm_campaign=Feed:%20dzone)
*   [HTTP 状态码 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/HTTP%E7%8A%B6%E6%80%81%E7%A0%81)
*   [理解 RESTful 架构 - 阮一峰的网络日志](http://www.ruanyifeng.com/blog/2011/09/restful.html)

# 部署

我们这里以项目 [flask-todo-app](https://github.com/ethan-funny/flask-todo-app) 为例，介绍如何将其部署到生产环境，主要有以下几个步骤：

*   创建项目的运行环境
*   使用 Gunicorn 启动 flask 程序
*   使用 supervisor 管理服务器进程
*   使用 Nginx 做反向代理

## 创建项目的运行环境

*   创建 Python 虚拟环境，以便隔离不同的项目
*   安装项目依赖包

```py
$ pip install virtualenvwrapper
$ source /usr/local/bin/virtualenvwrapper.sh
$ mkvirtualenv flask-todo-env   # 创建完后，会自动进入到该虚拟环境，以后可以使用 workon 命令
$ 
(flask-todo-env)$ git clone https://github.com/ethan-funny/flask-todo-app
(flask-todo-env)$ cd flask-todo-app
(flask-todo-env)$ pip install -r requirements.txt 
```

## 使用 Gunicorn 启动 flask 程序

我们在本地调试的时候经常使用命令 `python manage.py runserver` 或者 `python app.py` 等启动 Flask 自带的服务器，但是，Flask 自带的服务器性能无法满足生产环境的要求，因此这里我们采用 [Gunicorn](http://gunicorn.org/) 做 wsgi (Web Server Gateway Interface，Web 服务器网关接口) 容器，假设我们以 root 用户身份进行部署：

```py
(flask-todo-env)$ pip install gunicorn
(flask-todo-env)$ /home/root/.virtualenvs/flask-todo-env/bin/gunicorn -w 4 -b 127.0.0.1:7345 application.app:create_app() 
```

上面的命令中，-w 参数指定了 worker 的数量，-b 参数绑定了地址（包含访问端口）。

需要注意的是，由于我们这里将 Gunicorn 绑定在本机 127.0.0.1，因此它仅仅监听来自服务器自身的连接，也就是我们从外网访问该服务。在这种情况下，我们通常使用一个反向代理来作为外网和 Gunicorn 服务器的中介，而这也是推荐的做法，接下来也会介绍如何使用 nginx 做反向代理。不过，有时为了调试方便，我们可能需要从外网发送请求给 Gunicorn，这时我们可以让 Gunicorn 绑定 0.0.0.0，这样它就会监听来自外网的所有请求。

## 使用 supervisor 管理服务器进程

在上面，我们手动使用命令启动了 flask 程序，当程序挂掉的时候，我们又要再启动一次。另外，当我们想关闭程序的时候，我们需要找到 pid 进程号并 kill 掉。这里，我们采用一种更好的方式来管理服务器进程，我们将 supervisor 安装全局环境下，而不是在当前的虚拟环境：

```py
$ pip install supervisor
$ echo_supervisord_conf > supervisor.conf   # 生成 supervisor 默认配置文件
$ vi supervisor.conf    # 修改 supervisor 配置文件，添加 gunicorn 进程管理 
```

在 `supervisor.conf` 添加以下内容：

```py
[program:flask-todo-env]
directory=/home/root/flask-todo-app
command=/home/root/.virtualenvs/%(program_name)s/bin/gunicorn
  -w 4
  -b 127.0.0.1:7345
  --max-requests 2000
  --log-level debug
  --error-logfile=-
  --name %(program_name)s
  "application.app:create_app()"

environment=PATH="/home/root/.virtualenvs/%(program_name)s/bin"
numprocs=1
user=deploy
autostart=true
autorestart=true
redirect_stderr=true
redirect_stdout=true
stdout_logfile=/home/root/%(program_name)s-out.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=10
stderr_logfile=/home/root/%(program_name)s-err.log
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=10 
```

supervisor 的常用命令如下：

```py
supervisord -c supervisor.conf                             通过配置文件启动 supervisor
supervisorctl -c supervisor.conf status                    查看 supervisor 的状态
supervisorctl -c supervisor.conf reload                    重新载入 配置文件
supervisorctl -c supervisor.conf start [all]|[appname]     启动指定/所有 supervisor 管理的程序进程
supervisorctl -c supervisor.conf stop [all]|[appname]      关闭指定/所有 supervisor 管理的程序进程
supervisorctl -c supervisor.conf restart [all]|[appname]   重启指定/所有 supervisor 管理的程序进程 
```

## 使用 Nginx 做反向代理

将 Nginx 作为反向代理可以处理公共的 HTTP 请求，发送给 Gunicorn 并将响应带回给发送请求的客户端。在 ubuntu 上可以使用 `sudo apt-get install nginx` 安装 nginx，其他系统也类似。

要想配置 Nginx 作为运行在 127.0.0.1:7345 的 Gunicorn 的反向代理，我们可以在 /etc/nginx/sites-enabled 下给应用创建一个文件，不妨称之为 flask-todo-app.com，nginx 的类似配置如下：

```py
# Handle requests to exploreflask.com on port 80
server {
    listen 80;
    server_name flask-todo-app.com;

    # Handle all locations
    location / {
        # Pass the request to Gunicorn
        proxy_pass http://127.0.0.1:7345;

        # Set some HTTP headers so that our app knows where the request really came from
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
} 
```

常用的 nginx 使用命令如下：

```py
$ sudo service nginx start
$ sudo service nginx stop
$ sudo service nginx restart 
```

可以看到，我们上面的部署方式，都是手动部署的，如果有多台服务器要部署上面的程序，那就会是一个恶梦，有一个自动化部署的神器 [Fabric](http://www.fabfile.org/) 可以帮助我们解决这个问题，感兴趣的读者可以了解一下。

## 参考资料

*   [部署 | Flask 之旅](https://spacewander.github.io/explore-flask-zh/14-deployment.html)
*   [python web 部署：nginx + gunicorn + supervisor + flask 部署笔记 - 简书](http://www.jianshu.com/p/be9dd421fb8d)
*   [新手教程：建立网站的全套流程与详细解释 | 谢益辉](http://yihui.name/cn/2009/06/how-to-build-a-website-as-a-dummy/)
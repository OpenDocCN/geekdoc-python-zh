# Flask 扩展系列(八)–用户会话管理

在入门系列第四篇中，我们曾介绍过如果使用会话 Session 保存用户登录状态，同时在进阶系列的第四篇中，我们演示了如何写一个视图装饰器来验证当前请求的用户是否已登陆。其实这些用户登录及会话管理功能基本上是每个应用都必须有的，因此自然会存在一个 Flask 的扩展来帮助你完成这些功能，那就是 Flask-Login，也是本篇要介绍的内容。

### 系列文章

*   Flask 扩展系列(一)–Restful
*   Flask 扩展系列(二)–Mail
*   Flask 扩展系列(三)–国际化 I18N 和本地化 L10N
*   Flask 扩展系列(四)–SQLAlchemy
*   Flask 扩展系列(五)–MongoDB
*   Flask 扩展系列(六)–缓存
*   Flask 扩展系列(七)–表单
*   Flask 扩展系列(八)–用户会话管理
*   Flask 扩展系列(九)–HTTP 认证
*   Flask 扩展系列–自定义扩展

### 安装和启用

遵循标准的 Flask 扩展安装和启用方式，先通过 pip 来安装扩展：

```py
$ pip install Flask-Login
```

接下来创建扩展对象实例：

```py
from flask import Flask
from flask.ext.login import LoginManager

app = Flask(__name__)
login_manager = LoginManager(app)

```

同时，你可以对 LoginManager 对象赋上配置参数：

```py
# 设置登录视图的名称，如果一个未登录用户请求一个只有登录用户才能访问的视图，
# 则闪现一条错误消息，并重定向到这里设置的登录视图。
# 如果未设置登录视图，则直接返回 401 错误。
login_manager.login_view = 'login'
# 设置当未登录用户请求一个只有登录用户才能访问的视图时，闪现的错误消息的内容，
# 默认的错误消息是：Please log in to access this page.。
login_manager.login_message = 'Unauthorized User'
# 设置闪现的错误消息的类别
login_manager.login_message_category = "info"

```

### 编写用户类

使用 Flask-Login 之前，你需要先定义用户类，该类必须实现以下三个属性和一个方法：

1.  属性 **is_authenticated**

当用户登录成功后，该属性为 True。

3.  属性 **is_active**

如果该用户账号已被激活，且该用户已登录成功，则此属性为 True。

5.  属性 **is_anonymous**

是否为匿名用户（未登录用户）。

7.  方法 **get_id()**

每个用户都必须有一个唯一的标识符作为 ID，该方法可以返回当前用户的 ID，这里 ID 必须是 Unicode。

因为每次写个用户类很麻烦，Flask-Login 提供了”UserMixin”类，你可以直接继承它即可：

```py
from flask.ext.login import UserMixin

class User(UserMixin):
    pass

```

### 从会话或请求中加载用户

在编写登录登出视图前，我们要先写一个加载用户对象的方法。它的功能是根据传入的用户 ID，构造一个新的用户类的对象。为了简化范例，我们不引入数据库，而是在列表里定义用户记录。

```py
# 用户记录表
users = [
    {'username': 'Tom', 'password': '111111'},
    {'username': 'Michael', 'password': '123456'}
]

# 通过用户名，获取用户记录，如果不存在，则返回 None
def query_user(username):
    for user in users:
        if user['username'] == username:
            return user

# 如果用户名存在则构建一个新的用户类对象，并使用用户名作为 ID
# 如果不存在，必须返回 None
@login_manager.user_loader
def load_user(username):
    if query_user(username) is not None:
        curr_user = User()
        curr_user.id = username
        return curr_user

```

上述代码中，通过”@login_manager.user_loader”装饰器修饰的方法，既是我们要实现的加载用户对象方法。它是一个回调函数，在每次请求过来后，Flask-Login 都会从 Session 中寻找”user_id”的值，如果找到的话，就会用这个”user_id”值来调用此回调函数，并构建一个用户类对象。因此，没有这个回调的话，Flask-Login 将无法工作。

有一个问题，启用 Session 的话一定需要客户端允许 Cookie，因为 Session ID 是保存在 Cookie 中的，如果 Cookie 被禁用了怎么办？那我们的应用只好通过请求参数将用户信息带过来，一般情况下会使用一个动态的 Token 来表示登录用户的信息。此时，我们就不能依靠”@login_manager.user_loader”回调，而是使用”@login_manager.request_loader”回调。

```py
from flask import request

# 从请求参数中获取 Token，如果 Token 所对应的用户存在则构建一个新的用户类对象
# 并使用用户名作为 ID，如果不存在，必须返回 None
@login_manager.request_loader
def load_user_from_request(request):
    username = request.args.get('token')
    if query_user(username) is not None:
        curr_user = User()
        curr_user.id = username
        return curr_user

```

为了简化代码，上面的例子就直接使用用户名作为 Token 了，实际项目中，大家还是要用一个复杂的算法来验证 Token。

### 登录及登出

一切准备就绪，我们开始实现登录视图：

```py
from flask import render_template, redirect, url_for, flash
from flask.ext.login import login_user

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        user = query_user(username)
        # 验证表单中提交的用户名和密码
        if user is not None and request.form['password'] == user['password']:
            curr_user = User()
            curr_user.id = username

            # 通过 Flask-Login 的 login_user 方法登录用户
            login_user(curr_user)

            # 如果请求中有 next 参数，则重定向到其指定的地址，
            # 没有 next 参数，则重定向到"index"视图
            next = request.args.get('next')
            return redirect(next or url_for('index'))

        flash('Wrong username or password!')
    # GET 请求
    return render_template('login.html')

```

上述代码同之前 Login 视图最大的不同就是你在用户验证通过后，需要调用 Flask-Login 扩展提供的”login_user()”方法来让用户登录，该方法需传入用户类对象。这个”login_user()”方法会帮助你操作用户 Session，并且会在请求上下文中记录用户信息。另外，在具体实现时，建议大家对”next”参数值作验证，避免被 URL 注入攻击。

“login.html”模板很简单，就是显示一个用户名密码的表单：

```py
<!doctype html>
<title>Login Sample</title>
<h1>Login</h1>
{% with messages = get_flashed_messages() %}
    <div>{{ messages[0] }}</div>
{% endwith %}
<form action="{{ url_for('login') }}" method="POST">
    <input type="text" name="username" id="username" placeholder="Username"></input>
    <input type="password" name="password" id="password" placeholder="Password"></input>
    <input type="submit" name="submit"></input>
</form>

```

接下来，让我们写个 index 视图：

```py
from flask.ext.login import current_user, login_required

@app.route('/')
@login_required
def index():
    return 'Logged in as: %s' % current_user.get_id()

```

装饰器”@login_required”就如同我们在进阶系列第四篇中写的一样，确保只有登录用户才能访问这个 index 视图，Flask-Login 帮我们实现了这个装饰器。如果用户未登录，它就会将页面重定向到登录视图，也就是我们在第一节中配置的”login_manager.login_view”的视图。

同时，重定向的地址会自动加上”next”参数，参数的值是当前用户请求的地址，这样，登录成功后就会跳转回当前视图。可以看到我们对于用户登录所需要的操作，这个装饰器基本都实现了，很方便吧！

Flask-Login 还提供了”current_user”代理，可以访问到登录用户的用户类对象。我们在模板中也可以使用这个代理。让我们再写一个 home 视图：

```py
@app.route('/home')
@login_required
def home():
    return render_template('hello.html')

```

模板代码如下：

```py
<!doctype html>
<title>Login Sample</title>
{% if current_user.is_authenticated %}
  <h1>Hello {{ current_user.get_id() }}!</h1>
{% endif %}

```

在上面的模板代码中，我们直接访问了”current_user”对象的属性和方法。

登出视图也很简单，Flask-Login 提供了”logout_user()”方法来帮助你清理用户 Session。

```py
from flask.ext.login import logout_user

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return 'Logged out successfully!'

```

### 自定义未授权访问的处理方法

“@login_required”装饰器对于未登录用户访问的默认处理是重定向到登录视图，如果我们不想它这么做的话，可以自定义处理方法：

```py
@login_manager.unauthorized_handler
def unauthorized_handler():
    return 'Unauthorized'

```

这个”@login_manager.unauthorized_handler”装饰器所修饰的方法就会代替”@login_required”装饰器的默认处理方法。有了上面的代码，当未登录用户访问 index 视图时，页面就会直接返回”Unauthorized”信息。

### Remember Me

在登录视图中，调用”login_user()”方法时，传入”remember=True”参数，即可实现“记住我”功能：

```py
...
            login_user(curr_user, remember=True)
...

```

Flask-Login 是通过在 Cookie 实现的，它会在 Cookie 中添加一个”remember_token”字段来记住之前登录的用户信息，所以禁用 Cookie 的话，该功能将无法工作。

### Fresh 登录

当用户通过账号和密码登录后，Flask-Login 会将其标识为 Fresh 登录，即在 Session 中设置”_fresh”字段为 True。而用户通过 Remember Me 自动登录的话，则不标识为 Fresh 登录。对于”@login_required”装饰器修饰的视图，是否 Fresh 登录都可以访问，但是有些情况下，我们会强制要求用户登录一次，比如修改登录密码，这时候，我们可以用”@fresh_login_required”装饰器来修饰该视图。这样，通过 Remember Me 自动登录的用户，将无法访问该视图：

```py
from flask.ext.login import fresh_login_required

@app.route('/home')
@fresh_login_required
def home():
    return 'Logged in as: %s' % current_user.get_id()

```

### 会话保护

Flask-Login 自动启用会话保护功能。对于每个请求，它会验证用户标识，这个标识是由客户端 IP 地址和 User Agent 的值经 SHA512 编码而来。在用户登录成功时，Flask-Login 就会将这个值保存起来以便后续检查。默认的会话保护模式是”basic”，为了加强安全性，你可以启用强会话保护模式，方法是配置 LoginManager 实例对象中的”session_protection”属性：

```py
login_manager.session_protection = "strong"

```

在”strong”模式下，一旦用户标识检查失败，便会清空所用 Session 内容，并且 Remember Me 也失效。而”basic”模式下，只是将登录标为非 Fresh 登录。你还可以将”login_manager.session_protection”置为 None 来取消会话保护。

#### 更多参考资料

[Flask-Login 的官方文档](http://flask-login.readthedocs.org/en/latest/)
[Flask-Login 的源码](https://github.com/maxcountryman/flask-login)

本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext8.html)
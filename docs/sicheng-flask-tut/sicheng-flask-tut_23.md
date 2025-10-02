# Flask 扩展系列(九)–HTTP 认证

上一篇中，我们介绍了用户会话管理和登录验证，那么对于 HTTP 请求上的认证，比如 Restful API 请求的认证要怎么做呢？因为 Restful API 不保存状态，无法依赖 Cookie 及 Session 来保存用户信息，自然也无法使用 Flask-Login 扩展来实现用户认证。所以这里，我们就要介绍另一个扩展，Flask-HTTPAuth。

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

Flask-HTTPAuth 不属于官方扩展，所以你无法从[官方扩展列表](http://flask.pocoo.org/extensions/)上找到它。不过安装和启用同其他扩展基本一样，先通过 pip 来安装扩展：

```py
$ pip install Flask-HTTPAuth
```

接下来创建扩展对象实例：

```py
from flask import Flask
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth()

```

注意，初始化实例时不需要传入 app 对象，也不需要调用”auth.init_app(app)”注入应用对象。另外，Flask-HTTPAuth 提供了几种不同的 Auth 方法，比如 HTTPBasicAuth，HTTPTokenAuth，MultiAuth 和 HTTPDigestAuth。上例中我们使用了 HTTPBasicAuth，下文中也会分别介绍 HTTPTokenAuth 和 MultiAuth，HTTPDigestAuth 本篇不准备介绍，感兴趣的朋友们可以去查阅[官方文档](http://flask-httpauth.readthedocs.io/en/latest/)。

### 用户名及密码认证

为了方便理解，我们就不引入数据库了，把用户名密码存在一个数组里，我们所要做的，就是实现一个根据用户名获取密码的回调函数：

```py
users = [
    {'username': 'Tom', 'password': '111111'},
    {'username': 'Michael', 'password': '123456'}
]

@auth.get_password
def get_password(username):
    for user in users:
        if user['username'] == username:
            return user['password']
    return None

```

回调函数”get_password()”由装饰器”@auth.get_password”修饰。在函数里，我们根据传入的用户名，返回其密码；如果用户不存在，则返回空。接下来，我们就可以在任一视图函数上，加上”@auth.login_required”装饰器，来表示该视图需要认证：

```py
@app.route('/')
@auth.login_required
def index():
    return "Hello, %s!" % auth.username()

```

启动该应用，当你在浏览器里打开”http://localhost:5000/”，你会发现浏览器跳出了下面的登录框，输入正确的用户名密码（比如上例中的 Tom:111111）后，”Hello Tom!”的字样才会显示出来。
![Authentication Dialog](img/ede0a0787a6fad6fbd209ea18814aa88.jpg)

进入浏览器调试，你会发现认证并没有启用 Cookie，而是在请求头中加上了加密后的认证字段：

```py
Authorization: Basic TWljaGFlbDoxMjM0NTY=
```

这就是”HTTPBasicAuth”认证的功能，你也可以用 Curl 命令来测试：

```py
curl -u Tom:111111 -i -X GET http://localhost:5000/
```

#### 非明文密码

上例中”@auth.get_password”回调只对明文的密码有效，但是大部分情况，我们的密码都是经过加密后才保存的，这时候，我们要使用另一个回调函数”@auth.verify_password”。在演示代码之前，先要介绍 Werkzeug 库里提供的两个方法：

1.  **generate_password_hash**: 对于给定的字符串，生成其加盐的哈希值
2.  **check_password_hash**: 验证传入的哈希值及明文字符串是否相符

这两个方法都在”werkzeug.security”包下。现在，我们要利用这两个方法，来实现加密后的用户名密码验证：

```py
from werkzeug.security import generate_password_hash, check_password_hash

users = [
    {'username': 'Tom', 'password': generate_password_hash('111111')},
    {'username': 'Michael', 'password': generate_password_hash('123456')}
]

@auth.verify_password
def verify_password(username, password):
    for user in users:
        if user['username'] == username:
            if check_password_hash(user['password'], password):
                return True
    return False

```

在”@auth.verify_password”所修饰的回调函数里，我们验证传入的用户名密码，如果正确的话返回 True，否则就返回 False。现在，让我们再次使用浏览器或者 Curl 命令测试一下，是否同上例的结果一样？

#### 错误处理

在之前的例子中，如果未认证成功，服务端会返回 401 状态码及”Unauthorized Access”文本信息。你可以重写错误处理方法，并用”@auth.error_handler”装饰器来修饰它：

```py
from flask import make_response, jsonify

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)

```

有了上面的”unauthorized()”方法后，如果认证未成功，服务端返回 401 状态码，并返回 JSON 信息”{‘error’: ‘Unauthorized access’}”。

### 令牌（Token）认证

在对 HTTP 形式的 API 发请求时，大部分情况我们不是通过用户名密码做验证，而是通过一个令牌，也就是 Token 来做验证。此时，我们就要请出 Flask-HTTPAuth 扩展中的 HTTPTokenAuth 对象。

同 HTTPBasicAuth 类似，它也提供”login_required”装饰器来认证视图函数，”error_handler”装饰器来处理错误。区别是，它没有”verify_password”装饰器，相应的，它提供了”verify_token”装饰器来验证令牌。我们来看下代码，为了简化，我们将 Token 与用户的关系保存在一个字典中：

```py
from flask import Flask, g
from flask_httpauth import HTTPTokenAuth

app = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    "secret-token-1": "John",
    "secret-token-2": "Susan"
}

@auth.verify_token
def verify_token(token):
    g.user = None
    if token in tokens:
        g.user = tokens[token]
        return True
    return False

@app.route('/')
@auth.login_required
def index():
    return "Hello, %s!" % g.user

```

可以看到，在”verify_token()”方法里，我们验证传入的 Token 是否合法，是的话返回 True，否则返回 False。另外，我们通过 Token 获取了用户信息，并保存在全局变量 g 中，这样视图中可以获取它。注意，在第一节的例子中，我们使用了”auth.username()”来获取用户名，但这里不支持。

此外，细心的朋友们可能注意到，初始化 HTTPTokenAuth 对象时，我们传入了”scheme=’Bearer'”。这个 scheme，就是我们在发送请求时，在 HTTP 头”Authorization”中要用的 scheme 字段。

让我们启动上面的代码，并用 Curl 命令来测试它：

```py
curl -X GET -H "Authorization: Bearer secret-token-1" http://localhost:5000/
```

注意到这里的 HTTP 头信息”Authorization: Bearer secret-token-1″了吗？”Bearer”就是指定的 scheme，”secret-token-1″就是待验证的 Token。在上例中，”secret-token-1″对应着用户名”John”，所以 Token 验证成功，Curl 命令会返回响应内容”Hello, John!”。

#### 使用 itsdangerous 库来管理令牌

itsdangerous 库提供了对信息加签名（Signature）的功能，我们可以通过它来生成并验证令牌。使用前，先记得安装”pip install itsdangerous”。现在，让我们先来产生令牌，并打印出来看看：

```py
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret key here'
serializer = Serializer(app.config['SECRET_KEY'], expires_in=1800)

users = ['John', 'Susan']
for user in users:
    token = serializer.dumps({'username': user})
    print('Token for {}: {}\n'.format(user, token))

```

这里我们实例化了一个针对 JSON 的签名序列化对象 serializer，它是有时效性的，30 分钟后序列化后的签名即会失效。让我们运行下程序，在控制台上，会看到类似下面的内容：

```py
Token for John: eyJhbGciOiJIUzI1NiIsImV4cCI6MTQ2MzUzMzY4MCwiaWF0IjoxNDYzNTMxODgwfQ.eyJ1c2VybmFtZSI6IkpvaG4ifQ.ox-64Jbd2ngjQMV198nHYUsJ639KIZS6RJl48tC7-DU

Token for Susan: eyJhbGciOiJIUzI1NiIsImV4cCI6MTQ2MzUzMzY4MCwiaWF0IjoxNDYzNTMxODgwfQ.eyJ1c2VybmFtZSI6IlN1c2FuIn0.lRx6Z4YZMmjCmga7gs84KB44UIadHYRnhOr7b4AAKwo

```

接下来，改写”verify_token()”方法：

```py
@auth.verify_token
def verify_token(token):
    g.user = None
    try:
        data = serializer.loads(token)
    except:
        return False
    if 'username' in data:
        g.user = data['username']
        return True
    return False

```

我们通过序列化对象的”load()”方法，将签名反序列化为 JSON 对象，也就是 Python 里的字典。然后获取字典中的用户名，如果成功则返回 True，否则返回 False。这样，就实现了加密后的令牌认证了，让我们用 Curl 测试一下，还记得刚才控制台上打印出的令牌吗？

```py
curl -X GET -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsImV4cCI6MTQ2MzUzMzY4MCwiaWF0IjoxNDYzNTMxODgwfQ.eyJ1c2VybmFtZSI6IkpvaG4ifQ.ox-64Jbd2ngjQMV198nHYUsJ639KIZS6RJl48tC7-DU" http://localhost:5000/

```

更多关于 itsdangerous 库的内容，可以查阅其[官方文档](https://pythonhosted.org/itsdangerous/)，或者[中文文档](http://itsdangerous.readthedocs.io/en/latest/)。

### 多重认证

Flask-HTTPAuth 扩展还支持几种不同认证的组合，比如上面我们介绍了 HTTPBasicAuth 和 HTTPTokenAuth，我们可以将两者组合在一起，其中任意一个认证通过，即可以访问应用视图。实现起来也很简单，只需将不同的认证实例化为不同的对象，并将其传入 MultiAuth 对象即可。大体代码如下：

```py
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth, MultiAuth

...

basic_auth = HTTPBasicAuth()
token_auth = HTTPTokenAuth(scheme='Bearer')
multi_auth = MultiAuth(basic_auth, token_auth)

...

@basic_auth.verify_password
...

@token_auth.verify_token
...

@basic_auth.error_handler
...

@token_auth.error_handler
...

@app.route('/')
@multi_auth.login_required
def index():
    return 'Hello, %s!' % g.user

```

这里，每个认证都有自己的验证和错误处理函数，不过在视图上，我们使用”@multi_auth.login_required”来实现多重认证。大家可以使用 Curl 命令试验下。

上例中我们略去了很多业务代码，想要完整代码的朋友们可以下载示例。

### RESTFul 扩展集成

还记得本系列第一篇中介绍的 Flask-RESTful 扩展吗？如果能将上面介绍的 HTTP 认证方法，加入到 RESTful API 中去，那该多好啊。别担心，我们有办法。让我们先取回 Flask-RESTful 扩展的示例代码，再把上例中的认证代码，就 HTTPTokenAuth 部分吧，加上去。现在关键时刻到了，使用 Flask-RESTful 扩展时，我们并没有声明视图函数，那该怎么把”@auth.login_required”装饰器加到 API 视图中去呢？我们来看下代码：

```py
...

class User(Resource):
    decorators = [auth.login_required]
    ...

class UserList(Resource):
    decorators = [auth.login_required]
    ...

```

很简单吧，只需要在 Resource 类中，加上”decorators=[…]”的声明，就可以注入视图装饰器，而且可以同时注入多个。完整代码可以参考示例。

#### 更多参考资料

[Flask-HTTPAuth 的官方文档](http://flask-httpauth.readthedocs.io/en/latest/)
[Flask-HTTPAuth 的源码](https://github.com/miguelgrinberg/Flask-HTTPAuth)
pythondoc.com 上的[使用 Flask 设计 RESTful 的认证](http://www.pythondoc.com/flask-restful/third.html)

本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext9.html)
# Flask 入门系列(五)–错误处理及消息闪现

本篇将补充一些 Flask 的基本功能，包括错误处理，URL 重定向，日志功能，还有一个很有趣的消息闪现功能。

### 系列文章

*   Flask 入门系列(一)–Hello World
*   Flask 入门系列(二)–路由
*   Flask 入门系列(三)–模板
*   Flask 入门系列(四)–请求，响应及会话
*   Flask 入门系列(五)–错误处理及消息闪现
*   Flask 入门系列(六)–数据库集成

#### 错误处理

使用”abort()”函数可以直接退出请求，返回错误代码：

```py
from flask import abort

@app.route('/error')
def error():
    abort(404)

```

上例会显示浏览器的 404 错误页面。有时候，我们想要在遇到特定错误代码时做些事情，或者重写错误页面，可以用下面的方法：

```py
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

```

此时，当再次遇到 404 错误时，即会调用”page_not_found()”函数，其返回”404.html”的模板页。第二个参数代表错误代码。

不过，在实际开发过程中，我们并不会经常使用”abort()”来退出，常用的错误处理方法一般都是异常的抛出或捕获。装饰器”@app.errorhandler()”除了可以注册错误代码外，还可以注册指定的异常类型。让我们来自定义一个异常：

```py
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=400):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code

@app.errorhandler(InvalidUsage)
def invalid_usage(error):
    response = make_response(error.message)
    response.status_code = error.status_code
    return response

```

我们在上面的代码中定义了一个异常”InvalidUsage”，同时我们通过装饰器”@app.errorhandler()”修饰了函数”invalid_usage()”，装饰器中注册了我们刚定义的异常类。这也就意味着，一但遇到”InvalidUsage”异常被抛出，这个”invalid_usage()”函数就会被调用。写个路由试一试吧。

```py
@app.route('/exception')
def exception():
    raise InvalidUsage('No privilege to access the resource', status_code=403)

```

#### URL 重定向

重定向”redirect()”函数的使用在上一篇 logout 的例子中已有出现。作用就是当客户端浏览某个网址时，将其导向到另一个网址。常见的例子，比如用户在未登录时浏览某个需授权的页面，我们将其重定向到登录页要求其登录先。

```py
from flask import session, redirect

@app.route('/')
def index():
    if 'user' in session:
        return 'Hello %s!' % session['user']
    else:
        return redirect(url_for('login'), 302)

```

“redirect()”的第二个参数时 HTTP 状态码，可取的值有 301, 302, 303, 305 和 307，默认即 302（为什么没有 304？留给大家去思考）。

#### 日志

提到错误处理，那一定要说到日志。Flask 提供 logger 对象，其是一个标准的 Python Logger 类。修改上例中的”exception()”函数：

```py
@app.route('/exception')
def exception():
    app.logger.debug('Enter exception method')
    app.logger.error('403 error happened')
    raise InvalidUsage('No privilege to access the resource', status_code=403)

```

执行后，你会在控制台看到日志信息。在 debug 模式下，日志会默认输出到标准错误 stderr 中。你可以添加 FileHandler 来使其输出到日志文件中去，也可以修改日志的记录格式，下面演示一个简单的日志配置代码：

```py
server_log = TimedRotatingFileHandler('server.log','D')
server_log.setLevel(logging.DEBUG)
server_log.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))

error_log = TimedRotatingFileHandler('error.log', 'D')
error_log.setLevel(logging.DEBUG)
error_log.setFormatter(logging.Formatter(
    '%(asctime)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

app.logger.addHandler(server_log)
app.logger.addHandler(error_log)

```

上例中，我们在本地目录下创建了两个日志文件，分别是”server.log”记录所有级别日志；”error.log”只记录错误日志。我们分别给两个文件不同的内容格式。另外，我们使用了”TimedRotatingFileHandler”并给了参数”D”，这样日志每天会创建一个新的文件，并将旧文件加日期后缀来归档。

你还可以将错误信息发送邮件。更详细的日志使用可参阅 Python [logging 官方文档](https://docs.python.org/2/library/logging.html)。

#### 消息闪现

“Flask Message”是一个很有意思的功能，一般一个操作完成后，我们都希望在页面上闪出一个消息，告诉用户操作的结果。用户看完后，这个消息就不复存在了。Flask 提供的”flash”功能就是为了这个。我们还是拿用户登录来举例子：

```py
from flask import render_template, request, session, url_for, redirect, flash

@app.route('/')
def index():
    if 'user' in session:
        return render_template('hello.html', name=session['user'])
    else:
        return redirect(url_for('login'), 302)

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        session['user'] = request.form['user']
        flash('Login successfully!')
        return redirect(url_for('index'))
    else:
        return '''
        <form name="login" action="/login" method="post">
            Username: <input type="text" name="user" />
        </form>
        '''

```

上例中，当用户登录成功后，就用”flash()”函数闪出一个消息。让我们找回第三篇中的模板代码，在”layout.html”加上消息显示的部分：

```py
<!doctype html>
<title>Hello Sample</title>
<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul class="flash">
    {% for message in messages %}
      <li>{{ message }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endwith %}
<div class="page">
    {% block body %}
    {% endblock %}
</div>

```

上例中”get_flashed_messages()”函数就会获取我们在”login()”中通过”flash()”闪出的消息。从代码中我们可以看出，闪出的消息可以有多个。模板”hello.html”不用改。运行下试试。登录成功后，是不是出现了一条”Login successfully”文字？再刷新下页面，你会发现文字消失了。你可以通过 CSS 来控制这个消息的显示方式。

“flash()”方法的第二个参数是消息类型，可选择的有”message”, “info”, “warning”, “error”。你可以在获取消息时，同时获取消息类型，还可以过滤特定的消息类型。只需设置”get_flashed_messages()”方法的”with_categories”和”category_filter”参数即可。比如，Python 部分可改为：

```py
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        session['user'] = request.form['user']
        flash('Login successfully!', 'message')
        flash('Login as user: %s.' % request.form['user'], 'info')
        return redirect(url_for('index'))
    ...

```

layout 模板部分可改为：

```py
...
{% with messages = get_flashed_messages(with_categories=true, category_filter=["message","error"]) %}
  {% if messages %}
    <ul class="flash">
    {% for category, message in messages %}
      <li class="{{ category }}">{{ category }}: {{ message }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endwith %}
...

```

运行结果大家就自己试试吧。

本例中的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-5.html)
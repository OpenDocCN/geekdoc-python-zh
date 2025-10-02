# 浅入浅出 Flask 框架：用户会话

2014-06-28

session 用来记录用户的登录状态，一般基于 cookie 实现。

下面是一个简单的示例。

### 建立 Flask 项目

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

### 编辑`HelloWorld/index.py`

内容如下：

```py
from flask import Flask, render_template_string, \
    session, request, redirect, url_for

app = Flask(__name__)

app.secret_key = 'F12Zr47j\3yX R~X@H!jLwf/T'

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/login')
def login():
    page = '''
    <form action="{{ url_for('do_login') }}" method="post">
        <p>name: <input type="text" name="user_name" /></p>
        <input type="submit" value="Submit" />
    </form>
    '''
    return render_template_string(page)

@app.route('/do_login', methods=['POST'])
def do_login():
    name = request.form.get('user_name')
    session['user_name'] = name
    return 'success'

@app.route('/show')
def show():
    return session['user_name']

@app.route('/logout')
def logout():
    session.pop('user_name', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True) 
```

### 代码的含义

`app.secret_key`用于给 session 加密。

在`/login`中将向用户展示一个表单，要求输入一个名字，submit 后将数据以 post 的方式传递给`/do_login`，`/do_login`将名字存放在 session 中。

如果用户成功登录，访问`/show`时会显示用户的名字。此时，打开 firebug 等调试工具，选择 session 面板，会看到有一个 cookie 的名称为`session`。

`/logout`用于登出，通过将`session`中的`user_name`字段 pop 即可。Flask 中的 session 基于字典类型实现，调用 pop 方法时会返回 pop 的键对应的值；如果要 pop 的键并不存在，那么返回值是`pop()`的第二个参数。

另外，使用`redirect()`重定向时，一定要在前面加上`return`。

### 设置 sessin 的有效时间

下面这段代码来自[Is there an easy way to make sessions timeout in flask?](http://stackoverflow.com/questions/11783025/is-there-an-easy-way-to-make-sessions-timeout-in-flask)：

```py
from datetime import timedelta
from flask import session, app

session.permanent = True
app.permanent_session_lifetime = timedelta(minutes=5) 
```

这段代码将 session 的有效时间设置为 5 分钟。

### 相关资料

[class flask.session](http://flask.pocoo.org/docs/api/#flask.session)
[Step 5: The View Functions](http://flask.pocoo.org/docs/tutorial/views/)
[把 session 持久化到其他介质中](http://flask.pocoo.org/snippets/category/sessions/)
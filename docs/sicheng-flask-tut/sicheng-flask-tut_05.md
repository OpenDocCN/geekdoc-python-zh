# Flask 入门系列(六)–数据库集成

转眼，我们要进入本系列的最后一篇了。一个基本的 Web 应用功能其实已经讲完了，现在就让我们引入数据库。简单起见，我们就使用 SQLite3 作为例子。

### 系列文章

*   Flask 入门系列(一)–Hello World
*   Flask 入门系列(二)–路由
*   Flask 入门系列(三)–模板
*   Flask 入门系列(四)–请求，响应及会话
*   Flask 入门系列(五)–错误处理及消息闪现
*   Flask 入门系列(六)–数据库集成

### 集成数据库

既然前几篇都用用户登录作为例子，我们这篇就继续讲登录，只是登录的信息会由数据库来验证。让我们先准备 SQLite 环境吧。

#### 初始化数据库

怎么安装 SQLite 这里就不说了。我们先写个数据库表的初始化 SQL，保存在”init.sql”文件中：

```py
drop table if exists users;
create table users (
  id integer primary key autoincrement,
  name text not null,
  password text not null
);

insert into users (name, password) values ('visit', '111');
insert into users (name, password) values ('admin', '123');

```

运行 sqlite3 命令，初始化数据库。我们的数据库文件就放在”db”子目录下的”user.db”文件中。

```py
$ sqlite3 db/user.db < init.sql
```

#### 配置连接参数

创建配置文件"config.py"，保存配置信息：

```py
#coding:utf8
DATABASE = 'db/user.db'       # 数据库文件位置
DEBUG = True                  # 调试模式
SECRET_KEY = 'secret_key_1'   # 会话密钥

```

在创建 Flask 应用时，导入配置信息：

```py
from flask import Flask
import config

app = Flask(__name__)
app.config.from_object('config')

```

这里也可以用"app.config.from_envvar('FLASK_SETTINGS', silent=True)"方法来导入配置信息，此时程序会读取系统环境变量中"FLASK_SETTINGS"的值，来获取配置文件路径，并加载此文件。如果文件不存在，该语句返回 False。参数"silent=True"表示忽略错误。

#### 建立和释放数据库连接

这里要用到请求的上下文装饰器，我们会在进阶系列的第一篇里详细介绍上下文。

```py
@app.before_request
def before_request():
    g.db = sqlite3.connect(app.config['DATABASE'])

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

```

我们在"before_request()"里建立数据库连接，它会在每次请求开始时被调用；并在"teardown_request()"关闭它，它会在每次请求关闭前被调用。

#### 查询数据库

让我们取回上一篇登录部分的代码，"index()"和"logout()"请求不用修改，在"login()"请求中，我们会查询数据库，验证客户端输入的用户名和密码是否存在：

```py
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        name = request.form['user']
        passwd = request.form['passwd']
        cursor = g.db.execute('select * from users where name=? and password=?', [name, passwd])
        if cursor.fetchone() is not None:
            session['user'] = name
            flash('Login successfully!')
            return redirect(url_for('index'))
        else:
            flash('No such user!', 'error')
            return redirect(url_for('login'))
    else:
        return render_template('login.html')

```

模板中加上"login.html"文件

```py
{% extends "layout.html" %}
{% block body %}
<form name="login" action="/login" method="post">
    Username: <input type="text" name="user" /><br>
    Password: <input type="password" name="passwd" /><br>
    <input type="submit" value="Submit" />
</form>
{% endblock %}

```

终于一个真正的登录验证写完了（前几篇都是假的），打开浏览器登录下吧。因为比较懒，就不写 CSS 美化了，受不了这粗糙界面的朋友们就自己调吧。

到目前为止，Flask 的基础功能已经介绍完了，是否很想动手写个应用啦？其实 Flask 还有更强大的高级功能，之后会在进阶系列里介绍。

本例中的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-6.html)
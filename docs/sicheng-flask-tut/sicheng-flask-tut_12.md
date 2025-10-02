# Flask 进阶系列(七)–应用最佳实践

一个好的应用目录结构可以方便代码的管理和维护，一个好的应用管理维护方式也可以强化程序的可扩展性。在 Flask 的官方文档，和一些网上资料中都给出了 Flask 大型应用最佳实践的建议，虽然各有不同，但是宗旨还是类似的。本篇就按我个人的总结，跟大家聊聊 Flask 应用管理的最佳实践。

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

### 应用目录结构

假定我们的应用主目录是”flask-demo”，首先我们建议每个应用都放在一个独立的包下，假设包名是”myapp”。所以，整个应用的目录结构如下：

```py
flask-demo/
  ├ run.py           # 应用启动程序
  ├ config.py        # 环境配置
  ├ requirements.txt # 列出应用程序依赖的所有 Python 包
  ├ tests/           # 测试代码包
  │   ├ __init__.py 
  │   └ test_*.py    # 测试用例
  └ myapp/
      ├ admin/       # 蓝图目录
      ├ static/
      │   ├ css/     # css 文件目录
      │   ├ img/     # 图片文件目录
      │   └ js/      # js 文件目录
      ├ templates/   # 模板文件目录
      ├ __init__.py    
      ├ forms.py     # 存放所有表单，如果多，将其变为一个包
      ├ models.py    # 存放所有数据模型，如果多，将其变为一个包
      └ views.py     # 存放所有视图函数，如果多，将其变为一个包

```

应用的创建代码放在”__init__.py”中：

```py
from flask import Flask
from myapp.admin import admin
import config

app = Flask(__name__)
app.config.from_object('config')
app.register_blueprint(admin)

from myapp import views

```

我们把创建应用的代码与应用的启动剥离开，并且在应用对象创建之后，再导入视图模块，因为此时视图函数上的”@app.route()”才有效。视图模块包括了所有的视图函数及路由，如果有多个视图模块，则一并导入。在主目录下的”run.py”内，我们才放入应用的启动代码：

```py
from myapp import app

app.run(host='0.0.0.0')

```

此后，应用的启动都是通过执行”run.py”来完成。蓝图里的目录结构跟应用基本上一样，也是”static”目录放所有静态文件，”templates”目录放所有模板文件，”views.py”存放所有视图，”forms.py”存放所有表单，”models.py”存放所有数据模型，我就不画了。蓝图对象的初始化也放在”__init__.py”里，并在初始化后导入蓝图中的视图模块，这样风格可以跟应用保持一致：

```py
from flask import Blueprint

admin = Blueprint('admin', __name__, url_prefix='/admin')

from myapp.admin import views

```

还有一种风格，就是将蓝图的模板和静态文件也放在 myapp 的”templates”和”static”目录下，只是在这些目录下创建与蓝图同名的子目录，比如：

```py
flask-demo/
  ...
  └ myapp/
      ├ admin/       # 蓝图目录
      ├ static/
      │   ├ admin/   # 蓝图 admin 专有的 js, css, 图片文件
      │   ├ css/     # css 文件目录
      │   ├ img/     # 图片文件目录
      │   └ js/      # js 文件目录
      ├ templates/   # 模板文件目录
      │   └ admin/   # 蓝图 admin 的模板文件
      ├ __init__.py    
      ...

```

这样做的好处就是便于资源统一管理。此时蓝图中获取模板文件或静态文件，都需要加个前缀，比如”render_template(‘admin/hello.html’)”。个人觉得，如果你的蓝图只是为了划分模块，蓝图之间重用部分较多，可以用这个方法。如果蓝图之间比较独立，比如用户站点和管理员后台，就建议采用第一种方法。

### 应用工厂 App Factory

Flask 官方建议采用工厂的模式来创建应用。什么是应用工厂呢？让我们对上例中”myapp”下的”__init__.py”文件作如下修改：

```py
from flask import Flask
from flask.ext.mail import Mail
from flask.ext.sqlalchemy import SQLAlchemy
from werkzeug.utils import import_string

mail = Mail()
db = SQLAlchemy()

blueprints = [
    'myapp.main:main',
    'myapp.admin:admin',
]

def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)

    # Load extensions
    mail.init_app(app)
    db.init_app(app)

    # Load blueprints
    for bp_name in blueprints:
        bp = import_string(bp_name)
        app.register_blueprint(bp)

    return app

```

我们不在代码中直接创建应用，而是通过调用”create_app()”方法来返回一个应用对象，这个”create_app()”就是应用工厂方法。在工厂方法里，我们分别加载了配置，扩展和蓝图。这里要注意，因为没有一个全局的 app 对象，所以上一节例子中的应用视图就无法工作，因为无法执行”@app.route()”。怎么办？还好 Flask 有个蓝图功能，我们将主程序里的视图，数据模型，表单等也放到一个蓝图下，这里起名为”main”。创建蓝图时不指定”url_prefix”，这样它的 URL 前缀就是根路径。

```py
from flask import Blueprint

main = Blueprint('main', __name__)

from myapp.main import views

```

然后在视图函数里用蓝图对象路由即可，这个对象在蓝图创建后就可见了：

```py
from myapp.main import main

@main.route('/')
def index():
    return '<h1>Hello World from app factory!</h1>'

```

有了应用工厂后，我们怎么启动应用呢。这个就简单了，我们修改下主目录下的”run.py”文件：

```py
from myapp import create_app
import config

app = create_app('config')

app.run(host='0.0.0.0', debug=True)

```

这样应用就可以被启动了。可能还是有朋友会问，采用应用工厂到底有什么好处呢？主要有两个：

1.  在跑自动测试时，每个测试用例都通过应用工厂来获取各自的应用，这样测试用例之前不会互相污染
2.  方便获取同一应用的多个实例，比如 debug 版和 release 版，并根据需要启用，甚至于同时启用来服务不同的目的

### 多应用组合

上节最后说到了多个应用（在同一 Python 进程中）同时启用。这个怎么做到？我们先来深入下 Flask 中的”app.run()”方法。这个方法本质上是调用了 Werkzeug 中的”werkzeug.serving.run_simple()”方法来启动一个 WSGI 服务器，前面例子中”app.run(host=’0.0.0.0′, debug=True)”等同于调用：

```py
from werkzeug.serving import run_simple

# debug=True means use_reloader=True and use_debugger=True
run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True)

```

我们曾经说过 Flask 就是基于 Werkzeug 和 Jinja2 建立起来的，Werkzeug 帮助 Flask 封装了很多 WSGI 层面的操作。所以，要同时启用多个应用，就要利用 Werkzeug 的方法，大家看下面的例子：

```py
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple
from myapp import create_app
import config

release_app = create_app('config.release')
debug_app = create_app('config.debug')

app = DispatcherMiddleware(release_app, {'/test': debug_app})

run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True)

```

我们将这段代码保存在主目录下的”run_batch.py”中，执行它后你会发现。release 应用挂在了服务器的根路径上，而 debug 应用挂在了服务器的”/test”路径上了。这样，我们就实现了两个应用同时启用。”werkzeug.wsgi.DispatcherMiddleware”是一个调度中间件，它的实例化参数就是一组应用及其挂载路径的键值对，没提供挂载路径就意味着挂到根目录上。

此外，”run_simple()”方法只能用于开发环境，生产环境还是建议大家部署在一个强大的 Web 服务器上。我们会在下篇介绍。

本篇中的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad7.html)
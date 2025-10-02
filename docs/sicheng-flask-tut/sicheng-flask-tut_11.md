# Flask 进阶系列(六)–蓝图(Blueprint)

写进阶系列比入门系列累不少啊，很多地方都需要自己去反复验证，必要时还要翻翻源码，上一个视图写着写着就发现篇幅很长了。还好蓝图比较简单，这篇应该会比较简短，读者们请放心^_^

我们的应用经常会区分用户站点和管理员后台，比如本博客所使用的 WordPress，就有网站和后台两部分。两者虽然都在同一个应用中，但是风格迥异。把它们分成两个应用吧，总有些代码我们想重用；放在一起嘛，耦合度太高，代码不便于管理。所以 Flask 提供了蓝图(Blueprint)功能。蓝图使用起来就像应用当中的子应用一样，可以有自己的模板，静态目录，有自己的视图函数和 URL 规则，蓝图之间互相不影响。但是它们又属于应用中，可以共享应用的配置。对于大型应用来说，我们可以通过添加蓝图来扩展应用功能，而不至于影响原来的程序。不过有一点要注意，目前 Flask 蓝图的注册是静态的，不支持可插拔。

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

### 创建一个蓝图

比较好的习惯是将蓝图放在一个单独的包里，所以让我们先创建一个”admin”子目录，并创建一个空的”__init__.py”表示它是一个 Python 的包。现在我们来编写蓝图，将其存在”admin/admin_module.py”文件里：

```py
from flask import Blueprint

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/')
def index(name):
    return '<h1>Hello, this is admin blueprint</h1>'

```

我们创建了蓝图对象”admin_bp”，它使用起来类似于 Flask 应用的 app 对象，比如，它可以有自己的路由”admin_bp.route()”。初始化 Blueprint 对象的第一个参数’admin’指定了这个蓝图的名称，第二个参数指定了该蓝图所在的模块名，这里自然是当前文件。

接下来，我们在应用中注册该蓝图。在 Flask 应用主程序中，使用”app.register_blueprint()”方法即可：

```py
from flask import Flask
from admin.admin_module import admin_bp

app = Flask(__name__)
app.register_blueprint(admin_bp, url_prefix='/admin')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

“app.register_blueprint()”方法的”url_prefix”指定了这个蓝图的 URL 前缀。现在，访问”http://localhost:5000/admin/”就可以加载蓝图的 index 视图了。

你也可以在创建蓝图对象时指定其 URL 前缀：

```py
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

```

这样注册时就无需指定：

```py
app.register_blueprint(admin_bp)

```

### 蓝图资源

蓝图有自己的目录，它的所有资源都在其目录下。蓝图的资源目录是由创建 Blueprint 对象时传入的模块名”__name__”所在的位置决定的。同时，我们可以指定蓝图自己的模板目录和静态目录。比如我们创建蓝图时传入：

```py
admin_bp = Blueprint('admin', __name__,
                     template_folder='templates',
                     static_folder='static')

```

这样，该蓝图的模板目录就在”admin/templates”下，而静态目录就在”admin/static”下。当然，其实默认值就是这两个位置，不指定也没关系。我们可以用蓝图对象的”root_path”属性获取其主资源路径，”open_resource()”方法访问主资源路径下的某个文件，比如：

```py
# Assume current app is at /home/bjhee/flask-app,
# this will return /home/bjhee/flask-app/admin
print admin_bp.root_path
# Read file /home/bjhee/flask-app/admin/files/info.txt
with admin_bp.open_resource('files/info.txt') as f:
    info = f.read()
print info

```

### 构建 URL

我们曾在入门系列–路由中介绍过构建 URL 的方法”url_for()”。其第一个参数我们称为端点(Endpoint)，一般指向视图函数名或资源名。蓝图的端点名称都要加上蓝图名为前缀，还记得上例的蓝图名是什么吗？对，’admin’，创建 Blueprint 对象时的第一个参数。当我们通过端点名称来获取 URL 时，我们要这样做：

```py
from flask import url_for

url_for('admin.index')                          # return /admin/
url_for('admin.static', filename='style.css')   # return /admin/static/style.css

```

这样才能获得’admin’蓝图下视图或资源的 URL 地址。如果，”url_for()”函数的调用就在本蓝图下，那蓝图名可以省略，但必须留下”.”表示当前蓝图：

```py
url_for('.index')
url_for('.static', filename='style.css')

```

### 蓝图在国际化中的使用

在国际化的站点中，普遍采用的方法是通过 URL 前缀来区分语言，比如”www.abc.com/cn/”是中文主页，”www.abc.com/en/”是英文主页。在 Flask 中怎么实现呢，大家想到的肯定是在路由上加参数。对的，我们来实现下：

```py
@app.route('/<lang_code>/')
def index(lang_code):
    g.lang_code = lang_code
    return '<h1>Index of language %s</h1>' % g.lang_code

@app.route('/<lang_code>/path')
def path(lang_code):
    g.lang_code = lang_code
    return '<h1>Language base URL is %s</h1>' % url_for('index', lang_code=g.lang_code)

```

每个路由都要加”<lang_code>”参数，而且每个视图函数都要将这个参数保存在上下文环境变量中以便其他地方使用，能不能简化呢？让我们创建一个以参数做 URL 前缀的蓝图吧：

```py
from flask import Blueprint, g, url_for

lang_bp = Blueprint('lang', __name__, url_prefix='/<lang_code>')

@lang_bp.route('/')
def index():
    return '<h1>Index of language %s</h1>' % g.lang_code

@lang_bp.route('/path')
def path():
    return '<h1>Language base URL is %s</h1>' % url_for('.index', lang_code=g.lang_code)

```

将上面的代码保存在”lang_module.py”中，然后在应用主程序里注册：

```py
from lang_module import lang_bp

app.register_blueprint(lang_bp)

```

这样做的确省去了每个路由加”<lang_code>”参数的麻烦，但如果有朋友运行了该程序，会发现报错。因为在视图中没有”lang_code”传进来，所以也没地方设置这个”g.lang_code”变量。这里，我们就要用到 URL 预处理器了，让我们回到蓝图代码”lang_module.py”，加上下面的函数：

```py
@lang_bp.url_value_preprocessor
def get_lang_code_from_url(endpoint, view_args):
    g.lang_code = view_args.pop('lang_code')

```

这个”@lang_bp.url_value_preprocessor”装饰器表明，它所装饰的函数，会在视图函数被调用之前，URL 路径被预处理时执行。而且只针对当前蓝图的所有视图有效。它所传入的第二个参数，保存了当前请求 URL 路径上的所有参数的值。所以，上面的”get_lang_code_from_url()”函数就可以在 URL 预处理时，设置”g.lang_code”变量。这样，视图函数中就可以取到”g.lang_code”，而我们的程序也能够正常运行了。

等下，还有可以优化的地方。每次调用”url_for()”来构建路径时，必须给”lang_code”参数赋上值。这个是否也可以统一处理？我们再加上一个函数：

```py
from flask import current_app

@lang_bp.url_defaults
def add_language_code(endpoint, values):
    if 'lang_code' in values or not g.lang_code:
        return
    if current_app.url_map.is_endpoint_expecting(endpoint, 'lang_code'):
        values['lang_code'] = g.lang_code

```

这个”@lang_bp.url_defaults”装饰器所装饰的函数，会在每次调用”url_for()”时执行，也只对当前蓝图内的所有视图有效。它就可以在构建 URL 时，设置 url 规则上参数的默认值，你只需将参数名及其默认值保存在函数的第二个参数 values 里即可。”current_app.url_map.is_endpoint_expecting()”是用来检查当前的端点是否必须提供一个”lang_code”的参数值。因为我们这个蓝图里的所有端点都包含前缀”<lang_code>”，这种情况下”is_endpoint_expecting”检查可以省去，所以上面的函数可以简化为：

```py
@lang_bp.url_defaults
def add_language_code(endpoint, values):
    values.setdefault('lang_code', g.lang_code)

```

现在，我们就可以将视图函数”url_for()”简写为：

```py
@lang_bp.route('/path')
def path():
    return '<h1>Language base URL is %s</h1>' % url_for('.index')

```

本篇中的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad6.html)
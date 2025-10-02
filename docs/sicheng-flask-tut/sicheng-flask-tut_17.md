# Flask 扩展系列(三)–国际化 I18N 和本地化 L10N

在 Jinja2 系列中，我们曾经介绍过 Jinja2 模板的 i18n 扩展，它可以实现模板中内容的本地化翻译。这里，我们要介绍一个 Flask 扩展，Flask-Babel。它同 Jinja2 的 i18n 扩展一样，可以翻译 Jinja2 模板中的内容，以及 Flask 代码中的文字内容。同时它还可以翻译日期格式等等。它也是基于 Babel 和 gettext 等组件实现，有着非常简单友好的 API 接口，便于我们使用。

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

建议通过 pip 安装，简单方便：

```py
$ pip install Flask-Babel
```

我们可以采用下面的方法初始化一个 Flask-Babel 的实例：

```py
from flask import Flask
from flask.ext.babel import Babel

app = Flask(__name__)
babel = Babel(app)

```

### 设置语言和时区

Flask-Babel 提供了两个 Flask 应用配置项：

1.  BABEL_DEFAULT_LOCALE: 应用默认语言，不设置的话即为”en”
2.  BABEL_DEFAULT_TIMEZONE: 应用默认时区，不设置的话即为”UTC”

```py
app.config.update(
    DEBUG=True,
    BABEL_DEFAULT_LOCALE='zh'
)

```

当程序里没指定时，就会采用这些默认设置。那么如何在程序里指定呢？Flask-Babel 提供了两个装饰器”localeselector”和”timezoneselector”，分别用来设置语言和时区：

```py
@babel.localeselector
def get_locale():
    return 'zh'

@babel.timezoneselector
def get_timezone():
    return 'UTC'

```

这里的设置将会覆盖应用配置项中”BABEL_DEFAULT_LOCALE”和”BABEL_DEFAULT_TIMEZONE”。上面的程序不是个好例子，常见的情况是从当前用户会话中，或者从服务器环境中获取语言/时区设置。

装饰器”localeselector”和”timezoneselector”修饰的函数，被调用一次后就会被缓存，也就是不会被多次调用。但是有时候，当切换用户时，我们想从新用户会话中重新获取语言/时区设置，此时可以在登录请求中调用”refresh()”方法清缓存：

```py
from flask.ext.babel import refresh

@app.route('/login')
def login():
    ... # Get new user locale and timezone
    refresh()
    ... # Render response

```

### 在视图和模板中使用翻译

Flask-Babel 封装了 Python 的”gettext()”方法，你可以在视图函数中使用它：

```py
from flask.ext.babel import gettext, ngettext

@app.route('/trans')
@app.route('/trans/<int:num>')
def translate(num=None):
    if num is None:
        return gettext(u'No users')
    return ngettext(u'%(num)d user', u'%(num)d users', num)

```

关于”gettext”和”ngettext”的区别，大家可以参考下这篇文章的”单/复数支持”部分。

同样在模板中，我们也可以使用”gettext()”方法，更简单的我们可以用”_()”方法代替：

```py
<!doctype html>
<title>{{ _('Test Sample') }}</title>
<h1>{{ _('Hello World!') }}</h1>

```

在 Flask 请求中，我们来渲染此模板：

```py
@app.route('/')
def index():
    return render_template('hello.html')

```

现在让我们启动应用，访问上面的视图，验证下程序是否正常运行。大家应该可以看到”gettext()”方法里的文字被显示出来了，目前还没有被翻译。

### 创建本地化翻译文件

在介绍 Jinja2 模板的 i18n 扩展时，我们曾使用 Python 源码中的”pygettext.py”来创建翻译文件。这里，我们用个更方便的，就是 Babel 中的”pybabel”命令。步骤如下：

1.  首先让我们创建一个 Babel 的配置文件，文件名任意，这里我们取名为”babel.cfg”

```py
[python: **.py]
[jinja2: **/templates/**.html]
extensions=jinja2.ext.autoescape,jinja2.ext.with_

```

这个文件告诉”pybabel”要从当前目录及其子目录下所有的”*.py”文件，和 templates 目录及其子目录下所有的”*.html”文件里面搜寻可翻译的文字，即所有调用”gettext()”，”ngettext()”和”_()”方法时传入的字符串。同时它告诉”pybabel”，当前 Jinja2 模板启用了 autoescape 和 with 扩展。

*   接下来，在当前目录下，生成一个名为”messages.pot”的翻译文件模板

```py
$ pybabel extract -F babel.cfg -o messages.pot .
```

打开”messages.pot”，你会发现，上例中”No users”, “Test Sample”等文字都出现在”msgid”项中了，很强大吧。参数”-F”指定了 Babel 配置文件；”-o”指定了输出文件名。

*   修改翻译文件模板

首先记得将”messages.pot”中的”#, fuzzy”注释去掉，有这个注释在，将无法编译 po 文件。然后修改里面的项目信息内容如作者，版本等。

*   创建”.po”翻译文件

```py
$ pybabel init -i messages.pot -d translations -l zh
```

上面的命令就可以创建一个中文的 po 翻译文件了，文件会保存在当前目录下的”translations/zh/LC_MESSAGES”下，文件名为”messages.po”。参数”-i”指定了翻译文件模板；”-d”指定了翻译文件存放的子目录，上例中我们放在”translations”子目录下；”-l”指定了翻译的语言，同样也是第二级子目录的名称”zh”。

*   编辑”.po”翻译文件

打开刚才生成的中文 po 翻译文件，将我们要翻译的内容写入”msgstr”项中，并保存：

```py
#: flask-ext3.py:31
msgid "No users"
msgstr "没有用户"

#: flask-ext3.py:32
msgid "%(num)d user"
msgid_plural "%(num)d users"
msgstr[0] "%(num)d 个用户"

#: templates/hello.html:2
msgid "Test Sample"
msgstr "测试范例"

#: templates/hello.html:3
msgid "Hello World!"
msgstr "世界，你好！"

```

*   最后一步，编译 po 文件，并生成”*.mo”文件

```py
$ pybabel compile -d translations
```

“-d”指定了翻译文件存放的子目录。该命令执行后，”translations”目录下的所有 po 文件都会被编译成 mo 文件。

如果我们当前的语言”locale”已经设置为”zh”了，再次启动应用，访问根视图或者”/trans”视图，你会看到我们的文字都已经是中文的了。

之后，如果代码中的待翻译的文字被更改过，我们需要重新生成”messages.pot”翻译文件模板。此时，要是再通过”pybabel init”命令来创建 po 文件的话，会丢失之前已翻译好的内容，这个损失是很大的，其实我们可以通过下面的方法来更新 po 文件：

```py
$ pybabel update -i messages.pot -d translations
```

“-i”和”-d”参数就不用再解释了。执行”pybabel update”后，原先的翻译会被保留。不过要注意，因为有些字条 pybabel 无法确定，会将其标为”fuzzy”，你要将”fuzzy”注释去掉才能使其起效。

### 格式化日期

Flask-Babel 不仅可以翻译文字，还可以自动翻译日期格式，运行下面的例子：

```py
from flask.ext.babel import format_datetime
from datetime import datetime

@app.route('/now')
def current_time():
    return format_datetime(datetime.now())

```

假设当前系统时间是”2016-3-20 11:38:32″，在 locale 是 en 的情况下，会显示”Mar 20, 2016, 11:39:59 AM”；而在 locale 是 zh 的情况下，会显示”2016 年 3 月 20 日 上午 11:38:32″。

“format_datetime()”方法还可以带第二个参数指定输出格式，如”full”, “short”, “yyyy-MM-dd”等。详细的日期输出格式可参阅[Babel 的官方文档](http://babel.pocoo.org/en/latest/dates.html)。

### 格式化数字

Flask-Babel 提供了”format_number”和”format_decimal”方法来格式化数字，使用方法同上例中的”format_datetime”非常类似，只需传入待格式化的数字即可：

```py
from flask.ext.babel import format_decimal

@app.route('/num')
def get_num():
    return format_decimal(1234567.89)

```

上面的数字，在 locale 是 en 的情况下，会显示”1,234,567.90″；而在 locale 是 de 的情况下，会显示”1.234.567,89″。

### 格式化货币

既然可以格式化数字，自然也少不了货币格式化显示的功能了。我们可以使用”format_currency”方法，它同”format_decimal”的区别是它必须传入两个参数，第二个参数指定了货币类型：

```py
from flask.ext.babel import format_currency

@app.route('/currency')
def currency():
    return format_currency(1234.5, 'CNY')

```

上面的数字”1234.5″，在类型（即第二个参数）是”CNY”的情况下，会显示”￥1,234.50″；而在类型是”USD”的情况下，会显示”US$1,234.50″。

Flask-Babel 还提供了格式化百分数”format_percent”，和格式化科学计数”format_scientific”的方法，这里就不一一介绍了。

关于 Babel 的详细内容可以参考其[官方网站](http://babel.pocoo.org/en/latest/)。本篇中的示例参考了[Flask-Babel 的官方文档](http://pythonhosted.org/Flask-Babel/)和[Flask-Babel 的源码](https://github.com/python-babel/flask-babel)。本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext3.html)
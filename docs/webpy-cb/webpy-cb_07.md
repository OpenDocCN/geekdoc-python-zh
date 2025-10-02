# Templates 模板

# Templetor: web.py 模板系统

## Introduction

web.py 的模板语言叫做 `Templetor`，它能负责将 python 的强大功能传递给模板系统。 在模板中没有重新设计语法，它是类 python 的。 如果你会 python，你可以顺手拈来。

这是一个模板示例:

```py
$def with (name)
Hello $name! 
```

第一行表示模板定义了一个变量 `name`。 第二行中的 `$name` 将会用 `name` 的值来替换。

如果是从 web.py 0.2 升级请看这里 升级 部分。

## 使用模板系统

通用渲染模板的方法：

```py
render = web.template.render('templates')
return render.hello('world') 
```

`render` 方法从模板根目录查找模板文件，`render.hello(..)`表示渲染 hello.html 模板。实际上，系统会在根目录去查找叫 `hello`的所有文件，直到找到匹配的。(事实上他只支持 .html 和 .xml 两种)

除了上面的使用方式，你也可以直接用文件的方式来处理模板 `frender`：

```py
hello = web.template.frender('templates/hello.html')
render hello('world') 
```

直接使用字符串方式：

```py
template = "$def with (name)\nHello $name"
hello = web.template.Template(template)
return hello('world') 
```

## 语法

### 表达式用法

特殊字符 `$` 被用于特殊的 python 表达式。表达式能够被用于一些确定的组合当中 `()` 和 `{}`:

```py
Look, a $string. 
Hark, an ${arbitrary + expression}. 
Gawk, a $dictionary[key].function('argument'). 
Cool, a $(limit)ing. 
```

### 赋值

有时你可能需要定义一个新变量或给一些变量重新赋值，如下：

```py
$ bug = get_bug(id)
<h1>$bug.title</h1>
<div>
    $bug.description
<div> 
```

注意 `$`在赋值变量名称之前要有一个空格，这有区别于常规的赋值用法。

### 过滤

模板默认会使用 `web.websafe` 过滤 html 内容(encodeing 处理)。

```py
>>> render.hello("1 < 2")
"Hello 1 &lt; 2" 
```

不需要过滤可以在 `$` 之后 使用 `:`。示例：

```py
该 Html 内容不会被义
$:form.render() 
```

### 新起一行用法

在行末添加 `\` 代表显示层该内容不会被真实处理成一行。

```py
If you put a backslash \ 
at the end of a line \ 
(like these) \ 
then there will be no newline. 
```

### 转义 `$`

使用 `$$` 可以在输出的时候显示字符 `$`.

```py
Can you lend me $$50? 
```

### 注释

`$#` 是注释指示符。任何以 `$#` 开始的某行内容都被当做注释。

```py
$# this is a comment
Hello $name.title()! $# display the name in title case 
```

### 控制结构

模板系统支持 `for`, `while`, `if`, `elif` 和 `else`。像 python 一样，这里是需要缩进的。

```py
$for i in range(10): 
    I like $i

$for i in range(10): I like $i

$while a:
    hello $a.pop()

$if times > max: 
    Stop! In the name of love. 
$else: 
    Keep on, you can do it. 
```

`for` 循环内的成员变量只在循环内发生可用：

```py
loop.index: the iteration of the loop (1-indexed)
loop.index0: the iteration of the loop (0-indexed)
loop.first: True if first iteration
loop.last: True if last iteration
loop.odd: True if an odd iteration
loop.even: True if an even iteration
loop.parity: "odd" or "even" depending on which is true
loop.parent: the loop above this in nested loops 
```

有时候，他们使用起来很方便：

```py
<table>
$for c in ["a", "b", "c", "d"]:
    <tr class="$loop.parity">
        <td>$loop.index</td>
        <td>$c</td>
    </tr>
</table> 
```

## 其他

### 使用 `def`

可以使用 `$def` 定义一个新的模板函数，支持使用参数。

```py
$def say_hello(name='world'):
    Hello $name!

$say_hello('web.py')
$say_hello() 
```

其他示例：

```py
$def tr(values):
    <tr>
    $for v in values:
        <td>$v</td>
    </tr>

$def table(rows):
    <table>
    $for row in rows:
        $:row
    </table>

$ data = [['a', 'b', 'c'], [1, 2, 3], [2, 4, 6], [3, 6, 9] ]
$:table([tr(d) for d in data]) 
```

### 代码

可以在 `code` 块书写任何 python 代码： $code: x = "you can write any python code here" y = x.title() z = len(x + y)

```py
 def limit(s, width=10):
        """limits a string to the given width"""
        if len(s) >= width:
            return s[:width] + "..."
        else:
            return s

And we are back to template.
The variables defined in the code block can be used here.
For example, $limit(x) 
```

### 使用 `var`

`var` 块可以用来定义模板结果的额外属性：

```py
$def with (title, body)

$var title: $title
$var content_type: text/html

<div id="body">
$body
</div> 
```

以上模板内容的输出结果如下：

```py
>>> out = render.page('hello', 'hello world')
>>> out.title
u'hello'
>>> out.content_type
u'text/html'
>>> str(out)
'\n\n<div>\nhello world\n</div>\n' 
```

## 内置 和 全局

像 python 的任何函数一样，模板系统同样可以使用内置以及局部参数。很多内置的公共方法像 `range`，`min`，`max`等，以及布尔值 `True` 和 `False`，在模板中都是可用的。部分内置和全局对象也可以使用在模板中。

全局对象可以使用参数方式传给模板，使用 `web.template.render`：

```py
import web
import markdown

globals = {'markdown': markdown.markdown}
render = web.template.render('templates', globals=globals) 
```

内置方法是否可以在模板中也是可以被控制的：

```py
# 禁用所有内置方法
render = web.template.render('templates', builtins={}) 
```

## 安全

模板的设计想法之一是允许非高级用户来写模板，如果要使模板更安全，可在模板中禁用以下方法：

*   不安全部分像 `import`，`exec` 等；
*   允许属性开始部分使用 `_`；
*   不安全的内置方法 `open`, `getattr`, `setattr` 等。

如果模板中使用以上提及的会引发异常 `SecurityException`。

## 从 web.py 0.2 升级

新版本大部分兼容早期版本，但仍有部分使用方法会无法运行，看看以下原因：

*   Template output is always storage like `TemplateResult` object, however converting it to `unicode` or `str` gives the result as unicode/string.
*   重定义全局变量将无法正常运行，如果 x 是全局变量下面的写法是无法运行的。

    ```py
     $ x = x + 1 
    ```

以下写法仍被支持，但不被推荐。

*   如果你原来用 `\$` 反转美元字符串， 推荐用 `$$` 替换；
*   如果你有时会修改 `web.template.Template.globals`，建议通过向 `web.template.render` 传变量方式来替换。

# 站点布局模板

### 问题

如何让站点每个页面共享一个整站范围的模板？（在某些框架中，称为模板继承，比如 ASP.NET 中的母版页）

### 方法

我们可以用 base 属性来实现:

```py
render = web.template.render('templates/', base='layout') 
```

现在如果你调用`render.foo()`方法，将会加载`templates/foo.html` 模板，并且它将会被 `templates/layout.html`模板包裹。

"layout.html" 是一个简单模板格式文件，它包含了一个模板变量，如下:

```py
$def with (content)
<html>
<head>
    <title>Foo</title>
</head>
<body>
$:content
</body>
</html> 
```

在某些情况，如果不想使用基本模板，只需要创建一个没有 base 属性的 reander 对象，如下：

```py
render_plain = web.template.render('templates/') 
```

### Tip: 在布局文件（layout.html）中定义的页面标题变量，如何在其他模板文件中赋值，如下:

##### templates/index.html

```py
$var title: This is title.

<h3>Hello, world</h3> 
```

##### templates/layout.html

```py
$def with (content)
<html>
<head>
    <title>$content.title</title>
</head>
<body>
$:content
</body>
</html> 
```

### Tip: 在其他模板中引用 css 文件，如下:

#### templates/login.html

```py
$var cssfiles: static/login.css static/login2.css

hello, world. 
```

#### templates/layout.html

```py
$def with (content)
<html>
<head>
    <title>$content.title</title>

    $if content.cssfiles:
        $for f in content.cssfiles.split():
            <link rel="stylesheet" href="$f" type="text/css" media="screen" charset="utf-8"/>

</head>
<body>
$:content
</body>
</html> 
```

输入的 HTML 代码如下:

```py
<link rel="stylesheet" href="static/login.css" type="text/css" media="screen" charset="utf-8"/>
<link rel="stylesheet" href="static/login2.css" type="text/css" media="screen" charset="utf-8"/> 
```

# 交替风格

### 问题:

你想通过数据集合动态的生成交替背景色的列表.

### 方法:

Give templetor access to the `int` built-in and use modulo to test.

## code.py

```py
web.template.Template.globals['int'] = int 
```

## template.html

```py
<ul>
$var i: 0
$for track in tracks:
    $var i: ${int(self.i) + 1}
    <li class="
    $if int(self.i) % 2:
        odd
    $else:
        even
    ">$track.title</li>
</ul> 
```

## New Templetor

In the new implementation of templetor (which will be the default when version .3 is released), within any template loop you have access to a $loop variable. This works like so:

```py
<ul>
$for foo in foos:
    <li class="$loop.parity">
    $foo
    </li>
</ul> 
```

# Import functions into templates

`Problem`: How can I import a python module in template?

`Solution`:

While you write templates, inevitably you will need to write some functions which is related to display logic only. web.py gives you the flexibility to write large blocks of code, including defining functions, directly in the template using `$code` blocks (if you don't know what is $code block, please read the tutorial for Templator first). For example, the following code block will translate a status code from database to a human readable status message:

```py
def status(c):
    st = {}
    st[0] = 'Not Started'
    st[1] = 'In Progress'
    st[2] = 'Finished'
    return st[c] 
```

As you do more web.py development, you will write more such functions here and there in your templates. This makes the template messy and is a violation of the DRY (Don't Repeat Yourself) principle.

Naturally, you will want to write a module, say *displayLogic.py* and import that module into every templates that needs such functionalities. Unfortunately, `import` is disabled in template for security reason. However, it is easy to solve this problem, you can import any function via the global namespace into the template:

```py
#in your application.py:
def status(c):
    st = {}
    st[0] = 'Not Started'
    st[1] = 'In Progress'
    st[2] = 'Finished'
    return st[c]

render = web.template.render('templates', globals={'stat':status})

#in the template:
$def with(status)
... ...
<div>Status: $stat(status)</div> 
```

Remember that you can import more than one name into the *globals* dict. This trick is also used in importing session variable into template.

# i18n support in template file

## 模板文件中的 i18n 支持

### 问题:

在 web.py 的模板文件中, 如何得到 i18n 的支持?

### Solution:

项目目录结构:

```py
proj/
   |- code.py
   |- i18n/
       |- messages.po
       |- en_US/
            |- LC_MESSAGES/
                   |- messages.po
                   |- messages.mo
   |- templates/
       |- hello.html 
```

文件: proj/code.py

```py
#!/usr/bin/env python
# encoding: utf-8

import web
import gettext

urls = (
    '/.*', 'hello',
    )

# File location directory.
curdir = os.path.abspath(os.path.dirname(__file__))

# i18n directory.
localedir = curdir + '/i18n'

gettext.install('messages', localedir, unicode=True)   
gettext.translation('messages', localedir, languages=['en_US']).install(True)  
render = web.template.render(curdir + '/templates/', globals={'_': _})

class hello:
    def GET(self):
        return render.hello()

# 使用内建的 HTTP 服务器来运行.
app = web.application(urls, globals())
if __name__ == "__main__":
    app.run() 
```

模板文件: proj/templates/hello.html.

```py
$_("Message") 
```

创建一个 locale 目录并使用 python2.6 内建的 pygettext.py 从 python 脚本和模板文件中导出翻译:

```py
shell> cd /path/to/proj/
shell> mkdir -p i18n/en_US/LC_MESSAGES/
shell> python /path/to/pygettext.py -a -v -d messages -o i18n/messages.po *.py templates/*.html
Working on code.py
Working on templates/hello.html 
```

你将会得到 pot file: i18n/messages.po. 它的内容和下面的差不多 ('msgstr'包含了翻译后的信息):

```py
 # 文件 code.py:40
msgid "Message"
msgstr "This is translated message in file: code.py." 
```

拷贝文件'i18n/messages.po'到目录'i18n/en_US/LC_MESSAGES/'下, 然后翻译它. 使用 gettext 包的 msgfmt 工具或者使用 python2.6 内建的'msgfmt.py'文件将一个 pot 文件编译称 mo 文件:

```py
shell> msgfmt -o i18n/en_US/LC_MESSAGES/messages.mo i18n/en_US/LC_MESSAGES/messages.po 
```

运行 web.py 的服务器:

```py
shell> cd /path/to/proj/
shell> python code.py
http://0.0.0.0:8000/ 
```

打开你的浏览器, 比如说 firefox, 然后访问地址: [`192.168.0.3:8000/`](http://192.168.0.3:8000/), 你将会看过翻译过的信息.

# 在 webpy 中使用 Mako 模板引擎

### 问题

如何在 webpy 中使用 Mako 模板引擎?

### 解决方案

首先需要安装 Mako 和 web.py(0.3):[`www.makotemplates.org/`](http://www.makotemplates.org/) 然后尝试下面的代码:

```py
# encoding: utf-8
# File: code.py
import web
from web.contrib.template import render_mako
urls = (
        '/(.*)', 'hello'
        )
app = web.application(urls, globals(), autoreload=True)
# input_encoding and output_encoding is important for unicode
# template file. Reference:
# http://www.makotemplates.org/docs/documentation.html#unicode
render = render_mako(
        directories=['templates'],
        input_encoding='utf-8',
        output_encoding='utf-8',
        )

class hello:
    def GET(self, name):
        return render.hello(name=name)
        # Another way:
        #return render.hello(**locals())

if __name__ == "__main__":
    app.run() 
```

### 模板文件

```py
## File: templates/hello.html

Hello, ${name}. 
```

### 注意:

如果你使用 Apache+mod_wsgi 来部署 webpy 程序, 你也许会在 Apache 错误日志中得到下面的错误信息: [Sat Jun 21 21:56:22 2008] [error] [client 192.168.122.1] TopLevelLookupException: Cant locate template for uri 'index.html'

你必须使用绝对路径指出模板的位置. 你也可以使用相对路径来让它更简单一些:

```py
import os

render = render_mako(
        directories=[os.path.join(os.path.dirname(__file__), 'templates').replace('\\','/'),],
        input_encoding='utf-8',
        output_encoding='utf-8',
        ) 
```

### 参考:

[`code.google.com/p/modwsgi/wiki/ApplicationIssues`](http://code.google.com/p/modwsgi/wiki/ApplicationIssues)

# 在 webpy 中使用 Cheetah 模板引擎

### 问题：

怎样在 webpy 中使用 Cheetah 模板引擎？

### 解决：

您需要先安装 webpy(0.3)和 Cheetah：[`www.cheetahtemplate.org/`](http://www.cheetahtemplate.org/). 然后尝试使用下面的代码段：

```py
# encoding: utf-8
# File: code.py

import web
from web.contrib.template import render_cheetah

render = render_cheetah('templates/')

urls = (
    '/(first)', 'first',
    '/(second)', 'second'
    )

app = web.application(urls, globals(), web.reloader)

class first:
    def GET(self, name):
        # cheetah template takes only keyword arguments,
        # you should call it as:
        #   return render.hello(name=name)
        # Below is incorrect:
        #   return render.hello(name)
        return render.first(name=name)

class second:
    def GET(self, name):
        return render.first(**locals())

if __name__ == "__main__":
    app.run() 
```

模板文件

```py
## File: templates/first.html

hello, $name. 
```

# Use Jinja2 template engine in webpy

### 问题

如何在 web.py 中使用 Jinja2 ([`jinja.pocoo.org/2/`](http://jinja.pocoo.org/2/)) 模板引擎?

### 方案

首先需要安装 Jinja2 和 webpy(0.3), 然后使用下面的代码做测试:

```py
import web
from web.contrib.template import render_jinja

urls = (
        '/(.*)', 'hello'
        )

app = web.application(urls, globals())

render = render_jinja(
        'templates',   # 设置模板路径.
        encoding = 'utf-8', # 编码.
    )

#添加或者修改一些全局方法.
#render._lookup.globals.update(
#       var=newvar,
#       var2=newvar2,
#)

class hello:
    def GET(self, name):
        return render.hello(name=name)

if __name__ == "__main__":
    app.run() 
```

### 模板文件: templates/hello.html

```py
Hello, . 
```

# How to use templates on Google App Engine

## 问题

如何在 Google App Engine 上使用模板

## 解答

web.py templetor 把模板编译成 python 字节码，这需要访问标准库中的 parser 模块。不幸的是，由于安全原因 GAE 禁用了这个模块。

为了克服这个状况，web.py 支持把模板编译成 python 代码，从而避免在 GAE 上使用原来的模板。web.py 确保在应用这种方法的时候模板中的代码不需要任何改变。

为了编译一个文件夹中所有的模板（一旦有模板改动，就需要重新运行），运行：

```py
$ python web/template.py --compile templates 
```

以上命令把 templates/ 目录下的模板文件递归地全部编译，并且生产 `__init__.py`， 'web.template.render` 重新编写过，它将视 templates 为一个 python 模块。
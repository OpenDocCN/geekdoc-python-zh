# 高级应用

# web.ctx

## 问题

如何在代码中得到客户端信息？比如：来源页面(referring page)或是客户端浏览器类型

## 解法

使用 web.ctx 即可。首先讲一点架构的东西：web.ctx 基于 threadeddict 类，又被叫做 ThreadDict。这个类创建了一个类似字典(dictionary-like)的对象，对象中的值都是与线程 id 相对应的。这样做很妙,因为很多用户同时访问系统时，这个字典对象能做到仅为某一特定的 HTTP 请求提供数据(因为没有数据共享，所以对象是线程安全的)

web.ctx 保存每个 HTTP 请求的特定信息，比如客户端环境变量。假设，我们想知道正在访问某页面的用户是从哪个网页跳转而来的：

## 例子

```py
class example:
    def GET(self):
        referer = web.ctx.env.get('HTTP_REFERER', 'http://google.com')
        raise web.seeother(referer) 
```

上述代码用 web.ctx.env 获取 HTTP_REFERER 的值。如果 HTTP＿REFERER 不存在，就会将 google.com 做为默认值。接下来，用户就会被重定向回到之前的来源页面。

web.ctx 另一个特性，是它可以被 loadhook 赋值。例如：当一个请求被处理时，会话(Session)就会被设置并保存在 web.ctx 中。由于 web.ctx 是线程安全的，所以我们可以象使用普通的 python 对象一样，来操作会话(Session)。

## 'ctx'中的数据成员

### Request

*   `environ` 又被写做. `env` – 包含标准 WSGI 环境变量的字典。
*   `home` – 应用的 http 根路径(译注：可以理解为应用的起始网址，协议＋站点域名＋应用所在路径)例：*[`example.org/admin`](http://example.org/admin)*
*   `homedomain` – 应用所在站点(可以理解为协议＋域名) *[`example.org`](http://example.org)*
*   `homepath` – 当前应用所在的路径，例如： */admin*
*   `host` – 主机名（域名）＋用户请求的端口（如果没有的话，就是默认的 80 端口），例如： *example.org*, *example.org:8080*
*   `ip` – 用户的 IP 地址，例如： *xxx.xxx.xxx.xxx*
*   `method` – 所用的 HTTP 方法，例如： *GET*
*   `path` – 用户请求路径，它是基于当前应用的相对路径。在子应用中，匹配外部应用的那部分网址将被去掉。例如：主应用在`code.py`中，而子应用在`admin.py`中。在`code.py`中, 我们将`/admin`关联到`admin.app`。 在`admin.py`中, 将`/stories`关联到`stories`类。在 `stories`中, `web.ctx.path`就是`/stories`, 而非`/admin/stories`。形如： */articles/845*
*   `protocol` – 所用协议，例如： *https*
*   `query` – 跟在'？'字符后面的查询字符串。如果不存在查询参数，它就是一个空字符串。例如： *?fourlegs=good&twolegs=bad*
*   `fullpath` 可以视为 `path + query` – 包含查询参数的请求路径，但不包括'homepath'。例如：*/articles/845?fourlegs=good&twolegs=bad*

### Response

*   `status` – HTTP 状态码（默认是'200 OK') *401 Unauthorized 未经授权*
*   `headers` – 包含 HTTP 头信息(headers)的二元组列表。
*   `output` – 包含响应实体的字符串。

# Application processors

## 问题

如何使用应用处理器，加载钩子(loadhooks)和卸载钩子(unloadhook)？

## 解法

web.py 可以在处理请求之前或之后，通过添加处理器(processor)来完成某些操作。

```py
def my_processor(handler): 
    print 'before handling'
    result = handler() 
    print 'after handling'
    return result

app.add_processor(my_processor) 
```

可以用加载钩子(loadhook)和卸载钩子(unloadhook)的方式来完成同样的操作，它们分别在请求开始之前和结束之后工作。

```py
def my_loadhook():
    print "my load hook"

def my_unloadhook():
    print "my unload hook"

app.add_processor(web.loadhook(my_loadhook))
app.add_processor(web.unloadhook(my_unloadhook)) 
```

你可以在钩子中使用和修改全局变量，比如：web.header()

```py
def my_loadhook():
    web.header('Content-type', "text/html; charset=utf-8")

app.add_processor(web.loadhook(my_loadhook)) 
```

### 提示: 你也可以在钩子中使用 web.ctx 和 web.input() 。

```py
def my_loadhook():
    input = web.input()
    print input 
```

# 如何使用 web.background

*注意！！* web.backgrounder 已转移到 web.py 3.X 实验版本中，不再是发行版中的一部分。你可以在[这里](http://github.com/webpy/webpy/blob/686aafab4c1c5d0e438b4b36fab3d14d121ef99f/experimental/background.py)下载，要把它与 application.py 放置在同一目录下才能正运行。

## 介绍

web.background 和 web.backgrounder 都是 python 装饰器，它可以让某个函式在一个单独的 background 线程中运行，而主线程继续处理当前的 HTTP 请求，并在稍后报告 background 线程的状态(事实上，后台函式的标准输出(stdout)被返回给启动该线程的"backrounder")。 译注：我本来想将 background thread 翻译为后台线程，后来认为作者本意是想表达“被 background 修饰的函式所在的线程”，最后翻译采用“background 线程”

这样，服务器就可以在处理其他 http 请求的同时，快速及时地响应当前客户端请求。同时，background 线程继续执行需要长时间运行的函式。

## 例子

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from web import run, background, backgrounder
from datetime import datetime; now = datetime.now
from time import sleep

urls = (
    '/', 'index',
    )

class index:
    @backgrounder
    def GET(self):
        print "Started at %s" % now()
        print "hit f5 to refresh!"
        longrunning()

@background
def longrunning():
    for i in range(10):
        sleep(1)
        print "%s: %s" % (i, now())

if __name__ == '__main__':
    run(urls, globals()) 
```

在请求[`localhost:8080/时，将自动重定向到类似 http://localhost:8080/?_t=3080772748 的网址(t 后面的数字就是 background 线程 id)，接下来(在点击几次刷新之后)就会看到如下信息：`](http://localhost:8080/时，将自动重定向到类似 http://localhost:8080/?_t=3080772748 的网址(t 后面的数字就是 background 线程 id)，接下来(在点击几次刷新之后)就会看到如下信息：)

```py
Started at 2008-06-14 15:50:26.764474
hit f5 to refresh!
0: 2008-06-14 15:50:27.763813
1: 2008-06-14 15:50:28.763861
2: 2008-06-14 15:50:29.763844
3: 2008-06-14 15:50:30.763853
4: 2008-06-14 15:50:31.764778
5: 2008-06-14 15:50:32.763852
6: 2008-06-14 15:50:33.764338
7: 2008-06-14 15:50:34.763925
8: 2008-06-14 15:50:35.763854
9: 2008-06-14 15:50:36.763789 
```

## 提示

web.py 在 background.threaddb 字典中保存线程信息。这就很容易检查线程的状态；

```py
class threaddbviewer:
    def GET(self):
        for k, v in background.threaddb.items():
            print "%s - %s" % ( k, v ) 
```

web.py 并不会主动去清空 threaddb 词典，这使得输出(如[`localhost:8080/?_t=3080772748)会一直执行，直到内存被用满。`](http://localhost:8080/?_t=3080772748)会一直执行，直到内存被用满。)

通常是在 backgrounder 函式中做线程清理工作，是因为 backgrounder 可以获得线程 id(通过 web.input()得到"_t"的值，就是线程 id)，从而根据线程 id 来回收资源。这是因为虽然 background 能知道自己何时结束，但它无法获得自己的线程 id，所以 background 无法自己完成线程清理。

还要注意 [How not to do thread local storage with Python 在 python 中如何避免多线程本地存储](http://blogs.gnome.org/jamesh/2008/06/11/tls-python/) - 线程 ID 有时会被重用(可能会引发错误)

在使用 web.background 时，还是那句话－－“小心为上”

# 自定义 NotFound 消息

## 问题

如何定义 NotFound 消息和其他消息？

## 解法

```py
import web

urls = (...)
app =  web.application(urls, globals())

def notfound():
    return web.notfound("Sorry, the page you were looking for was not found.")

    # You can use template result like below, either is ok:
    #return web.notfound(render.notfound())
    #return web.notfound(str(render.notfound()))

app.notfound = notfound 
```

要返回自定义的 NotFound 消息，这么做即可：

```py
class example:
    def GET(self):
        raise web.notfound() 
```

也可以用同样的方法自定义 500 错误消息：

```py
def internalerror():
    return web.internalerror("Bad, bad server. No donut for you.")

app.internalerror = internalerror 
```

# 如何流传输大文件

### 问题

如何流传输大文件？

### 解法

要流传输大文件，需要添加传输译码(Transfer-Encoding)区块头，这样才能一边下载一边显示。否则，浏览器将缓冲所有数据直到下载完毕才显示。

如果这样写：直接修改基础字符串(例中就是 j)，然后用 Yield 返回－－是没有效果的。如果要使用 Yield,就要向对所有内容使用 yield。因为这个函式此时是一个产生器。(注：请处请详看 Yield 文档，在此不做过多论述。)

例子

```py
# Simple streaming server demonstration
# Uses time.sleep to emulate a large file read
import web
import time

urls = (
    "/",    "count_holder",
    "/(.*)",  "count_down",
    )
app = web.application(urls, globals())

class count_down:
    def GET(self,count):
        # These headers make it work in browsers
        web.header('Content-type','text/html')
        web.header('Transfer-Encoding','chunked')        
        yield '<h2>Prepare for Launch!</h2>'
        j = '<li>Liftoff in %s...</li>'
        yield '<ul>'
        count = int(count)
        for i in range(count,0,-1):
            out = j % i
            time.sleep(1)
            yield out
        yield '</ul>'
        time.sleep(1)
        yield '<h1>Lift off</h1>'

class count_holder:
    def GET(self):
        web.header('Content-type','text/html')
        web.header('Transfer-Encoding','chunked')        
        boxes = 4
        delay = 3
        countdown = 10
        for i in range(boxes):
            output = '<iframe src="/%d" width="200" height="500"></iframe>' % (countdown - i)
            yield output
            time.sleep(delay)

if __name__ == "__main__":
    app.run() 
```

# 管理自带 webserver 日志

## 问题

如何操作 web.py 自带的 webserver 的日志？

## 解法

我们可以用[wsgilog](http://pypi.python.org/pypi/wsgilog/)来操作内置的 webserver 的日志，并做其为中间件加到应用中。

如下，写一个 Log 类继承 wsgilog.WsgiLog，在*init*中把参数传给基类，如[这个例子](http://github.com/harryf/urldammit/blob/234bcaae6deb65240e64ee3199213712ed62883a/dammit/log.py)：

```py
import sys, logging
from wsgilog import WsgiLog, LogIO
import config

class Log(WsgiLog):
    def __init__(self, application):
        WsgiLog.__init__(
            self,
            application,
            logformat = '%(message)s',
            tofile = True,
            file = config.log_file,
            interval = config.log_interval,
            backups = config.log_backups
            )
        sys.stdout = LogIO(self.logger, logging.INFO)
        sys.stderr = LogIO(self.logger, logging.ERROR) 
```

接下来，当应用运行时，传递一个引用给上例中的 Log 类即可(假设上面代码是'mylog'模块的一部分，代码如下)：

```py
from mylog import Log
application = web.application(urls, globals())
application.run(Log) 
```

# 用 cherrypy 提供 SSL 支持

## 问题

如何用内置的 cheerypy 提供 SSL 支持？

## 解法

```py
import web

from web.wsgiserver import CherryPyWSGIServer

CherryPyWSGIServer.ssl_certificate = "path/to/ssl_certificate"
CherryPyWSGIServer.ssl_private_key = "path/to/ssl_private_key"

urls = ("/.*", "hello")
app = web.application(urls, globals())

class hello:
    def GET(self):
        return 'Hello, world!'

if __name__ == "__main__":
    app.run() 
```

# 实时语言切换

## 问题:

如何实现实时语言切换？

## 解法:

*   首先你必须阅读 模板语言中的 i18n 支持, 然后尝试下面的代码。

文件: code.py

```py
import os
import sys
import gettext
import web

# File location directory.
rootdir = os.path.abspath(os.path.dirname(__file__))

# i18n directory.
localedir = rootdir + '/i18n'

# Object used to store all translations.
allTranslations = web.storage()

def get_translations(lang='en_US'):
    # Init translation.
    if allTranslations.has_key(lang):
        translation = allTranslations[lang]
    elif lang is None:
        translation = gettext.NullTranslations()
    else:
        try:
            translation = gettext.translation(
                    'messages',
                    localedir,
                    languages=[lang],
                    )
        except IOError:
            translation = gettext.NullTranslations()
    return translation

def load_translations(lang):
    """Return the translations for the locale."""
    lang = str(lang)
    translation  = allTranslations.get(lang)
    if translation is None:
        translation = get_translations(lang)
        allTranslations[lang] = translation

        # Delete unused translations.
        for lk in allTranslations.keys():
            if lk != lang:
                del allTranslations[lk]
    return translation

def custom_gettext(string):
    """Translate a given string to the language of the application."""
    translation = load_translations(session.get('lang'))
    if translation is None:
        return unicode(string)
    return translation.ugettext(string)

urls = (
'/', 'index'
)

render = web.template.render('templates/',
        globals={
            '_': custom_gettext,
            }
        )

app = web.application(urls, globals())

# Init session.
session = web.session.Session(app,
        web.session.DiskStore('sessions'),
        initializer={
            'lang': 'en_US',
            }
        )

class index:
    def GET(self):
        i = web.input()
        lang = i.get('lang', 'en_US')

        # Debug.
        print >> sys.stderr, 'Language:', lang

        session['lang'] = lang
        return render.index()

if __name__ == "__main__": app.run() 
```

模板文件: templates/index.html.

```py
$_('Hello') 
```

不要忘记生成必要的 po&mo 语言文件。参考: 模板语言中的 i18n 支持

现在运行 code.py:

```py
$ python code.py
http://0.0.0.0:8080/ 
```

然后用你喜欢的浏览器访问下面的地址，检查语言是否改变:

```py
http://your_server:8080/
http://your_server:8080/?lang=en_US
http://your_server:8080/?lang=zh_CN 
```

你必须:

*   确保语言文件(en_US、zh_CN 等)可以动态改变。
*   确保 custom_gettext()调用越省资源约好。

参考:

*   这里有使用 app.app_processor()的 [另一个方案](http://groups.google.com/group/webpy/browse_thread/thread/a215837aa30e8f80)。
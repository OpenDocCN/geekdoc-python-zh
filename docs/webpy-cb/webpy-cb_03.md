# 基本应用

# Hello World!

## 问题

如何用 web.py 实现 Hello World!？

## 解法

```py
import web

urls = ("/.*", "hello")
app = web.application(urls, globals())

class hello:
    def GET(self):
        return 'Hello, world!'

if __name__ == "__main__":
    app.run() 
```

### 提示：要保证网址有无'/'结尾，都能指向同一个类。就要多写几行代码，如下：

在 URL 开头添加代码：

```py
'/(.*)/', 'redirect', 
```

然后用 redirect 类处理以'/'结尾的网址：

```py
class redirect:
    def GET(self, path):
        web.seeother('/' + path) 
```

# 提供静态文件 (诸如 js 脚本, css 样式表和图象文件)

## 问题

如何在 web.py 自带的 web server 中提供静态文件访问？

## 解法

### web.py 服务器

在当前应用的目录下，创建一个名为 static 的目录，把要提供访问的静态文件放在里面即可。

例如, 网址 `http://localhost/static/logo.png` 将发送 `./static/logo.png` 给客户端。

### Apache

在 Apache 中可以使用 [Alias](http://httpd.apache.org/docs/2.2/mod/mod_alias.html#alias) 指令，在处理 web.py 之前将请求映射到指定的目录。

这是一个在 Unix like 系统上虚拟主机配置的例子：

```py
<VirtualHost *:80>
    ServerName example.com:80
    DocumentRoot /doc/root/
    # mounts your application if mod_wsgi is being used
    WSGIScriptAlias / /script/root/code.py
    # the Alias directive
    Alias /static /doc/root/static

    <Directory />
        Order Allow,Deny
        Allow From All
        Options -Indexes
    </Directory>

    # because Alias can be used to reference resources outside docroot, you
    # must reference the directory with an absolute path
    <Directory /doc/root/static>
        # directives to effect the static directory
        Options +Indexes
    </Directory>
</VirtualHost> 
```

# 理解 URL 控制

`问题`: 如何为整个网站设计一个 URL 控制方案 / 调度模式

`解决`:

web.py 的 URL 控制模式是简单的、强大的、灵活的。在每个应用的最顶部，你通常会看到整个 URL 调度模式被定义在元组中:

```py
urls = (
    "/tasks/?", "signin",
    "/tasks/list", "listing",
    "/tasks/post", "post",
    "/tasks/chgpass", "chgpass",
    "/tasks/act", "actions",
    "/tasks/logout", "logout",
    "/tasks/signup", "signup"
) 
```

这些元组的格式是: *URL 路径*, *处理类* 这组定义有多少可以定义多少。如果你并不知道 URL 路径和处理类之间的关系，请在阅读 cookbook 之前先阅读 Hello World example，或者快速入门。

`路径匹配`

你可以利用强大的正则表达式去设计更灵活的 URL 路径。比如 /(test1|test2) 可以捕捉 /test1 或 /test2。要理解这里的关键，匹配是依据 URL 路径的。比如下面的 URL:

```py
http://localhost/myapp/greetings/hello?name=Joe 
```

这个 URL 的路径是 */myapp/greetings/hello*。web.py 会在内部给 URL 路径加上^ 和$ ，这样 */tasks/* 不会匹配 */tasks/addnew*。URL 匹配依赖于“路径”，所以不能这样使用，如： */tasks/delete?name=(.+)* ,?之后部分表示是“查询”，并不会被匹配。阅读 URL 组件的更多细节，请访问 web.ctx。

`捕捉参数`

你可以捕捉 URL 的参数，然后用在处理类中:

```py
/users/list/(.+), "list_users" 
```

在 *list/*后面的这块会被捕捉，然后作为参数被用在 GET 或 POST:

```py
class list_users:
    def GET(self, name):
        return "Listing info about user: {0}".format(name) 
```

你可以根据需要定义更多参数。同时要注意 URL 查询的参数(?后面的内容)也可以用 web.input()取得。

`开发子程序的时候注意`

为了更好的控制大型 web 应用，web.py 支持子程序。在为子程序设计 URL 模式的时候，记住取到的路径(web.ctx.path)是父应用剥离后的。比如，你在主程序定义了 URL"/blog"跳转到'blog'子程序，那没在你 blog 子程序中所有 URL 都是以"/"开头的，而不是"/blog"。查看 web.ctx 取得更多信息。

# 使用子应用

## 问题

如何在当前应用中包含定义在其他文件中的某个应用？

## 解法

在`blog.py`中:

```py
import web
urls = (
  "", "reblog",
  "/(.*)", "blog"
)

class reblog:
    def GET(self): raise web.seeother('/')

class blog:
    def GET(self, path):
        return "blog " + path

app_blog = web.application(urls, locals()) 
```

当前的主应用`code.py`:

```py
import web
import blog
urls = (
  "/blog", blog.app_blog,
  "/(.*)", "index"
)

class index:
    def GET(self, path):
        return "hello " + path

app = web.application(urls, locals())

if __name__ == "__main__":
    app.run() 
```

# 提供 XML 访问

### 问题

如何在 web.py 中提供 XML 访问？

如果需要为第三方应用收发数据，那么提供 xml 访问是很有必要的。

### 解法

根据要访问的 xml 文件(如 response.xml)创建一个 XML 模板。如果 XML 中有变量，就使用相应的模板标签进行替换。下面是一个例子：

```py
$def with (code)
<?xml version="1.0"?>
<RequestNotification-Response>
<Status>$code</Status>
</RequestNotification-Response> 
```

为了提供这个 XML，需要创建一个单独的 web.py 程序(如 response.py)，它要包含下面的代码。注意：要用"web.header('Content-Type', 'text/xml')"来告知客户端－－正在发送的是一个 XML 文件。

```py
import web

render = web.template.render('templates/', cache=False)

urls = (
    '/(.*)', 'index'
)

app = web.application(urls, globals())

class index:
    def GET(self, code):
        web.header('Content-Type', 'text/xml')
        return render.response(code)

web.webapi.internalerror = web.debugerror
if __name__ == '__main__': app.run() 
```

# 从 post 读取原始数据

## 介绍

有时候，浏览器会通过 post 发送很多数据。在 webpy，你可以这样操作。

## 代码

```py
class RequestHandler():
    def POST():
        data = web.data() # 通过这个方法可以取到数据 
```
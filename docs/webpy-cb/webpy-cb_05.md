# Sessions and user state 会话和用户状态

# Sessions

### 问题

如何在 web.py 中使用 session

### 解法

*注意！！！：session 并不能在调试模式(Debug mode)下正常工作，这是因为 session 与调试模试下的重调用相冲突(有点类似 firefox 下著名的 Firebug 插件，使用 Firebug 插件分析网页时，会在火狐浏览器之外单独对该网页发起请求，所以相当于同时访问该网页两次)，下一节中我们会给出在调试模式下使用 session 的解决办法。*

`web.session`模块提供 session 支持。下面是一个简单的例子－－统计有多少人正在使用 session(session 计数器)：

```py
import web
web.config.debug = False
urls = (
    "/count", "count",
    "/reset", "reset"
)
app = web.application(urls, locals())
session = web.session.Session(app, web.session.DiskStore('sessions'), initializer={'count': 0})

class count:
    def GET(self):
        session.count += 1
        return str(session.count)

class reset:
    def GET(self):
        session.kill()
        return ""

if __name__ == "__main__":
    app.run() 
```

web.py 在处理请求之前，就加载 session 对象及其数据；在请求处理完之后，会检查 session 数据是否被改动。如果被改动，就交由 session 对象保存。

上例中的`initializer`参数决定了 session 初始化的值，它是个可选参数。

如果用数据库代替磁盘文件来存储 session 信息，只要用`DBStore`代替`DiskStore`即可。使用 DBStore 需要建立一个表，结构如下：

```py
 create table sessions (
    session_id char(128) UNIQUE NOT NULL,
    atime timestamp NOT NULL default current_timestamp,
    data text
); 
```

`DBStore`被创建要传入两个参数：`db`对象和 session 的表名。

```py
db = web.database(dbn='postgres', db='mydatabase', user='myname', pw='')
store = web.session.DBStore(db, 'sessions')
session = web.session.Session(app, store, initializer={'count': 0}) 
```

｀web.config｀中的`sessions_parameters`保存着 session 的相关设置，`sessions_parameters`本身是一个字典，可以对其修改。默认设置如下：

```py
web.config.session_parameters['cookie_name'] = 'webpy_session_id'
web.config.session_parameters['cookie_domain'] = None
web.config.session_parameters['timeout'] = 86400, #24 * 60 * 60, # 24 hours   in seconds
web.config.session_parameters['ignore_expiry'] = True
web.config.session_parameters['ignore_change_ip'] = True
web.config.session_parameters['secret_key'] = 'fLjUfxqXtfNoIldA0A0J'
web.config.session_parameters['expired_message'] = 'Session expired' 
```

*   cookie_name - 保存 session id 的 Cookie 的名称
*   cookie_domain - 保存 session id 的 Cookie 的 domain 信息
*   timeout - session 的有效时间 ，以秒为单位
*   ignore_expiry - 如果为 True，session 就永不过期
*   ignore_change_ip - 如果为 False，就表明只有在访问该 session 的 IP 与创建该 session 的 IP 完全一致时，session 才被允许访问。
*   secret_key - 密码种子，为 session 加密提供一个字符串种子
*   expired_message - session 过期时显示的提示信息。

# 在调试模式下使用 session

## 问题

如何在调试模式下使用 session?

## 解法

使用 web.py 自带的 webserver 提供 web 服务时，web.py 就运行在调试模式下。当然最简单的办法就是禁用调试，只要令`web.config.debug = False`即可。

```py
import web
web.config.debug = False

# rest of your code 
```

如果非要用调试模式下使用 session，可以用非主流的一些办法。哈哈

因为调试模式支持模块重载入(重载入，绝非重载。是 reload,而非 override)，所以 reloader 会载入主模块两次，因此，就会创建两个 session 对象。但我们只要把 session 存储在全局的数据容器中，就能避免二次创建 session。

下面这个例子就是把 session 保存在 `web.config`中：

```py
import web
urls = ("/", "hello")

app = web.application(urls, globals())

if web.config.get('_session') is None:
    session = web.session.Session(app, web.session.DiskStore('sessions'), {'count': 0})
    web.config._session = session
else:
    session = web.config._session

class hello:
   def GET(self):
       print 'session', session
       session.count += 1
       return 'Hello, %s!' % session.count

if __name__ == "__main__":
   app.run() 
```

# 在 template 中使用 session

`问题`: 我想在模板中使用 session（比如：读取并显示 session.username）

`解决`:

在应用程序中的代码:

```py
render = web.template.render('templates', globals={'context': session}) 
```

在模板中的代码:

```py
<span>You are logged in as <b>$context.username</b></span> 
```

你可以真正的使用任何符合语法的 python 变量名，比如上面用的*context*。我更喜欢在应用中直接使用'session'。

# 如何操作 Cookie

## 问题

如何设置和获取用户的 Cookie?

## 解法

对 web.py 而言，设置/获取 Cookie 非常方便。

### 设置 Cookies

#### 概述

```py
setcookie(name, value, expires="", domain=None, secure=False): 
```

*   *name* `(string)` - Cookie 的名称，由浏览器保存并发送至服务器。
*   *value* `(string)` -Cookie 的值，与 Cookie 的名称相对应。
*   *expires* `(int)` - Cookie 的过期时间，这是个可选参数，它决定 cookie 有效时间是多久。以秒为单位。它必须是一个整数，而绝不能是字符串。
*   *domain* `(string)` - Cookie 的有效域－在该域内 cookie 才是有效的。一般情况下，要在某站点内可用，该参数值该写做站点的域（比如.webpy.org），而不是站主的主机名（比如 wiki.webpy.org）
*   *secure* `(bool)`- 如果为 True，要求该 Cookie 只能通过 HTTPS 传输。.

#### 示例

用`web.setcookie()` 设置 cookie,如下:

```py
class CookieSet:
    def GET(self):
        i = web.input(age='25')
        web.setcookie('age', i.age, 3600)
        return "Age set in your cookie" 
```

用 GET 方式调用上面的类将设置一个名为 age,默认值是 25 的 cookie(实际上，默认值 25 是在 web.input 中赋予 i.age 的，从而间接赋予 cookie，而不是在 setcookie 函式中直接赋予 cookie 的)。这个 cookie 将在一小时后(即 3600 秒)过期。

`web.setcookie()`的第三个参数－"expires"是一个可选参数，它用来设定 cookie 过期的时间。如果是负数，cookie 将立刻过期。如果是正数，就表示 cookie 的有效时间是多久，以秒为单位。如果该参数为空，cookie 就永不过期。

### 获得 Cookies

#### 概述

获取 Cookie 的值有很多方法，它们的区别就在于找不到 cookie 时如何处理。

##### 方法 1（如果找不到 cookie，就返回 None）：

```py
web.cookies().get(cookieName)  
    #cookieName is the name of the cookie submitted by the browser 
```

##### 方法 2（如果找不到 cookie，就抛出 AttributeError 异常）：

```py
foo = web.cookies()
foo.cookieName 
```

##### 方法 3（如果找不到 cookie，可以设置默认值来避免抛出异常）：

```py
foo = web.cookies(cookieName=defaultValue)
foo.cookieName   # return the value (which could be default)
    #cookieName is the name of the cookie submitted by the browser 
```

#### 示例：

用`web.cookies()` 访问 cookie. 如果已经用`web.setcookie()`设置了 Cookie, 就可以象下面这样获得 Cookie:

```py
class CookieGet:
    def GET(self):
        c = web.cookies(age="25")
        return "Your age is: " + c.age 
```

这个例子为 cookie 设置了默认值。这么做的原因是在访问时，若 cookie 不存在，web.cookies()就会抛出异常，如果事先设置了默认值就不会出现这种情况。

如果要确认 cookie 值是否存在，可以这样做：

```py
class CookieGet:
    def GET(self):
        try: 
             return "Your age is: " + web.cookies().age
        except:
             # Do whatever handling you need to, etc. here.
             return "Cookie does not exist." 
```

或

```py
class CookieGet:
    def GET(self):
        age=web.cookies().get('age')
        if age:
            return "Your age is: %s" % age
        else:
            return "Cookie does not exist." 
```

# 用户认证

## 原作者没有写完，但是可以参照下一节，写得很详细

## 问题

如何完成一个用户认证系统？

## 解法

用户认证系统由这几个部分组成：用户添加，用户登录，用户注销以及验证用户是否已登录。用户认证系统一般都需要一个数据库。在这个例子中，我们要用到 MD5 和 SQLite。

## #

```py
import hashlib
import web    

def POST(self):
    i = web.input()

    authdb = sqlite3.connect('users.db')
    pwdhash = hashlib.md5(i.password).hexdigest()
    check = authdb.execute('select * from users where username=? and password=?', (i.username, pwdhash))
    if check: 
        session.loggedin = True
        session.username = i.username
        raise web.seeother('/results')   
    else: return render.base("Those login details don't work.") 
```

## 注意

这仅仅是个例子，可不要在真实的生产环境中应用哦。

# 在 PostgreSQL 下实现用户认证

## 问题

*   如何利用 PostgreSQL 数据库实现一个用户认证系统？

## 解法

*   用户认证系统有很多功能。在这个例子中，将展示如何在 PostgreSQL 数据库环境下一步一步完成一个用户认证系统

## 必需

*   因为要用到 make 模板和 postgreSQL 数据库，所以要: import web from web.contrib.template import render_mako import pg

## 第一步：创建数据库

首先，为创建一个用户表。虽然这个表结构非常简单，但对于大部分项目来说都足够用了。

```py
CREATE TABLE example_users
(
  id serial NOT NULL,
  user character varying(80) NOT NULL,
  pass character varying(80) NOT NULL,
  email character varying(100) NOT NULL,
  privilege integer NOT NULL DEFAULT 0,
  CONSTRAINT utilisateur_pkey PRIMARY KEY (id)
) 
```

## 第二步：确定网址

登录和注销对应两个网址：

*   "Login" 对应登录页

*   "Reset" 对应注销页

```py
urls = (
    '/login', 'login',
    '/reset', 'reset',
     ) 
```

## 第三步：判断用户是否登录

要判断用户是否已登录，是非常简单的，只要有个变量记录用户登录的状态即可。在 login/reset 类中使用这段代码:

```py
def logged():
    if session.login==1:
        return True
    else:
        return False 
```

## 第四步：简单的权限管理

我把我的用户划为四类：管理员，用户，读者（已登录），访客（未登录）。根据 example_users 表中定义的不同权限，选择不同的模板路径。

```py
def create_render(privilege):
    if logged():
        if privilege==0:
            render = render_mako(
                directories=['templates/reader'],
                input_encoding='utf-8',
                output_encoding='utf-8',
                )
        elif privilege==1:
            render = render_mako(
                directories=['templates/user'],
                input_encoding='utf-8',
                output_encoding='utf-8',
                )
        elif privilege==2:
            render = render_mako(
                directories=['templates/admin'],
                input_encoding='utf-8',
                output_encoding='utf-8',
                )
    else:
        render = render_mako(
            directories=['templates/communs'],
            input_encoding='utf-8',
            output_encoding='utf-8',
            )
    return render 
```

## 第五：登录(Login)和注销(Reset)的 python 类

现在，让我们用个轻松的方法来解决： - 如果你已登录，就直接重定向到 login_double.html 模板文件 - 否则，还是到 login.html。

```py
class login:
    def GET(self):
        if logged():
            render = create_render(session.privilege)
            return "%s" % (
                render.login_double()               )
        else:
            render = create_render(session.privilege)
            return "%s" % (
                render.login()
                ) 
```

*   好了。现在写 POST()方法。从.html 文件中，我们得到表单提交的变量值(见 login.html)，并根据变量值得到 example_users 表中对应的 user 数据
*   如果登录通过了，就重定向到 login_ok.html。
*   如果没通过，就重定向到 login_error.html。

```py
 def POST(self):
        user, passwd = web.input().user, web.input().passwd
        ident = db.query("select * from example_users where user = '%s'" % (user)).getresult()
        try:
            if passwd==ident[0][2]:
                session.login=1
                session.privilege=ident[0][4]
                render = create_render(session.privilege)
                return "%s" % (
                        render.login_ok()
                        )
            else:
                session.login=0
                session.privilege=0
                render = create_render(session.privilege)
                return "%s" % (
                    render.login_error()
                    )
        except:
            session.login=0
            session.privilege=0
            render = create_render(session.privilege)
            return "%s" % (
                render.login_error()
                ) 
```

对于 reset 方法，只要清除用户 session，再重定向到 logout.html 模板页即可。

```py
class reset:
    def GET(self):
        session.login=0
        session.kill()
        render = create_render(session.privilege)
        return "%s" % (
            render.logout()
            ) 
```

## 6th: 第六步：HTML 模板帮助

嗯，我认为没有人想看这个，但我喜欢把所有的信息都提供出来。最重要的就是 login.html。

```py
<FORM action=/login method=POST>
    <table id="login">
        <tr>
            <td>User: </td>
            <td><input type=text name='user'></td>
        </tr>
        <tr>
            <td>Password: </td>
            <td><input type="password" name=passwd></td>
        </tr>
        <tr>
            <td></td>
            <td><input type=submit value=LOGIN></td>
        </tr>
    </table>
</form> 
```

## 第七：问题或疑问？

*   邮件：您可以联想我，我的邮箱是 guillaume(at)process-evolution(dot)fr
*   IRC：#webpy on irc.freenode.net (pseudo: Ephedrax)
*   翻译：我是法国人，我的英文不好...你可以修改我的文档(译注：哈哈，谦虚啥，你那是没见过 wrongway 的山东英文...)

# 在子应用下使用 session

## 提示

这个解决方案是来自 web.py 邮件列表。[this](http://www.mail-archive.com/webpy@googlegroups.com/msg02557.html)

## 问题

如何在子应用中使用 session？

## 解法

web.py 默认 session 信息只能在主应用中共享，即便在其他模块中 import Session 都不行。在 app.py（或 main.py）可以这样初始化 session：

```py
session = web.session.Session(app, web.session.DiskStore('sessions'),
initializer = {'test': 'woot', 'foo':''}) 
```

.. 接下来创建一个被 web.loadhook 加载的处理器(processor)

```py
def session_hook():
    web.ctx.session = session

app.add_processor(web.loadhook(session_hook)) 
```

.. 在子应用(假设是 sub-app.py)中，可以这样操作 session:

```py
print web.ctx.session.test
web.ctx.session.foo = 'bar' 
```
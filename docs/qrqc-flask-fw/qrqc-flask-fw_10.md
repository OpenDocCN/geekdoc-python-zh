# 浅入浅出 Flask 框架：Cookie

2014-06-28

Cookie 是存储在客户端的记录访问者状态的数据。具体原理，请见[`zh.wikipedia.org/wiki/Cookie`](http://zh.wikipedia.org/wiki/Cookie)。常用的用于记录用户登录状态的 session 大多是基于 cookie 实现的。

cookie 可以借助`flask.Response`来实现。`flask.Response`在浅入浅出 Flask 框架：处理客户端通过 POST 方法传送的数据有过介绍。下面是一个示例。

### 建立 Flask 项目

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

### 代码

修改`HelloWorld/index.py`：

```py
from flask import Flask, request, Response, make_response
import time

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/add')
def login():
    res = Response('add cookies')
    res.set_cookie(key='name', value='letian', expires=time.time()+6*60)
    return res

@app.route('/show')
def show():
    return request.cookies.__str__()

@app.route('/del')
def del_cookie():
    res = Response('delete cookies')
    res.set_cookie('name', '', expires=0)
    # print res.headers
    # print res.data
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 
```

由上可以看到，可以使用`Response.set_cookie`添加和删除 cookie。`expires`参数用来设置 cookie 有效时间，它的值可以是`datetime`对象或者 unix 时间戳，笔者使用的是 unix 时间戳。

```py
res.set_cookie(key='name', value='letian', expires=time.time()+6*60) 
```

上面的 expire 参数的值表示 cookie 在从现在开始的 6 分钟内都是有效的。

要删除 cookie，将 expire 参数的值设为 0 即可：

```py
res.set_cookie('name', '', expires=0) 
```

`set_cookie()`函数的原型如下：

> set_cookie(key, value=’’, max_age=None, expires=None, path=’/‘, domain=None, secure=None, httponly=False)
> 
> Sets a cookie. The parameters are the same as in the cookie Morsel object in the Python standard library but it accepts unicode data, too.
> Parameters:
> 
>      key – the key (name) of the cookie to be set.
>      value – the value of the cookie.
>      max_age – should be a number of seconds, or None (default) if the cookie should last only as long as the client’s browser session.
> 
>      expires – should be a datetime object or UNIX timestamp.
> 
>      domain – if you want to set a cross-domain cookie. For example, domain=”.example.com” will set a cookie that is readable by the domain www.example.com, foo.example.com etc. Otherwise, a cookie will only be readable by the domain that set it.
> 
>      path – limits the cookie to a given path, per default it will span the whole domain.

### 运行与测试

运行`HelloWorld/index.py`：

```py
$ python HelloWorld/index.py 
```

使用浏览器打开`http://127.0.0.1:5000/add`，浏览器界面会显示

```py
add cookies 
```

下面查看一下 cookie，如果使用 firefox 浏览器，可以用 firebug 插件查看。打开 firebug，选择`Cookies`选项，刷新页面，可以看到名为`name`的 cookie，其值为`letian`。

在“网络”选项中，可以查看响应头中设置 cookie 的 HTTP“指令”：

```py
Set-Cookie: name=letian; Expires=Sun, 29-Jun-2014 05:16:27 GMT; Path=/ 
```

在 cookie 有效期间，使用浏览器访问`http://127.0.0.1:5000/show`，可以看到：

```py
{'name': u'letian'} 
```

### 相关资料：

[Flask: How to remove cookies?](http://stackoverflow.com/questions/14386304/flask-how-to-remove-cookies)
[Flask API](http://flask.pocoo.org/docs/api/)
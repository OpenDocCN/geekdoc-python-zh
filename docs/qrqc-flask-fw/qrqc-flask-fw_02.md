# 浅入浅出 Flask 框架：获取 URL 参数

2014-6-23

URL 参数是出现在 url 中的键值对，例如`http://127.0.0.1:5000/?disp=3`中的 url 参数是`{'disp':3}`。

## 建立 Flask 项目

* * *

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

## 列出所有的 url 参数

* * *

在 index.py 中添加以下内容：

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    params = request.args.items()
    return params.__str__()

if __name__ == '__main__':
    app.run() 
```

在浏览器中访问`http://127.0.0.1:5000/?disp=3&catalog=0&sort=time&p=7`，将显示：

```py
[('disp', u'3'), ('sort', u'time'), ('catalog', u'0'), ('p', u'7')] 
```

较新的浏览器也支持直接在 url 中输入中文（浏览器会帮忙转换），在浏览器中访问`http://127.0.0.1:5000/?info=这是爱，`，将显示：

```py
[('info', u'\u8fd9\u662f\u7231\uff0c')] 
```

> 小提示：可以通过`request.full_path`和`request.path`获取客户端请求的 url（去掉域名或者 IP:port）。

## 获取某个指定的参数

* * *

例如，要获取键`info`对应的值，如下修改`index.py`：

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return request.args.get('info')

if __name__ == '__main__':
    app.run() 
```

运行 index.py，在浏览器中访问`http://127.0.0.1:5000/?info=hello`，浏览器将显示：

```py
hello 
```

不过，当我们访问`http://127.0.0.1:5000/`时候却出现了 500 错误，浏览器显示：

```py
Internal Server Error

The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application. 
```

Flask 在终端中输出：

```py
127.0.0.1 - - [23/Jun/2014 17:25:56] "GET / HTTP/1.1" 500 - 
```

这是因为没有在 URL 参数中找到`info`。所以`request.args.get('info')`返回 None。打开调试：

```py
if __name__ == '__main__':
    app.run(debug=True) 
```

浏览器重新请求，会指出错误所在：

```py
...
ValueError: View function did not return a response
... 
```

对此，第一种解决办法是：

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    r = request.args.get('info')
    return r.__str__()

if __name__ == '__main__':
    app.run(debug=True) 
```

此时，浏览器会显示`None`，但是这种方式并不合理，于是干脆判断一下 info 对应的值：

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    r = request.args.get('info')
    if r==None:
        # do something
        return ''
    return r

if __name__ == '__main__':
    app.run(debug=True) 
```

当然，也可以设置默认值：

```py
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    r = request.args.get('info', 'hi')
    return r

if __name__ == '__main__':
    app.run(debug=True) 
```

函数`request.args.get`的第二个参数用来设置默认值。此时在浏览器访问`http://127.0.0.1:5000/`，将显示：

```py
hi 
```
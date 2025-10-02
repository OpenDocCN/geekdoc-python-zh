# 浅入浅出 Flask 框架：使用 redirect

2014-06-28

`redirect`函数用于重定向，实现机制很简单，就是向客户端（浏览器）发送一个重定向的 HTTP 报文，浏览器会去访问报文中指定的 url。

## 示例

* * *

### 建立 Flask 项目

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

### 编写代码

使用`redirect`时，给它一个字符串类型的参数就行了。

编辑`HelloWorld/index.py`：

```py
from flask import Flask, url_for, redirect

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/test1')
def test1():
    print 'this is test1'
    return redirect(url_for('test2'))

@app.route('/test2')
def test2():
    print 'this is test2'
    return 'this is test2'

if __name__ == '__main__':
    app.run(debug=True) 
```

运行`HelloWorld/index.py`，在浏览器中访问`http://127.0.0.1:5000/test1`，浏览器的 url 会变成`http://127.0.0.1:5000/test2`，并显示：

```py
this is test2 
```

而`HelloWorld/index.py`的输出信息为：

```py
this is test1
127.0.0.1 - - [28/Jun/2014 18:56:23] "GET /test1 HTTP/1.1" 302 -
this is test2
127.0.0.1 - - [28/Jun/2014 18:56:24] "GET /test2 HTTP/1.1" 200 - 
```
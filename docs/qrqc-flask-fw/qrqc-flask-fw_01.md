# 浅入浅出 Flask 框架：从 HelloWorld 开始 Flask

2014-06-01

本文主要内容：使用 Flask 写一个显示”hello world”的 web 程序，如何配置、调试 Flask。

### Hello World

* * *

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

`static`和`templates`目录是默认配置，其中`static`用来存放静态资源，例如图片、js、css 文件等。`templates`存放模板文件。
我们的网站逻辑基本在`index.py`文件中，当然，也可以给这个文件起其他的名字。

在`index.py`中加入以下内容：

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run() 
```

运行 index.py：

```py
$ python index.py 
 * Running on http://127.0.0.1:5000/ 
```

打开浏览器访问`http://127.0.0.1:5000/`，浏览页面上将出现`Hello World!`。
终端里会显示下面的信息：

```py
127.0.0.1 - - [16/May/2014 10:29:08] "GET / HTTP/1.1" 200 - 
```

变量 app 是一个 Flask 实例，通过下面的方式：

```py
@app.route('/')
def hello_world():
    return 'Hello World!' 
```

当客户端访问`/`时，将响应`hello_world()`函数返回的内容。注意，这不是返回`Hello World!`这么简单，`Hello World!`只是 HTTP 响应报文的实体部分，状态码等信息既可以由 Flask 自动处理，也可以通过编程来制定。

### 修改 Flask 的目录配置

* * *

Flask 使用`static`目录存放静态资源，这是可以更改的，请在 index.py 的：

```py
app = Flask(__name__) 
```

中为 Flask 多加几个参数值，这些参数请参考`__doc__`：

```py
from flask import Flask
print Flask.__doc__ 
```

### 调试

* * *

上面的 index.py 中以`app.run()`方式运行，这种方式下，如果服务器端出现错误是不会在客户端显示的。但是在开发环境中，显示错误信息是很有必要的，要显示错误信息，应该以下面的方式运行 Flask：

```py
app.run(debug=True) 
```

### 绑定网络接口和端口

* * *

默认情况下，Flask 绑定 IP 为`127.0.0.1`，端口为`5000`，可以通过下面的方式指定：

```py
app.run(host='0.0.0.0', port=80, debug=True) 
```

由于绑定了 80 端口，需要使用 root 权限运行 index.py。
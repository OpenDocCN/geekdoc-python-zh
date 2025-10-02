# Flask 入门系列(一)–Hello World

项目开发中，经常要写一些小系统来辅助，比如监控系统，配置系统等等。用传统的 Java 写，太笨重了，连 PHP 都嫌麻烦。一直在寻找一个轻量级的后台框架，学习成本低，维护简单。发现 Flask 后，我立马被它的轻巧所吸引，它充分发挥了 Python 语言的优雅和轻便，连 Django 这样强大的框架在它面前都觉得繁琐。可以说简单就是美。这里我们不讨论到底哪个框架语言更好，只是从简单这个角度出发，Flask 绝对是佼佼者。这一系列文章就会给大家展示 Flask 的轻巧之美。

### 系列文章

*   Flask 入门系列(一)–Hello World
*   Flask 入门系列(二)–路由
*   Flask 入门系列(三)–模板
*   Flask 入门系列(四)–请求，响应及会话
*   Flask 入门系列(五)–错误处理及消息闪现
*   Flask 入门系列(六)–数据库集成

### Hello World

程序员的经典学习方法，从 Hello World 开始。不要忘了，先安装 python, pip，然后运行”pip install Flask”，环境就装好了。当然本人还是强烈建议使用 virtualenv 来安装环境。细节就不多说了，让我们写个 Hello World 吧：

```py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Hello World</h1>'

if __name__ == '__main__':
    app.run()

```

一个 Web 应用的代码就写完了，对，就是这么简单！保存为”hello.py”，打开控制台，到该文件目录下，运行

```py
$ python hello.py
```

看到”* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)”字样后，就说明服务器启动完成。打开你的浏览器，访问”http://127.0.0.1:5000/”，一个硕大的”Hello World”映入眼帘:)。

#### 简单解释下这段代码

1.  首先引入了 Flask 包，并创建一个 Web 应用的实例”app”

```py
from flask import Flask
app = Flask(__name__)

```

这里给的实例名称就是这个 python 模块名。

*   定义路由规则

```py
@app.route('/')

```

这个函数级别的注解指明了当地址是根路径时，就调用下面的函数。可以定义多个路由规则，会在下篇文章里详细介绍。说的高大上些，这里就是 MVC 中的 Contoller。

*   处理请求

```py
def index():
    return '<h1>Hello World</h1>'

```

当请求的地址符合路由规则时，就会进入该函数。可以说，这里是 MVC 的 Model 层。你可以在里面获取请求的 request 对象，返回的内容就是 response。本例中的 response 就是大标题”Hello World”。

*   启动 Web 服务器

```py
if __name__ == '__main__':
    app.run()

```

当本文件为程序入口（也就是用 python 命令直接执行本文件）时，就会通过”app.run()”启动 Web 服务器。如果不是程序入口，那么该文件就是一个模块。Web 服务器会默认监听本地的 5000 端口，但不支持远程访问。如果你想支持远程，需要在”run()”方法传入”host=0.0.0.0″，想改变监听端口的话，传入”port=端口号”，你还可以设置调试模式。具体例子如下：

```py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)

```

注意，Flask 自带的 Web 服务器主要还是给开发人员调试用的，在生产环境中，你最好是通过 WSGI 将 Flask 工程部署到类似 Apache 或 Nginx 的服务器上。

本例中的代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-1.html)
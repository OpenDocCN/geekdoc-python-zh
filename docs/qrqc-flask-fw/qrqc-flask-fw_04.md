# 浅入浅出 Flask 框架：RESTful URL

2014-06-28

简单来说，restful url 可以看做是对 url 参数的替代。

## 入门

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

编辑 HelloWorld/index.py：

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/user/<username>')
def user(username):
    print username
    print type(username)
    return 'hello world'

@app.route('/user/<username>/friends')
def user_friends(username):
    print username
    print type(username)
    return 'hello world'

if __name__ == '__main__':
    app.run(debug=True) 
```

运行 HelloWorld/index.py。使用浏览器访问`http://127.0.0.1:5000/user/letian`，HelloWorld/index.py 将输出：

```py
letian
<type 'unicode'> 
```

而访问`http://127.0.0.1:5000/user/letian/`，响应为 404。

访问`http://127.0.0.1:5000/user/letian/friends`，能够得到期望的结果。HelloWorld/index.py 输出：

```py
letian
<type 'unicode'> 
```

## 转换类型

* * *

由上面的示例可以看出，使用 restful url 得到的变量默认为 unicode 对象。如果我们需要通过分页显示查询结果，那么需要在 url 中有数字来指定页数。按照上面方法，可以在获取 unicode 类型页数变量后，将其转换为 int 类型。不过，还有更方面的方法，就是在 route 中指定该如何转换。将 HelloWorld/index.py 修改如下：

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/page/<int:num>')
def page(num):
    print num
    print type(num)
    return 'hello world'

if __name__ == '__main__':
    app.run(debug=True) 
```

`@app.route('/page/<int:num>')`会将 num 变量自动转换成 int 类型。

运行上面的程序，在浏览器中访问`http://127.0.0.1:5000/page/1`，HelloWorld/index.py 将输出如下内容：

```py
1
<type 'int'> 
```

如果访问的是`http://127.0.0.1:5000/page/asd`，我们会得到 404 响应。

在官方资料中，说是有 3 个默认的转换器：

```py
int     accepts integers
float     like int but for floating point values
path     like the default but also accepts slashes 
```

肯定够用。

## 一个有趣的用法

* * *

如下编写`HelloWorld/index.py`：

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/page/<int:num1>-<int:num2>')
def page(num1, num2):
    print num1
    print num2
    return 'hello world'

if __name__ == '__main__':
    app.run(debug=True) 
```

在浏览器中访问`http://127.0.0.1:5000/page/11-22`，`HelloWorld/index.py`会输出：

```py
11
22 
```

## 编写转换器

* * *

自定义的转换器是一个继承`werkzeug.routing.BaseConverter`的类，修改`to_python`和`to_url`方法即可。`to_python`方法用于将 url 中的变量转换后供被`@app.route`包装的函数使用，`to_url`方法用于`flask.url_for`中的参数转换。

下面是一个示例，将`HelloWorld/index.py`修改如下：

```py
from flask import Flask, url_for

from werkzeug.routing import BaseConverter

class MyIntConverter(BaseConverter):

    def __init__(self, url_map):
        super(MyIntConverter, self).__init__(url_map)

    def to_python(self, value):
        return int(value)

    def to_url(self, value):
        return 'hi'

app = Flask(__name__)

app.url_map.converters['my_int'] = MyIntConverter

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/page/<my_int:num>')
def page(num):
    print num
    print url_for('page', num='123')
    return 'hello world'

if __name__ == '__main__':
    app.run(debug=True) 
```

浏览器访问`http://127.0.0.1:5000/page/123`后，`HelloWorld/index.py`的输出信息是：

```py
123
/page/hi 
```

## 资料

* * *

关于 restful url，可以参考[理解 RESTful 架构](http://www.ruanyifeng.com/blog/2011/09/restful.html)。
URL 参数，可以参考浅入浅出 Flask 框架：获取 URL 参数。
关于转换器，可以参考[Custom Converters](http://werkzeug.pocoo.org/docs/routing/)和[Does Flask support regular expressions in its URL routing?](http://stackoverflow.com/questions/5870188/does-flask-support-regular-expressions-in-its-url-routing)。
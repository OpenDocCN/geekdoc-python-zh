# 浅入浅出 Flask 框架：flashing system

2014-06-29

Flask 的闪存系统（flashing system）用于向用户提供反馈信息，这些反馈信息一般是对用户上一次操作的反馈。反馈信息是存储在服务器端的，当用户获得了反馈信息后，这些反馈信息会被服务器端删除。

下面是一个示例。

### 建立 Flask 项目

按照以下命令建立 Flask 项目 HelloWorld:

```py
mkdir HelloWorld
mkdir HelloWorld/static
mkdir HelloWorld/templates
touch HelloWorld/index.py 
```

### 编写 HelloWorld/index.py

内容如下：

```py
from flask import Flask, flash, get_flashed_messages
import time

app = Flask(__name__)
app.secret_key = 'some_secret'

@app.route('/')
def index():
    return 'hi'

@app.route('/gen')
def gen():
    info = 'access at '+ time.time().__str__()
    flash(info)
    return info

@app.route('/show1')
def show1():
    return get_flashed_messages().__str__()

@app.route('/show2')
def show2():
    return get_flashed_messages().__str__()

if __name__ == "__main__":
    app.run() 
```

### 运行与观察

运行服务器：

```py
$ python HelloWorld/index.py 
```

打开浏览器，访问`http://127.0.0.1:5000/gen`，浏览器界面显示（注意，时间戳是动态生成的，每次都会不一样，除非并行访问）：

```py
access at 1404020982.83 
```

查看浏览器的 cookie，可以看到`session`，其对应的内容是：

```py
.eJyrVopPy0kszkgtVrKKrlZSKIFQSUpWSknhYVXJRm55UYG2tkq1OlDRyHC_rKgIvypPdzcDTxdXA1-XwHLfLEdTfxfPUn8XX6DKWCAEAJKBGq8.BpE6dg.F1VURZa7VqU9bvbC4XIBO9-3Y4Y 
```

再一次访问`http://127.0.0.1:5000/gen`，浏览器界面显示：

```py
access at 1404021130.32 
```

cookie 中`session`发生了变化，新的内容是：

```py
.eJyrVopPy0kszkgtVrKKrlZSKIFQSUpWSknhYVXJRm55UYG2tkq1OlDRyHC_rKgIvypPdzcDTxdXA1-XwHLfLEdTfxfPUn8XX6DKWLBaMg1yrfCtciz1rfIEGxRbCwAhGjC5.BpE7Cg.Cb_B_k2otqczhknGnpNjQ5u4dqw 
```

然后使用浏览器访问`http://127.0.0.1:5000/show1`，浏览器界面显示：

```py
['access at 1404020982.83', 'access at 1404021130.32'] 
```

这个列表中的内容也就是上面的两次访问`http://127.0.0.1:5000/gen`得到的内容。此时，cookie 中已经没有`session`了。

如果使用浏览器访问`http://127.0.0.1:5000/show1`或者`http://127.0.0.1:5000/show2`，只会得到：

```py
[] 
```

### 高级用法

flash 系统也支持对 flash 的内容进行分类。修改`HelloWorld/index.py`内容：

```py
from flask import Flask, flash, get_flashed_messages
import time

app = Flask(__name__)
app.secret_key = 'some_secret'

@app.route('/')
def index():
    return 'hi'

@app.route('/gen')
def gen():
    info = 'access at '+ time.time().__str__()
    flash('1 '+info, category='show1')
    flash('2 '+info, category='show2')
    return info

@app.route('/show1')
def show1():
    return get_flashed_messages(category_filter='show1').__str__()

@app.route('/show2')
def show2():
    return get_flashed_messages(category_filter='show2').__str__()

if __name__ == "__main__":
    app.run() 
```

某一时刻，浏览器访问`http://127.0.0.1:5000/gen`，浏览器界面显示：

```py
access at 1404022326.39 
```

不过，由上面的代码可以知道，此时生成了两个 flash 信息。

使用浏览器访问`http://127.0.0.1:5000/show1`，得到如下内容：

```py
['1 access at 1404022326.39'] 
```

而继续访问`http://127.0.0.1:5000/show2`，得到的内容为空：

```py
[] 
```

### 在模板文件中获取 flash 的内容

在 Flask 中，`get_flashed_messages()`默认已经集成到`Jinja2`模板引擎中，易用性很强。下面是来自官方的一个示例：
![](img/2014-06-29-flask-flash.jpg)
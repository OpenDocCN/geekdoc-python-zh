# 用瓶子显影

> 原文：<https://realpython.com/developing-with-bottle-part-1/>

我爱[瓶](http://bottlepy.org/docs/stable/)。这是一个简单、快速而强大的 Python 微框架，非常适合小型 web 应用程序和快速原型开发。对于刚刚开始 web 开发的人来说，这也是一个很好的学习工具。

让我们看一个简单的例子。

> **注**:本教程假设您正在运行一个基于 Unix 的环境——例如，Mac OS X、Linux 版本或通过虚拟机驱动的 Linux 版本。

**更新于 2015 年 6 月 13 日:**更新了代码示例和说明

## 启动

首先，让我们创建一个工作目录:

```py
$ mkdir bottle && cd bottle
```

接下来，您需要安装 [pip](https://realpython.com/what-is-pip/) 、virtualenv 和 git。

virtualenv 是一个 Python 工具，使得[很容易管理特定项目所需的 Python 包](https://realpython.com/python-virtual-environments-a-primer/)；它防止一个项目中的包与其他项目中的包发生冲突。 [pip](https://pypi.python.org/pypi/pip) 同时是一个包管理器，用于管理 [Python 包](https://realpython.com/python-modules-packages/)的安装。

要获得在 Unix 环境中安装 pip(及其依赖项)的帮助，请遵循本[要点](https://gist.github.com/mjhea0/5692708)中的说明。如果你在 Windows 环境下，请观看这个[视频](http://www.youtube.com/watch?v=MIHYflJwyLk)寻求帮助。

一旦安装了 [pip](https://realpython.com/courses/what-is-pip/) ，运行以下命令安装 virtualenv:

```py
$ pip install virtualenv==12.0.7
```

现在，我们可以轻松设置我们的本地环境:

```py
$ virtualenv venv
$ source venv/bin/activate
```

安装瓶子:

```py
$ pip install bottle==0.12.8
$ pip freeze > requirements.txt
```

最后，让我们使用 Git 对我们的应用程序进行版本控制。有关 Git 的更多信息，请[查看本文](https://realpython.com/python-git-github-intro/)，其中也包括安装说明。

```py
$ git init
$ git add .
$ git commit -m "initial commit"
```

[*Remove ads*](/account/join/)

## 编写您的应用程序

我们已经准备好编写我们的瓶子应用程序。打开 [Sublime Text 3](https://realpython.com/setting-up-sublime-text-3-for-full-stack-python-development/) 或您选择的文本编辑器。创建您的应用程序文件， *app.py* ，它将保存我们第一个应用程序的*整体*:

```py
import os
from bottle import route, run, template

index_html = '''My first web app! By <strong>{{ author }}</strong>.'''

@route('/')
def index():
    return template(index_html, author='Real Python')

@route('/name/<name>')
def name(name):
    return template(index_html, author=name)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    run(host='0.0.0.0', port=port, debug=True)
```

保存文件。

现在，您可以在本地运行您的应用:

```py
$ python app.py
```

您应该能够连接到 [http://localhost:8080/](http://localhost:8080/) 并看到您的应用程序正在运行！

```py
My first web app! By RealPython.
```

因此，`@route` [装饰器](https://realpython.com/primer-on-python-decorators/)将一个函数绑定到路由。在第一个路由`/`中，`index()`函数被绑定到该路由，该路由呈现`index_html`模板并传入一个[变量](https://realpython.com/python-variables/)、`author`，作为关键字参数。然后可以在模板中访问这个变量。

现在导航到下一条路线，确保在路线的末尾添加您的姓名，即[http://localhost:8080/name/Michael](http://localhost:8080/name/Michael)。您应该会看到类似这样的内容:

```py
My first web app! By Michael.
```

**到底怎么回事？**

1.  同样，`@route`装饰器将一个函数绑定到路由。在这种情况下，我们使用包含通配符`<name>`的动态路由。
2.  然后这个通配符作为参数传递给视图函数- `def name(name)`。
3.  然后我们将它作为关键字参数传递给模板- `author=name`
4.  然后，模板呈现作者变量- `{{ author }}`。

## 外壳脚本

想快速上手？使用这个 Shell 脚本在几秒钟内生成 starter 应用程序。

```py
mkdir bottle
cd bottle
pip install virtualenv==12.0.7
virtualenv venv
source venv/bin/activate
pip install bottle==0.12.8
pip freeze > requirements.txt
git init
git add .
git commit -m "initial commit"

cat >app.py <<EOF
import os
from bottle import route, run, template

index_html = '''My first web app! By <strong>{{ author }}</strong>.'''

@route('/')
def index():
 return template(index_html, author='Real Python')

@route('/name/<name>')
def name(name):
 return template(index_html, author=name)

if __name__ == '__main__':
 port = int(os.environ.get('PORT', 8080))
 run(host='0.0.0.0', port=port, debug=True)
EOF

chmod a+x app.py

git init
git add .
git commit -m "Updated"
```

从这个[要点](https://gist.github.com/mjhea0/5784132)下载这个脚本，然后使用以下命令运行它:

```py
$ bash bottle.sh
```

## 接下来的步骤

从这一点来看，创建新页面就像添加新的`@route`修饰函数一样简单。

创建 HTML 很简单:在上面的应用程序中，我们只是在文件本身内联了 HTML。很容易修改它，从文件中加载模板。例如:

```py
@route('/main')
def main(name):
    return template('main_template')
```

这将加载模板文件`main_template.tpl`，该文件必须放在项目结构的`views`文件夹中，并呈现给最终用户。

更多信息请参考瓶子[文档](http://bottlepy.org/docs/dev/)。

* * *

我们将在随后的帖子中看看如何添加额外的页面和模板。然而，我强烈建议你自己尝试一下。如有任何问题，请在下面留言。

**看看[第二部](https://realpython.com/developing-with-bottle-part-2-plot-ly-api/)！***
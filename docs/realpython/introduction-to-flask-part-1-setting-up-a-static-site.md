# 探索 Flask，第 1 部分——设置静态站点

> 原文：<https://realpython.com/introduction-to-flask-part-1-setting-up-a-static-site/>

欢迎来到 Realp Python *探索烧瓶*系列…

系列概述

访问[discoverflask.com](http://discoverflask.com)查看系列摘要——博客帖子和视频的链接。

Flask 是一个由 Python 支持的微型 web 框架。它的 API 相当小，容易学习和使用。但是不要让这欺骗了你，因为它足够强大，可以支持处理大量流量的企业级应用程序。

你可以从一个完全包含在一个文件中的应用程序开始，然后随着你的站点变得越来越复杂，以一种结构良好的方式慢慢扩展到多个文件和文件夹。

这是一个很好的开始框架，你将在真正的“真正的 Python 风格”中学习:通过有趣的实践例子。

> **注**:本教程最初发布于 2013 年 1 月 29 日。我们修改了它，由于所做的改变的数量，我们决定“退休”旧教程，并创建一个全新的教程。如果你有兴趣查看旧教程的代码和视频，请访问这个[回购](https://github.com/mjhea0/flask-intro)。

查看随附的[视频](#video)。

## 要求

本教程假设你已经安装了 [Python 2.7.x](https://www.python.org/download/releases/2.7) 、 [pip](http://pip.readthedocs.org/en/latest/installing.html) 和 [virtualenv](http://virtualenv.readthedocs.org/en/latest/) 。

理想情况下，您应该对命令行或终端以及 Python 有基本的了解。如果没有，您将学到足够的知识，然后随着您继续使用 Flask，您的开发技能也会提高。如果你确实想要额外的帮助，看看真正的 Python 系列，从头开始学习 Python 和 web 开发。

你还需要一个代码编辑器或者 IDE，比如 [Sublime Text](http://www.sublimetext.com/) 、 [gedit](https://wiki.gnome.org/Apps/Gedit) 、 [Notepad++](http://notepad-plus-plus.org/) ，或者 [VIM](http://vimdoc.sourceforge.net/) 等。如果您确定要使用什么，请查看 Sublime Text，它是一个轻量级但功能强大的跨平台代码编辑器。

[*Remove ads*](/account/join/)

## 惯例

1.  本教程中的所有例子都使用了 Unix 风格的提示符:`$ python hello-world.py`。*【记住，美元符号不是命令的一部分，Windows 中对应的命令是:`C:\Sites> python hello-world.py`。]*

2.  所有的例子都在崇高的文本 3 编码。

3.  所有例子都使用了 Python 2.7.7。不过，您可以使用任何版本的 2.7.x。

4.  Github [repo](https://github.com/realpython/flask-intro) 中的`requirements.txt`文件中列出了额外的需求和依赖版本。

## 设置

1.  导航到一个方便的目录，如“桌面”或“文档”文件夹
2.  创建一个名为“flask-intro”的新目录来存放您的项目
3.  激活虚拟
4.  用 Pip `$ pip install Flask`安装烧瓶

## 结构

如果你熟悉 [Django](https://www.djangoproject.com/) 、 [web2py](http://www.web2py.com/) 或任何其他高级(或[全栈](https://wiki.python.org/moin/WebFrameworks))框架，那么你知道每一个都有特定的结构。然而，由于它的极简本质，Flask 没有提供集合结构，这对初学者来说可能很困难。幸运的是，这很容易弄清楚，特别是如果您对 Flask 组件使用单个文件。

在“flask-intro”文件夹中创建以下项目结构:

```py
├── app.py
├── static
└── templates
```

这里，我们简单地为 Flask 应用程序创建了一个名为 *app.py* 的文件，然后创建了两个文件夹，“静态”和“模板”。前者存放我们的样式表、 [JavaScript](https://realpython.com/python-vs-javascript/) 文件和图像，而后者存放 HTML 文件。这是一个很好的起点。我们已经在考虑前端和后端了。 *app.py* 将在后端利用模型-视图-控制器(MVC)设计模式来处理请求并向最终用户发出响应。

简单地说，当一个请求进来时，处理我们应用程序的业务逻辑的控制器决定如何处理它。

例如，控制器可以直接与数据库通信(如 [MySQL](https://realpython.com/python-mysql/) 、 [SQLite](https://realpython.com/python-sqlite-sqlalchemy/) 、PostgreSQL、MongoDB 等)。)来获取请求的数据，并通过视图返回一个响应，其中包含适当格式的适当数据(如 HTML 或 JSON)。或者最终用户请求的是不存在的资源——在这种情况下，控制器将响应 404 错误。

从这种结构开始将有助于将你的应用扩展到不同的文件和文件夹中，因为前端和后端之间已经有了逻辑上的分离。如果你对 MVC 模式不熟悉，在这里阅读更多关于它的内容。习惯它吧，因为几乎每个 web 框架都使用某种形式的 MVC。

## 路线

打开您最喜欢的编辑器，将以下代码添加到您的 *app.py* 文件中:

```py
# import the Flask class from the flask module
from flask import Flask, render_template

# create the application object
app = Flask(__name__)

# use decorators to link the function to a url
@app.route('/')
def home():
    return "Hello, World!"  # return a string

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')  # render a template

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
```

这相当简单。

导入`Flask`类后，我们创建(或实例化)应用程序对象，定义响应请求的视图，然后启动服务器。

`route` [装饰器](http://flask.pocoo.org/docs/patterns/viewdecorators/)用于将一个 URL 关联(或映射)到一个函数。URL `/`与`home()`函数相关联，因此当最终用户请求该 URL 时，视图将使用一个字符串进行响应。类似地，当请求`/welcome` URL 时，视图将呈现*welcome.html*模板。

简言之，主应用程序对象被实例化，然后用于将 URL 映射到函数。

更详细的解释，请阅读 Flask 的快速入门[教程](http://flask.pocoo.org/docs/quickstart/)。

[*Remove ads*](/account/join/)

## 测试

是时候进行理智检查了。启动您的开发服务器:

```py
$ python app.py
```

导航到 [http://localhost:5000/](http://localhost:5000/) 。你应该看到“你好，世界！”盯着你。然后请求下一个 URL，[http://localhost:5000/welcome](http://localhost:5000/welcome)。您应该会看到一个“TemplateNotFound”错误。为什么？因为我们还没有建立我们的模板，*welcome.html*。弗拉斯克正在找，但不在那里。就这么办吧。首先，从你的终端按下 `Ctrl` + `C` 杀死服务器。

> 关于实际回应的更多信息，请查看本文附带的[视频](#video)。

## 模板

在你的模板目录中创建一个名为*welcome.html*的新文件。在代码编辑器中打开该文件，然后添加以下 HTML:

```py
<!DOCTYPE html>
<html>
  <head>
    <title>Flask Intro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
  </head>
  <body>
    <div class="container">
      <h1>Welcome to Flask!</h2>
      <br>
      <p>Click <a href="/">here</a> to go home.</p>
    </div>
  </body>
</html>
```

保存。再次运行您的服务器。当你请求[http://localhost:5000/welcome](http://localhost:5000/welcome)的时候你现在看到了什么？测试链接。它能工作，但是不太漂亮。让我们改变这一点。这一次，在我们进行更改时，让服务器保持运行。

## 引导程序

好吧。让我们通过添加样式表来利用这些静态文件夹。你听说过 Bootstrap 吗？如果你的答案是否定的，那么请看[这篇](https://realpython.com/getting-started-with-bootstrap-3/)博客文章了解详情。

下载 [Bootstrap](http://getbootstrap.com/) ，然后将 *bootstrap.min.css* 和 *bootstrap.min.js* 文件添加到你的“静态”文件夹中。

更新模板:

```py
<!DOCTYPE html>
<html>
  <head>
    <title>Flask Intro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="static/bootstrap.min.css" rel="stylesheet" media="screen">
  </head>
  <body>
    <div class="container">
      <h1>Welcome to Flask!</h2>
      <br>
      <p>Click <a href="/">here</a> to go home.</p>
    </div>
  </body>
</html>
```

我们只是包含了 CSS 样式表；我们将在后面的教程中添加 JavaScript 文件。

返回浏览器。

还记得我们让服务器运行吗？嗯，当 Flask 处于[调试模式](http://flask.pocoo.org/docs/quickstart/#debug-mode)，`app.run(debug=True)`时，有一个自动重新加载机制在代码改变时生效。因此，我们只需在浏览器中点击“刷新”,就可以看到新模板正盯着我们。

很好。

## 结论

你最初的想法是什么？下面评论。抓取[代码](https://github.com/realpython/flask-intro)。观看[视频](#video)。

在不到 30 分钟的时间里，你学会了 Flask 的基础知识，并为一个更大的应用程序打下了基础。如果你以前使用过 Django，你可能会立即注意到 Flask 不会妨碍你的开发，让你可以自由地以你认为合适的方式构建和设计你的应用程序。

由于缺乏结构，真正的初学者可能会有点吃力，但是这是一个宝贵的学习经验，从长远来看，无论您是继续使用 Flask 还是继续使用更高级别的框架，都将使您受益。

在下一个教程中，我们将看看添加一些动态内容。

干杯！

[*Remove ads*](/account/join/)

## 视频

[https://www.youtube.com/embed/WfpFUmV1d0w?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/WfpFUmV1d0w?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)***
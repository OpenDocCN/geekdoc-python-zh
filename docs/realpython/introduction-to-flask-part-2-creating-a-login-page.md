# 探索 Flask，第 2 部分——创建登录页面

> 原文：<https://realpython.com/introduction-to-flask-part-2-creating-a-login-page/>

欢迎来到真正的 Python *探索烧瓶*系列…

系列概述

访问[discoverflask.com](http://discoverflask.com)查看系列摘要——博客帖子和视频的链接。

* * *

上次[时间](https://realpython.com/introduction-to-flask-part-1-setting-up-a-static-site/)我们讨论了如何建立一个基本的 Flask 结构，然后开发了一个静态站点，风格为 Bootstrap。在本系列的第二部分中，我们将为最终用户添加一个登录页面。

基于上一教程中的代码，我们需要:

*   添加路由以处理对登录 URL 的请求；和
*   为登录页面添加模板

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

## 添加一个路由来处理对登录 URL 的请求

确保您的 virtualenv 已激活。在您的代码编辑器中打开 *app.py* ，并添加以下路径:

```py
# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)
```

确保您还更新了导入:

```py
from flask import Flask, render_template, redirect, url_for, request
```

[*Remove ads*](/account/join/)

### 这是怎么回事？

1.  首先，请注意，我们为路由指定了适用的 HTTP 方法 GET 和 POST，作为路由装饰器中的一个参数。

2.  GET 是默认方法。因此，如果没有显式定义方法，Flask 假设唯一可用的[方法](http://flask.pocoo.org/docs/quickstart/#http-methods)是 GET，就像前面两条路线`/`和`/welcome`一样。

3.  对于新的`/login`路由，我们需要指定 POST 方法和 GET，以便最终用户可以用他们的登录凭证向那个`/login`端点发送 POST 请求。

4.  `login()`函数中的逻辑测试凭证是否正确。如果它们是正确的，那么用户将被重定向到主路由`/`，如果凭证不正确，则会出现一个错误。这些凭证从何而来？POST 请求，你马上就会看到。

5.  在 GET 请求的情况下，简单地呈现登录页面。

> **注意**:[`url_for()`](http://flask.pocoo.org/docs/api/#flask.url_for)函数为所提供的方法生成一个端点。

## 为登录页面添加模板

创建一个名为*login.html*的新文件，将其添加到“模板”目录中:

```py
<html>
  <head>
    <title>Flask Intro - login page</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="static/bootstrap.min.css" rel="stylesheet" media="screen">
  </head>
  <body>
    <div class="container">
      <h1>Please login</h1>
      <br>
      <form action="" method="post">
        <input type="text" placeholder="Username" name="username" value="{{
 request.form.username }}">
         <input type="password" placeholder="Password" name="password" value="{{
 request.form.password }}">
        <input class="btn btn-default" type="submit" value="Login">
      </form>
      {% if error %}
        <p class="error"><strong>Error:</strong> {{ error }}
      {% endif %}
    </div>
  </body>
</html>
```

是时候做一个快速测试了…

1.  启动服务器。导航到[http://localhost:5000/log in](http://localhost:5000/login)。

2.  输入不正确的凭证，然后按登录。您应该得到这样的响应:“错误:无效的凭证。请再试一次。”

3.  现在使用“admin”作为用户名和密码，您应该会被重定向到`/` URL。

4.  你能看出这里发生了什么吗？当表单被提交时，POST 请求连同表单数据`value="{{request.form.username }}"`和`value="{{request.form.password }}"`一起被发送到控制器`app.py`，然后控制器处理请求，或者用错误消息响应，或者将用户重定向到`/` URL。**一定要看看附带的[视频](https://www.youtube.com/watch?v=bLA6eBGN-_0)来用 Chrome 开发者工具深入挖掘这一点！**

5.  最后，我们的模板中有一些逻辑。最初，我们没有为错误传递[或](https://realpython.com/null-in-python/)。如果错误不是 None，那么我们显示实际的错误消息，它从视图:`<p class="error"><strong>Error:</strong> {{ error }}</p>`传递给模板。要了解这是如何工作的，请查看[这篇](https://realpython.com/primer-on-jinja-templating/#.U5CtZJRdUZ0)博文，了解更多关于 Jinja2 模板引擎的信息。

## 结论

你怎么想呢?简单吧？不要太兴奋，因为我们在用户管理方面还有很多工作要做…

**免费奖励:** [点击此处获得免费的 Flask + Python 视频教程](https://realpython.com/bonus/discover-flask-video-tutorial/)，向您展示如何一步一步地构建 Flask web 应用程序。

既然用户能够登录，我们需要保护 URL `/`免受未授权的访问。换句话说，当最终用户点击该端点时，除非他们已经登录，否则应该立即将他们发送到登录页面。下次吧。在那之前，去[练习](https://realpython.com/learn/jquery-practice/)一些 jQuery。

**一定要抢[码](https://github.com/realpython/flask-intro)，看[视频](#video)。**

## 视频

[https://www.youtube.com/embed/bLA6eBGN-_0?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com](https://www.youtube.com/embed/bLA6eBGN-_0?autoplay=1&modestbranding=1&rel=0&showinfo=0&origin=https://realpython.com)*
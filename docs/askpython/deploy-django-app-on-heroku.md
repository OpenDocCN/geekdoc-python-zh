# 在 Heroku 上部署 Django 应用程序——简单的分步指南

> 原文：<https://www.askpython.com/django/deploy-django-app-on-heroku>

大家好！在本教程中，我们将讨论如何在 Heroku 上免费部署 Django 应用程序。因此，让我们开始在 Heroku 平台上部署我们的第一个 Django 应用程序的激动人心的旅程，并且零成本。

如果你不知道 Django，我们有一个 [Django 教程](https://www.askpython.com/django/django-forms)系列，你可以看完。

## 什么是 Heroku？

**[Heroku](https://heroku.com)** 是一个基于云的平台，它使世界各地的开发者能够开发或构建、运行和操作各种类型的 web 应用。它属于 PaaS(平台即服务),这是广泛使用和最流行的云计算服务产品之一。

Heroku 完全支持学生的学习，这就是为什么除了付费功能外，它还提供免费服务，以方便实验和部署。人们可以在 Heroku 平台上轻松使用编程语言，如 *Java、* *Node.js、Scala、Clojure、 **Python** 、PHP 和 Go* ，因为它支持所有这些。

## 为什么要在 Heroku 上部署我们的 Django 应用？

每当我们学习任何编程语言或框架，如 Python 中的 Django，我们都在本地计算机上进行各种开发工作，这足以学习和调试东西。但是当我们完成了开发工作并且我们的项目已经准备好被一些真实世界的用户使用之后，就有必要在一些 web 服务器上部署项目或者应用程序了。

以便所有潜在用户都可以访问它。最重要的是，它对我们的开发工作产生了非常深刻和积极的影响，因为它在互联网上直播，人们可以很容易地看到实时工作的东西。

## 在 Heroku 上部署 Django 应用程序的步骤

以下是在 Heroku 上部署 Django 应用程序的五个关键步骤。

### 1.创建一个您希望部署在 Heroku 上的 Django 应用程序

如果您已经创建并开发了 Django 项目(一个 web 应用程序或网站)，那就太好了。你可以跳过这一步。对于那些现在还没有 Django 项目，但仍然想在 Heroku 上学习 Django 应用程序的部署过程的人。您可以运行以下命令来创建一个新的 Django 项目和应用程序。

```py
> python -m pip install Django

```

```py
> django-admin startproject <your_project_name>

```

```py
> python manage.py migrate

```

```py
> python manage.py runserver

```

**输出:**

```py
Django version 3.2.6, using settings '<your_project_name>.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.

```

**注意:**在继续部署之前，不要忘记在本地服务器(http://127.0.0.1:8000/)上运行 Django 应用程序。

### 2.在 GitHub 上创建 Django 应用程序的存储库

登录你的 GitHub 账户，创建一个全新的 repo (repository)来存储你的 Django 项目的所有文件夹、文件和代码。此外，将当前 Django 项目的目录设为 git repo，并将其连接到远程 GitHub 存储库。然后暂存所有内容，提交，最后将所有内容推送到远程 GitHub repo。

### 3.对 Django 项目文件进行以下更改

*   在 Django 项目的目录中创建一个文件名为 **Procfile** 的新文件，并将下面的代码复制到其中。

```py
web: gunicorn <your_project_name>.wsgi --log-file -

```

*   从命令行界面或虚拟环境(如果有)安装以下依赖项。

```py
> python -m pip install gunicorn

```

```py
> python -m pip install whitenoise

```

*   修改项目子文件夹中的`settings.py`文件，添加允许的主机，并将`DEBUG`参数设置为`False`，如下所示。

```py
DEBUG = False

ALLOWED_HOSTS = ['127.0.0.1', '<site_name>.herokuapp.com']

```

*   再次修改`settings.py`文件，通过以下方式用 whitenoise 依赖关系更新`MIDDLEWARE`代码。

```py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
     # Added Following One Line Of Code
    'whitenoise.middleware.WhiteNoiseMiddleware', 
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

```

*   按照以下方式更新项目子文件夹中的`settings.py`文件，这是*媒体*和*静态*文件顺利工作的需要。

```py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include("home.urls")),
    # Added Following Two Lines Of Code
    url(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}), 
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}), 
]

```

*   使用下面的命令创建一个`requirements.txt`文件，该文件将告诉服务器 Django 项目的各种依赖项以及 Django 应用程序顺利部署和运行所需的版本。

```py
> python -m pip freeze > requirements.txt

```

**注意:**不要忘记将修改或变更暂存、提交，然后推送到远程(GitHub 存储库)。

### 4.在 Heroku 平台上创建一个免费帐户

前往[www.heroku.com](https://www.heroku.com/)创建一个免费的 Heroku 账户，只需提供以下必填信息。

*   西方人名的第一个字
*   姓
*   电子邮件地址
*   作用
*   初级开发语言

如果你已经有一个 Heroku 帐号，不需要创建一个新的，只需要在你默认的浏览器上登录就可以了。

### 5.在 Heroku 仪表板上创建和设置一个新的 Heroku 应用程序

以下是创建和设置新 Heroku 应用程序的步骤。

*   转到 Heroku **仪表板**并点击**新**按钮。
*   从下拉菜单中选择**创建新应用**选项。
*   选择一个合适的可用的**应用名称**。
*   进入**应用设置**面板，在 Buildpacks 部分选择 **Python** 。
*   切换到**App Deploy**panel，在**部署方式**部分连接你的 GitHub 账号。
*   搜索包含 Django 项目的 GitHub repo 并选择它。
*   在**手动部署**部分选择 git 分支，通常是主/主分支，点击**部署分支**按钮。

万岁！您已经在 Heroku 服务器上成功启动了您的 Django 应用程序或网站。

## 总结

在本教程中，我们学习了 Heroku 平台、部署需求、在 Heroku 平台上部署 Django 应用或网站的步骤。希望你理解了部署过程，并对在 Heroku 上部署你的 Django 应用或网站感到兴奋。感谢阅读！请继续关注我们，了解更多关于 Python 的精彩学习内容。
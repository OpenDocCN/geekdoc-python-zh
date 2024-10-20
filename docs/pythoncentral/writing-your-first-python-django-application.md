# 编写您的第一个 Python Django 应用程序

> 原文：<https://www.pythoncentral.io/writing-your-first-python-django-application/>

## 创建 Django 项目

上一篇文章[介绍 Python 的 Django](https://www.pythoncentral.io/introduction-to-pythons-django/ "Introduction to Python's Django") 介绍了 Django 框架的概况。在本文中，我们将从头开始编写一个简单的 Django 应用程序。第一步是使用 Django 的一个内置命令`django-admin.py`创建一个项目。在 Virtualenv 中，键入以下命令:
【shell】
django-admin . py start project my blog

`django-admin.py`是一个方便的 shell 可执行文件，它提供了一系列子命令来管理 Django 应用程序。在前面的例子中，子命令`startproject`在当前目录中创建一个 Django 项目目录结构:

```py

myblog/

manage.py

myblog/

__init__.py

settings.py

urls.py

wsgi.py

```

*   `myblog`是 Django 项目`myblog`的父目录。它可以被重命名为你喜欢的任何名字，因为它只是一个容器。
*   `manage.py`是一个命令行实用程序，可以让您以各种方式与 Django 项目`myblog`进行交互。这个实用程序对于调试或练习代码非常有帮助。
*   `myblog/myblog`是包含项目实际 Python 包的目录。因为它是一个普通的 Python 包，所以可以使用普通的 Python 语法导入其中的任何模块或包。例如，`import myblog.settings`导入`myblog`包内的设置模块。
*   `myblog/myblog/settings.py`是 Django 项目的设置或配置。它包含在整个项目中使用的全局配置列表。
*   `myblog/myblog/urls.py`声明此项目的 URL 路由。它包含一个 URL 映射列表，告诉项目哪个视图函数如何处理 HTTP(S)请求。
*   `myblog/myblog/wsgi.py`是在兼容 WSGI 的 web 服务器上运行 Django 项目的脚本。

既然我们有一个 Django 项目，就让我们来运行它吧！

```py

python manage.py runserver

Validating models...
发现 0 个错误
2013 年 3 月 20 日- 21:15:27 
 Django 版本 1.5，使用设置‘my blog . settings’
开发服务器运行在 http://127.0.0.1:8000/ 
用 CONTROL-C 退出服务器

```

在你最喜欢的浏览器中访问`http://127.0.0.1:8000/`会显示一个欢迎页面。Django 应用程序的默认 IP 地址是`127.0.0.1`，只能从本地机器访问。如果你想在其他计算机上显示你的应用程序，你可以修改 ip 地址和端口，比如`python manage.py runserver 0.0.0.0:8000`，它允许你的 Django 应用程序监听所有的公共 IP。

# 在 Django 建立一个数据库

任何重要网站最重要的一个方面是数据库后端。由于数据库后端通常被配置为全局环境变量，Django 在`myblog/settings.py`中提供了一个方便的配置`default`项来处理各种数据库配置。

## Django 和 MySQL

在包含 Django 应用程序的 Virtualenv 中，运行以下命令:
【shell】
pip install MySQL-python

这将安装 MySQL Python 驱动程序。Django 的对象关系映射(ORM)后端将使用这个驱动程序在原始 SQL 语句中与 MySQL 服务器通信。

然后，在 mysql shell 中执行以下语句，为 Django 应用程序创建一个用户和一个数据库。
【shell】
MySQL>创建用户‘python central’@‘localhost’标识为‘12345’；
mysql >创建数据库 myblog
mysql > GRANT ALL ON myblog。* TO ' python central ' @ ' localhost '；

现在，修改`myblog/settings.py`。

```py

DATABASES = {

'default': {

# Add 'postgresql_psycopg2', 'mysql', 'sqlite3' or 'oracle'

'ENGINE': 'django.db.backends.mysql',

# Or the path to the database file if using sqlite3

'NAME': 'myblog',

# The following settings are not used with sqlite3

'USER': 'pythoncentral',

'PASSWORD': '12345',

# Empty for localhost through domain sockets or '127.0.0.1' for localhost through TCP

'HOST': '',

# Set to empty string for default

'PORT': '',

}

}

```

然后，运行下面的命令来初始化 Django 应用程序的数据库。
【shell】
python manage . py syncdb
创建表...
创建表 auth_permission
创建表 auth_group_permissions
创建表 auth_group
创建表 auth_user_groups
创建表 auth_user_user_permissions
创建表 django_content_type
创建表 django_session
创建表 django_site

您刚刚安装了 Django 的 auth 系统，这意味着您没有定义任何超级用户。您想现在创建一个吗？(是/否):是
用户名(留空使用‘我的用户名’):
邮箱:
密码:
密码(再次):
超级用户创建成功。
安装自定义 SQL...
安装索引...
从 0 个夹具安装了 0 个对象

最后，一个全新的 MySQL 数据库后端已经创建！您可以通过 django.contrib.auth.models 中的
【shell】
python manage . py shell
>>>与新数据库进行交互导入用户
>>>User . objects . all()
<用户:my username>

## Django 和 PostgreSQL

像 MySQL 一样，我们在 virtualenv 中安装了一个 Python PostgreSQL 驱动程序。
【shell】
pip 安装 psycopg2

然后，在 PostgreSQL shell 中执行以下语句，为 Django 项目创建一个用户和一个数据库。
【shell】
postgres = #创建用户 pythoncentral，密码为‘12345’；
创建角色
postgres=#创建数据库 myblog
创建数据库
postgres=#将数据库 myblog 上的所有权限授予 pythoncentral
授予

现在，修改`myblog/settings.py`。
【python】
DATABASES = {
' default ':{
# Add ' PostgreSQL _ psycopg 2 '，' mysql '，' sqlite3 '或' Oracle '
' ENGINE ':' django . db . backends . PostgreSQL _ psycopg 2 '，
#如果使用 sqlite3
'NAME': 'myblog '，
#以下设置不适用于 sqlite3
' USER ':' python central '，'

然后，运行下面的命令来初始化 Django 应用程序的数据库后端。
【shell】
python manage . py syncdb
创建表...
创建表 auth_permission
创建表 auth_group_permissions
创建表 auth_group
创建表 auth_user_groups
创建表 auth_user_user_permissions
创建表 django_content_type
创建表 django_session
创建表 django_site

您刚刚安装了 Django 的 auth 系统，这意味着您没有定义任何超级用户。您想现在创建一个吗？(是/否):是
用户名(留空使用‘我的用户名’):
邮箱:
密码:
密码(再次):
超级用户创建成功。
安装自定义 SQL...
安装索引...
从 0 个夹具安装了 0 个对象

最后，您可以使用内置的 shell 与新的 Django 应用程序进行交互。
【shell】
python manage . py shell
>>>from django . contrib . auth . models 导入用户
>>>User . objects . all()
<用户:my username>

注意，当您查询所有的`User`对象时，Django 的 shell 对于数据库后端是不可知的。相同的代码`User.objects.all()`用于`MySQL`和`PostgreSQL`检索数据库中所有用户的列表。

# Django 简介摘要

在本文中，我们创建了我们的第一个 Django 应用程序`myblog`，并针对一个`MySQL`后端和一个`PostgreSQL`后端进行了测试。使用 Django 的`ORM`，我们可以编写数据库`CRUD`(创建、读取、更新、删除)操作，而无需关心底层数据库后端。在本系列的后续文章中，我们将对几乎所有的数据库代码广泛使用 Django 的`ORM`。
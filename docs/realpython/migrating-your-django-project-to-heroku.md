# 将 Django 项目迁移到 Heroku

> 原文：<https://realpython.com/migrating-your-django-project-to-heroku/>

在本教程中，我们将采用一个简单的本地 Django 项目，由一个 [MySQL 数据库](https://realpython.com/python-mysql/)支持，并将其转换为在 Heroku 上运行。亚马逊 S3 将用于托管我们的静态文件，而 Fabric 将自动化部署过程。

该项目是一个简单的消息系统。它可以是一个 todo 应用程序，一个博客，甚至是一个 Twitter 的克隆。为了模拟真实场景，该项目将首先使用 MySQL 后端创建，然后转换为 Postgres 以部署在 Heroku 上。我个人有五六个项目需要做这样的事情:把一个本地项目，用 MySQL 支持，转换成 Heroku 上的一个实时应用。

## 设置

### 先决条件

1.  在 Heroku 阅读 Django 官方快速入门指南。读一下吧。这将有助于你对我们将在本教程中完成的内容有所了解。我们将使用官方教程作为我们自己更高级的部署过程的指南。
2.  创建一个 AWS 帐户并设置一个有效的 S3 存储桶。
3.  安装 MySQL。

[*Remove ads*](/account/join/)

### 让我们开始吧

从这里下载测试项目[开始](https://realpython.com/files/django_heroku_deploy.zip)，解压，然后[激活一个虚拟项目](https://realpython.com/python-virtual-environments-a-primer/):

```py
$ cd django_heroku_deploy
$ virtualenv --no-site-packages myenv
$ source myenv/bin/activate
```

在 Github 上创建一个新的资源库:

```py
$ curl -u 'USER' https://api.github.com/user/repos -d '{"name":"REPO"}'
```

> 确保用您自己的设置替换所有大写关键字。例如:`curl -u 'mjhea0' https://api.github.com/user/repos -d '{"name":"django-deploy-heroku-s3"}'`

添加一个 readme 文件，[初始化本地 Git repo](https://realpython.com/python-git-github-intro/) ，然后将本地副本推送到 Github:

```py
$ touch README.md
$ git init
$ git add .
$ git commit -am "initial"
$ git remote add origin https://github.com/username/Hello-World.git
$ git push origin master
```

> 确保将 URL 更改为您在上一步中创建的 repo 的 URL。

建立一个名为 *django_deploy* 的新 MySQL 数据库:

```py
$  mysql.server  start $  mysql  -u  root  -p Enter  password: Welcome  to  the  MySQL  monitor.  Commands  end  with  ;  or  \g.  Your  MySQL  connection  id  is  1 Type  'help;'  or  '\h'  for  help.  Type  '\c'  to  clear  the  buffer. mysql> mysql>  CREATE  DATABASE  django_deploy; Query  OK,  1  row  affected  (0.01  sec) mysql> mysql>  quit Bye
```

更新 *settings.py* :

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'django_deploy',
        'USER': 'root',
        'PASSWORD': 'your_password',
    }
}
```

安装依赖项:

```py
$ pip install -r requirements.txt
$ python manage.py syncdb
$ python manage.py runserver
```

在[http://localhost:8000/admin/](http://localhost:8000/admin/)运行服务器，确保可以登录 admin。向`Whatever`对象添加一些项目。关掉服务器。

## 从 MySQL 转换到 Postgres

> **注意:**在这个假设的情况下，让我们假设你已经使用 MySQL 在这个项目上工作了一段时间，现在你想把它转换成 Postgres。

安装依赖项:

```py
$ pip install psycopg2
$ pip install py-mysql2pgsql
```

建立 Postgres 数据库:

```py
$  psql  -h  localhost psql  (9.2.4) Type  "help"  for  help. michaelherman=#  CREATE  DATABASE  django_deploy; CREATE  DATABASE michaelherman=#  \q
```

迁移数据:

```py
$ py-mysql2pgsql
```

该命令创建一个名为 *mysql2pgsql.yml* 的文件，包含以下信息:

```py
mysql: hostname:  localhost port:  3306 socket:  /tmp/mysql.sock username:  foo password:  bar database:  your_database_name compress:  false destination: postgres: hostname:  localhost port:  5432 username:  foo password:  bar database:  your_database_name
```

> 为您的配置更新此内容。这个例子只是涵盖了基本的转换。您还可以包括或排除某些表。完整示例见[此处](https://github.com/philipsoutham/py-mysql2pgsql)。

传输数据:

```py
$ py-mysql2pgsql -v -f mysql2pgsql.yml
```

一旦数据传输完毕，请务必更新您的 *settings.py* 文件:

```py
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "your_database_name",
        "USER": "foo",
        "PASSWORD": "bar",
        "HOST": "localhost",
        "PORT": "5432",
    }
}
```

最后，重新同步数据库，运行测试服务器，并向数据库添加另一项，以确保转换成功。

[*Remove ads*](/account/join/)

## 添加一个 local_settings.py 文件

通过添加一个 *local_settings.py* 文件，您可以使用与您的本地环境相关的设置来扩展 *settings.py* 文件，而主 *settings.py* 文件仅用于您的试运行和生产环境。

确保将 *local_settings.py* 添加到您的*中。gitignore* 文件，以便将该文件排除在您的存储库之外。那些想要使用你的项目或者为你的项目做贡献的人可以克隆这个 repo，然后创建他们自己的 *local_settings.py* 文件，这个文件专门针对他们自己的本地环境。

> 尽管这种使用两个设置文件的方法已经成为惯例很多年了，但是许多 Python 开发人员现在使用另一种叫做[的模式，一种真正的方式](https://speakerdeck.com/jacobian/the-best-and-worst-of-django?slide=81)。我们可以在以后的教程中研究这个模式。

### 更新 settings.py

我们需要对当前的 *settings.py* 文件进行三处修改:

将`DEBUG`模式更改为假:

```py
DEBUG = False
```

将以下代码添加到文件的底部:

```py
# Allow all host hosts/domain names for this site
ALLOWED_HOSTS = ['*']

# Parse database configuration from $DATABASE_URL
import dj_database_url

DATABASES = { 'default' : dj_database_url.config()}

# Honor the 'X-Forwarded-Proto' header for request.is_secure()
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# try to load local_settings.py if it exists
try:
  from local_settings import *
except Exception as e:
  pass
```

更新数据库设置:

```py
# we only need the engine name, as heroku takes care of the rest
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
    }
}
```

创建您的 *local_settings.py* 文件:

```py
$ touch local_settings.py
$ pip install dj_database_url
```

然后添加以下代码:

```py
from settings import PROJECT_ROOT, SITE_ROOT
import os

DEBUG = True
TEMPLATE_DEBUG = True

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "django_deploy",
        "USER": "foo",
        "PASSWORD": "bar",
        "HOST": "localhost",
        "PORT": "5432",
    }
}
```

启动测试服务器，确保一切正常。向数据库中再添加一些记录。

## Heroku 设置

将 Procfile 添加到主目录:

```py
$ touch Procfile
```

并将以下代码添加到文件中:

```py
web: python manage.py runserver 0.0.0.0:$PORT --noreload
```

安装 Heroku 工具带:

```py
$ pip install django-toolbelt
```

冻结依赖关系:

```py
$ pip freeze > requirements.txt
```

更新 *wsgi.py* 文件:

```py
from django.core.wsgi import get_wsgi_application
from dj_static import Cling

application = Cling(get_wsgi_application())
```

在本地测试您的 Heroku 设置:

```py
$ foreman start
```

导航到 [http://localhost:5000/](http://localhost:5000/) 。

好看吗？让亚马逊 S3 开始运行吧。

[*Remove ads*](/account/join/)

## 亚马逊 S3

尽管假设可以在 Heroku repo 中托管静态文件，但最好使用第三方主机，尤其是如果您有一个面向客户的应用程序。S3 很容易使用，只需要对你的 *settings.py* 文件做一些改动。

安装依赖项:

```py
$ pip install django-storages
$ pip install boto
```

在“settings.py”中将`storages`和`boto`添加到您的`INSTALLED_APPS`中

将以下代码添加到“settings.py”的底部:

```py
# Storage on S3 settings are stored as os.environs to keep settings.py clean
if not DEBUG:
   AWS_STORAGE_BUCKET_NAME = os.environ['AWS_STORAGE_BUCKET_NAME']
   AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
   AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
   STATICFILES_STORAGE = 'storages.backends.s3boto.S3BotoStorage'
   S3_URL = 'http://%s.s3.amazonaws.com/' % AWS_STORAGE_BUCKET_NAME
   STATIC_URL = S3_URL
```

AWS 环境相关设置存储为环境变量。所以我们不必在每次运行开发服务器时从终端设置这些，我们可以在我们的 virtualenv `activate`脚本中设置这些。从 S3 那里获取 AWS 桶名、访问密钥 ID 和秘密访问密钥。打开`myenv/bin/activate`并添加以下代码(确保添加您刚从 S3 获得的具体信息):

```py
# S3 deployment info
export AWS_STORAGE_BUCKET_NAME=[YOUR AWS S3 BUCKET NAME]
export AWS_ACCESS_KEY=XXXXXXXXXXXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXXXXXX
```

停用并重新激活您的 virtualenv，然后启动本地服务器以确保更改生效:

```py
$ foreman start
```

杀死服务器，然后更新 *requirements.txt* 文件:

```py
$ pip freeze > requirements.txt
```

## 推送至 Github 和 Heroku

在推送到 Heroku 之前，让我们将文件备份到 Github:

```py
$ git add .
$ git commit -m "update project for heroku and S3"
$ git push -u origin master
```

创建 Heroku 项目/回购:

```py
$ heroku create <name>
```

> 随便你怎么命名。

推到 Heroku:

```py
$ git push heroku master
```

将 AWS 环境变量发送到 Heroku

```py
$ heroku config:set AWS_STORAGE_BUCKET_NAME=[YOUR AWS S3 BUCKET NAME]
$ heroku config:set AWS_ACCESS_KEY=XXXXXXXXXXXXXXXXXXXX
$ heroku config:set AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXXXXXX
```

收集静态文件并发送给 Amazon:

```py
$ heroku run python manage.py collectstatic
```

添加开发数据库:

```py
$ heroku addons:add heroku-postgresql:dev
Adding heroku-postgresql on deploy_django... done, v13 (free)
Attached as HEROKU_POSTGRESQL_COPPER_URL
Database has been created and is available
! This database is empty. If upgrading, you can transfer
! data from another database with pgbackups:restore.
Use `heroku addons:docs heroku-postgresql` to view documentation.
$ heroku pg:promote HEROKU_POSTGRESQL_COPPER_URL
Promoting HEROKU_POSTGRESQL_COPPER_URL to DATABASE_URL... done
```

现在同步数据库:

```py
$ heroku run python manage.py syncdb
```

[*Remove ads*](/account/join/)

## 数据传输

我们需要将数据从本地数据库转移到生产数据库。

安装 Heroku PGBackups 附件:

```py
$ heroku addons:add pgbackups
```

转储您的本地数据库:

```py
$ pg_dump -h localhost  -Fc library  > db.dump
```

为了让 Heroku 访问 db dump，您需要将它上传到互联网的某个地方。你可以使用个人网站、dropbox 或 S3。我只是把它上传到了 S3 桶。

将转储导入 Heroku:

```py
$ heroku pgbackups:restore DATABASE http://www.example.com/db.dump
```

## 测试

让我们测试一下以确保一切正常。

首先，在 *settings.py* 中将允许的主机更新到您的特定域:

```py
ALLOWED_HOSTS = ['[your-project-name].herokuapp.com']
```

查看您的应用:

```py
$ heroku open
```

## 织物

Fabric 用于自动化应用程序的部署。

安装:

```py
$ pip install fabric
```

创建 fabfile:

```py
$ touch fabfile.py
```

然后添加以下代码:

```py
from fabric.api import local

def deploy():
   local('pip freeze > requirements.txt')
   local('git add .')
   print("enter your git commit comment: ")
   comment = raw_input()
   local('git commit -m "%s"' % comment)
   local('git push -u origin master')
   local('heroku maintenance:on')
   local('git push heroku master')
   local('heroku maintenance:off')
```

测试:

```py
$ fab deploy
```

有问题或意见吗？加入下面的讨论。****
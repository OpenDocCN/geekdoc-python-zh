# 通过 Docker 开发和部署 Cookiecutter-Django

> 原文：<https://realpython.com/development-and-deployment-of-cookiecutter-django-via-docker/>

让我们看看如何引导一个预先加载了基本需求的 Django 项目，以便快速启动并运行项目。此外，除了项目结构之外，大多数自举项目还负责设置开发和生产环境设置，不会给用户带来太多麻烦——所以我们也来看看这一点。

**更新**:

*   *04/15/2019* :更新至最新版本的 cookiecutter (v [1.6.0](https://github.com/audreyr/cookiecutter/releases/tag/1.6.0) )、cookiecutter-django、Django (v [2.0](https://docs.djangoproject.com/en/2.2/) )、Docker (v [18.09.2](https://github.com/docker/docker-ce/releases/tag/v18.09.2) )、Docker Compose (v [1.23.2](https://github.com/docker/compose/releases/tag/1.23.2) )、Docker Machine (v [0.16.1](https://github.com/docker/machine/releases/tag/v0.16.1) )。

*   *10/04/2016* :更新至最新版本的 cookiecutter (v [1.4.0](https://github.com/audreyr/cookiecutter/releases/tag/1.4.0) )、cookiecutter-django、Django (v [1.10.1](https://docs.djangoproject.com/en/1.10/releases/1.10.1/) )、Docker (v [1.12.1](https://github.com/docker/docker/releases/tag/v1.12.1) )、Docker Compose (v [1.8.1](https://github.com/docker/compose/releases/tag/1.8.1) )、Docker Machine (v [0.8.2](https://github.com/docker/machine/releases/tag/v0.8.2) )。

我们将使用流行的 [cookiecutter-django](https://github.com/pydanny/cookiecutter-django) 作为 django 项目的引导程序，与 [Docker](https://www.docker.com/) 一起管理我们的应用程序环境。

我们开始吧！

## 本地设置

从全局安装 [cookiecutter](https://github.com/audreyr/cookiecutter) 开始:

```py
$ pip install cookiecutter==1.6.0
```

现在执行下面的命令来生成一个引导的 django 项目:

```py
$ cookiecutter https://github.com/pydanny/cookiecutter-django.git
```

该命令使用 [cookiecutter-django](https://github.com/pydanny/cookiecutter-django) repo 运行 cookiecutter，允许我们输入特定于项目的详细信息:

```py
project_name [My Awesome Project]: django_cookiecutter_docker
project_slug [django_cookiecutter_django]: django_cookiecutter_docker
description [Behold My Awesome Project!]: Tutorial on bootstrapping django projects
author_name [Daniel Roy Greenfeld]: Michael Herman
domain_name [example.com]: realpython.com
email [michael-herman@example.com]: michael@realpython.com
version [0.1.0]: 0.1.0
Select open_source_license:
1 - MIT
2 - BSD
3 - GPLv3
4 - Apache Software License 2.0
5 - Not open source
Choose from 1, 2, 3, 4, 5 (1, 2, 3, 4, 5) [1]: 1
timezone [UTC]: UTC
windows [n]: 
use_pycharm [n]: 
use_docker [n]: y
Select postgresql_version:
1 - 10.5
2 - 10.4
3 - 10.3
4 - 10.2
5 - 10.1
6 - 9.6
7 - 9.5
8 - 9.4
9 - 9.3
Choose from 1, 2, 3, 4, 5, 6, 7, 8, 9 (1, 2, 3, 4, 5, 6, 7, 8, 9) [1]: 1
Select js_task_runner:
1 - None
2 - Gulp
Choose from 1, 2 (1, 2) [1]: 2
Select cloud_provider:
1 - AWS
2 - GCE
Choose from 1, 2 (1, 2) [1]: 
custom_bootstrap_compilation [n]: 
use_compressor [n]: 
use_celery [n]: 
use_mailhog [n]: 
use_sentry [n]: 
use_whitenoise [n]: 
use_heroku [n]: 
use_travisci [n]: 
keep_local_envs_in_vcs [y]: 
debug [n]: 
 [SUCCESS]: Project initialized, keep up the good work!
```

[*Remove ads*](/account/join/)

## 项目结构

快速查看生成的项目结构，特别注意以下目录:

1.  “配置”包括本地和生产环境的所有设置。
2.  “需求”包含了所有的需求文件——*base . txt*、 *local.txt* 、*production . txt*——你可以对其进行修改，然后通过`pip install -r file_name`进行安装。
3.  “django_cookiecutter_docker”是主项目目录，由“static”、“contrib”和“templates”目录以及包含与用户认证相关的模型和样板代码的`users`应用程序组成。

有些服务可能需要环境变量。您可以在*中找到每个服务的环境文件。envs* 目录并添加所需的变量。

## 对接设置

按照说明安装[对接引擎](https://docs.docker.com/install/)和所需的对接组件——引擎、机器和合成。

检查版本:

```py
$ docker --version
Docker version 18.09.2, build 6247962

$ docker-compose --version
docker-compose version 1.23.2, build 1110ad01

$ docker-machine --version
docker-machine version 0.16.1, build cce350d7
```

### 对接机

安装完成后，在新创建的 Django 项目的根目录下创建一个新的 Docker 主机:

```py
$ docker-machine create --driver virtualbox dev
$ eval $(docker-machine env dev)
```

> **注意** : `dev`你想取什么名字都可以。例如，如果您有不止一个开发环境，您可以将它们命名为`djangodev1`、`djangodev2`，等等。

要查看所有计算机，请运行:

```py
$ docker-machine ls
```

您还可以通过运行以下命令来查看`dev`机器的 IP:

```py
$ docker-machine ip dev
```

### 坞站组成〔t0〕

现在，我们可以通过 Docker Compose 来启动任何东西，例如 Django 和 Postgres:

```py
$ docker-compose -f local.yml build
$ docker-compose -f local.yml up -d
```

> 您可能需要将您的对接机 IP ( `docker-machine ip dev`)添加到`config/settings/local.py`中的`ALLOWED_HOSTS`列表中。
> 
> 运行 Windows？打这个错误- `Interactive mode is not yet supported on Windows`？见[此评论](https://realpython.com/development-and-deployment-of-cookiecutter-django-via-docker/#comment-2442262433)。

第一次构建需要一段时间。由于[缓存](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache)，后续构建将运行得更快。

[*Remove ads*](/account/join/)

## 健全性检查

现在我们可以通过[应用迁移](https://realpython.com/django-migrations-a-primer/)然后运行服务器来测试我们的 Django 项目:

```py
$ docker-compose -f local.yml run django python manage.py makemigrations
$ docker-compose -f local.yml run django python manage.py migrate
$ docker-compose -f local.yml run django python manage.py createsuperuser
```

在浏览器中导航到`dev` IP(端口 8000 ),查看项目快速启动页面，打开调试模式，安装并运行更多面向开发环境的特性。

停止容器(`docker-compose -f local.yml down`)，初始化一个新的 git repo，提交，并推送到 GitHub。

## 部署设置

因此，我们已经使用 cookiecutter-django 成功地在本地设置了 Django 项目，并通过 Docker 使用传统的 *manage.py* 命令行实用程序提供了它。

**注意:**如果你想采用不同的方法来部署你的 Django 应用程序，可以查看一下关于使用 Gunicorn 和 Nginx 的真实 Python 的[教程](https://realpython.com/django-nginx-gunicorn/)或[视频课程](https://realpython.com/courses/django-app-with-gunicorn-nginx/)。

在这一节中，我们继续讨论部署部分，在这里 web 服务器的角色开始发挥作用。我们将在[数字海洋](https://www.digitalocean.com/?refcode=d8f211a4b4c2) droplet 上建立一个 Docker 机器，用 Postgres 作为我们的数据库，用 [Nginx](https://nginx.org/) 作为我们的网络服务器。

与此同时，我们将使用 guni corn T1 代替 Django 的 T2 单线程开发服务器 T3 来运行服务器进程。

### 为什么选择 Nginx？

除了作为一个高性能的 HTTP 服务器(市场上几乎每一个好的 web 服务器都是这样)，Nginx 还有一些非常好的特性使它脱颖而出——即它:

*   可以耦合成一个[反向代理服务器](https://en.wikipedia.org/wiki/Reverse_proxy)。
*   可以托管多个站点。
*   采用异步方式处理 web 请求，这意味着由于它不依赖线程来处理 web 请求，因此在处理多个请求时具有更高的性能。

### 为什么是 Gunicorn？

[Gunicorn](https://gunicorn.org/) 是一个 Python WSGI HTTP 服务器，可以轻松定制，在可靠性方面比 Django 的单线程开发服务器在生产环境中提供更好的性能。

### 数字海洋设置

在本教程中，我们将使用一个数字海洋服务器。在您[注册](https://www.digitalocean.com/?refcode=d8f211a4b4c2)(如果需要的话)，[生成](https://www.digitalocean.com/community/tutorials/how-to-use-the-digitalocean-api-v2)一个个人访问令牌，然后运行以下命令:

```py
$ docker-machine create \
-d digitalocean \
--digitalocean-access-token ADD_YOUR_TOKEN_HERE \
prod
```

这应该只需要几分钟就可以提供数字 Ocean droplet 并设置一个名为`prod`的新 Docker 机器。当你等待的时候，导航到[数字海洋控制面板](https://cloud.digitalocean.com)；您应该会看到一个新的液滴正在被创建，再次被称为`prod`。

一旦完成，现在应该有两台机器在运行，一台在本地(`dev`)，一台在数字海洋(`prod`)。运行`docker-machine ls`确认:

```py
NAME   ACTIVE   DRIVER         STATE     URL                         SWARM   DOCKER     ERRORS
dev    *        virtualbox     Running   tcp://192.168.99.100:2376           v18.09.2
prod   -        digitalocean   Running   tcp://104.131.50.131:2376           v18.09.2
```

将`prod`设置为活动机器，然后将 Docker 环境加载到 shell 中:

```py
$ eval $(docker-machine env prod)
```

[*Remove ads*](/account/join/)

### 复合码头(取 2)

在*内。envs/。生产/。django* 更新`DJANGO_ALLOWED_HOSTS`变量以匹配数字海洋 IP 地址——即`DJANGO_ALLOWED_HOSTS=104.131.50.131`。

现在，我们可以创建构建，然后在云中启动服务:

```py
$ docker-compose -f production.yml build
$ docker-compose -f production.yml up -d
```

### 健全性检查(取 2)

应用所有迁移:

```py
$ docker-compose run django python manage.py makemigrations
$ docker-compose run django python manage.py migrate
```

就是这样！

现在只需访问与数字海洋水滴相关的服务器 IP 地址，并在浏览器中查看。

你应该可以走了。

* * *

为了进一步参考，只需从[库](https://github.com/realpython/django_cookiecutter_docker)中获取代码。非常感谢你的阅读！期待各位的提问。***
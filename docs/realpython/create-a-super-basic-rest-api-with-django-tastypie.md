# 用 Django Tastypie 创建一个超级基础的 REST API

> 原文：<https://realpython.com/create-a-super-basic-rest-api-with-django-tastypie/>

让我们用 [Django Tastypie](http://tastypieapi.org/) 建立一个 RESTful [API](https://realpython.com/api-integration-in-python/) 。

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

*更新:*

*   07/10/2016:升级到 Python (v [3.5.1](https://www.python.org/downloads/release/python-351/) )、Django (v [1.9.7](https://docs.djangoproject.com/en/1.9/releases/1.9.7/) )和 django-tastypie (v [13.3](https://github.com/django-tastypie/django-tastypie/releases/tag/v0.13.3) )的最新版本。

## 项目设置

要么按照下面的步骤创建您的示例项目，要么从 [Github](https://github.com/mjhea0/django-tastypie-tutorial) 克隆 repo。

创建一个新的项目目录，创建并激活一个 virtualenv，安装 [Django](https://realpython.com/get-started-with-django-1/) 和[所需的依赖项](https://realpython.com/what-is-pip/):

```py
$ mkdir django-tastypie-tutorial
$ cd django-tastypie-tutorial
$ pyvenv-3.5 env
$ source env/bin/activate
$ pip install Django==1.9.7
$ pip install django-tastypie==0.13.3
$ pip install defusedxml==0.4.1
$ pip install lxml==3.6.0
```

创建一个基本 Django 项目和应用程序:

```py
$ django-admin.py startproject django19
$ cd django19
$ python manage.py startapp whatever
```

确保将应用添加到 *settings.py* 中的`INSTALLED_APPS`部分:

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'whatever',
]
```

在 *settings.py* 中添加对 [SQLite](https://realpython.com/python-sqlite-sqlalchemy/) (或者您选择的 RDBMS)的支持:

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'test.db'),
    }
}
```

更新您的 *models.py* 文件:

```py
from django.db import models

class Whatever(models.Model):
    title = models.CharField(max_length=200)
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

[创建迁移](https://realpython.com/django-migrations-a-primer/):

```py
$ python manage.py makemigrations
```

现在迁移它们:

```py
$ python manage.py migrate --fake-initial
```

> **注意**:如果您必须对现有迁移进行故障排除，则`fake-initial`可选参数是必需的。如果不存在迁移，请忽略。

启动 Django Shell 并填充数据库:

>>>

```py
$ python manage.py shell
>>> from whatever.models import Whatever
>>> w = Whatever(title="What Am I Good At?", body="What am I good at? What is my talent? What makes me stand out? These are the questions we ask ourselves over and over again and somehow can not seem to come up with the perfect answer. This is because we are blinded, we are blinded by our own bias on who we are and what we should be. But discovering the answers to these questions is crucial in branding yourself.")
>>> w.save()

>>> w = Whatever(title="Charting Best Practices: Proper Data Visualization", body="Charting data and determining business progress is an important part of measuring success. From recording financial statistics to webpage visitor tracking, finding the best practices for charting your data is vastly important for your company’s success. Here is a look at five charting best practices for optimal data visualization and analysis.")
>>> w.save()

>>> w = Whatever(title="Understand Your Support System Better With Sentiment Analysis", body="There’s more to evaluating success than monitoring your bottom line. While analyzing your support system on a macro level helps to ensure your costs are going down and earnings are rising, taking a micro approach to your business gives you a thorough appreciation of your business’ performance. Sentiment analysis helps you to clearly see whether your business practices are leading to higher customer satisfaction, or if you’re on the verge of running clients away.")
>>> w.save()
```

完成后退出 shell。

[*Remove ads*](/account/join/)

## 任务类型设置

在你的应用中创建一个名为 *api.py* 的新文件。

```py
from tastypie.resources import ModelResource
from tastypie.constants import ALL

from whatever.models import Whatever

class WhateverResource(ModelResource):
    class Meta:
        queryset = Whatever.objects.all()
        resource_name = 'whatever'
        filtering = {'title': ALL}
```

更新 *urls.py* :

```py
from django.conf.urls import url, include
from django.contrib import admin

from django19.api import WhateverResource

whatever_resource = WhateverResource()

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^api/', include(whatever_resource.urls)),
]
```

## 开始吧！

1.  启动服务器。
2.  导航到[http://localhost:8000/API/whatever/？format=json](http://localhost:8000/api/whatever/?format=json) 获取 json 格式的数据
3.  导航到[http://localhost:8000/API/whatever/？format=xml](http://localhost:8000/api/whatever/?format=json) 获取 xml 格式的数据

还记得我们放在`WhateverResource`类上的过滤器吗？

```py
filtering = {'title': ALL}
```

嗯，我们可以按标题过滤对象。尝试各种关键词:

1.  [http://localhost:8000/API/whatever/？格式=json &标题 _ _ 包含=什么](http://localhost:8000/api/whatever/?format=json&title__contains=what)
2.  [http://localhost:8000/API/whatever/？格式=json &标题 _ _ 包含=测试](http://localhost:8000/api/whatever/?format=json&title__contains=test)

简单，对！？！

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

Tastypie 可以配置的东西太多了。查看官方[文档](http://tastypieapi.org/)了解更多信息。如有疑问，请在下方评论。

同样，您可以从 [repo](https://github.com/mjhea0/django-tastypie-tutorial) 下载代码。*
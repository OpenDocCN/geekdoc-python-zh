# Django 的 LinkedIn 社交认证

> 原文：<https://realpython.com/linkedin-social-authentication-in-django/>

社交认证(或社交登录)是一种简化最终用户登录的方法，它使用来自流行社交网络服务的现有登录信息，如[脸书](https://www.facebook.com/)、[推特](https://www.twitter.com/)、[谷歌](https://realpython.com/flask-google-login/)、 [LinkedIn](https://www.linkedin.com/) (本文重点)等等。大多数需要用户登录的网站利用社交登录平台来获得更好的认证/注册体验，而不是开发自己的系统。

[Python Social Auth](http://psa.matiasaguirre.net/) 提供了一种机制，可以轻松地建立一个认证/注册系统，该系统支持多种框架和认证提供者。

在本教程中，我们将详细演示如何将这个库集成到您的 Django 项目中，以便使用 OAuth 2.0 通过 LinkedIn 提供用户认证。

## 什么是 OAuth 2.0？

[OAuth 2.0](http://oauth.net/2/) 是一个授权框架，它允许应用程序通过流行的社交网络服务访问最终用户的帐户进行身份验证/注册。最终用户可以选择应用程序可以访问哪些细节。它专注于简化开发工作流，同时为 web 应用程序和桌面应用程序、移动电话和 IOT(物联网)设备提供特定的授权流。

[*Remove ads*](/account/join/)

## 环境和 Django 设置

我们将使用:

*   python3.4.2
*   Django v1.8.4
*   python-social-auth v0.2.12

> 如果你已经有了一个 [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) 和一个 [Django 项目](https://realpython.com/get-started-with-django-1/)并准备好了，请随意跳过这一部分。

### 创建虚拟人

```py
$ mkvirtualenv --python='/usr/bin/python3.4' django_social_project
$ pip install django==1.8.4
```

### 开始一个新的 Django 项目

引导 Django 应用程序:

```py
$ django-admin.py startproject django_social_project
$ cd django_social_project
$ python manage.py startapp django_social_app
```

不要忘记将应用添加到 *settings.py* 中的`INSTALLED_APPS`元组，让我们的项目知道我们已经创建了一个应用，它需要作为 Django 项目的一部分。

### 设置初始表格

```py
$ python manage.py migrate
Operations to perform:
 Synchronize unmigrated apps: messages, staticfiles
 Apply all migrations: sessions, admin, auth, contenttypes
Synchronizing apps without migrations:
 Creating tables...
 Running deferred SQL...
 Installing custom SQL...
Running migrations:
 Rendering model states... DONE
 Applying contenttypes.0001_initial... OK
 Applying auth.0001_initial... OK
 Applying admin.0001_initial... OK
 Applying contenttypes.0002_remove_content_type_name... OK
 Applying auth.0002_alter_permission_name_max_length... OK
 Applying auth.0003_alter_user_email_max_length... OK
 Applying auth.0004_alter_user_username_opts... OK
 Applying auth.0005_alter_user_last_login_null... OK
 Applying auth.0006_require_contenttypes_0002... OK
 Applying sessions.0001_initial... OK
```

### 添加超级用户

```py
$ python manage.py syncdb
```

### 创建一个模板目录

在项目根目录下创建一个名为“templates”的新目录，然后在 *settings.py* 文件中添加`TEMPLATES`的正确路径:

```py
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

### 运行健全性检查

启动开发服务器- `python manage.py runserver`以确保一切正常，然后导航到 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 。你应该看到“成功了！”页面。

您的项目应该如下所示:

```py
└── django_social_project
    ├── db.sqlite3
    ├── django_social_app
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── migrations
    │   │   └── __init__.py
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    ├── django_social_project
    │   ├── __init__.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── manage.py
    └── templates
```

[*Remove ads*](/account/join/)

## Python 社交认证设置

遵循以下步骤和/或[官方安装指南](http://psa.matiasaguirre.net/docs/installing.html)来安装和设置基本配置，以使我们的应用程序能够通过任何社交网络服务处理社交登录。

### 安装

用 [pip](https://realpython.com/what-is-pip/) 安装:

```py
$ pip install python-social-auth==0.2.12
```

### 配置

更新 *settings.py* 以在我们的项目中包含/注册该库:

```py
INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_social_app',
    'social.apps.django_app.default',
)

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.core.context_processors.debug',
                'django.core.context_processors.i18n',
                'django.core.context_processors.media',
                'django.core.context_processors.static',
                'django.core.context_processors.tz',
                'django.contrib.messages.context_processors.messages',
                'social.apps.django_app.context_processors.backends',
                'social.apps.django_app.context_processors.login_redirect',
            ],
        },
    },
]
```

> **注意**:由于我们使用 LinkedIn 社交认证，我们需要 Linkedin OAuth2 后端:

```py
AUTHENTICATION_BACKENDS = (
    'social.backends.linkedin.LinkedinOAuth2',
    'django.contrib.auth.backends.ModelBackend',
)
```

### 运行迁移

一旦注册，[更新数据库](https://realpython.com/django-migrations-a-primer/):

```py
$ python manage.py makemigrations
$ python manage.py migrate
```

### 更新网址

在项目的 *urls.py* 文件中，更新 urlpatters 以包含主 auth URLs:

```py
urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url('', include('social.apps.django_app.urls', namespace='social')),
]
```

接下来，我们需要从 LinkedIn 应用程序中获取所需的认证密钥。这一过程与许多流行的社交网络类似，比如 Twitter、脸书和谷歌。

### LinkedIn 认证密钥

为了让我们的应用程序清楚地识别我们实现的 LinkedIn 社交登录，我们需要一些特定于应用程序的凭据，以区分我们的应用程序登录和网络上的其他社交登录。

在[https://www.linkedin.com/developer/apps](https://www.linkedin.com/developer/apps)创建一个新的应用程序，并确保使用一个回调/重定向 URL[http://127 . 0 . 0 . 1:8000/complete/LinkedIn-oauth 2/](http://127.0.0.1:8000/complete/linkedin-oauth2/)(非常重要！).请记住，这个 URL 是特定于 OAuth 2.0 的。

> **注意**:上面使用的回调 URL 仅在本地开发中有效，当您转移到生产或试运行环境时需要更改。

在“django_social_project”目录中，添加一个名为 *config.py* 的新文件。然后从 LinkedIn 获取消费者密钥(API 密钥)和消费者秘密(API 秘密),并将它们添加到文件中:

```py
SOCIAL_AUTH_LINKEDIN_OAUTH2_KEY = 'update me'
SOCIAL_AUTH_LINKEDIN_OAUTH2_SECRET = 'update me'
```

让我们将以下 URL 添加到 *config.py* 文件中，以指定登录和重定向 URL(在用户验证之后):

```py
SOCIAL_AUTH_LOGIN_REDIRECT_URL = '/home/'
SOCIAL_AUTH_LOGIN_URL = '/'
```

将以下导入添加到 *settings.py*

```py
from config import *
```

为了让我们的应用程序成功完成登录过程，我们需要定义与这些 URL 相关联的模板和视图。让我们现在做那件事。

[*Remove ads*](/account/join/)

## 友好的观点

为了检查我们的应用程序是否工作，我们只需要两个视图- *登录*和*主页*。

### URLs

更新项目的 *urls.py* 文件中的 urlpatterns，以将我们的 URL 映射到我们将在后续部分中看到的视图:

```py
urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url('', include('social.apps.django_app.urls', namespace='social')),
    url(r'^$', 'django_social_app.views.login'),
    url(r'^home/$', 'django_social_app.views.home'),
    url(r'^logout/$', 'django_social_app.views.logout'),
]
```

### 视图

现在，将视图添加到应用程序的 *views.py* 中，以使我们的路线知道当特定路线被击中时应该做什么。

```py
from django.shortcuts import render_to_response, redirect
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.template.context import RequestContext

def login(request):
    return render_to_response('login.html', context=RequestContext(request))

@login_required(login_url='/')
def home(request):
    return render_to_response('home.html')

def logout(request):
    auth_logout(request)
    return redirect('/')
```

因此，在登录函数中，我们使用`RequestContext`获取登录的用户。

### 模板

将两个模板添加到“模板”文件夹-*login.html*:

```py
<!-- login.html -->
{% if user and not user.is_anonymous %}
  <a>Hello, {{ user.get_full_name }}!</a>
  <br>
  <a href="/logout">Logout</a>
{% else %}
  <a href="{% url 'social:begin' backend='linkedin-oauth2' %}">Login with Linkedin</a>
{% endif %}
```

还有*home.html*:

```py
<!-- home.html -->
<h1>Welcome</h1>
<br>
<p><a href="/logout">Logout</a>
```

您的项目现在应该如下所示:

```py
└── django_social_project
    ├── db.sqlite3
    ├── django_social_app
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── migrations
    │   │   └── __init__.py
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    ├── django_social_project
    │   ├── __init__.py
    │   ├── config.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── manage.py
    └── templates
        ├── home.html
        └── login.html
```

## 测试！

现在只需再次启动服务器进行测试:

```py
$ python manage.py runserver
```

只需浏览到 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 就会看到一个“用 LinkedIn 登录”的超链接。测试一下，确保一切正常。

抢码[这里](https://github.com/realpython/python-social-auth)。干杯！***
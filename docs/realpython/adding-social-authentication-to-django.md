# 向 Django 添加社会认证

> 原文：<https://realpython.com/adding-social-authentication-to-django/>

Python Social Auth 是一个库，它提供了“一个易于设置的社交认证/注册机制，支持多个[框架](http://psa.matiasaguirre.net/docs/intro.html#supported-frameworks)和[认证提供者](http://psa.matiasaguirre.net/docs/intro.html#auth-providers)”。在本教程中，我们将详细介绍如何将这个库集成到一个 [Django 项目](https://realpython.com/get-started-with-django-1/)中来提供用户认证。

**我们使用的是什么**:

*   Django==1.7.1
*   python-social-auth==0.2.1

## Django 设置

> 如果你已经准备好了一个项目，可以跳过这一部分。

创建并激活一个 virtualenv，安装 Django，然后启动一个新的 Django 项目:

```py
$ django-admin.py startproject django_social_project
$ cd django_social_project
$ python manage.py startapp django_social_app
```

[设置初始表](https://realpython.com/django-migrations-a-primer/)并添加超级用户:

```py
$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, contenttypes, auth, sessions
Running migrations:
 Applying contenttypes.0001_initial... OK
 Applying auth.0001_initial... OK
 Applying admin.0001_initial... OK
 Applying sessions.0001_initial... OK

$ python manage.py createsuperuser
Username (leave blank to use 'michaelherman'): admin
Email address: ad@min.com
Password:
Password (again):
Superuser created successfully.
```

在项目根目录下创建一个名为“templates”的新目录，然后将正确的路径添加到 *settings.py* 文件中:

```py
TEMPLATE_DIRS = (
    os.path.join(BASE_DIR, 'templates'),
)
```

运行开发服务器以确保一切正常，然后导航到 [http://localhost:8000/](http://localhost:8000/) 。你应该看到“成功了！”页面。

您的项目应该如下所示:

```py
└── django_social_project
    ├── db.sqlite3
    ├── django_social_app
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── migrations
    │   │   └── __init__.py
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    ├── django_social_project
    │   ├── __init__.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── manage.py
    └── templates
```

[*Remove ads*](/account/join/)

## Python 社交认证设置

按照以下步骤和/或[官方安装指南](http://psa.matiasaguirre.net/docs/index.html)安装和设置基本配置。

### 安装

使用 pip 安装:

```py
$ pip install python-social-auth==0.2.1
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
    'django_social_project',
    'social.apps.django_app.default',
)

TEMPLATE_CONTEXT_PROCESSORS = (
    'django.contrib.auth.context_processors.auth',
    'django.core.context_processors.debug',
    'django.core.context_processors.i18n',
    'django.core.context_processors.media',
    'django.core.context_processors.static',
    'django.core.context_processors.tz',
    'django.contrib.messages.context_processors.messages',
    'social.apps.django_app.context_processors.backends',
    'social.apps.django_app.context_processors.login_redirect',
)

AUTHENTICATION_BACKENDS = (
    'social.backends.facebook.FacebookOAuth2',
    'social.backends.google.GoogleOAuth2',
    'social.backends.twitter.TwitterOAuth',
    'django.contrib.auth.backends.ModelBackend',
)
```

注册后，更新数据库:

```py
$ python manage.py makemigrations
Migrations for 'default':
 0002_auto_20141109_1829.py:
 - Alter field user on usersocialauth

$ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, default, contenttypes, auth, sessions
Running migrations:
 Applying default.0001_initial... OK
 Applying default.0002_auto_20141109_1829... OK
```

在 *urls.py* 中更新项目的`urlpatterns`,以包含主要的授权 URL:

```py
urlpatterns = patterns(
    '',
    url(r'^admin/', include(admin.site.urls)),
    url('', include('social.apps.django_app.urls', namespace='social')),
)
```

接下来，您需要从想要包含的每个社交应用程序中获取所需的认证密钥。这个过程对于许多流行的社交网络来说是相似的——像推特、脸书和谷歌。让我们以 Twitter 为例…

## Twitter 认证密钥

在[https://apps.twitter.com/app/new](https://apps.twitter.com/app/new)创建一个新的应用程序，并确保使用回调 URL[http://127 . 0 . 0 . 1:8000/complete/Twitter](http://127.0.0.1:8000/complete/twitter)。

在“django_social_project”目录中，添加一个名为 *config.py* 的新文件。从 Twitter 的“Keys and Access Tokens”选项卡下获取`Consumer Key (API Key)`和`Consumer Secret (API Secret)`,并将其添加到配置文件中，如下所示:

```py
SOCIAL_AUTH_TWITTER_KEY = 'update me'
SOCIAL_AUTH_TWITTER_SECRET = 'update me'
```

让我们将以下 URL 添加到 *config.py* 中，以指定登录和重定向 URL(在用户验证之后):

```py
SOCIAL_AUTH_LOGIN_REDIRECT_URL = '/home/'
SOCIAL_AUTH_LOGIN_URL = '/'
```

将以下导入添加到 *settings.py* :

```py
from config import *
```

确保将 *config.py* 添加到您的*中。gitignore* 文件，因为*不想*将此文件添加到版本控制中，因为它包含敏感信息。

欲了解更多信息，请查看[官方文件](http://python-social-auth.readthedocs.org/en/latest/backends/twitter.html?highlight=twitter)。

[*Remove ads*](/account/join/)

## 健全性检查

让我们来测试一下。启动服务器并导航到[http://127 . 0 . 0 . 1:8000/log in/Twitter](http://127.0.0.1:8000/login/twitter)，授权应用程序，如果一切正常，你应该会被重定向到[http://127 . 0 . 0 . 1:8000/home/](http://127.0.0.1:8000/home/)(与`SOCIAL_AUTH_LOGIN_REDIRECT_URL`相关联的 URL)。您应该会看到 404 错误，因为我们还没有设置路由、视图或模板。

让我们现在就开始吧…

## 友好的观点

现在，我们只需要两个视图——登录和主页。

### URLs

更新 *urls.py* 中的 URL 模式:

```py
urlpatterns = patterns(
    '',
    url(r'^admin/', include(admin.site.urls)),
    url('', include('social.apps.django_app.urls', namespace='social')),
    url(r'^$', 'django_social_app.views.login'),
    url(r'^home/$', 'django_social_app.views.home'),
    url(r'^logout/$', 'django_social_app.views.logout'),
)
```

除了`/`和`home/`航线，我们还增加了`logout/`航线。

### 视图

接下来，添加以下视图函数:

```py
from django.shortcuts import render_to_response, redirect, render
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
# from django.template.context import RequestContext

def login(request):
    # context = RequestContext(request, {
    #     'request': request, 'user': request.user})
    # return render_to_response('login.html', context_instance=context)
    return render(request, 'login.html')

@login_required(login_url='/')
def home(request):
    return render_to_response('home.html')

def logout(request):
    auth_logout(request)
    return redirect('/')
```

在`login()`函数中，我们用`RequestContext`获取登录的用户。作为参考，实现这一点的更明确的方法被注释掉了。

### 模板

添加两个模板*home.html*和 login.htmlT2。

**home.html**

```py
<h1>Welcome</h1>
<p><a href="/logout">Logout</a>
```

**login.html**

```py
{% if user and not user.is_anonymous %}
  <a>Hello, {{ user.get_full_name }}!</a>
  <br>
  <a href="/logout">Logout</a>
{% else %}
  <a href="{% url 'social:begin' 'twitter' %}?next={{ request.path }}">Login with Twitter</a>
{% endif %}
```

您的项目现在应该如下所示:

```py
└── django_social_project
    ├── db.sqlite3
    ├── django_social_app
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── migrations
    │   │   └── __init__.py
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    ├── django_social_project
    │   ├── __init__.py
    │   ├── config.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── manage.py
    └── templates
        ├── home.html
        └── login.html
```

再测试一次。启动服务器。确保首先注销，因为用户应该已经从最后一次测试中登录，然后测试登录和注销。登录后，用户应该被重定向到`/home`。

[*Remove ads*](/account/join/)

## 接下来的步骤

在这一点上，你可能想要添加更多的认证提供者——像[脸书](http://python-social-auth.readthedocs.org/en/latest/backends/facebook.html)和[谷歌](http://python-social-auth.readthedocs.org/en/latest/backends/google.html)。添加新的社交身份验证提供者的工作流程很简单:

1.  在提供商的网站上创建新的应用程序。
2.  设置回调 URL。
3.  获取密钥/令牌并将其添加到 *config.py* 中。
4.  将新的提供者添加到 *settings.py* 中的`AUTHENTICATION_BACKENDS`元组中。
5.  通过添加新的 URL 来更新登录模板，如 so - `<a href="{% url 'social:begin' 'ADD AUTHENTICATION PROVIDER NAME' %}?next={{ request.path }}">Login with AUTHENTICATION PROVIDER NAME</a>`。

请查看[正式文件](http://psa.matiasaguirre.net/docs/)了解更多信息。请在下面留下评论和问题。感谢阅读！

哦——一定要从[回购](https://github.com/realpython/django-social-auth-example)中获取代码。***
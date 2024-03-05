# 使用 Django、Vue 和 GraphQL 创建一个博客

> 原文：<https://realpython.com/python-django-blog/>

你经常使用 Django 吗？你有没有发现自己想要将后端和前端解耦？您是否希望在 API 中处理数据持久性，同时使用 React 或 Vue 等客户端框架在浏览器中的单页应用程序(SPA)中显示数据？你很幸运。本教程将带你完成构建 Django 博客后端和前端的过程，使用 [GraphQL](https://graphql.org/) 在它们之间进行通信。

[项目](https://realpython.com/intermediate-python-project-ideas/)是学习和巩固概念的有效途径。本教程是一个循序渐进的项目，因此您可以通过实践的方式进行学习，并根据需要进行休息。

**在本教程中，您将学习如何:**

*   将你的 **Django 模型**转换成 **GraphQL API**
*   在你的电脑上同时运行 **Django 服务器**和 **Vue 应用**
*   在 **Django admin** 中管理您的博客文章
*   在 Vue 中使用 graph QL API**在浏览器中显示数据**

您可以点击下面的链接，下载所有用于构建 Django 博客应用程序的源代码:

**获取源代码:** [单击此处获取源代码，您将在本教程中使用](https://realpython.com/bonus/django-blog-project-code/)用 Django、Vue 和 GraphQL 构建一个博客应用程序。

## 演示:一个 Django 博客管理员，一个 GraphQL API 和一个 Vue 前端

博客应用程序是一个常见的入门项目，因为它们涉及创建、读取、更新和删除(CRUD)操作。在这个项目中，您将使用 Django admin 来完成繁重的 CRUD 提升工作，并专注于为您的博客数据提供 GraphQL API。

这是一个完整项目的实际演示:

[https://player.vimeo.com/video/540329665?background=1](https://player.vimeo.com/video/540329665?background=1)

接下来，在开始构建您的博客应用程序之前，您将确保您拥有所有必要的背景信息和工具。

[*Remove ads*](/account/join/)

## 项目概述

您将创建一个具有一些基本功能的小型博客应用程序。作者可以写很多帖子。帖子可以有许多标签，可以是已发布的，也可以是未发布的。

您将在 Django 中构建这个博客的后端，并配备一名管理员来添加新的博客内容。然后将内容数据作为 GraphQL API 公开，并使用 Vue 在浏览器中显示这些数据。您将通过几个高级步骤来实现这一点:

1.  建立 Django 博客
2.  创建 Django 博客管理员
3.  建立石墨烯-Django
4.  设置`django-cors-headers`
5.  设置 vue . js
6.  设置 Vue 路由器
7.  创建 Vue 组件
8.  获取数据

每个部分都将提供任何必要资源的链接，并给你一个暂停并根据需要返回的机会。

## 先决条件

如果您已经对一些 web 应用程序概念有了坚实的基础，那么您将最适合学习本教程。你应该明白 [HTTP 请求和响应](https://realpython.com/python-requests/)以及 API 是如何工作的。您可以查看[Python&API:读取公共数据的成功组合](https://realpython.com/python-api/)，以了解使用 GraphQL APIs 与 REST APIs 的细节。

因为您将使用 Django 为您的博客构建后端，所以您将希望熟悉开始 Django 项目的[和定制 Django 管理的](https://realpython.com/django-setup/)和[。如果您以前没有怎么使用过 Django，您可能还想先尝试构建另一个仅支持 Django 的项目。要获得好的介绍，请查看](https://realpython.com/customize-django-admin-python/)[Django 入门第 1 部分:构建投资组合应用](https://realpython.com/get-started-with-django-1/)。

因为您将在前端使用 Vue，所以一些关于 reactive [JavaScript](https://realpython.com/python-vs-javascript/) 的经验也会有所帮助。如果你过去只在类似于 [jQuery](https://jquery.com/) 的框架中使用过 DOM 操作范例，那么 [Vue 简介](https://vuejs.org/v2/guide/)是一个很好的基础。

熟悉 JSON 也很重要，因为 GraphQL 查询类似于 JSON，并以 JSON 格式返回数据。你可以阅读关于在 Python 中使用 JSON 数据的[作为介绍。你还需要](https://realpython.com/python-json/)[安装 Node.js](https://nodejs.org/en/download/package-manager) 在本教程后面的前端工作。

## 第一步:建立 Django 博客

在深入之前，您需要一个目录，在其中您可以组织项目的代码。首先创建一个名为`dvg/`的，是 Django-Vue-GraphQL 的缩写:

```py
$ mkdir dvg/
$ cd dvg/
```

您还将完全分离前端和后端代码，因此立即开始创建这种分离是个好主意。在您的项目目录中创建一个`backend/`目录:

```py
$ mkdir backend/
$ cd backend/
```

您将把您的 Django 代码放在这个目录中，与您将在本教程后面创建的 Vue 代码完全隔离。

### 安装 Django

现在您已经准备好开始构建 Django 应用程序了。为了将这个项目与其他项目的依赖项分开，创建一个**虚拟环境**，在其中安装项目的需求。你可以在 [Python 虚拟环境:初级读本](https://realpython.com/python-virtual-environments-a-primer/)中阅读更多关于虚拟环境的内容。本教程的其余部分假设您将在活动的虚拟环境中运行与 Python 和 Django 相关的命令。

现在您已经有了一个安装需求的虚拟环境，在`backend/`目录中创建一个`requirements.txt`文件，并定义您需要的第一个需求:

```py
Django==3.1.7
```

一旦保存了`requirements.txt`文件，就用它来安装 Django:

```py
(venv) $ python -m pip install -r requirements.txt
```

现在您可以开始创建您的 Django 项目了。

[*Remove ads*](/account/join/)

### 创建 Django 项目

现在 Django 已经安装好了，使用`django-admin`命令[初始化您的 Django 项目](https://realpython.com/django-setup/):

```py
(venv) $ django-admin startproject backend .
```

这将在`backend/`目录中创建一个`manage.py`模块和一个`backend`包，因此您的项目目录结构应该如下所示:

```py
dvg
└── backend
    ├── manage.py
    ├── requirements.txt
    └── backend
        ├── __init__.py
        ├── asgi.py
        ├── settings.py
        ├── urls.py
        └── wsgi.py
```

本教程不会涵盖或需要所有这些文件，但它不会伤害他们的存在。

### 运行 Django 迁移

在向您的应用程序添加任何特定的东西之前，您还应该运行 Django 的初始**迁移**。如果你以前没有处理过迁移，那么看看 [Django 迁移:初级读本](https://realpython.com/django-migrations-a-primer/)。使用`migrate`管理命令运行迁移:

```py
(venv) $ python manage.py migrate
```

您应该会看到一个很长的迁移列表，每个后面都有一个`OK`:

```py
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying auth.0010_alter_group_name_max_length... OK
  Applying auth.0011_update_proxy_permissions... OK
  Applying auth.0012_alter_user_first_name_max_length... OK
  Applying sessions.0001_initial... OK
```

这将创建一个名为`db.sqlite3`的 SQLite 数据库文件，该文件也将存储项目的其余数据。

### 创建超级用户

现在你有了数据库，你可以创建一个超级用户。您将需要这个用户，这样您最终可以登录到 Django 管理界面。使用`createsuperuser`管理命令创建一个:

```py
(venv) $ python manage.py createsuperuser
```

在下一节中，您将能够使用在这一步中提供的用户名和密码登录 Django admin。

### 第一步总结

现在您已经安装了 Django，创建了 Django 项目，运行了 Django 迁移，并创建了一个超级用户，您就有了一个功能完整的 Django 应用程序。现在，您应该能够启动 Django 开发服务器，并在浏览器中查看它。使用`runserver`管理命令启动服务器，默认情况下它将监听端口`8000`:

```py
(venv) $ python manage.py runserver
```

现在在浏览器中访问`http://localhost:8000`。您应该看到 Django 启动页面，表明安装成功。您还应该能够访问`http://localhost:8000/admin`，在那里您会看到一个登录表单。

使用您为超级用户创建的用户名和密码登录 Django admin。如果一切正常，那么你将被带到 **Django 管理仪表板**页面。这个页面目前还很空，但是在下一步中你会让它变得更有趣。

[*Remove ads*](/account/join/)

## 步骤 2:创建 Django 博客管理员

现在您已经有了 Django 项目的基础，可以开始为您的博客创建一些核心业务逻辑了。在这一步中，您将创建用于创作和管理博客内容的**数据模型**和**管理配置**。

### 创建 Django 博客应用程序

请记住，一个 Django 项目可以包含许多 Django 应用程序。您应该将特定于博客的行为分离到它自己的 Django 应用程序中，以便它与您将来构建到项目中的任何应用程序保持区别。使用`startapp`管理命令创建应用程序:

```py
(venv) $ python manage.py startapp blog
```

这将创建一个包含几个框架文件的`blog/`目录:

```py
blog
├── __init__.py
├── admin.py
├── apps.py
├── migrations
│   └── __init__.py
├── models.py
├── tests.py
└── views.py
```

在本教程的后面部分，您将对其中一些文件进行更改和添加。

### 启用 Django 博客应用程序

默认情况下，创建 Django 应用程序不会使它在您的项目中可用。为了确保项目知道您的新`blog`应用程序，您需要将它添加到已安装应用程序的列表中。更新`backend/settings.py`中的`INSTALLED_APPS`变量:

```py
INSTALLED_APPS = [
  ...
  "blog",
]
```

这将有助于 Django 发现关于您的应用程序的信息，比如它包含的数据模型和 URL 模式。

### 创建 Django 博客数据模型

既然 Django 可以发现您的`blog`应用程序，您就可以创建数据模型了。首先，您将创建三个模型:

1.  **`Profile`** 存储博客用户的附加信息。
2.  **`Tag`** 代表博客帖子可以分组的类别。
3.  **`Post`** 存储每篇博文的内容和元数据。

您将把这些型号添加到`blog/models.py`中。首先，[导入](https://realpython.com/python-import/) Django 的`django.db.models`模块:

```py
from django.db import models
```

你的每个模型都将从`models.Model`类继承。

#### `Profile`型号

`Profile`模型将有几个字段:

*   **`user`** 是与配置文件关联的 Django 用户的一对一关联。
*   **`website`** 是一个可选的网址，您可以在这里了解有关用户的更多信息。
*   **`bio`** 是一个可选的、推文大小的广告，用于快速了解用户的更多信息。

首先需要从 Django 导入`settings`模块:

```py
from django.conf import settings
```

然后创建`Profile`模型，它应该类似于下面的代码片段:

```py
class Profile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
    )
    website = models.URLField(blank=True)
    bio = models.CharField(max_length=240, blank=True)

    def __str__(self):
        return self.user.get_username()
```

`__str__`方法将使您创建的`Profile`对象以更加人性化的方式出现在管理站点上。

#### `Tag`型号

`Tag`模型只有一个字段`name`，它为标签存储一个简短的、惟一的名称。创建`Tag`模型，它应该类似于下面的代码片段:

```py
class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name
```

同样，`__str__`将使您创建的`Tag`对象以更加人性化的方式出现在管理站点上。

#### `Post`型号

如你所想，模型是最复杂的。它将有几个字段:

| 字段名 | 目的 |
| --- | --- |
| `title` | 向读者显示的文章的唯一标题 |
| `subtitle` | 帖子内容的可选澄清器，帮助读者了解他们是否想阅读 |
| `slug` | 帖子在 URL 中使用的唯一可读标识符 |
| `body` | 帖子的内容 |
| `meta_description` | 用于 Google 等搜索引擎的可选描述 |
| `date_created` | 帖子创建的时间戳 |
| `date_modified` | 帖子最近一次编辑的时间戳 |
| `publish_date` | 帖子发布时的可选时间戳 |
| `published` | 文章当前是否对读者可用 |
| `author` | 对撰写帖子的用户个人资料的引用 |
| `tags` | 与帖子相关联的标签列表(如果有) |

因为博客通常首先显示最近的帖子，所以您也希望`ordering`按照发布日期显示，最近的放在最前面。创建`Post`模型，它应该类似于下面的代码片段:

```py
class Post(models.Model):
    class Meta:
        ordering = ["-publish_date"]

    title = models.CharField(max_length=255, unique=True)
    subtitle = models.CharField(max_length=255, blank=True)
    slug = models.SlugField(max_length=255, unique=True)
    body = models.TextField()
    meta_description = models.CharField(max_length=150, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_modified = models.DateTimeField(auto_now=True)
    publish_date = models.DateTimeField(blank=True, null=True)
    published = models.BooleanField(default=False)

    author = models.ForeignKey(Profile, on_delete=models.PROTECT)
    tags = models.ManyToManyField(Tag, blank=True)
```

`author`的`on_delete=models.PROTECT`参数确保您不会意外删除仍在博客上发表文章的作者。与`Tag`的`ManyToManyField`关系允许您将一篇文章与零个或多个标签相关联。每个标签可以关联到许多文章。

[*Remove ads*](/account/join/)

### 创建模型管理配置

现在模型已经准备好了，您需要告诉 Django 它们应该如何在管理界面中显示。在`blog/admin.py`中，首先导入 Django 的`admin`模块和您的模型:

```py
from django.contrib import admin

from blog.models import Profile, Post, Tag
```

然后为`Profile`和`Tag`创建并注册管理类，它们只需要指定的`model`:

```py
@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    model = Profile

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    model = Tag
```

就像模型一样，`Post`的管理类更加复杂。帖子包含大量信息，因此更明智地选择显示哪些信息有助于避免界面拥挤。

在所有帖子的列表中，您将指定 Django 应该只显示每个帖子的以下信息:

1.  身份证明
2.  标题
3.  小标题
4.  鼻涕虫
5.  出版日期
6.  发布状态

为了使浏览和编辑帖子更加流畅，您还将告诉 Django 管理系统采取以下操作:

*   允许按已发布或未发布的帖子过滤帖子列表。
*   允许按发布日期过滤帖子。
*   允许编辑所有显示的字段，ID 除外。
*   允许使用标题、副标题、段落和正文搜索帖子。
*   使用标题和副标题字段预填充 slug 字段。
*   使用所有帖子的发布日期创建一个可浏览的日期层次结构。
*   在列表顶部显示按钮以保存更改。

创建并注册`PostAdmin`类:

```py
@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    model = Post

    list_display = (
        "id",
        "title",
        "subtitle",
        "slug",
        "publish_date",
        "published",
    )
    list_filter = (
        "published",
        "publish_date",
    )
    list_editable = (
        "title",
        "subtitle",
        "slug",
        "publish_date",
        "published",
    )
    search_fields = (
        "title",
        "subtitle",
        "slug",
        "body",
    )
    prepopulated_fields = {
        "slug": (
            "title",
            "subtitle",
        )
    }
    date_hierarchy = "publish_date"
    save_on_top = True
```

你可以在[用 Python 定制 Django 管理](https://realpython.com/customize-django-admin-python/)中阅读更多关于 Django 管理提供的所有选项。

### 创建模型迁移

Django 拥有管理和保存博客内容所需的所有信息，但是您首先需要更新数据库以支持这些更改。在本教程的前面，您运行了 Django 内置模型的迁移。现在，您将为您的模型创建并运行迁移。

首先，使用`makemigrations`管理命令创建迁移:

```py
(venv) $ python manage.py makemigrations
Migrations for 'blog':
 blog/migrations/0001_initial.py
 - Create model Tag
 - Create model Profile
 - Create model Post
```

这将创建一个默认名称为`0001_initial.py`的迁移。使用`migrate`管理命令运行该迁移:

```py
(venv) $ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, blog, contenttypes, sessions
Running migrations:
 Applying blog.0001_initial... OK
```

请注意，迁移名称后面应该有`OK`。

[*Remove ads*](/account/join/)

### 第二步总结

现在您已经准备好了所有的数据模型，并且已经配置了 Django admin，这样您就可以添加和编辑这些模型了。

启动或重启 Django 开发服务器，在`http://localhost:8000/admin`访问管理界面，探索发生了什么变化。您应该会看到标签、个人资料和文章列表的链接，以及添加或编辑它们的链接。尝试添加和编辑其中的一些，看看管理界面是如何响应的。

## 第三步:建立石墨烯-Django

在这一点上，你已经完成了足够的后端，你*可以*决定一头扎进 Django 方向。您可以使用 Django 的 URL 路由和模板引擎来构建页面，向读者显示您在 admin 中创建的所有帖子内容。相反，您将把自己创建的后端封装在 GraphQL API 中，以便最终可以从浏览器中使用它，并提供更丰富的客户端体验。

GraphQL 允许您只检索您需要的数据，与 RESTful APIs 中常见的非常大的响应相比，这是非常有用的。GraphQL 还在投影数据方面提供了更多的灵活性，因此您可以经常以新的方式检索数据，而无需更改提供 GraphQL API 的服务的逻辑。

您将使用 [Graphene-Django](https://docs.graphene-python.org/projects/django/en/latest/) 将您目前创建的内容集成到 GraphQL API 中。

### 安装石墨烯-Django

要开始使用 Graphene-Django，首先将其添加到项目的需求文件中:

```py
graphene-django==2.14.0
```

然后使用更新的需求文件安装它:

```py
(venv) $ python -m pip install -r requirements.txt
```

将`"graphene_django"`添加到项目的`settings.py`模块的`INSTALLED_APPS`变量中，这样 Django 就会找到它:

```py
INSTALLED_APPS = [
  ...
  "blog",
  "graphene_django",
]
```

Graphene-Django 现在已经安装完毕，可以进行配置了。

### 配置石墨烯-Django

要让 Graphene-Django 在您的项目中工作，您需要配置几个部分:

1.  更新`settings.py`以便项目知道在哪里寻找 GraphQL 信息。
2.  添加一个 URL 模式来服务 GraphQL API 和 GraphQL 的可探索接口 GraphQL。
3.  创建 Graphene-Django 的 GraphQL 模式，这样 Graphene-Django 就知道如何将您的模型转换成 GraphQL。

#### 更新 Django 设置

`GRAPHENE`设置将 Graphene-Django 配置为在特定位置寻找 GraphQL 模式。将它指向`blog.schema.schema` Python 路径，您将很快创建该路径:

```py
GRAPHENE = {
  "SCHEMA": "blog.schema.schema",
}
```

注意，这个添加可能会导致 Django 产生一个导入错误，您可以在创建 GraphQL 模式时解决这个错误。

#### 为 GraphQL 和 graph QL 添加 URL 模式

为了让 Django 服务于 GraphQL 端点和 graph QL 接口，您将向`backend/urls.py`添加一个新的 URL 模式。你会把网址指向 Graphene-Django 的`GraphQLView`。因为您没有使用 Django 模板引擎的[跨站点请求伪造(CSRF)](https://en.wikipedia.org/wiki/Cross-site_request_forgery) 保护特性，所以您还需要导入 Django 的`csrf_exempt`装饰器来将视图标记为免于 CSRF 保护:

```py
from django.views.decorators.csrf import csrf_exempt
from graphene_django.views import GraphQLView
```

然后，将新的 URL 模式添加到`urlpatterns`变量中:

```py
urlpatterns = [
    ...
    path("graphql", csrf_exempt(GraphQLView.as_view(graphiql=True))),
]
```

`graphiql=True`参数告诉 Graphene-Django 使 GraphiQL 接口可用。

#### 创建 GraphQL 模式

现在您将创建 GraphQL 模式，这应该与您之前创建的管理配置类似。该模式由几个类组成，每个类都与一个特定的 Django 模型相关联，还有一个类指定如何解决前端需要的一些重要类型的查询。

在`blog/`目录下创建一个新的`schema.py`模块。导入 Graphene-Django 的`DjangoObjectType`，您的`blog`模型，以及 Django 的`User`模型:

```py
from django.contrib.auth import get_user_model
from graphene_django import DjangoObjectType

from blog import models
```

为您的每个模型和`User`模型创建一个相应的类。它们每个都应该有一个以`Type`结尾的名字，因为每个都代表一个 [GraphQL 类型](https://graphql.org/learn/schema/#type-system)。您的类应该如下所示:

```py
class UserType(DjangoObjectType):
    class Meta:
        model = get_user_model()

class AuthorType(DjangoObjectType):
    class Meta:
        model = models.Profile

class PostType(DjangoObjectType):
    class Meta:
        model = models.Post

class TagType(DjangoObjectType):
    class Meta:
        model = models.Tag
```

您需要创建一个继承自`graphene.ObjectType`的`Query`类。这个类将集合您创建的所有类型类，并且您将向它添加方法来指示您的模型可以被查询的方式。你需要先导入`graphene`:

```py
import graphene
```

`Query`类由许多属性组成，这些属性或者是`graphene.List`或者是`graphene.Field`。如果查询应该返回单个项目，您将使用`graphene.Field`，如果查询将返回多个项目，您将使用`graphene.List`。

对于这些属性中的每一个，您还将创建一个方法来解析查询。通过获取查询中提供的信息并返回相应的 Django queryset 来解析查询。

每个解析器的方法必须以`resolve_`开头，名称的其余部分应该匹配相应的属性。例如，为属性`all_posts`解析 queryset 的方法必须命名为`resolve_all_posts`。

您将创建查询来获取:

*   所有的帖子
*   具有给定用户名的作者
*   具有给定 slug 的帖子
*   给定作者的所有帖子
*   带有给定标签的所有帖子

现在创建`Query`类。它应该类似于下面的代码片段:

```py
class Query(graphene.ObjectType):
    all_posts = graphene.List(PostType)
    author_by_username = graphene.Field(AuthorType, username=graphene.String())
    post_by_slug = graphene.Field(PostType, slug=graphene.String())
    posts_by_author = graphene.List(PostType, username=graphene.String())
    posts_by_tag = graphene.List(PostType, tag=graphene.String())

    def resolve_all_posts(root, info):
        return (
            models.Post.objects.prefetch_related("tags")
            .select_related("author")
            .all()
        )

    def resolve_author_by_username(root, info, username):
        return models.Profile.objects.select_related("user").get(
            user__username=username
        )

    def resolve_post_by_slug(root, info, slug):
        return (
            models.Post.objects.prefetch_related("tags")
            .select_related("author")
            .get(slug=slug)
        )

    def resolve_posts_by_author(root, info, username):
        return (
            models.Post.objects.prefetch_related("tags")
            .select_related("author")
            .filter(author__user__username=username)
        )

    def resolve_posts_by_tag(root, info, tag):
        return (
            models.Post.objects.prefetch_related("tags")
            .select_related("author")
            .filter(tags__name__iexact=tag)
        )
```

现在您已经拥有了模式的所有类型和解析器，但是请记住您创建的`GRAPHENE`变量指向`blog.schema.schema`。创建一个`schema`变量，将您的`Query`类包装在`graphene.Schema`中，以便将它们联系在一起:

```py
schema = graphene.Schema(query=Query)
```

该变量与您在本教程前面为 Graphene-Django 配置的`"blog.schema.schema"`值相匹配。

[*Remove ads*](/account/join/)

### 第三步总结

您已经充实了您的博客的数据模型，现在您还用 Graphene-Django 包装了您的数据模型，以将该数据作为 GraphQL API。

运行 Django 开发服务器并访问`http://localhost:8000/graphql`。您应该看到 GraphiQL 界面，其中有一些解释如何使用该工具的注释文本。

展开屏幕右上方的*文档*部分，点击*查询:查询*。您应该会看到您在模式中配置的每个查询和类型。

如果您还没有创建任何测试博客内容，现在就创建吧。尝试以下查询，它将返回您创建的所有帖子的列表:

```py
{
  allPosts {
    title
    subtitle
    author {
      user {
        username
      }
    }
    tags {
      name
    }
  }
}
```

响应应该返回一个帖子列表。每个帖子的结构应该与查询的形状相匹配，如下例所示:

```py
{ "data":  { "allPosts":  [ { "title":  "The Great Coney Island Debate", "subtitle":  "American or Lafayette?", "author":  { "user":  { "username":  "coney15land" } }, "tags":  [ { "name":  "food" }, { "name":  "coney island" } ] } ] } }
```

如果你保存了一些帖子，并在回复中看到了它们，那么你就准备好继续了。

## 第四步:设置`django-cors-headers`

您还需要再走一步才能称后端工作完成。因为后端和前端将在本地不同的端口上运行，并且因为它们可能在生产环境中完全不同的域上运行，[跨源资源共享(CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) 开始发挥作用。如果不处理 CORS，从前端到后端的请求通常会被您的浏览器阻止。

这个项目让与 CORS 打交道变得相当轻松。您将使用它来告诉 Django 响应来自其他来源的请求，这将允许前端与 GraphQL API 正确通信。

### 安装`django-cors-headers`

首先，将`django-cors-headers`添加到您的需求文件中:

```py
django-cors-headers==3.6.0
```

然后使用更新的需求文件安装它:

```py
(venv) $ python -m pip install -r requirements.txt
```

将`"corsheaders"`添加到项目的`settings.py`模块的`INSTALLED_APPS`列表中:

```py
INSTALLED_APPS = [
  ...
  "corsheaders",
]
```

然后将`"corsheaders.middleware.CorsMiddleware"`添加到`MIDDLEWARE`变量的末尾:

```py
MIDDLEWARE = [
  "corsheaders.middleware.CorsMiddleware",
  ...
]
```

`django-cors-headers`文档建议将中间件尽可能早地放在`MIDDLEWARE`列表中。你可以把它放在这个项目列表的最顶端。

[*Remove ads*](/account/join/)

### 配置`django-cors-headers`

CORS 的存在是有充分理由的。您不希望您的应用程序暴露在互联网上的任何地方。您可以使用两个设置来非常精确地定义您希望打开 GraphQL API 的程度:

1.  **`CORS_ORIGIN_ALLOW_ALL`** 定义 Django 默认是全开还是全关。
2.  **`CORS_ORIGIN_WHITELIST`** 定义 Django 应用程序将允许哪些域的请求。

将以下设置添加到`settings.py`:

```py
CORS_ORIGIN_ALLOW_ALL = False
CORS_ORIGIN_WHITELIST = ("http://localhost:8080",)
```

这些设置将只允许来自前端的请求，您最终将在本地端口`8080`上运行这些请求。

### 第 4 步总结

后端完成！您有一个工作数据模型、一个工作管理界面、一个可以使用 GraphQL 探索的工作 GraphQL API，以及从您接下来要构建的前端查询 API 的能力。如果你已经有一段时间没有休息了，这是一个休息的好地方。

## 第五步:设置 Vue.js

您将使用 Vue 作为您博客的前端。要设置 Vue，您将创建 Vue 项目，安装几个重要的插件，并运行 Vue 开发服务器，以确保您的应用程序及其依赖项能够正常工作。

### 创建 Vue 项目

很像 Django，Vue 提供了一个**命令行界面**，用于创建一个项目，而不需要完全从零开始。您可以将其与 Node 的`npx`命令配对，以引导其他人发布的基于 JavaScript 的命令。使用这种方法，您不需要手动安装启动和运行 Vue 项目所需的各种独立的依赖项。现在使用`npx`创建您的 Vue 项目:

```
$ cd /path/to/dvg/
$ npx @vue/cli create frontend --default
...
🎉  Successfully created project frontend.
...
$ cd frontend/
```py

这将在现有的`backend/`目录旁边创建一个`frontend/`目录，安装一些 JavaScript 依赖项，并为应用程序创建一些框架文件。

### 安装检视外挂程式

你需要一些插件让 Vue 进行适当的浏览器路由，并与你的 GraphQL API 进行交互。这些插件有时会影响你的文件，所以最好在开始的时候安装它们，这样它们就不会覆盖任何东西，然后再配置它们。安装 Vue 路由器和 Vue Apollo 插件，在出现提示时选择默认选项:

```
$ npx @vue/cli add router
$ npx @vue/cli add apollo
```py

这些命令将花费一些时间来安装依赖项，它们将添加或更改项目中的一些文件，以配置和安装 Vue 项目中的每个插件。

### 第五步总结

您现在应该能够运行 Vue 开发服务器了:

```
$ npm run serve
```py

现在，Django 应用程序在`http://localhost:8000`运行，Vue 应用程序在`http://localhost:8080`运行。

在浏览器中访问`http://localhost:8080`。您应该会看到 Vue 启动页面，这表明您已经成功安装了所有东西。如果您看到 splash 页面，那么您已经准备好开始创建自己的组件了。

[*Remove ads*](/account/join/)

## 步骤 6:设置 Vue 路由器

客户端应用程序的一个重要部分是处理路由，而不必向服务器发出新的请求。Vue 中一个常见的解决方案是您之前安装的 [Vue 路由器](https://router.vuejs.org/)插件。你将使用 Vue 路由器代替普通的 HTML 锚标签来链接到你博客的不同页面。

### 创建路线

现在您已经安装了 Vue 路由器，您需要配置 Vue 来使用 Vue 路由器。您还需要为 Vue 路由器配置它应该路由的 URL 路径。

在`src/`目录下创建一个`router.js`模块。这个文件将保存关于哪个 URL 映射到哪个 Vue 组件的所有配置。从导入 Vue 和 Vue 路由器开始:

```
import  Vue  from  'vue' import  VueRouter  from  'vue-router'
```py

添加以下导入，每个导入对应于您稍后将创建的一个组件:

```
import  Post  from  '@/components/Post' import  Author  from  '@/components/Author' import  PostsByTag  from  '@/components/PostsByTag' import  AllPosts  from  '@/components/AllPosts'
```py

注册 Vue 路由器插件:

```
Vue.use(VueRouter)
```py

现在，您将创建路线列表。每条路线都有两个属性:

1.  **`path`** 是一个 URL 模式，可选地包含类似于 Django URL 模式的捕获变量。
2.  **`component`** 是当浏览器导航到与路径模式匹配的路线时显示的 Vue 组件。

添加这些路线作为一个`routes`变量。它们应该如下所示:

```
const  routes  =  [ {  path:  '/author/:username',  component:  Author  }, {  path:  '/post/:slug',  component:  Post  }, {  path:  '/tag/:tag',  component:  PostsByTag  }, {  path:  '/',  component:  AllPosts  }, ]
```py

创建一个新的`VueRouter`实例，并将其从`router.js`模块中导出，以便其他模块可以使用它:

```
const  router  =  new  VueRouter({ routes:  routes, mode:  'history', }) export  default  router
```py

在下一节中，您将在另一个模块中导入`router`变量。

### 安装路由器

在`src/main.js`的顶部，从您在上一节中创建的模块导入`router`:

```
import  router  from  '@/router'
```py

然后将路由器传递给 Vue 实例:

```
new  Vue({ router, ... })
```py

这就完成了 Vue 路由器的配置。

[*Remove ads*](/account/join/)

### 第六步总结

您已经为您的前端创建了路由，它将一个 URL 模式映射到将在该 URL 显示的组件。这些路径还不能工作，因为它们指向尚不存在的组件。您将在下一步中创建这些组件。

## 步骤 7:创建 Vue 组件

现在，您已经启动了 Vue 并运行了将到达您的组件的路由，您可以开始创建最终将显示来自 GraphQL 端点的数据的组件。目前，您只需要让它们显示一些静态内容。下表描述了您将创建的组件:

| 成分 | 显示 |
| --- | --- |
| `AuthorLink` | 给定作者页面的链接(在`Post`和`PostList`中使用) |
| `PostList` | 给定的博客帖子列表(在`AllPosts`、`Author`和`PostsByTag`中使用) |
| `AllPosts` | 所有帖子的列表，最新的放在最前面 |
| `PostsByTag` | 与给定标签相关的文章列表，最新的放在最前面 |
| `Post` | 给定帖子的元数据和内容 |
| `Author` | 关于作者的信息和他们写的文章列表 |

在下一步中，您将使用动态数据更新这些组件。

### `AuthorLink`组件

您将创建的第一个组件显示一个指向作者的链接。

在`src/components/`目录下创建一个`AuthorLink.vue`文件。该文件是一个 Vue 单文件组件(SFC)。sfc 包含正确呈现组件所需的 HTML、JavaScript 和 CSS。

`AuthorLink`接受一个`author`属性，其结构对应于 GraphQL API 中关于作者的数据。该组件应该显示用户的名字和姓氏(如果提供的话),否则显示用户的用户名。

您的`AuthorLink.vue`文件应该如下所示:

```
<template>
  <router-link
      :to="`/author/${author.user.username}`"
  >{{ displayName }}</router-link>
</template>

<script> export  default  { name:  'AuthorLink', props:  { author:  { type:  Object, required:  true, }, }, computed:  { displayName  ()  { return  ( this.author.user.firstName  && this.author.user.lastName  && `${this.author.user.firstName}  ${this.author.user.lastName}` )  ||  `${this.author.user.username}` }, }, } </script>
```py

这个组件不会直接使用 GraphQL。相反，其他组件将使用`author`属性传入作者信息。

### `PostList`组件

`PostList`组件接受一个`posts`属性，它的结构对应于 GraphQL API 中关于文章的数据。该组件还接受一个[布尔](https://realpython.com/python-boolean/) `showAuthor`属性，您将在作者的页面上将它设置为`false`，因为它是冗余信息。该组件应显示以下特征:

*   文章的标题和副标题，将它们链接到文章的页面
*   使用`AuthorLink`链接到文章作者(如果`showAuthor`是`true`)
*   帖子发布的日期
*   文章的元描述
*   与帖子相关联的标签列表

在`src/components/`目录中创建一个`PostList.vue` SFC。组件模板应该如下所示:

```
<template>
  <div>
    <ol class="post-list">
      <li class="post" v-for="post in publishedPosts" :key="post.title">
          <span class="post__title">
            <router-link
              :to="`/post/${post.slug}`"
            >{{ post.title }}: {{ post.subtitle }}</router-link>
          </span>
          <span v-if="showAuthor">
            by <AuthorLink :author="post.author" />
          </span>
          <div class="post__date">{{ displayableDate(post.publishDate) }}</div>
        <p class="post__description">{{ post.metaDescription }}</p>
        <ul>
          <li class="post__tags" v-for="tag in post.tags" :key="tag.name">
            <router-link :to="`/tag/${tag.name}`">#{{ tag.name }}</router-link>
          </li>
        </ul>
      </li>
    </ol>
  </div>
</template>
```py

`PostList`组件的 JavaScript 应该如下所示:

```
<script> import  AuthorLink  from  '@/components/AuthorLink' export  default  { name:  'PostList', components:  { AuthorLink, }, props:  { posts:  { type:  Array, required:  true, }, showAuthor:  { type:  Boolean, required:  false, default:  true, }, }, computed:  { publishedPosts  ()  { return  this.posts.filter(post  =>  post.published) } }, methods:  { displayableDate  (date)  { return  new  Intl.DateTimeFormat( 'en-US', {  dateStyle:  'full'  }, ).format(new  Date(date)) } }, } </script>
```py

`PostList`组件以`prop`的形式接收数据，而不是直接使用 GraphQL。

您可以添加一些可选的 CSS 样式，使帖子列表在呈现后更具可读性:

```
<style> .post-list  { list-style:  none; } .post  { border-bottom:  1px  solid  #ccc; padding-bottom:  1rem; } .post__title  { font-size:  1.25rem; } .post__description  { color:  #777; font-style:  italic; } .post__tags  { list-style:  none; font-weight:  bold; font-size:  0.8125rem; } </style>
```py

这些样式增加了一些间距，消除了一些混乱，区分了不同的信息，有助于浏览。

[*Remove ads*](/account/join/)

### `AllPosts`组件

您将创建的下一个组件是博客上所有帖子的列表。它需要显示两条信息:

1.  最近的帖子标题
2.  帖子列表，使用`PostList`

在`src/components/`目录下创建`AllPosts.vue` SFC。它应该如下所示:

```
<template>
  <div>
    <h2>Recent posts</h2>
    <PostList v-if="allPosts" :posts="allPosts" />
  </div>
</template>

<script> import  PostList  from  '@/components/PostList' export  default  { name:  'AllPosts', components:  { PostList, }, data  ()  { return  { allPosts:  null, } }, } </script>
```py

在本教程的后面，您将使用 GraphQL 查询动态填充`allPosts`变量。

### `PostsByTag`组件

`PostsByTag`组件与`AllPosts`组件非常相似。标题文本不同，在下一步中，您将查询一组不同的文章。

在`src/components/`目录下创建`PostsByTag.vue` SFC。它应该如下所示:

```
<template>
  <div>
    <h2>Posts in #{{ $route.params.tag }}</h2>
    <PostList :posts="posts" v-if="posts" />
  </div>
</template>

<script> import  PostList  from  '@/components/PostList' export  default  { name:  'PostsByTag', components:  { PostList, }, data  ()  { return  { posts:  null, } }, } </script>
```py

在本教程的后面，您将使用 GraphQL 查询填充`posts`变量。

### `Author`组件

`Author`组件充当作者的个人资料页面。它应该显示以下信息:

*   带有作者姓名的标题
*   作者网站的链接，如果提供的话
*   作者的传记，如果提供的话
*   作者的帖子列表，其中`showAuthor`设置为`false`

现在在`src/components/`目录下创建`Author.vue` SFC。它应该如下所示:

```
<template>
  <div v-if="author">
    <h2>{{ displayName }}</h2>
    <a
      :href="author.website"
      target="_blank"
      rel="noopener noreferrer"
    >Website</a>
    <p>{{ author.bio }}</p>

    <h3>Posts by {{ displayName }}</h3>
    <PostList :posts="author.postSet" :showAuthor="false" />
  </div>
</template>

<script> import  PostList  from  '@/components/PostList' export  default  { name:  'Author', components:  { PostList, }, data  ()  { return  { author:  null, } }, computed:  { displayName  ()  { return  ( this.author.user.firstName  && this.author.user.lastName  && `${this.author.user.firstName}  ${this.author.user.lastName}` )  ||  `${this.author.user.username}` }, }, } </script>
```py

在本教程的后面，您将使用 GraphQL 查询动态填充`author`变量。

### `Post`组件

就像数据模型一样，`Post`组件是最有趣的，因为它负责显示所有帖子的信息。该组件应显示关于 post 的以下信息:

*   标题和副标题，作为标题
*   作者，作为链接使用`AuthorLink`
*   出版日期
*   元描述
*   内容体
*   作为链接的关联标签列表

由于您的数据建模和组件架构，您可能会惊讶于这需要的代码如此之少。在`src/components/`目录下创建`Post.vue` SFC。它应该如下所示:

```
<template>
  <div class="post" v-if="post">
      <h2>{{ post.title }}: {{ post.subtitle }}</h2>
      By <AuthorLink :author="post.author" />
      <div>{{ displayableDate(post.publishDate) }}</div>
    <p class="post__description">{{ post.metaDescription }}</p>
    <article>
      {{ post.body }}
    </article>
    <ul>
      <li class="post__tags" v-for="tag in post.tags" :key="tag.name">
        <router-link :to="`/tag/${tag.name}`">#{{ tag.name }}</router-link>
      </li>
    </ul>
  </div>
</template>

<script> import  AuthorLink  from  '@/components/AuthorLink' export  default  { name:  'Post', components:  { AuthorLink, }, data  ()  { return  { post:  null, } }, methods:  { displayableDate  (date)  { return  new  Intl.DateTimeFormat( 'en-US', {  dateStyle:  'full'  }, ).format(new  Date(date)) } }, } </script>
```py

在本教程的后面，您将使用 GraphQL 查询动态填充`post`变量。

### `App`组件

在看到工作成果之前，需要更新 Vue setup 命令创建的`App`组件。它应该显示`AllPosts`组件，而不是显示 Vue 启动页面。

打开`src/`目录下的`App.vue` SFC。您可以删除其中的所有内容，因为您需要用显示以下特性的代码来替换它:

*   链接到主页的带有博客标题的标题
*   `<router-view>`，一个 Vue 路由器组件，呈现当前路由的正确组件

您的`App`组件应该如下所示:

```
<template>
    <div id="app">
        <header>
          <router-link to="/">
            <h1>Awesome Blog</h1>
          </router-link>
        </header>
        <router-view />
    </div>
</template>

<script> export  default  { name:  'App', } </script>
```py

您还可以添加一些可选的 CSS 样式来稍微修饰一下显示:

```
<style> *  { margin:  0; padding:  0; } body  { margin:  0; padding:  1.5rem; } *  +  *  { margin-top:  1.5rem; } #app  { margin:  0; padding:  0; } </style>
```py

这些样式为页面上的大多数元素提供了一点喘息的空间，并删除了大多数浏览器默认添加的整个页面周围的空间。

### 第七步总结

如果你以前没怎么用过 Vue，这一步可能会很难消化。不过，你已经到达了一个重要的里程碑。您已经有了一个可用的 Vue 应用程序，包括准备好显示数据的路线和视图。

您可以通过启动 Vue 开发服务器并访问`http://localhost:8080`来确认您的应用程序正在运行。您应该会看到您的博客标题和最近的文章标题。如果您这样做了，那么您就准备好进行最后一步了，您将使用 Apollo 查询您的 GraphQL API 来将前端和后端结合在一起。

## 第八步:获取数据

现在，您已经为显示可用数据做好了一切准备，是时候从 GraphQL API 获取数据了。

Apollo 使得查询 GraphQL APIs 更加方便。您之前安装的 Vue Apollo 插件将 Apollo 集成到了 Vue 中，使得在 Vue 项目中查询 GraphQL 更加方便。

### 配置 Vue 阿波罗

Vue Apollo 大部分配置都是开箱即用的，但是您需要告诉它要查询的正确端点。您可能还想关闭它默认尝试使用的 WebSocket 连接，因为这会在浏览器的网络和控制台选项卡中产生噪音。编辑`src/main.js`模块中的`apolloProvider`定义，指定`httpEndpoint`和`wsEndpoint`属性:

```
new  Vue({ ... apolloProvider:  createProvider({ httpEndpoint:  'http://localhost:8000/graphql', wsEndpoint:  null, }), ... })
```py

现在，您已经准备好开始添加查询来填充页面。您将通过向几个 sfc 添加一个`created()`函数来实现这一点。`created()`是一个特殊的 [Vue 生命周期挂钩](https://vuejs.org/v2/guide/instance.html#Instance-Lifecycle-Hooks)，当一个组件将要呈现在页面上时执行。您可以使用这个钩子来查询想要呈现的数据，以便在组件呈现时可以使用这些数据。您将为以下组件创建一个查询:

*   `Post`
*   `Author`
*   `PostsByTag`
*   `AllPosts`

您可以从创建`Post`查询开始。

### `Post`查询

对单个帖子的查询接受所需帖子的`slug`。它应该返回所有必要的信息来显示文章信息和内容。

您将使用`$apollo.query`帮助器和`gql`帮助器在`Post`组件的`created()`函数中构建查询，最终使用响应来设置组件的`post`,以便可以呈现它。`created()`应该如下图所示:

```
<script> import  gql  from  'graphql-tag' ... export  default  { ... async  created  ()  { const  post  =  await  this.$apollo.query({ query:  gql`query ($slug: String!) {
 postBySlug(slug: $slug) {
 title
 subtitle
 publishDate
 metaDescription
 slug
 body
 author {
 user {
 username
 firstName
 lastName
 }
 }
 tags {
 name
 }
 }
 }`, variables:  { slug:  this.$route.params.slug, }, }) this.post  =  post.data.postBySlug }, ... } </script>
```py

这个查询获取了关于文章及其相关作者和标签的大部分数据。注意，查询中使用了`$slug`占位符，传递给`$apollo.query`的`variables`属性用于填充占位符。`slug`属性在名称上与`$slug`占位符匹配。您将在其他一些查询中再次看到这种模式。

### `Author`查询

在对`Post`的查询中，您获取了单个帖子的数据和一些关于作者的嵌套数据，而在`Author`查询中，您需要获取作者数据和作者所有帖子的列表。

author 查询接受所需作者的`username`,并应该返回所有必要的信息以显示作者及其帖子列表。它应该如下所示:

```
<script> import  gql  from  'graphql-tag' ... export  default  { ... async  created  ()  { const  user  =  await  this.$apollo.query({ query:  gql`query ($username: String!) {
 authorByUsername(username: $username) {
 website
 bio
 user {
 firstName
 lastName
 username
 }
 postSet { title
 subtitle
 publishDate
 published
 metaDescription
 slug
 tags {
 name
 }
 }
 }
 }`, variables:  { username:  this.$route.params.username, }, }) this.author  =  user.data.authorByUsername }, ... } </script>
```py

这个查询使用了`postSet`，如果您过去做过一些 Django 数据建模，可能会觉得很熟悉。“post set”这个名字来自 Django 为一个`ForeignKey`字段创建的反向关系。在这种情况下，帖子对其作者有一个[外键关系](https://en.wikipedia.org/wiki/Foreign_key)，它与名为`post_set`的帖子有一个反向关系。Graphene-Django 已经在 GraphQL API 中自动将其公开为`postSet`。

### `PostsByTag`查询

对`PostsByTag`的查询应该与您创建的第一个查询非常相似。该查询接受所需的`tag`，并返回匹配文章的列表。`created()`应该像下面这样:

```
<script> import  gql  from  'graphql-tag' ... export  default  { ... async  created  ()  { const  posts  =  await  this.$apollo.query({ query:  gql`query ($tag: String!) {
 postsByTag(tag: $tag) {
 title
 subtitle
 publishDate
 published
 metaDescription
 slug
 author {
 user {
 username
 firstName
 lastName
 }
 }
 tags {
 name
 }
 }
 }`, variables:  { tag:  this.$route.params.tag, }, }) this.posts  =  posts.data.postsByTag }, ... } </script>
```py

您可能会注意到每个查询的某些部分看起来非常相似。虽然本教程不会涉及，但是您可以使用 [GraphQL 片段](https://dgraph.io/docs/graphql/api/fragments/)来减少查询代码中的重复。

### `AllPosts`查询

对`AllPosts`的查询不需要任何输入信息，并返回与`PostsByTag`查询相同的信息集。它应该如下所示:

```
<script> import  gql  from  'graphql-tag' export  default  { ... async  created  ()  { const  posts  =  await  this.$apollo.query({ query:  gql`query {
 allPosts {
 title
 subtitle
 publishDate
 published
 metaDescription
 slug
 author {
 user {
 username
 firstName
 lastName
 }
 }
 tags {
 name
 }
 }
 }`, }) this.allPosts  =  posts.data.allPosts }, ... } </script>
```

这是目前的最后一个查询，但是您应该重温最后几个步骤，以便让它们深入了解。如果您希望将来添加具有新数据视图的新页面，只需创建一个路由、一个组件和一个查询。

### 第八步总结

现在每个组件都在获取它需要显示的数据，您已经到达了一个功能正常的博客。运行 Django 开发服务器和 Vue 开发服务器。访问`http://localhost:8080`并浏览您的博客。如果你能在浏览器中看到作者、帖子、标签和帖子的内容，你就成功了！

## 接下来的步骤

您首先创建了一个 Django 博客后端来管理、持久化和服务博客数据。然后，您创建了一个 Vue 前端来消费和显示这些数据。你让这两个用石墨烯和阿波罗与 GraphQL 通信。

你可能已经在想下一步该怎么做了。要进一步验证您的博客是否按预期运行，您可以尝试以下方法:

*   **添加更多用户**和帖子，以查看按作者分类的用户和帖子。
*   **发布一些未发布的帖子**以确认它们不会出现在博客上。

如果你对自己正在做的事情充满信心和冒险精神，你还可以进一步发展你的系统:

*   **扩展您的数据模型**在您的 Django 博客中创建新的行为。
*   **创建新的查询**为您的博客数据提供有趣的视图。
*   **探索 GraphQL 突变**除了读取数据，还要写入数据。
*   将 CSS 添加到你的单文件组件中，让博客更加引人注目。

您已经组合在一起的数据建模和组件架构具有显著的可扩展性，所以您可以随心所欲地使用它！

如果你想让你的 Django 应用程序为黄金时间做好准备，请阅读[将 Django + Python3 + PostgreSQL 部署到 AWS Elastic Beanstalk](https://realpython.com/deploying-a-django-app-and-postgresql-to-aws-elastic-beanstalk/) 或[在 Fedora 上开发和部署 Django](https://realpython.com/development-and-deployment-of-cookiecutter-django-on-fedora/)。你也可以使用亚马逊网络服务或者类似 [Netlify](https://netlify.com) 的东西来部署你的 Vue 项目。

## 结论

您已经看到了如何使用 GraphQL 构建数据的类型化、灵活的视图。您可以在已经构建或计划构建的现有 Django 应用程序上使用这些技术。像其他 API 一样，您也可以在几乎任何客户端框架中使用您的 API。

**在本教程中，您学习了如何:**

*   构建 Django 博客**数据模型**和**管理界面**
*   使用 Graphene-Django 将您的数据模型包装在一个 **GraphQL API** 中
*   为数据的每个视图创建并路由单独的 **Vue 组件**
*   **使用 Apollo 动态查询 GraphQL API** 来填充您的 Vue 组件

你覆盖了很多领域，所以试着找出一些新的方法在不同的环境中使用这些概念来巩固你的学习。快乐编码，快乐写博客！

您可以通过单击下面的链接下载该项目的完整源代码:

**获取源代码:** [单击此处获取源代码，您将在本教程中使用](https://realpython.com/bonus/django-blog-project-code/)用 Django、Vue 和 GraphQL 构建一个博客应用程序。**********
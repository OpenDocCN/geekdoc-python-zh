# Django RESTful API 的测试驱动开发

> 原文：<https://realpython.com/test-driven-development-of-a-django-restful-api/>

**这篇文章介绍了使用 Django 和 [Django REST 框架](http://www.django-rest-framework.org/)开发基于 CRUD 的 RESTful API 的过程，Django REST 框架用于快速构建基于 Django 模型的 RESTful API。**

该应用程序使用:

*   python 3 . 6 . 0 版
*   Django v1.11.0
*   Django REST 框架 v3.6.2
*   Postgres v9.6.1
*   Psycopg2 v2.7.1

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

> **注意:**查看第三个[真实 Python](https://realpython.com) 课程，获得关于 Django REST 框架的更深入的教程。

## 目标

本教程结束时，您将能够…

1.  讨论使用 Django REST 框架引导 RESTful API 开发的好处
2.  使用序列化程序验证模型查询集
3.  欣赏 Django REST 框架的可浏览 API 特性，获得一个更干净、文档更完整的 API 版本
4.  实践测试驱动的开发

[*Remove ads*](/account/join/)

## 为什么选择 Django REST 框架？

Django REST Framework(REST Framework)提供了许多开箱即用的强大特性，这些特性与惯用的 Django 非常匹配，包括:

1.  [可浏览的 API](http://www.django-rest-framework.org/topics/browsable-api/#the-browsable-api) :用一个人友好的 HTML 输出来记录你的 API，提供一个漂亮的类似表单的界面，使用标准的 HTTP 方法向资源提交数据并从中获取数据。
2.  [身份验证支持](http://www.django-rest-framework.org/api-guide/authentication/) : REST 框架对各种身份验证协议以及权限和节流策略提供了丰富的支持，这些策略可以基于每个视图进行配置。
3.  序列化器(serializer):序列化器是一种验证模型查询集/实例并将其转换为可以轻松呈现为 JSON 和 XML 的原生 Python 数据类型的优雅方式。
4.  [节流](http://www.django-rest-framework.org/api-guide/throttling/):节流是决定一个请求是否被授权的方式，可以和不同的权限集成。它通常用于对单个用户的 API 请求进行速率限制。

此外，该文档易于阅读，并且充满了示例。如果你正在构建一个 RESTful 的 API，在你的 API 端点和你的模型之间有一对一的关系，那么 REST 框架是一个不错的选择。

## Django 项目设置

创建并激活虚拟设备:

```py
$ mkdir django-puppy-store
$ cd django-puppy-store
$ python3.6 -m venv env
$ source env/bin/activate
```

安装 Django 并[建立一个新项目](https://realpython.com/django-setup/):

```py
(env)$ pip install django==1.11.0
(env)$ django-admin startproject puppy_store
```

您当前的项目结构应该如下所示:

```py
└── puppy_store
    ├── manage.py
    └── puppy_store
        ├── __init__.py
        ├── settings.py
        ├── urls.py
        └── wsgi.py
```

## Django 应用和 REST 框架设置

首先创建`puppies`应用程序，然后[在你的虚拟机中安装 REST 框架](https://realpython.com/what-is-pip/):

```py
(env)$ cd puppy_store
(env)$ python manage.py startapp puppies
(env)$ pip install djangorestframework==3.6.2
```

现在我们需要配置 Django 项目来利用 REST 框架。

首先，将`puppies` app 和`rest_framework`添加到*puppy _ store/puppy _ store/settings . py*内的`INSTALLED_APPS`部分:

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'puppies',
    'rest_framework'
]
```

接下来，在单个字典中定义 REST 框架的全局[设置](http://www.django-rest-framework.org/tutorial/quickstart/#settings)，同样是在 *settings.py* 文件中:

```py
REST_FRAMEWORK = {
    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    'DEFAULT_PERMISSION_CLASSES': [],
    'TEST_REQUEST_DEFAULT_FORMAT': 'json'
}
```

这允许无限制地访问 API，并将所有请求的默认测试格式设置为 JSON。

> **注意:**不受限制的访问对于本地开发来说很好，但是在生产环境中，您可能需要限制对某些端点的访问。一定要更新这个。查看[文档](http://www.django-rest-framework.org/api-guide/permissions/#setting-the-permission-policy)了解更多信息。

您当前的项目结构看起来应该是这样的:

```py
└── puppy_store
    ├── manage.py
    ├── puppies
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── apps.py
    │   ├── migrations
    │   │   └── __init__.py
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    └── puppy_store
        ├── __init__.py
        ├── settings.py
        ├── urls.py
        └── wsgi.py
```

[*Remove ads*](/account/join/)

## 数据库和模型设置

让我们设置 Postgres 数据库，并对其应用所有的迁移。

> **注意**:您可以随意将 Postgres 替换为您选择的关系数据库！

一旦您的系统上有了一个正常工作的 Postgres 服务器，打开 Postgres 交互式 shell 并创建数据库:

```py
$  psql #  CREATE  DATABASE  puppy_store_drf; CREATE  DATABASE #  \q
```

安装 [psycopg2](https://github.com/psycopg/psycopg2) 以便我们可以通过 Python 与 Postgres 服务器进行交互:

```py
(env)$ pip install psycopg2==2.7.1
```

更新 *settings.py* 中的数据库配置，添加适当的用户名和密码:

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'puppy_store_drf',
        'USER': '<your-user>',
        'PASSWORD': '<your-password>',
        'HOST': '127.0.0.1',
        'PORT': '5432'
    }
}
```

接下来，在*django-puppy-store/puppy _ store/puppy/models . py*中定义一个具有一些基本属性的小狗模型:

```py
from django.db import models

class Puppy(models.Model):
    """
 Puppy Model
 Defines the attributes of a puppy
 """
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    breed = models.CharField(max_length=255)
    color = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_breed(self):
        return self.name + ' belongs to ' + self.breed + ' breed.'

    def __repr__(self):
        return self.name + ' is added.'
```

现在[应用迁移](https://realpython.com/django-migrations-a-primer/):

```py
(env)$ python manage.py makemigrations
(env)$ python manage.py migrate
```

## 健全性检查

再次跳转到`psql`并验证`puppies_puppy`已经创建:

```py
$  psql #  \c  puppy_store_drf You  are  now  connected  to  database  "puppy_store_drf". puppy_store_drf=#  \dt List  of  relations Schema  |  Name  |  Type  |  Owner --------+----------------------------+-------+----------------
  public  |  auth_group  |  table  |  michael.herman public  |  auth_group_permissions  |  table  |  michael.herman public  |  auth_permission  |  table  |  michael.herman public  |  auth_user  |  table  |  michael.herman public  |  auth_user_groups  |  table  |  michael.herman public  |  auth_user_user_permissions  |  table  |  michael.herman public  |  django_admin_log  |  table  |  michael.herman public  |  django_content_type  |  table  |  michael.herman public  |  django_migrations  |  table  |  michael.herman public  |  django_session  |  table  |  michael.herman public  |  puppies_puppy  |  table  |  michael.herman (11  rows)
```

> **注意:**如果您想查看实际的表细节，可以运行`\d+ puppies_puppy`。

在继续之前，让我们为小狗模型写一个[快速单元测试](https://realpython.com/python-testing/#unit-tests-vs-integration-tests)。

将以下代码添加到名为 *test_models.py* 的新文件中，该文件位于“django-puppy-store/puppy _ store/puppy”内名为“tests”的新文件夹中:

```py
from django.test import TestCase
from ..models import Puppy

class PuppyTest(TestCase):
    """ Test module for Puppy model """

    def setUp(self):
        Puppy.objects.create(
            name='Casper', age=3, breed='Bull Dog', color='Black')
        Puppy.objects.create(
            name='Muffin', age=1, breed='Gradane', color='Brown')

    def test_puppy_breed(self):
        puppy_casper = Puppy.objects.get(name='Casper')
        puppy_muffin = Puppy.objects.get(name='Muffin')
        self.assertEqual(
            puppy_casper.get_breed(), "Casper belongs to Bull Dog breed.")
        self.assertEqual(
            puppy_muffin.get_breed(), "Muffin belongs to Gradane breed.")
```

在上面的测试中，我们通过来自`django.test.TestCase`的`setUp()`方法向 puppy 表中添加了虚拟条目，并断言`get_breed()`方法返回了正确的字符串。

添加一个 *__init__。py* 文件到“tests”并从“django-puppy-store/puppy _ store/puppy”中删除 *tests.py* 文件。

让我们运行第一个测试:

```py
(env)$ python manage.py test
Creating test database for alias 'default'...
.
----------------------------------------------------------------------
Ran 1 test in 0.007s

OK
Destroying test database for alias 'default'...
```

太好了！我们的第一个单元测试已经通过了！

[*Remove ads*](/account/join/)

## 串行器

在继续创建实际的 API 之前，让我们为 Puppy 模型定义一个[序列化器](http://www.django-rest-framework.org/api-guide/serializers/)，它验证模型[查询集](https://docs.djangoproject.com/en/1.10/ref/models/querysets/)并生成 Pythonic 数据类型。

将以下代码片段添加到*django-puppy-store/puppy _ store/puppy/serializer . py*:

```py
from rest_framework import serializers
from .models import Puppy

class PuppySerializer(serializers.ModelSerializer):
    class Meta:
        model = Puppy
        fields = ('name', 'age', 'breed', 'color', 'created_at', 'updated_at')
```

在上面的代码片段中，我们为我们的小狗模型定义了一个`ModelSerializer`，验证了所有提到的字段。简而言之，如果你的 API 端点和你的模型之间有一对一的关系——如果你正在创建一个 RESTful API，你可能应该这样做——那么你可以使用一个 [ModelSerializer](http://www.django-rest-framework.org/api-guide/serializers/#modelserializer) 来创建一个序列化器。

有了我们的数据库，我们现在可以开始构建 RESTful API 了…

## RESTful 结构

在 RESTful API 中，端点(URL)定义了 API 的结构，以及最终用户如何使用 HTTP 方法(GET、POST、PUT、DELETE)从我们的应用程序中访问数据。端点应该围绕*集合*和*元素*进行逻辑组织，两者都是资源。

在我们的例子中，我们有一个单独的资源，`puppies`，所以我们将使用下面的 URL-`/puppies/`和`/puppies/<id>`，分别用于集合和元素:

| 端点 | HTTP 方法 | CRUD 方法 | 结果 |
| --- | --- | --- | --- |
| `puppies` | 得到 | 阅读 | 得到所有小狗 |
| `puppies/:id` | 得到 | 阅读 | 养一只小狗 |
| `puppies` | 邮政 | 创造 | 添加一只小狗 |
| `puppies/:id` | 放 | 更新 | 更新一只小狗 |
| `puppies/:id` | 删除 | 删除 | 删除一只小狗 |

## 路线和测试(TDD)

我们将采用测试优先的方法，而不是彻底的测试驱动的方法，其中我们将经历以下过程:

*   添加一个单元测试，代码足够失败
*   然后更新代码，使其通过测试。

一旦测试通过，重新开始新的测试。

首先创建一个新文件，django-puppy-store/puppy _ store/puppy/tests/test _ views . py，保存我们的视图的所有测试，并为我们的应用程序创建一个新的测试客户端:

```py
import json
from rest_framework import status
from django.test import TestCase, Client
from django.urls import reverse
from ..models import Puppy
from ..serializers import PuppySerializer

# initialize the APIClient app
client = Client()
```

在开始所有的 API 路由之前，让我们首先创建一个所有返回空响应的视图函数的框架，并将它们映射到文件*django-puppy-store/puppy _ store/puppy/views . py*中相应的 URL:

```py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Puppy
from .serializers import PuppySerializer

@api_view(['GET', 'DELETE', 'PUT'])
def get_delete_update_puppy(request, pk):
    try:
        puppy = Puppy.objects.get(pk=pk)
    except Puppy.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    # get details of a single puppy
    if request.method == 'GET':
        return Response({})
    # delete a single puppy
    elif request.method == 'DELETE':
        return Response({})
    # update details of a single puppy
    elif request.method == 'PUT':
        return Response({})

@api_view(['GET', 'POST'])
def get_post_puppies(request):
    # get all puppies
    if request.method == 'GET':
        return Response({})
    # insert a new record for a puppy
    elif request.method == 'POST':
        return Response({})
```

创建各自的 URL 以匹配*django-puppy-store/puppy _ store/puppy/URLs . py*中的视图:

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(
        r'^api/v1/puppies/(?P<pk>[0-9]+)$',
        views.get_delete_update_puppy,
        name='get_delete_update_puppy'
    ),
    url(
        r'^api/v1/puppies/$',
        views.get_post_puppies,
        name='get_post_puppies'
    )
]
```

更新*django-puppy-store/puppy _ store/puppy _ store/URLs . py*同样:

```py
from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^', include('puppies.urls')),
    url(
        r'^api-auth/',
        include('rest_framework.urls', namespace='rest_framework')
    ),
    url(r'^admin/', admin.site.urls),
]
```

[*Remove ads*](/account/join/)

## 可浏览的 API

现在，所有的路由都与视图函数连接起来了，让我们打开 REST 框架的可浏览 API 接口，验证所有的 URL 是否都按预期工作。

首先，启动开发服务器:

```py
(env)$ python manage.py runserver
```

确保注释掉我们的`settings.py`文件的`REST_FRAMEWORK`部分中的所有属性，以绕过登录。现在参观`http://localhost:8000/api/v1/puppies`

您将看到 API 响应的交互式 HTML 布局。同样，我们可以测试其他网址，并验证所有网址都工作得非常好。

让我们从每条路线的单元测试开始。

## 路线

### 获取全部

从验证提取的记录的测试开始:

```py
class GetAllPuppiesTest(TestCase):
    """ Test module for GET all puppies API """

    def setUp(self):
        Puppy.objects.create(
            name='Casper', age=3, breed='Bull Dog', color='Black')
        Puppy.objects.create(
            name='Muffin', age=1, breed='Gradane', color='Brown')
        Puppy.objects.create(
            name='Rambo', age=2, breed='Labrador', color='Black')
        Puppy.objects.create(
            name='Ricky', age=6, breed='Labrador', color='Brown')

    def test_get_all_puppies(self):
        # get API response
        response = client.get(reverse('get_post_puppies'))
        # get data from db
        puppies = Puppy.objects.all()
        serializer = PuppySerializer(puppies, many=True)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
```

运行测试。您应该会看到以下错误:

```py
self.assertEqual(response.data, serializer.data)
AssertionError: {} != [OrderedDict([('name', 'Casper'), ('age',[687 chars])])]
```

更新视图以通过测试。

```py
@api_view(['GET', 'POST'])
def get_post_puppies(request):
    # get all puppies
    if request.method == 'GET':
        puppies = Puppy.objects.all()
        serializer = PuppySerializer(puppies, many=True)
        return Response(serializer.data)
    # insert a new record for a puppy
    elif request.method == 'POST':
        return Response({})
```

在这里，我们获得小狗的所有记录，并使用`PuppySerializer`验证每个记录。

运行测试以确保它们全部通过:

```py
Ran 2 tests in 0.072s

OK
```

### 获得单身

获取一只小狗涉及两个测试案例:

1.  获得有效的小狗-例如，小狗存在
2.  得到无效的小狗-例如，小狗不存在

添加测试:

```py
class GetSinglePuppyTest(TestCase):
    """ Test module for GET single puppy API """

    def setUp(self):
        self.casper = Puppy.objects.create(
            name='Casper', age=3, breed='Bull Dog', color='Black')
        self.muffin = Puppy.objects.create(
            name='Muffin', age=1, breed='Gradane', color='Brown')
        self.rambo = Puppy.objects.create(
            name='Rambo', age=2, breed='Labrador', color='Black')
        self.ricky = Puppy.objects.create(
            name='Ricky', age=6, breed='Labrador', color='Brown')

    def test_get_valid_single_puppy(self):
        response = client.get(
            reverse('get_delete_update_puppy', kwargs={'pk': self.rambo.pk}))
        puppy = Puppy.objects.get(pk=self.rambo.pk)
        serializer = PuppySerializer(puppy)
        self.assertEqual(response.data, serializer.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_get_invalid_single_puppy(self):
        response = client.get(
            reverse('get_delete_update_puppy', kwargs={'pk': 30}))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
```

进行测试。您应该会看到以下错误:

```py
self.assertEqual(response.data, serializer.data)
AssertionError: {} != {'name': 'Rambo', 'age': 2, 'breed': 'Labr[109 chars]26Z'}
```

更新视图:

```py
@api_view(['GET', 'UPDATE', 'DELETE'])
def get_delete_update_puppy(request, pk):
    try:
        puppy = Puppy.objects.get(pk=pk)
    except Puppy.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    # get details of a single puppy
    if request.method == 'GET':
        serializer = PuppySerializer(puppy)
        return Response(serializer.data)
```

在上面的代码片段中，我们用一个 ID 得到了小狗。运行测试以确保它们都通过。

[*Remove ads*](/account/join/)

### 帖子

插入新记录也涉及两种情况:

1.  插入有效的小狗
2.  插入一只残疾小狗

首先，为它编写测试:

```py
class CreateNewPuppyTest(TestCase):
    """ Test module for inserting a new puppy """

    def setUp(self):
        self.valid_payload = {
            'name': 'Muffin',
            'age': 4,
            'breed': 'Pamerion',
            'color': 'White'
        }
        self.invalid_payload = {
            'name': '',
            'age': 4,
            'breed': 'Pamerion',
            'color': 'White'
        }

    def test_create_valid_puppy(self):
        response = client.post(
            reverse('get_post_puppies'),
            data=json.dumps(self.valid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_create_invalid_puppy(self):
        response = client.post(
            reverse('get_post_puppies'),
            data=json.dumps(self.invalid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
```

进行测试。您应该会看到两个失败:

```py
self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
AssertionError: 200 != 400

self.assertEqual(response.status_code, status.HTTP_201_CREATED)
AssertionError: 200 != 201
```

同样，更新视图以通过测试:

```py
@api_view(['GET', 'POST'])
def get_post_puppies(request):
    # get all puppies
    if request.method == 'GET':
        puppies = Puppy.objects.all()
        serializer = PuppySerializer(puppies, many=True)
        return Response(serializer.data)
    # insert a new record for a puppy
    if request.method == 'POST':
        data = {
            'name': request.data.get('name'),
            'age': int(request.data.get('age')),
            'breed': request.data.get('breed'),
            'color': request.data.get('color')
        }
        serializer = PuppySerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

在这里，我们通过在插入数据库之前序列化和验证请求数据来插入新记录。

再次运行测试以确保它们通过。

您也可以使用可浏览的 API 来测试这一点。再次启动开发服务器，并导航到[http://localhost:8000/API/v1/puppies/](http://localhost:8000/api/v1/puppies/)。然后，在 POST 表单中，提交以下内容作为`application/json`:

```py
{ "name":  "Muffin", "age":  4, "breed":  "Pamerion", "color":  "White" }
```

一定要得到全部，也要得到单个工作。

### 放

从更新记录的测试开始。与添加记录类似，我们也需要测试有效和无效的更新:

```py
class UpdateSinglePuppyTest(TestCase):
    """ Test module for updating an existing puppy record """

    def setUp(self):
        self.casper = Puppy.objects.create(
            name='Casper', age=3, breed='Bull Dog', color='Black')
        self.muffin = Puppy.objects.create(
            name='Muffy', age=1, breed='Gradane', color='Brown')
        self.valid_payload = {
            'name': 'Muffy',
            'age': 2,
            'breed': 'Labrador',
            'color': 'Black'
        }
        self.invalid_payload = {
            'name': '',
            'age': 4,
            'breed': 'Pamerion',
            'color': 'White'
        }

    def test_valid_update_puppy(self):
        response = client.put(
            reverse('get_delete_update_puppy', kwargs={'pk': self.muffin.pk}),
            data=json.dumps(self.valid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

    def test_invalid_update_puppy(self):
        response = client.put(
            reverse('get_delete_update_puppy', kwargs={'pk': self.muffin.pk}),
            data=json.dumps(self.invalid_payload),
            content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
```

进行测试。

```py
self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
AssertionError: 405 != 400

self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
AssertionError: 405 != 204
```

更新视图:

```py
@api_view(['GET', 'DELETE', 'PUT'])
def get_delete_update_puppy(request, pk):
    try:
        puppy = Puppy.objects.get(pk=pk)
    except Puppy.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    # get details of a single puppy
    if request.method == 'GET':
        serializer = PuppySerializer(puppy)
        return Response(serializer.data)

    # update details of a single puppy
    if request.method == 'PUT':
        serializer = PuppySerializer(puppy, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_204_NO_CONTENT)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # delete a single puppy
    elif request.method == 'DELETE':
        return Response({})
```

在上面的代码片段中，类似于插入，我们序列化并验证请求数据，然后做出适当的响应。

再次运行测试以确保所有测试都通过。

[*Remove ads*](/account/join/)

### 删除

要删除单个记录，需要 ID:

```py
class DeleteSinglePuppyTest(TestCase):
    """ Test module for deleting an existing puppy record """

    def setUp(self):
        self.casper = Puppy.objects.create(
            name='Casper', age=3, breed='Bull Dog', color='Black')
        self.muffin = Puppy.objects.create(
            name='Muffy', age=1, breed='Gradane', color='Brown')

    def test_valid_delete_puppy(self):
        response = client.delete(
            reverse('get_delete_update_puppy', kwargs={'pk': self.muffin.pk}))
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

    def test_invalid_delete_puppy(self):
        response = client.delete(
            reverse('get_delete_update_puppy', kwargs={'pk': 30}))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
```

进行测试。您应该看到:

```py
self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
AssertionError: 200 != 204
```

更新视图:

```py
@api_view(['GET', 'DELETE', 'PUT'])
def get_delete_update_puppy(request, pk):
    try:
        puppy = Puppy.objects.get(pk=pk)
    except Puppy.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    # get details of a single puppy
    if request.method == 'GET':
        serializer = PuppySerializer(puppy)
        return Response(serializer.data)

    # update details of a single puppy
    if request.method == 'PUT':
        serializer = PuppySerializer(puppy, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_204_NO_CONTENT)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # delete a single puppy
    if request.method == 'DELETE':
        puppy.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

再次运行测试。确保他们都通过。确保在可浏览的 API 中测试更新和删除功能！

## 结论和后续步骤

在本教程中，我们通过测试优先的方法，使用 Django REST 框架完成了创建 RESTful API 的过程。

**免费奖励:** ，并获得 Python + REST API 原则的实际操作介绍以及可操作的示例。

下一步是什么？为了使我们的 RESTful API 健壮和安全，我们可以为生产环境实现权限和节流，以允许基于认证凭证和速率限制的受限访问，从而避免任何类型的 DDoS 攻击。此外，不要忘记防止可浏览 API 在生产环境中被访问。

欢迎在下面的评论中分享你的评论、问题或建议。完整的代码可以在 [django-puppy-store](https://github.com/realpython/django-puppy-store) 存储库中找到。******
# 如何编写可安装的 Django 应用程序

> 原文：<https://realpython.com/installable-django-app/>

在 [Django](https://realpython.com/get-started-with-django-1/) 框架中，**项目**指的是特定网站的配置文件和代码的集合。Django 将业务逻辑分组到它所谓的**应用**中，这些应用是 Django 框架的模块。有很多关于如何构建项目和其中的应用程序的文档，但是当要打包一个可安装的 Django 应用程序时，信息就很难找到了。

在本教程中，你将学习如何从 Django 项目中取出一个应用程序并打包，使其可安装。一旦你打包了你的应用，你就可以在 [PyPI](https://pypi.org/) 上分享它，这样其他人就可以通过`pip install`获取它。

在本教程中，您将学习:

*   编写**独立应用**和在项目中编写**应用有什么区别**
*   如何创建一个 **`setup.cfg`文件**来发布你的 Django 应用
*   如何在 Django 项目之外引导 Django ,以便测试你的应用
*   如何使用 **`tox`** 跨多个版本的 Python 和 Django 进行测试
*   如何使用 **Twine** 将可安装的 Django 应用程序发布到 PyPI

请务必通过以下链接下载源代码来了解示例:

**下载示例代码:** [单击此处获取代码，您将使用](https://realpython.com/bonus/installable-django-app/)在本教程中学习如何编写可安装的 Django 应用程序。

## 先决条件

本教程要求对 [Django](https://www.djangoproject.com/) 、`pip`、 [PyPI](https://pypi.org) 、`pyenv`(或者一个等效的虚拟环境工具)和`tox`有所熟悉。要了解有关这些主题的更多信息，请访问:

*   [Django 教程](https://realpython.com/tutorials/django/)
*   [Pip 是什么？新蟒蛇指南](https://realpython.com/what-is-pip/)
*   [如何将开源 Python 包发布到 PyPI](https://realpython.com/pypi-publish-python-package/)
*   [使用 pyenv 管理多个 Python 版本](https://realpython.com/intro-to-pyenv/)
*   [Python 测试入门](https://realpython.com/python-testing/)

[*Remove ads*](/account/join/)

## 在项目中启动一个示例 Django 应用程序

本教程包括一个工作包，帮助你完成制作一个可安装的 Django 应用程序的过程。您可以从下面的链接下载源代码:

**下载示例代码:** [单击此处获取代码，您将使用](https://realpython.com/bonus/installable-django-app/)在本教程中学习如何编写可安装的 Django 应用程序。

即使你最初打算把你的 Django 应用作为一个包提供，你也可能从一个项目开始。为了演示从 Django 项目到可安装的 Django 应用程序的过程，我在 repo 中提供了两个分支。**项目分支**是 Django 项目中一个应用的开始状态。**主分支**就是完成的可安装 app。

你也可以在[PyPI real python-django-receipts package 页面](https://pypi.org/project/realpython-django-receipts/)下载完成的 app。你可以通过运行`pip install realpython-django-receipts`来安装包。

示例应用程序是收据上的行项目的简短表示。在项目分支中，您会发现一个名为`sample_project`的目录，其中包含一个正在运行的 Django 项目。该目录如下所示:

```py
sample_project/
│
├── receipts/
│   ├── fixtures/
│   │   └── receipts.json
│   │
│   ├── migrations/
│   │   ├── 0001_initial.py
│   │   └── __init__.py
│   │
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
│
├── sample_project/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── db.sqlite3
├── manage.py
├── resetdb.sh
└── runserver.sh
```

撰写本教程时，Django 的最新版本是 3.0.4，所有测试都是用 Python 3.7 完成的。本教程中概述的所有步骤都不应该与 Django 的早期版本不兼容——我从 Django 1.8 就开始使用这些技术了。但是，如果您使用 Python 2，一些更改是必要的。为了让例子简单，我假设代码库都是 Python 3.7。

### 从头开始创建 Django 项目

示例项目和收据应用程序是使用 Django `admin`命令和一些小的编辑创建的。首先，在干净的虚拟环境中运行以下代码:

```py
$ python -m pip install Django
$ django-admin startproject sample_project
$ cd sample_project
$ ./manage.py startapp receipts
```

这将创建一个`sample_project`项目目录结构和一个`receipts` app 子目录，其中包含用于创建可安装 Django 应用程序的模板文件。

接下来，`sample_project/settings.py`文件需要一些修改:

*   将`'127.0.0.1'`添加到`ALLOWED_HOSTS`设置中，这样您就可以进行本地测试。
*   将`'receipts'`添加到`INSTALLED_APPS`列表中。

您还需要在`sample_project/urls.py`文件中注册`receipts`应用程序的 URL。为此，将`path('receipts/', include('receipts.urls'))`添加到`url_patterns`列表中。

### 探索收据示例应用程序

app 由两个 ORM 模型类组成:`Item`和`Receipt`。`Item`类包含描述和成本的数据库字段声明。成本包含在 [`DecimalField`](https://docs.djangoproject.com/en/3.0/ref/models/fields/#decimalfield) 中。使用浮点数来表示货币是危险的——在处理货币时，你应该总是使用定点数。

`Receipt`类是`Item`对象的收集点。这是通过指向`Receipt`的`Item`上的`ForeignKey`实现的。`Receipt`还包括`total()`，用于获取`Receipt`中包含的`Item`对象的总成本:

```py
# receipts/models.py
from decimal import Decimal
from django.db import models

class Receipt(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Receipt(id={self.id})"

    def total(self) -> Decimal:
        return sum(item.cost for item in self.item_set.all())

class Item(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    description = models.TextField()
    cost = models.DecimalField(max_digits=7, decimal_places=2)
    receipt = models.ForeignKey(Receipt, on_delete=models.CASCADE)

    def __str__(self):
        return f"Item(id={self.id}, description={self.description}, " \
            f"cost={self.cost})"
```

模型对象为您提供了数据库的内容。一个简短的 Django 视图返回一个 [JSON 字典](https://realpython.com/lessons/deserializing-json-data/)，其中包含数据库中的所有`Receipt`对象及其`Item`对象:

```py
# receipts/views.py
from django.http import JsonResponse
from receipts.models import Receipt

def receipt_json(request):
    results = {
        "receipts":[],
    }

    for receipt in Receipt.objects.all():
        line = [str(receipt), []]
        for item in receipt.item_set.all():
            line[1].append(str(item))

        results["receipts"].append(line)

    return JsonResponse(results)
```

`receipt_json()`视图遍历所有的`Receipt`对象，创建一对`Receipt`对象和一个包含在其中的`Item`对象列表。所有这些都被放入字典，并通过 Django 的`JsonResponse()`返回。

为了使模型在 [Django 管理界面](https://realpython.com/lessons/django-admin-interface/)中可用，您使用一个`admin.py`文件来注册模型:

```py
# receipts/admin.py
from django.contrib import admin

from receipts.models import Receipt, Item

@admin.register(Receipt)
class ReceiptAdmin(admin.ModelAdmin):
    pass

@admin.register(Item)
class ItemAdmin(admin.ModelAdmin):
    pass
```

这段代码为每个`Receipt`和`Item`类创建一个 Django `ModelAdmin`，并向 Django admin 注册它们。

最后，一个`urls.py`文件根据一个 URL 在应用程序中注册了一个视图:

```py
# receipts/urls.py
from django.urls import path

from receipts import views

urlpatterns = [
    path("receipt_json/", views.receipt_json),
]
```

现在，您可以将`receipts/urls.py`包含在项目的`url.py`文件中，使收据视图在您的网站上可用。

一切就绪后，您可以运行`./manage.py makemigrations receipts`，使用 Django admin 添加数据，然后访问`/receipts/receipt_json/`查看结果:

```py
$ curl -sS http://127.0.0.1:8000/receipts/receipt_json/ | python3.8 -m json.tool
{
 "receipts": [
 [
 "Receipt(id=1)",
 [
 "Item(id=1, description=wine, cost=15.25)",
 "Item(id=2, description=pasta, cost=22.30)"
 ]
 ],
 [
 "Receipt(id=2)",
 [
 "Item(id=3, description=beer, cost=8.50)",
 "Item(id=4, description=pizza, cost=12.80)"
 ]
 ]
 ]
}
```

在上面的块中，您使用 [`curl`](https://en.wikipedia.org/wiki/CURL) 来访问`receipt_json`视图，得到一个包含`Receipt`对象及其对应的`Item`对象的 JSON 响应。

[*Remove ads*](/account/join/)

### 测试项目中的应用程序

Django 用自己的[测试功能](https://realpython.com/testing-in-django-part-2-model-mommy-vs-django-testing-fixtures/)增强了 Python `unittest`包，使您能够将夹具预加载到数据库中并运行您的测试。receipts 应用程序定义了一个`tests.py`文件和一个用于测试的夹具。这个测试并不全面，但它是一个足够好的概念证明:

```py
# receipts/tests.py
from decimal import Decimal
from django.test import TestCase
from receipts.models import Receipt

class ReceiptTest(TestCase):
    fixtures = ["receipts.json", ]

    def test_receipt(self):
        receipt = Receipt.objects.get(id=1)
        total = receipt.total()

        expected = Decimal("37.55")
        self.assertEqual(expected, total)
```

夹具创建两个`Receipt`对象和四个相应的`Item`对象。点击下面的可折叠部分，仔细查看夹具的代码。



Django 测试夹具是数据库中对象的**序列化**。下面的 JSON 代码创建了用于测试的`Receipt`和`Item`对象:

```py
[ { "model":  "receipts.receipt", "pk":  1, "fields":  { "created":  "2020-03-24T18:16:39.102Z" } }, { "model":  "receipts.receipt", "pk":  2, "fields":  { "created":  "2020-03-24T18:16:41.005Z" } }, { "model":  "receipts.item", "pk":  1, "fields":  { "created":  "2020-03-24T18:16:59.357Z", "description":  "wine", "cost":  "15.25", "receipt":  1 } }, { "model":  "receipts.item", "pk":  2, "fields":  { "created":  "2020-03-24T18:17:25.548Z", "description":  "pasta", "cost":  "22.30", "receipt":  1 } }, { "model":  "receipts.item", "pk":  3, "fields":  { "created":  "2020-03-24T18:19:37.359Z", "description":  "beer", "cost":  "8.50", "receipt":  2 } }, { "model":  "receipts.item", "pk":  4, "fields":  { "created":  "2020-03-24T18:19:51.475Z", "description":  "pizza", "cost":  "12.80", "receipt":  2 } } ]
```

上面的 fixture 在`ReceiptTestCase`类中被引用，并由 Django 测试工具自动加载。

您可以使用 Django `manage.py`命令测试 receipts 应用程序:

```py
$ ./manage.py test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
.
----------------------------------------------------------------------
Ran 1 test in 0.013s

OK
Destroying test database for alias 'default'...
```

运行`manage.py test`运行`receipts/tests.py`中定义的单一测试并显示结果。

## 制作可安装的 Django 应用程序

您的目标是在没有项目的情况下共享 receipts 应用程序，并让其他人可以重复使用它。您可以压缩`receipts/`目录并分发出去，但这多少有些限制。相反，你想把应用程序分离成一个包，这样它就可以安装了。

创建可安装的 Django 应用程序的最大挑战是 Django 需要一个项目。没有项目的 app 只是一个包含代码的目录。没有项目，Django 不知道如何处理你的代码，包括运行测试。

### 将 Django 应用程序移出项目

保留一个示例项目是个好主意，这样您就可以运行 Django dev 服务器并使用您的应用程序的实时版本。您不会将这个示例项目包含在应用程序包中，但是它仍然可以存在于您的存储库中。按照这个想法，您可以开始打包您的可安装 Django 应用程序，方法是将它移到一个目录中:

```py
$ mv receipts ..
```

目录结构现在看起来像这样:

```py
django-receipts/
│
├── receipts/
│   ├── fixtures/
│   │   └── receipts.json
│   │
│   ├── migrations/
│   │   ├── 0001_initial.py
│   │   └── __init__.py
│   │
│   ├── __init__.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
│   ├── admin.py
│   └── apps.py
│
├── sample_project/
│   ├── sample_project/
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   │
│   ├── db.sqlite3
│   ├── manage.py
│   ├── resetdb.sh
│   └── runserver.sh
│
├── LICENSE
└── README.rst
```

要打包您的应用程序，您需要将其从项目中取出。移动它是第一步。我通常会保留原始项目进行测试，但不会将它包含在最终的包中。

### 在项目外引导 Django

现在你的应用程序在 Django 项目之外，你需要告诉 Django 如何找到它。如果你想测试你的应用，那么运行一个 Django shell，它可以找到你的应用或者运行你的[迁移](https://realpython.com/django-migrations-a-primer/)。您需要配置 Django 并使其可用。

Django 的`settings.configure()`和`django.setup()`是在项目之外与你的应用程序交互的关键。Django 文档中提供了关于这些调用的更多信息[。](https://docs.djangoproject.com/en/3.0/topics/settings/#using-settings-without-setting-django-settings-module)

您可能在几个地方需要 Django 的这种配置，所以在函数中定义它是有意义的。创建一个名为`boot_django.py`的文件，包含以下代码:

```py
 1# boot_django.py
 2#
 3# This file sets up and configures Django. It's used by scripts that need to
 4# execute as if running in a Django server.
 5import os
 6import django
 7from django.conf import settings
 8
 9BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "receipts"))
10
11def boot_django():
12    settings.configure(
13        BASE_DIR=BASE_DIR,
14        DEBUG=True,
15        DATABASES={
16            "default":{
17                "ENGINE":"django.db.backends.sqlite3",
18                "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
19            }
20        },
21        INSTALLED_APPS=(
22            "receipts",
23        ),
24        TIME_ZONE="UTC",
25        USE_TZ=True,
26    )
27    django.setup()
```

**12 号线和 27 号线**设置 Django 环境。`settings.configure()`调用接受一个参数列表，这些参数等同于在`settings.py`文件中定义的变量。你在`settings.py`中运行应用程序所需的任何东西都会被传递到`settings.configure()`中。

上面的代码是一个相当精简的配置。receipts 应用不对会话或模板做任何事情，所以`INSTALLED_APPS`只需要`"receipts"`，并且您可以跳过任何中间件定义。`USE_TZ=True`值是必需的，因为`Receipt`模型包含一个`created`时间戳。否则，您会在加载测试夹具时遇到问题。

[*Remove ads*](/account/join/)

### 使用可安装的 Django 应用程序运行管理命令

现在您已经有了`boot_django.py`，您可以用一个非常短的脚本运行任何 Django 管理命令:

```py
#!/usr/bin/env python
# makemigrations.py

from django.core.management import call_command
from boot_django import boot_django

boot_django()
call_command("makemigrations", "receipts")
```

Django 允许您通过`call_command()`以编程方式调用管理命令。您现在可以通过导入并调用`boot_django()`然后调用`call_command()`来运行任何管理命令。

你的应用现在在项目之外，允许你对它做各种 Django-y 的事情。我经常定义四个实用程序脚本:

1.  **`load_tests.py`** 测试你的 app
2.  **`makemigrations.py`** 创建迁移文件
3.  **`migrate.py`** 执行表迁移
4.  **`djangoshell.py`** 生成一个 Django shell，它可以感知你的应用

### 测试可安装的 Django 应用程序

`load_test.py`文件可以像`makemigrations.py`脚本一样简单，但是它只能同时运行所有的测试。通过几行额外的代码，您可以将[命令行参数](https://realpython.com/python-command-line-arguments/)传递给测试运行程序，允许您运行选择性测试:

```py
 1#!/usr/bin/env python
 2# load_tests.py
 3import sys
 4from unittest import TestSuite
 5from boot_django import boot_django
 6
 7boot_django()
 8
 9default_labels = ["receipts.tests", ]
10
11def get_suite(labels=default_labels):
12    from django.test.runner import DiscoverRunner
13    runner = DiscoverRunner(verbosity=1)
14    failures = runner.run_tests(labels)
15    if failures:
16        sys.exit(failures)
17
18    # In case this is called from setuptools, return a test suite
19    return TestSuite()
20
21if __name__ == "__main__":
22    labels = default_labels
23    if len(sys.argv[1:]) > 0:
24        labels = sys.argv[1:]
25
26    get_suite(labels)
```

Django 的`DiscoverRunner`是与 Python 的`unittest`兼容的测试发现类。它负责建立测试环境，构建测试套件，建立数据库，运行测试，然后全部拆除。从**第 11 行**开始，`get_suite()`取一个测试标签列表，直接调用标签上的`DiscoverRunner`。

这个脚本类似于 Django 管理命令`test`的功能。`__main__`块将任何命令行参数传递给`get_suite()`，如果没有，则传递给应用程序的测试套件`receipts.tests`。您现在可以使用测试标签参数调用`load_tests.py`,并运行一个单独的测试。

**第 19 行**是一个特例，有助于用`tox`进行测试。在稍后的[章节](#testing-multiple-versions-with-tox)中，您将了解更多关于`tox`的内容。你也可以在下面的可折叠部分找到`DiscoverRunner`的潜在替代品。



我写的一个可安装的 Django 应用是 [django-awl](https://pypi.org/project/django-awl/) 。这是我多年来编写 Django 项目时积累的一个松散的实用程序集合。软件包中包含了一个名为`WRunner`的`DiscoverRunner`的替代品。

使用`WRunner`的关键优势是它支持测试标签的**通配符匹配**。传入一个以等号(`=`)开头的标签将匹配任何包含该标签作为子串的测试套件或方法名称。例如，标签`=rec`将匹配并在`receipt/tests.py`运行测试`ReceiptTest.test_receipt()`。

### 用`setup.cfg` 定义你的可安装包

要将可安装的 Django 应用程序放在 PyPI 上，首先需要将其放在一个包中。PyPI 期望一个 [`egg`](https://packaging.python.org/glossary/#term-egg) ， [`wheel`](https://realpython.com/python-wheels/) ，或者源分布。这些都是用`setuptools`建造的。为此，您需要在与您的`receipts`目录相同的目录级别创建一个`setup.cfg`文件和一个`setup.py`文件。

不过，在深入研究之前，您需要确保您有一些文档。您可以在`setup.cfg`中包含一个项目描述，它会自动显示在 PyPI 项目页面上。一定要写一个`README.rst`或类似的关于你的包裹的信息。

PyPI 默认支持 [reStructuredText](https://docutils.sourceforge.io/rst.html) 格式，但是它也可以使用[额外参数](https://stackoverflow.com/questions/26737222/how-to-make-pypi-description-markdown-work/26737258#26737258)处理[降价](https://github.com/theacodes/cmarkgfm):

```py
 1# setup.cfg 2[metadata] 3name  =  realpython-django-receipts 4version  =  1.0.3 5description  =  Sample installable django app 6long_description  =  file:README.rst 7url  =  https://github.com/realpython/django-receipts 8license  =  MIT 9classifiers  = 10  Development Status :: 4 - Beta 11  Environment :: Web Environment 12  Intended Audience :: Developers 13  License :: OSI Approved :: MIT License 14  Operating System :: OS Independent 15  Programming Language :: Python :: 3 :: Only 16  Programming Language :: Python :: 3.7 17  Programming Language :: Python :: Implementation :: CPython 18  Topic :: Software Development :: Libraries :: Application Frameworks 19  Topic :: Software Development :: Libraries :: Python Modules 20
21[options] 22include_package_data  =  true 23python_requires  =  >=3.6 24setup_requires  = 25  setuptools >=  38.3.0 26install_requires  = 27  Django>=2.2
```

这个`setup.cfg`文件描述了您将要构建的包。**第 6 行**使用`file:`指令读入你的`README.rst`文件。这样你就不用在两个地方写很长的描述了。

第 26 行**上的`install_requires`条目告诉任何安装者，比如`pip install`，关于你的应用程序的依赖关系。您总是希望将您的可安装 Django 应用程序绑定到其最低支持版本的 Django 上。**

如果您的代码有任何只需要运行测试的依赖项，那么您可以添加一个`tests_require =`条目。例如，在`mock`成为标准 Python 库的一部分之前，在`setup.cfg`中看到`tests_require = mock>=2.0.0`是很常见的。

在包中包含一个`pyproject.toml`文件被认为是最佳实践。Brett Cannon 的[关于这个主题的优秀文章](https://snarky.ca/what-the-heck-is-pyproject-toml/)可以带你浏览细节。示例代码中还包含一个`pyproject.toml`文件。

您几乎已经准备好为您的可安装 Django 应用程序构建包了。测试它最简单的方法是使用您的示例项目——这是保留一个示例项目的另一个好理由。`pip install`命令支持本地定义的包。这可用于确保您的应用程序仍可用于项目。然而，有一点需要注意的是，在这种情况下,`setup.cfg`不会自己工作。你还必须创建一个`setup.py`的填充版本:

```py
#!/usr/bin/env python

if __name__ == "__main__":
    import setuptools
    setuptools.setup()
```

这个脚本将自动使用您的`setup.cfg`文件。你现在可以安装一个**本地可编辑的**版本的包来从`sample_project`内部测试它。为了更加确定，最好从一个全新的虚拟环境开始。在`sample_project`目录中添加以下`requirements.txt`文件:

```py
# requirements.txt
-e ../../django-receipts
```

`-e`告诉`pip`这是一个本地可编辑的安装。您现在可以安装:

```py
$ pip install -r requirements.txt
Obtaining django-receipts (from -r requirements.txt (line 1))
Collecting Django>=3.0
 Using cached Django-3.0.4-py3-none-any.whl (7.5 MB)
Collecting asgiref~=3.2
 Using cached asgiref-3.2.7-py2.py3-none-any.whl (19 kB)
Collecting pytz
 Using cached pytz-2019.3-py2.py3-none-any.whl (509 kB)
Collecting sqlparse>=0.2.2
 Using cached sqlparse-0.3.1-py2.py3-none-any.whl (40 kB)
Installing collected packages: asgiref, pytz, sqlparse, Django, realpython-django-receipts
 Running setup.py develop for realpython-django-receipts
Successfully installed Django-3.0.4 asgiref-3.2.7 pytz-2019.3 realpython-django-receipts sqlparse-0.3.1
```

`setup.cfg`中的`install_requires`列表告诉`pip install`它需要 Django。姜戈需要`asgiref`、`pytz`和`sqlparse`。所有的依赖关系都已经处理好了，现在您应该能够运行您的`sample_project` Django dev 服务器了。恭喜您，您的应用程序现在已经打包并在示例项目中引用了！

[*Remove ads*](/account/join/)

### 用`tox` 测试多个版本

Django 和 Python 都在不断前进。如果你要与世界分享你的可安装 Django 应用，那么你可能需要在多种环境下进行[测试](https://realpython.com/python-testing/)。这个工具需要一点帮助来测试你的 Django 应用。继续在`setup.cfg`中进行以下更改:

```py
 1# setup.cfg 2[metadata] 3name  =  realpython-django-receipts 4version  =  1.0.3 5description  =  Sample installable django app 6long_description  =  file:README.rst 7url  =  https://github.com/realpython/django-receipts 8license  =  MIT 9classifiers  = 10  Development Status :: 4 - Beta 11  Environment :: Web Environment 12  Intended Audience :: Developers 13  License :: OSI Approved :: MIT License 14  Operating System :: OS Independent 15  Programming Language :: Python :: 3 :: Only 16  Programming Language :: Python :: 3.7 17  Programming Language :: Python :: Implementation :: CPython 18  Topic :: Software Development :: Libraries :: Application Frameworks 19  Topic :: Software Development :: Libraries :: Python Modules 20
21[options] 22include_package_data  =  true 23python_requires  =  >=3.6 24setup_requires  = 25  setuptools >=  38.3.0 26install_requires  = 27  Django>=2.2 28test_suite  =  load_tests.get_suite
```

**第 28 行**告诉包管理器使用`load_tests.py`脚本来获得它的测试套件。`tox`实用程序使用它来运行它的测试。回忆`load_tests.py`中的`get_suite()`:

```py
 1# Defined inside load_tests.py
 2def get_suite(labels=default_labels):
 3    from django.test.runner import DiscoverRunner
 4    runner = DiscoverRunner(verbosity=1)
 5    failures = runner.run_tests(labels)
 6    if failures:
 7        sys.exit(failures)
 8
 9    # If this is called from setuptools, then return a test suite
10    return TestSuite()
```

这里发生的事情确实有点奇怪。通常情况下，`setup.cfg`中的`test_suite`字段指向一个返回一组测试的方法。当`tox`调用`setup.py`时，它读取`test_suite`参数并运行`load_tests.get_suite()`。

如果这个调用没有返回一个`TestSuite`对象，那么`tox`就会抱怨。奇怪的是，你实际上并不希望`tox`得到一套测试，因为`tox`并不知道 Django 测试环境。相反，`get_suite()`创建一个`DiscoverRunner`并在**第 10 行返回一个空的`TestSuite`对象。**

您不能简单地让`DiscoverRunner`返回一组测试，因为您必须调用`DiscoverRunner.run_tests()`来正确执行 Django 测试环境的设置和拆卸。仅仅将正确的测试传递给`tox`是行不通的，因为数据库不会被创建。`get_suite()`运行所有的测试，但是作为函数调用的副作用，而不是作为返回测试套件给`tox`执行的正常情况。

`tox`工具允许您测试多种组合。一个`tox.ini`文件决定测试哪些环境组合。这里有一个例子:

```py
[tox] envlist  =  py{36,37}-django220, py{36,37}-django300 [testenv] deps  = django220: Django>=2.2,<3 django300: Django>=3 commands= python setup.py test
```

该文件声明应该结合 Django 2.2 和 3.0 运行 Python 3.6 和 3.7 的测试。总共有四个测试环境。`commands=`部分是你告诉`tox`通过`setup.py`调用测试的地方。这就是你在`setup.cfg`中调用`test_suite = load_tests.get_suite`钩子的方法。

**注:**`setup.py`的`test`子命令已被[弃用](https://github.com/pypa/setuptools/issues/1684)。Python 中的打包目前变化很快。虽然一般不建议打电话给`python setup.py test`，但是在这种特定的情况下，打电话是可行的。

## 发布到 PyPI

最后，是时候在 PyPI 上分享你的可安装 Django 应用了。上传包有多种工具，但在本教程中，您将重点关注 [Twine](https://twine.readthedocs.io/en/latest/) 。以下代码构建包并调用 Twine:

```py
$ python -m pip install -U wheel twine setuptools
$ python setup.py sdist
$ python setup.py bdist_wheel
$ twine upload dist/*
```

前两个命令构建包的源代码和二进制发行版。对`twine`的调用上传到 PyPI。如果您的主目录中有一个`.pypirc`文件，那么您可以预设您的用户名，这样唯一提示您的就是您的密码:

```py
[disutils] index-servers  = pypi [pypi] username: <YOUR_USERNAME>
```

我经常用一个小的 shell 脚本从代码中`grep`出版本号。然后我调用`git tag`用版本号标记 repo，删除旧的`build/`和`dist/`目录，调用上面三个命令。

关于使用 Twine 的更多细节，请参见[如何将开源 Python 包发布到 PyPI](https://realpython.com/pypi-publish-python-package/) 。Twine 的两个流行替代品是[诗歌](https://python-poetry.org/)和 [Flit](https://flit.readthedocs.io/en/latest/) 。Python 中的包管理变化很快。PEP 517 和 [PEP 518](https://www.python.org/dev/peps/pep-0518/) 正在重新定义如何描述 Python 包和依赖关系。

## 结论

Django 应用程序依赖于 Django 项目结构，因此单独打包它们需要额外的步骤。您已经看到了如何通过从项目中提取、打包并在 PyPI 上共享来制作可安装的 Django 应用程序。请务必从以下链接下载示例代码:

**下载示例代码:** [单击此处获取代码，您将使用](https://realpython.com/bonus/installable-django-app/)在本教程中学习如何编写可安装的 Django 应用程序。

**在本教程中，您已经学会了如何:**

*   在项目之外使用 Django 框架
*   在独立于项目的应用上调用 Django **管理命令**
*   编写一个调用 **Django 测试**的脚本，可选地使用一个测试标签
*   构建一个 **`setup.py`文件**来定义你的包
*   修改`setup.py`脚本以适应 **`tox`**
*   使用 **Twine** 上传你的可安装 Django 应用

你已经准备好与全世界分享你的下一款应用了。编码快乐！

[*Remove ads*](/account/join/)

## 延伸阅读

Django、打包和测试都是非常深入的话题。外面有很多信息。要深入了解，请查看以下资源:

*   [Django 文档](https://docs.djangoproject.com/en/3.0/intro/tutorial01/)
*   [Django 入门:构建投资组合应用](https://realpython.com/courses/django-portfolio-project/)
*   [Django 教程](https://realpython.com/tutorials/django/)
*   [使用 pyenv 管理多个 Python 版本](https://realpython.com/intro-to-pyenv/)
*   [pip 是什么？新蟒蛇指南](https://realpython.com/what-is-pip/)
*   [如何将开源 Python 包发布到 PyPi](https://realpython.com/pypi-publish-python-package/)
*   [Python 测试入门](https://realpython.com/python-testing/)
*   [诗歌](https://python-poetry.org/)
*   [掠过](https://flit.readthedocs.io/en/latest/)

PyPI 有大量值得一试的可安装 Django 应用。以下是一些最受欢迎的:

*   姜戈-CSP
*   Django reCAPTCHA
*   姜戈-阿劳斯
*   [草堆](https://github.com/django-haystack/django-haystack)
*   [响应式 Django 管理员](https://github.com/douglasmiranda/django-admin-bootstrap)
*   [Django 调试工具栏](https://github.com/jazzband/django-debug-toolbar)*****
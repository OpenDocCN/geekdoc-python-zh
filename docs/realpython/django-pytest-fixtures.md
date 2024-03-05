# 如何在 Pytest 中为 Django 模型提供测试夹具

> 原文：<https://realpython.com/django-pytest-fixtures/>

如果你在 [Django](https://realpython.com/getting-started-with-django-channels/) ， [`pytest` fixtures](https://pytest.org/en/latest/fixture.html) 工作，可以帮助你为你的模型创建测试，维护起来并不复杂。编写好的测试是维持一个成功应用的关键步骤，而**夹具**是让你的测试套件高效且有效的关键因素。夹具是作为测试基线的小块数据。

随着您的测试场景的变化，添加、修改和维护您的设备可能会很痛苦。但是不用担心。本教程将向你展示如何使用[`pytest-django`插件](https://pytest-django.readthedocs.io/en/latest/)来使编写新的测试用例及夹具变得轻而易举。

在本教程中，您将学习:

*   如何在 Django 中创建和加载**测试夹具**
*   如何为 Django 模型创建和加载 **`pytest`夹具**
*   如何使用**工厂**为`pytest`中的 Django 模型创建测试夹具
*   如何使用**工厂作为夹具**模式来创建测试夹具之间的依赖关系

本教程中描述的概念适用于任何使用 [`pytest`](https://realpython.com/pytest-python-testing/) 的 Python 项目。为了方便起见，示例使用了 Django ORM，但是结果可以在其他类型的 ORM 中重现，甚至可以在不使用 ORM 或数据库的项目中重现。

**免费奖励:** [点击此处获取免费的 Django 学习资源指南(PDF)](#) ，该指南向您展示了构建 Python + Django web 应用程序时要避免的技巧和窍门以及常见的陷阱。

## Django 的固定装置

首先，您将建立一个新的 Django 项目。在本教程中，您将使用[内置认证模块](https://docs.djangoproject.com/en/3.0/topics/auth/default/)编写一些测试。

[*Remove ads*](/account/join/)

### 设置 Python 虚拟环境

当你创建一个新项目时，最好也为它创建一个虚拟环境。虚拟环境允许您将该项目与计算机上的其他项目隔离开来。这样，不同的项目可以使用不同版本的 Python、Django 或任何其他包，而不会互相干扰。

以下是在新目录中创建虚拟环境的方法:

```py
$ mkdir django_fixtures
$ cd django_fixtures
django_fixtures $ python -m venv venv
```

关于如何创建虚拟环境的分步说明，请查看 [Python 虚拟环境:初级教程](https://realpython.com/python-virtual-environments-a-primer/)。

运行这个命令将创建一个名为`venv`的新目录。该目录将存储您在虚拟环境中安装的所有软件包。

### 建立 Django 项目

现在您已经有了一个全新的虚拟环境，是时候建立一个 Django 项目了。在您的终端中，激活虚拟环境并安装 Django:

```py
$ source venv/bin/activate
$ pip install django
```

现在您已经安装了 Django，您可以创建一个名为`django_fixtures`的新 Django 项目:

```py
$ django-admin startproject django_fixtures
```

运行这个命令后，您会看到 Django 创建了新的文件和目录。关于如何开始一个新的 Django 项目，请查看[开始一个 Django 项目](https://realpython.com/django-setup/)。

为了完成 Django 项目的设置，为内置模块应用[迁移](https://realpython.com/django-migrations-a-primer/):

```py
$ cd django_fixtures
$ python manage.py migrate
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
 Applying sessions.0001_initial... OK
```

输出列出了 Django 应用的所有迁移。当开始一个新项目时，Django 会为内置的应用程序如`auth`、`sessions`和`admin`进行迁移。

现在您已经准备好开始编写测试和夹具了！

### 创建 Django 装置

Django 为来自文件的模型提供了自己的[创建和加载 fixture](https://docs.djangoproject.com/en/3.0/howto/initial-data/#providing-data-with-fixtures)的方法。Django fixture 文件可以用 [JSON](https://realpython.com/courses/working-json-data-python/) 或 YAML 编写。在本教程中，您将使用 JSON 格式。

创建 Django fixture 最简单的方法是使用现有的对象。启动 Django shell:

```py
$ python manage.py shell
Python 3.8.0 (default, Oct 23 2019, 18:51:26)
[GCC 9.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
```

在 Django shell 中，创建一个名为`appusers`的新组:

>>>

```py
>>> from django.contrib.auth.models import Group
>>> group = Group.objects.create(name="appusers")
>>> group.pk
1
```

[`Group`](https://docs.djangoproject.com/en/3.0/ref/contrib/auth/#django.contrib.auth.models.Group) 模型是 [Django 的认证系统](https://docs.djangoproject.com/en/3.0/ref/contrib/auth/)的一部分。组对于管理 Django 项目中的权限非常有用。

您创建了一个名为`appusers`的新群组。您刚刚创建的群组的**主键**是`1`。为了给组`appusers`创建一个夹具，您将使用[Django 管理命令`dumpdata`](https://docs.djangoproject.com/en/3.0/ref/django-admin/#dumpdata) 。

用`exit()`退出 Django shell，并从您的终端执行以下命令:

```py
$ python manage.py dumpdata auth.Group --pk 1 --indent 4 > group.json
```

在本例中，您将使用`dumpdata`命令从现有模型实例中生成夹具文件。让我们来分解一下:

*   **`auth.Group`** :描述要转储哪个型号。格式为`<app_label>.<model_name>`。

*   **`--pk 1`** :描述要转储哪个对象。该值是逗号分隔的主键列表，如`1,2,3`。

*   **`--indent 4`** :这是一个可选的格式参数，告诉 Django 在生成的文件中的每个缩进层次之前要添加多少个空格。使用缩进使夹具文件更具可读性。

*   **`> group.json`** :描述在哪里写命令的输出。在这种情况下，输出将被写入一个名为`group.json`的文件。

接下来，检查夹具文件`group.json`的内容:

```py
[ { "model":  "auth.group", "pk":  1, "fields":  { "name":  "appusers", "permissions":  [] } } ]
```

夹具文件包含一个对象列表。在这种情况下，列表中只有一个对象。每个对象包括一个带有模型名和主键的**头**，以及一个带有模型中每个字段值的**字典**。您可以看到 fixture 包含了组名`appusers`。

您可以手动创建和编辑夹具文件，但是事先创建对象并使用 Django 的`dumpdata`命令创建夹具文件通常更方便。

[*Remove ads*](/account/join/)

### 加载 Django 夹具

现在您已经有了一个 fixture 文件，您希望将它加载到数据库中。但是在这之前，您应该打开一个 Django shell 并删除您已经创建的组:

>>>

```py
>>> from django.contrib.auth.models import Group
>>> Group.objects.filter(pk=1).delete()
(1, {'auth.Group_permissions': 0, 'auth.User_groups': 0, 'auth.Group': 1})
```

现在该组已被删除，使用 [`loaddata`命令](https://docs.djangoproject.com/en/3.0/ref/django-admin/#django-admin-loaddata)加载夹具:

```py
$ python manage.py loaddata group.json
Installed 1 object(s) from 1 fixture(s)
```

要确保加载了新组，请打开 Django shell 并获取它:

>>>

```py
>>> from django.contrib.auth.models import Group
>>> group = Group.objects.get(pk=1)
>>> vars(group)
{'_state': <django.db.models.base.ModelState at 0x7f3a012d08b0>,
 'id': 1,
 'name': 'appusers'}
```

太好了！该组已加载。您刚刚创建并加载了您的第一个 Django 设备。

### 在测试中加载 Django 装置

到目前为止，您已经从命令行创建并加载了一个 fixture 文件。现在你如何用它来测试呢？要查看 Django 测试中如何使用 fixtures，创建一个名为`test.py`的新文件，并添加以下测试:

```py
from django.test import TestCase
from django.contrib.auth.models import Group

class MyTest(TestCase):
    def test_should_create_group(self):
        group = Group.objects.get(pk=1)
        self.assertEqual(group.name, "appusers")
```

该测试获取主键为`1`的组，并测试其名称是否为`appusers`。

从您的终端运行测试:

```py
$ python manage.py test test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
E
======================================================================
ERROR: test_should_create_group (test.MyTest)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "/django_fixtures/django_fixtures/test.py", line 9, in test_should_create_group
 group = Group.objects.get(pk=1)
 File "/django_fixtures/venv/lib/python3.8/site-packages/django/db/models/manager.py", line 82, in manager_method
 return getattr(self.get_queryset(), name)(*args, **kwargs)
 File "/django_fixtures/venv/lib/python3.8/site-packages/django/db/models/query.py", line 415, in get
 raise self.model.DoesNotExist(
django.contrib.auth.models.Group.DoesNotExist: Group matching query does not exist. 
----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)
Destroying test database for alias 'default'...
```

测试失败，因为主键为`1`的组不存在。

要在测试中加载 fixture，您可以使用[类`TestCase`的一个特殊属性，称为`fixtures`](https://docs.djangoproject.com/en/3.0/topics/testing/tools/#fixture-loading) :

```py
from django.test import TestCase
from django.contrib.auth.models import Group

class MyTest(TestCase):
    fixtures = ["group.json"]
     def test_should_create_group(self):
        group = Group.objects.get(pk=1)
        self.assertEqual(group.name, "appusers")
```

将这个属性添加到一个`TestCase`中，告诉 Django 在执行每个测试之前加载 fixtures。注意`fixtures`接受一个数组，所以您可以在每次测试之前提供多个 fixture 文件来加载。

现在运行测试会产生以下输出:

```py
$ python manage.py test test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
.
----------------------------------------------------------------------
Ran 1 test in 0.005s

OK
Destroying test database for alias 'default'...
```

太神奇了！该组已加载，测试已通过。现在您可以在您的测试中使用组`appusers`。

[*Remove ads*](/account/join/)

### 引用 Django Fixtures 中的相关对象

到目前为止，您只对一个对象使用了一个文件。然而，大多数情况下，你的应用程序中会有很多模型，在测试中你会需要不止一个模型。

要查看 Django fixtures 中对象之间的依赖关系，创建一个新的用户实例，然后将其添加到您之前创建的`appusers`组中:

>>>

```py
>>> from django.contrib.auth.models import User, Group
>>> appusers = Group.objects.get(name="appusers")
>>> haki = User.objects.create_user("haki")
>>> haki.pk
1
>>> haki.groups.add(appusers)
```

用户`haki`现在是`appusers`组的成员。要查看带有外键的 fixture 是什么样子，为用户`1`生成一个 fixture:

```py
$ python manage.py dumpdata auth.User --pk 1 --indent 4
[
{
 "model": "auth.user",
 "pk": 1,
 "fields": {
 "password": "!M4dygH3ZWfd0214U59OR9nlwsRJ94HUZtvQciG8y",
 "last_login": null,
 "is_superuser": false,
 "username": "haki",
 "first_name": "",
 "last_name": "",
 "email": "",
 "is_staff": false,
 "is_active": true,
 "date_joined": "2019-12-07T09:32:50.998Z",
 "groups": [ 1 ], "user_permissions": []
 }
}
]
```

夹具的结构与您之前看到的相似。

一个用户可以与多个组相关联，因此字段`group`包含该用户所属的所有组的 id。在这种情况下，用户属于主键为`1`的组，也就是您的`appusers`组。

使用主键来引用 fixtures 中的对象并不总是一个好主意。组的主键是数据库在创建组时分配给该组的任意标识符。在另一个环境中，或者在另一台计算机上，`appusers`组可以有不同的 ID，这不会对对象产生任何影响。

为了避免使用任意标识符，Django 定义了自然键的概念。自然键是对象的唯一标识符，不一定是主键。在组的情况下，两个组不能有相同的名称，所以组的自然关键字可以是它的名称。

要使用自然键而不是主键来引用 Django fixture 中的相关对象，请将`--natural-foreign`标志添加到`dumpdata`命令中:

```py
$ python manage.py dumpdata auth.User --pk 1 --indent 4 --natural-foreign
[
{
 "model": "auth.user",
 "pk": 1,
 "fields": {
 "password": "!f4dygH3ZWfd0214X59OR9ndwsRJ94HUZ6vQciG8y",
 "last_login": null,
 "is_superuser": false,
 "username": "haki",
 "first_name": "",
 "last_name": "",
 "email": "benita",
 "is_staff": false,
 "is_active": true,
 "date_joined": "2019-12-07T09:32:50.998Z",
 "groups": [ [ `appusers` ] ], "user_permissions": []
 }
}
]
```

Django 为用户生成了 fixture，但是它没有使用`appusers`组的主键，而是使用了组名。

您还可以添加`--natural-primary`标志来从 fixture 中排除一个对象的主键。当`pk`为空时，主键将在运行时设置，通常由数据库设置。

### 维护 Django 设备

Django fixtures 很棒，但也带来了一些挑战:

*   **保持夹具更新** : Django 夹具必须包含模型的所有必需字段。如果您添加一个不可空的新字段，您必须更新 fixtures。否则，它们将无法加载。当你有很多 Django 设备时，保持它们的更新会成为一种负担。

*   维护 fixture 之间的依赖关系:依赖于其他 fixture 的 Django fixtures 必须按照特定的顺序一起加载。随着新测试用例的增加和旧测试用例的修改，跟上夹具的步伐可能是一个挑战。

由于这些原因，Django 灯具对于经常更换的车型来说并不是一个理想的选择。例如，很难维护 Django fixtures 来表示应用程序中的核心对象，如销售、订单、交易或预订。

另一方面，Django 设备是以下用例的绝佳选择:

*   **常量数据**:这适用于很少变化的型号，比如国家代码和邮政编码。

*   **初始数据**:这适用于存储你的应用的查找数据的模型，比如产品类别、用户组、用户类型。

[*Remove ads*](/account/join/)

## `pytest`姜戈的固定装置

在上一节中，您使用了 Django 提供的内置工具来创建和加载装置。Django 提供的 fixtures 对于某些用例来说很棒，但是对于其他用例来说并不理想。

在本节中，您将使用一种非常不同的夹具进行实验:夹具`pytest`。`pytest`提供了一个非常广泛的 fixture 系统，您可以使用它来创建一个可靠的、可维护的测试套件。

### 为 Django 项目设置`pytest`

要开始使用`pytest`，你首先需要安装`pytest`和`pytest` 的 [Django 插件。激活虚拟环境时，在终端中执行以下命令:](https://github.com/pytest-dev/pytest-django)

```py
$ pip install pytest
$ pip install pytest-django
```

`pytest-django`插件由`pytest`开发团队维护。它为使用`pytest`为 Django 项目编写测试提供了有用的工具。

接下来，您需要让`pytest`知道它可以在哪里找到您的 Django 项目设置。在项目的根目录下创建一个名为`pytest.ini`的新文件，并在其中添加以下几行:

```py
[pytest] DJANGO_SETTINGS_MODULE=django_fixtures.settings
```

这是使`pytest`与您的 Django 项目一起工作所需的最小配置量。还有更多[配置选项](https://docs.pytest.org/en/latest/reference.html#configuration-options)，但这已经足够开始了。

最后，为了测试您的设置，用这个虚拟测试替换`test.py`的内容:

```py
def test_foo():
    assert True
```

要运行虚拟测试，从您的终端使用`pytest`命令:

```py
$ pytest test.py
============================== test session starts ======================
platform linux -- Python 3.7.4, pytest-5.2.0, py-1.8.0, pluggy-0.13.0
Django settings: django_fixtures.settings (from ini file)
rootdir: /django_fixtures, inifile: pytest.ini
plugins: django-3.5.1

test.py .
 [100%]
============================= 1 passed in 0.05s =========================
```

您刚刚用`pytest`完成了一个新 Django 项目的设置！现在，您已经准备好深入挖掘了。

关于如何设置`pytest`和编写测试的更多信息，请查看使用`pytest` 的[测试驱动开发。](https://realpython.com/courses/test-driven-development-pytest/)

### 从测试中访问数据库

在本节中，您将使用内置认证模块`django.contrib.auth`编写测试。本模块中最熟悉的车型是`User`和`Group`。

要开始使用 Django 和`pytest`，编写一个测试来检查 Django 提供的函数`create_user()`是否正确设置了用户名:

```py
from django.contrib.auth.models import User

def test_should_create_user_with_username() -> None:
    user = User.objects.create_user("Haki")
    assert user.username == "Haki"
```

现在，尝试从命令中执行测试，如下所示:

```py
$ pytest test.py
================================== test session starts ===============
platform linux -- Python 3.7.4, pytest-5.2.0, py-1.8.0, pluggy-0.13.0
Django settings: django_fixtures.settings (from ini file)
rootdir: /django-django_fixtures/django_fixtures, inifile: pytest.ini
plugins: django-3.5.1
collected 1 item

test.py F

=============================== FAILURES =============================
____________________test_should_create_user_with_username ____________

 def test_should_create_user_with_username() -> None:
>       user = User.objects.create_user("Haki")

self = <mydbengine.base.DatabaseWrapper object at 0x7fef66ed57d0>, name = None

 def _cursor(self, name=None):
>       self.ensure_connection()

E   Failed: Database access not allowed, use the "django_db" mark, or the "db"
 or "transactional_db" fixtures to enable it.
```

命令失败，测试没有执行。这个错误消息给了你一些有用的信息:为了在测试中访问数据库，你需要注入一个叫做`db` 的特殊夹具[。`db` fixture 是您之前安装的`django-pytest`插件的一部分，它需要在测试中访问数据库。](https://pytest-django.readthedocs.io/en/latest/helpers.html#pytest-mark-django-db-request-database-access)

将`db`夹具注入测试中:

```py
from django.contrib.auth.models import User

def test_should_create_user_with_username(db) -> None:
    user = User.objects.create_user("Haki")
    assert user.username == "Haki"
```

再次运行测试:

```py
$ pytest test.py
================================== test session starts ===============
platform linux -- Python 3.7.4, pytest-5.2.0, py-1.8.0, pluggy-0.13.0
Django settings: django_fixtures.settings (from ini file)
rootdir: /django_fixtures, inifile: pytest.ini
plugins: django-3.5.1
collected 1 item

test.py .
```

太好了！命令成功完成，您的测试通过。您现在知道了如何在测试中访问数据库。您还将 fixture 注入到测试用例中。

[*Remove ads*](/account/join/)

### 为 Django 模型创建夹具

现在您已经熟悉了 Django 和`pytest`，编写一个测试来检查用`set_password()`设置的密码是否按照预期得到了验证。用该测试替换`test.py`的内容:

```py
from django.contrib.auth.models import User

def test_should_check_password(db) -> None:
    user = User.objects.create_user("A")
    user.set_password("secret")
    assert user.check_password("secret") is True

def test_should_not_check_unusable_password(db) -> None:
    user = User.objects.create_user("A")
    user.set_password("secret")
    user.set_unusable_password()
    assert user.check_password("secret") is False
```

第一个测试检查 Django 是否验证了一个拥有可用密码的用户。第二个测试检查一种边缘情况，在这种情况下，用户的密码是不可用的，不应该被 Django 验证。

这里有一个重要的区别:上面的测试用例不测试`create_user()`。他们测试`set_password()`。这意味着对`create_user()`的改变不应该影响这些测试用例。

另外，请注意,`User`实例被创建了两次，每个测试用例一次。一个大型项目可以有许多需要一个`User`实例的测试。如果每个测试用例都会创建自己的用户，那么如果`User`模型发生变化，你将来可能会有麻烦。

为了在许多测试用例中重用一个对象，您可以创建一个[测试夹具](https://pytest-django.readthedocs.io/en/latest/helpers.html#fixtures):

```py
import pytest
from django.contrib.auth.models import User

@pytest.fixture
def user_A(db) -> User:
    return User.objects.create_user("A")

def test_should_check_password(db, user_A: User) -> None:
    user_A.set_password("secret")
    assert user_A.check_password("secret") is True

def test_should_not_check_unusable_password(db, user_A: User) -> None:
    user_A.set_password("secret")
    user_A.set_unusable_password()
    assert user_A.check_password("secret") is False
```

在上面的代码中，您创建了一个名为`user_A()`的函数，该函数创建并返回一个新的`User`实例。为了将这个函数标记为 fixture，您用 [`pytest.fixture`装饰器](https://docs.pytest.org/en/latest/reference.html#pytest-fixture)来装饰它。一旦一个函数被标记为 fixture，它就可以被注入到测试用例中。在这种情况下，您将 fixture `user_A`注入到两个测试用例中。

### 需求变化时维护夹具

假设您已经向您的应用程序添加了一个新的需求，现在每个用户都必须属于一个特殊的`"app_user"`组。该组中的用户可以查看和更新他们自己的个人详细信息。为了测试您的应用程序，您需要您的测试用户也属于`"app_user"`组:

```py
import pytest
from django.contrib.auth.models import User, Group, Permission

@pytest.fixture
def user_A(db) -> Group:
    group = Group.objects.create(name="app_user")
    change_user_permissions = Permission.objects.filter(
        codename__in=["change_user", "view_user"],
    )
    group.permissions.add(*change_user_permissions)
    user = User.objects.create_user("A")
    user.groups.add(group)
    return user

def test_should_create_user(user_A: User) -> None:
    assert user_A.username == "A"

def test_user_is_in_app_user_group(user_A: User) -> None:
    assert user_A.groups.filter(name="app_user").exists()
```

在 fixture 中，您创建了组`"app_user"`，并为其添加了相关的`change_user`和`view_user`权限。然后，您创建了测试用户，并将他们添加到`"app_user"`组中。

以前，您需要检查创建用户的每个测试用例，并将其添加到组中。使用 fixtures，你可以只做一次改变。一旦你改变了夹具，同样的改变出现在你注入`user_A`的每个测试用例中。使用 fixtures，您可以避免重复，并使您的测试更易于维护。

### 将夹具注入其他夹具

大型应用程序通常不止有一个用户，经常需要用多个用户来测试它们。在这种情况下，您可以添加另一个夹具来创建测试`user_B`:

```py
import pytest
from django.contrib.auth.models import User, Group, Permission

@pytest.fixture
def user_A(db) -> User:
    group = Group.objects.create(name="app_user")
    change_user_permissions = Permission.objects.filter(
        codename__in=["change_user", "view_user"],
    )
    group.permissions.add(*change_user_permissions)
 user = User.objects.create_user("A")    user.groups.add(group)
    return user

@pytest.fixture
def user_B(db) -> User:
    group = Group.objects.create(name="app_user")
    change_user_permissions = Permission.objects.filter(
        codename__in=["change_user", "view_user"],
    )
    group.permissions.add(*change_user_permissions)
 user = User.objects.create_user("B")    user.groups.add(group)
    return user

def test_should_create_two_users(user_A: User, user_B: User) -> None:
    assert user_A.pk != user_B.pk
```

在您的终端中，尝试运行测试:

```py
$ pytest test.py
==================== test session starts =================================
platform linux -- Python 3.7.4, pytest-5.2.0, py-1.8.0, pluggy-0.13.0
Django settings: django_fixtures.settings (from ini file)
rootdir: /django_fixtures, inifile: pytest.ini
plugins: django-3.5.1
collected 1 item

test.py E
 [100%]
============================= ERRORS ======================================
_____________ ERROR at setup of test_should_create_two_users ______________

self = <django.db.backends.utils.CursorWrapper object at 0x7fc6ad1df210>,
sql ='INSERT INTO "auth_group" ("name") VALUES (%s) RETURNING "auth_group"."id"'
,params = ('app_user',)

 def _execute(self, sql, params, *ignored_wrapper_args):
 self.db.validate_no_broken_transaction()
 with self.db.wrap_database_errors:
 if params is None:
 # params default might be backend specific.
 return self.cursor.execute(sql)
 else:
>               return self.cursor.execute(sql, params)
E               psycopg2.IntegrityError: duplicate key value violates
 unique constraint "auth_group_name_key" E               DETAIL:  Key (name)=(app_user) already exists.   ======================== 1 error in 4.14s ================================
```

新的测试抛出一个`IntegrityError`。错误消息来自数据库，因此根据您使用的数据库，它看起来可能会有所不同。根据错误消息，测试违反了组名的唯一约束。当你看着你的固定装置，它是有意义的。`"app_user"`组创建两次，一次在夹具`user_A`中，另一次在夹具`user_B`中。

到目前为止，我们忽略的一个有趣的观察是夹具`user_A`正在使用夹具`db`。这意味着**夹具可以注入到其他夹具**中。你可以用这个特性来解决上面的`IntegrityError`。在夹具中仅创建一次`"app_user"`组，并将其注入到`user_A`和`user_B`夹具中。

为此，重构您的测试并添加一个`"app user"`组 fixture:

```py
import pytest
from django.contrib.auth.models import User, Group, Permission

@pytest.fixture
def app_user_group(db) -> Group:
    group = Group.objects.create(name="app_user")
    change_user_permissions = Permission.objects.filter(
        codename__in=["change_user", "view_user"],
    )
    group.permissions.add(*change_user_permissions)
    return group

@pytest.fixture
def user_A(db, app_user_group: Group) -> User:
    user = User.objects.create_user("A")
    user.groups.add(app_user_group)
    return user

@pytest.fixture
def user_B(db, app_user_group: Group) -> User:
    user = User.objects.create_user("B")
    user.groups.add(app_user_group)
    return user

def test_should_create_two_users(user_A: User, user_B: User) -> None:
    assert user_A.pk != user_B.pk
```

在您的终端中，运行您的测试:

```py
$ pytest test.py
================================== test session starts ===============
platform linux -- Python 3.7.4, pytest-5.2.0, py-1.8.0, pluggy-0.13.0
Django settings: django_fixtures.settings (from ini file)
rootdir: /django_fixtures, inifile: pytest.ini
plugins: django-3.5.1
collected 1 item

test.py .
```

太神奇了！你的测试通过了。group fixture 封装了与`"app user"`组相关的逻辑，比如设置权限。然后，您将该组注入到两个独立的用户设备中。通过以这种方式构建您的 fixtures，您已经使您的测试变得不那么复杂，易于阅读和维护。

[*Remove ads*](/account/join/)

### 使用工厂

到目前为止，您已经创建了很少参数的对象。然而，有些对象可能更复杂，具有许多参数和许多可能的值。对于这样的对象，您可能想要创建几个测试夹具。

例如，如果您为 [`create_user()`](https://docs.djangoproject.com/en/3.0/ref/contrib/auth/#django.contrib.auth.models.Userapp_user.create_user) 提供所有参数，那么 fixture 看起来会是这样的:

```py
import pytest
from django.contrib.auth.models import User

@pytest.fixture
def user_A(db, app_user_group: Group) -> User
    user = User.objects.create_user(
        username="A",
        password="secret",
        first_name="haki",
        last_name="benita",
        email="me@hakibenita.com",
        is_staff=False,
        is_superuser=False,
        is_active=True,
    )
    user.groups.add(app_user_group)
    return user
```

你的夹具变得更复杂了！用户实例现在可以有许多不同的变体，例如超级用户、职员用户、非活动职员用户和非活动普通用户。

在前面的章节中，您了解到在每个测试夹具中维护复杂的设置逻辑是很困难的。因此，为了避免每次创建用户时都必须重复所有的值，可以添加一个函数，使用`create_user()`根据应用程序的特定需求创建用户:

```py
from typing import List, Optional
from django.contrib.auth.models import User, Group

def create_app_user(
    username: str,
    password: Optional[str] = None,
    first_name: Optional[str] = "first name",
    last_name: Optional[str] = "last name",
    email: Optional[str] = "foo@bar.com",
    is_staff: str = False,
    is_superuser: str = False,
    is_active: str = True,
    groups: List[Group] = [],
) -> User:
    user = User.objects.create_user(
        username=username,
        password=password,
        first_name=first_name,
        last_name=last_name,
        email=email,
        is_staff=is_staff,
        is_superuser=is_superuser,
        is_active=is_active,
    )
    user.groups.add(*groups)
    return user
```

该函数创建一个应用程序用户。根据应用程序的具体要求，每个参数都设置了合理的默认值。例如，您的应用程序可能要求每个用户都有一个电子邮件地址，但是 Django 的内置函数不会强制这样的限制。相反，您可以在函数中强制要求。

创建对象的函数和类通常被称为**工厂**。为什么？这是因为这些函数充当了生产特定类的实例的工厂。关于 Python 中工厂的更多信息，请查看[工厂方法模式及其在 Python 中的实现](https://realpython.com/factory-method-python/)。

上面的函数是一个工厂的简单实现。它没有状态，也没有实现任何复杂的逻辑。您可以重构您的测试，以便它们使用工厂函数在您的 fixtures 中创建用户实例:

```py
@pytest.fixture
def user_A(db, app_user_group: Group) -> User:
 return create_user(username="A", groups=[app_user_group]) 
@pytest.fixture
def user_B(db, app_user_group: Group) -> User:
 return create_user(username="B", groups=[app_user_group]) 
def test_should_create_user(user_A: User, app_user_group: Group) -> None:
    assert user_A.username == "A"
    assert user_A.email == "foo@bar.com"
    assert user_A.groups.filter(pk=app_user_group.pk).exists()

def test_should_create_two_users(user_A: User, user_B: User) -> None:
    assert user_A.pk != user_B.pk
```

您的夹具变得更短，并且您的测试现在对变化更有弹性。例如，如果您使用了一个[定制用户模型](https://docs.djangoproject.com/en/3.0/topics/auth/customizing/#auth-custom-user)，并且您刚刚向该模型添加了一个新字段，那么您只需要更改`create_user()`就可以让您的测试按预期工作。

### 利用工厂作为固定设备

复杂的设置逻辑使得编写和维护测试变得更加困难，使得整个套件变得脆弱，对变化的适应能力更差。到目前为止，您已经通过创建 fixture、创建 fixture 之间的依赖关系以及使用一个工厂来抽象尽可能多的设置逻辑解决了这个问题。

但是在您的测试夹具中仍然有一些设置逻辑:

```py
@pytest.fixture
def user_A(db, app_user_group: Group) -> User:
 return create_user(username="A", groups=[app_user_group]) 
@pytest.fixture
def user_B(db, app_user_group: Group) -> User:
 return create_user(username="B", groups=[app_user_group])
```

两个夹具都注射了`app_user_group`。目前这是必要的，因为工厂功能`create_user()`无法访问`app_user_group`夹具。在每个测试中都有这样的设置逻辑会使修改变得更加困难，并且在将来的测试中更容易被忽略。相反，您希望封装创建用户的整个过程，并从测试中抽象出来。这样，您可以专注于手头的场景，而不是设置独特的测试数据。

为了向用户工厂提供对`app_user_group` fixture 的访问，您可以使用一个名为 [factory 的模式作为 fixture](https://docs.pytest.org/en/latest/fixture.html#factories-as-fixtures) :

```py
from typing import List, Optional

import pytest
from django.contrib.auth.models import User, Group, Permission

@pytest.fixture
def app_user_group(db) -> Group:
    group = Group.objects.create(name="app_user")
    change_user_permissions = Permission.objects.filter(
        codename__in=["change_user", "view_user"],
    )
    group.permissions.add(*change_user_permissions)
    return group

@pytest.fixture
def app_user_factory(db, app_user_group: Group):
    # Closure
    def create_app_user(
        username: str,
        password: Optional[str] = None,
        first_name: Optional[str] = "first name",
        last_name: Optional[str] = "last name",
        email: Optional[str] = "foo@bar.com",
        is_staff: str = False,
        is_superuser: str = False,
        is_active: str = True,
        groups: List[Group] = [],
    ) -> User:
        user = User.objects.create_user(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            email=email,
            is_staff=is_staff,
            is_superuser=is_superuser,
            is_active=is_active,
        )
        user.groups.add(app_user_group)
        # Add additional groups, if provided.
        user.groups.add(*groups)
        return user
    return create_app_user
```

这离你已经做的不远了，所以让我们来分解一下:

*   `app_user_group`夹具保持不变。它创建了特殊的`"app user"`组，拥有所有必要的权限。

*   添加了一个名为`app_user_factory`的新夹具，它与`app_user_group`夹具一起注入。

*   fixture `app_user_factory`创建一个闭包，并返回一个名为`create_app_user()`的[内部函数](https://realpython.com/inner-functions-what-are-they-good-for/)。

*   `create_app_user()`类似于您之前实现的函数，但是现在它可以访问 fixture `app_user_group`。通过访问该组，您现在可以在工厂功能中将用户添加到`app_user_group`。

要使用`app_user_factory` fixture，将其注入另一个 fixture 并使用它创建一个用户实例:

```py
@pytest.fixture
def user_A(db, app_user_factory) -> User:
    return app_user_factory("A")

@pytest.fixture
def user_B(db, app_user_factory) -> User:
    return app_user_factory("B")

def test_should_create_user_in_app_user_group(
    user_A: User,
    app_user_group: Group,
) -> None:
    assert user_A.groups.filter(pk=app_user_group.pk).exists()

def test_should_create_two_users(user_A: User, user_B: User) -> None:
    assert user_A.pk != user_B.pk
```

注意，与之前不同，您创建的 fixture 提供了一个*函数*，而不是一个*对象*。这是 fixture 模式工厂背后的主要概念:**工厂 fixture 创建了一个闭包，它为内部函数提供了对 fixture 的访问。**

关于 Python 中闭包的更多信息，请查看 [Python 内部函数——它们有什么用处？](https://realpython.com/inner-functions-what-are-they-good-for/)

现在您已经有了自己的工厂和设备，这是您的测试的完整代码:

```py
from typing import List, Optional

import pytest
from django.contrib.auth.models import User, Group, Permission

@pytest.fixture
def app_user_group(db) -> Group:
    group = Group.objects.create(name="app_user")
    change_user_permissions = Permission.objects.filter(
        codename__in=["change_user", "view_user"],
    )
    group.permissions.add(*change_user_permissions)
    return group

@pytest.fixture
def app_user_factory(db, app_user_group: Group):
    # Closure
    def create_app_user(
        username: str,
        password: Optional[str] = None,
        first_name: Optional[str] = "first name",
        last_name: Optional[str] = "last name",
        email: Optional[str] = "foo@bar.com",
        is_staff: str = False,
        is_superuser: str = False,
        is_active: str = True,
        groups: List[Group] = [],
    ) -> User:
        user = User.objects.create_user(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            email=email,
            is_staff=is_staff,
            is_superuser=is_superuser,
            is_active=is_active,
        )
        user.groups.add(app_user_group)
        # Add additional groups, if provided.
        user.groups.add(*groups)
        return user
    return create_app_user

@pytest.fixture
def user_A(db, app_user_factory) -> User:
    return app_user_factory("A")

@pytest.fixture
def user_B(db, app_user_factory) -> User:
    return app_user_factory("B")

def test_should_create_user_in_app_user_group(
    user_A: User,
    app_user_group: Group,
) -> None:
    assert user_A.groups.filter(pk=app_user_group.pk).exists()

def test_should_create_two_users(user_A: User, user_B: User) -> None:
    assert user_A.pk != user_B.pk
```

打开终端并运行测试:

```py
$ pytest test.py
======================== test session starts ========================
platform linux -- Python 3.8.1, pytest-5.3.3, py-1.8.1, pluggy-0.13.1
django: settings: django_fixtures.settings (from ini)
rootdir: /django_fixtures/django_fixtures, inifile: pytest.ini
plugins: django-3.8.0
collected 2 items

test.py ..                                                     [100%]

======================== 2 passed in 0.17s ==========================
```

干得好！您已经在测试中成功地实现了工厂作为夹具模式。

[*Remove ads*](/account/join/)

### 工厂作为实践中的固定装置

工厂作为夹具模式是非常有用的。如此有用，事实上，你可以在`pytest`本身提供的夹具中找到它。比如`pytest`提供的 [`tmp_path`](https://docs.pytest.org/en/latest/tmpdir.html#the-tmp-path-fixture) 夹具，就是夹具厂 [`tmp_path_factory`](https://docs.pytest.org/en/latest/tmpdir.html#the-tmp-path-factory-fixture) 创造的。同样， [`tmpdir`](https://docs.pytest.org/en/latest/tmpdir.html#the-tmpdir-fixture) 夹具由夹具厂 [`tmpdir_factory`](https://docs.pytest.org/en/latest/tmpdir.html#the-tmpdir-factory-fixture) 创建。

掌握工厂作为 fixture 模式可以消除许多与编写和维护测试相关的麻烦。

## 结论

您已经成功实现了一个提供 Django 模型实例的 fixture 工厂。您还维护和实现了夹具之间的依赖关系，这种方式消除了编写和维护测试的一些麻烦。

**在本教程中，您已经学习了:**

*   如何在 Django 中创建和加载**夹具**
*   如何在`pytest`中为 Django 车型提供**测试夹具**
*   如何使用**工厂**为`pytest`中的 Django 模型创建夹具
*   如何实现工厂作为夹具的模式来创建测试夹具之间的依赖关系

您现在能够实现和维护一个可靠的测试套件，这将帮助您更快地生成更好、更可靠的代码！*******
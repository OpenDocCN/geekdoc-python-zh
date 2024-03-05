# 使用 Flask 的 Python Web 应用程序–第三部分

> 原文：<https://realpython.com/python-web-applications-with-flask-part-iii/>

请注意:这是来自 Real Python 的 Michael Herman 和来自 [De Deo Designs](http://dedeodesigns.com/) 的 Python 开发者 Sean Vieira 的合作作品。

* * *

**本系列文章:**

1.  第一部分:[应用程序设置](https://realpython.com/python-web-applications-with-flask-part-i/)
2.  第二部分:[设置用户账户、模板、静态文件](https://realpython.com/python-web-applications-with-flask-part-ii/)
3.  **第三部分:测试(单元和集成)、调试和错误处理← *当前文章***

欢迎回到烧瓶跟踪开发系列！对于那些刚刚加入我们的人来说，我们正在实现一个符合[这个餐巾纸规范](https://realpython.com/python-web-applications-with-flask-part-i/#toc_1)的网络分析应用。对于所有在家的人来说，你可以查看今天的代码:

```py
$ git checkout v0.3
```

或者你可以从 Github 的[发布页面下载。那些刚刚加入我们的人可能也希望](https://github.com/mjhea0/flask-tracking/releases)[读一读关于存储库结构的注释](https://realpython.com/python-web-applications-with-flask-part-i/#toc_5)。

[在前面的部分](https://realpython.com/python-web-applications-with-flask-part-ii/)中，我们在应用程序中添加了用户账户。本周我们将致力于实现一个测试框架，讨论一下为什么测试很重要，然后为我们的应用程序编写一些测试。之后，我们将讨论一下应用程序和日志中的调试错误。

## 为什么要测试

在我们实际编写任何测试之前，让我们来谈谈[为什么测试是重要的](https://realpython.com/python-testing/)。如果你还记得《T2》第一部中的 Python 之禅，你可能已经注意到“简单比复杂好”就在“复杂比复杂好”的正上方。*简单是理想，复杂往往是现实。*尤其是 Web 应用程序，有许多可移动的部分，可以很快地从简单变得复杂。

随着应用程序复杂性的增加，我们希望确保我们创建的各种活动部件能够继续以和谐的方式一起工作。我们不想改变一个实用函数的签名，它破坏了生产中一个看似不相关的特性。此外，我们希望确保我们的更改仍然保留正确的 T2 功能。一个总是返回同一个`datetime`实例的方法每天有效和正确两次，但在其余时间都有效和不正确。

测试是很好的调试辅助工具。编写一个产生我们所看到的无效行为的测试有助于我们从不同的角度来看待我们的代码。此外，一旦我们通过了测试，我们就确保了我们不会再次引入这个 bug(至少以那种特定的方式)。

测试也是文档的极好来源。因为它们必须处理预期的输入和输出，所以阅读测试套件将澄清被测代码预期要做什么。这将阐明我们编写的文档中不清楚的部分(或者在简单的情况下，甚至替换它)。

最后，测试可以是一个很好的探索工具——在我们写代码之前，勾画出我们希望 T1 如何与我们的代码交互，揭示更简单的 API，并帮助我们掩盖一个领域的内部复杂性。[“测试驱动开发”](http://en.wikipedia.org/wiki/Test-driven_development)是对这个过程的最终承诺。在 TDD 中，我们首先编写测试来覆盖代码的功能，然后才编写代码。

测试将使它显而易见:

*   当代码不工作时，
*   什么代码被破坏了，还有
*   我们当初为什么要写这段代码。

每次我们向应用程序添加功能、修复 bug 或更改代码时，我们都应该确保我们的代码被测试充分覆盖，并且在我们完成后测试全部通过。

**做:**

*   添加测试以涵盖代码的基本功能。
*   添加测试来覆盖你能想到的尽可能多的代码的角落/边缘情况。
*   添加测试来覆盖您回去后没有想到的角落/边缘情况，并修复它们。
*   提醒您的编码同行充分测试他们的代码。
*   关于没有通过测试的代码的 Bug。

**不要:**

*   不经测试提交代码。
*   提交没有通过或破坏测试的代码。
*   更改您的测试，以便您的代码通过测试而不修复问题。

既然我们已经知道了为什么测试如此重要，让我们开始为我们的应用程序编写一些测试。

[*Remove ads*](/account/join/)

## 设置

每个功能块都需要测试。为了简洁明了地做到这一点，每个包中都有一个`tests.py`模块。这样我们就知道每个包的测试在哪里，如果我们需要把它从应用程序中分离出来，它们就包含在包中。

我们将使用 [`Flask-Testing`](http://pythonhosted.org/Flask-Testing/) 扩展，因为它有一堆有用的测试特性，我们无论如何都要设置它们。继续将`Flask-Testing==0.4`添加到`requirements.txt`的底部，然后运行`pip install -r requirements.txt`。

烧瓶测试消除了几乎所有为单元测试设置烧瓶的样板文件。剩下的一小部分我们将放入新模块`test_base.py`:

```py
# flask_testing/test_base.py
from flask.ext.testing import TestCase

from . import app, db

class BaseTestCase(TestCase):
    """A base test case for flask-tracking."""

    def create_app(self):
        app.config.from_object('config.TestConfiguration')
        return app

    def setUp(self):
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
```

这个测试用例没有做任何引人注目的事情——它只是用我们的测试配置来配置应用程序，在每个测试开始时创建所有的表，在每个测试结束时删除所有的表。这样，每个测试用例都是从一个干净的石板开始的，我们可以花更多的时间编写测试，花更少的时间调试我们的测试用例。由于每个测试用例都将继承我们新的`BaseTestCase()`类，我们将避免复制和粘贴这个配置到我们为应用程序创建的每个包中。

我们做的另外一件事是模块化我们的配置。最初的`config.py`模块只支持一种配置——我们可以更新它以适应不同环境之间的差异。提醒一下，这是《T2》第二部中`config.py`的样子:

```py
# config.py
from os.path import abspath, dirname, join

_cwd = dirname(abspath(__file__))

SECRET_KEY = 'flask-session-insecure-secret-key'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + join(_cwd, 'flask-tracking.db')
SQLALCHEMY_ECHO = True
```

现在看起来几乎是一样的——我们只是创建了一个保存所有这些配置值的类:

```py
from os.path import abspath, dirname, join

_cwd = dirname(abspath(__file__))

class BaseConfiguration(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'flask-session-insecure-secret-key'
    HASH_ROUNDS = 100000
    # ... etc. ...
```

我们可以从中继承:

```py
class TestConfiguration(BaseConfiguration):
    TESTING = True
    WTF_CSRF_ENABLED = False

    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # + join(_cwd, 'testing.db')

    # Since we want our unit tests to run quickly
    # we turn this down - the hashing is still done
    # but the time-consuming part is left out.
    HASH_ROUNDS = 1
```

通过这种方式，可以轻松共享所有环境通用的设置，并且我们可以轻松地在特定于环境的配置中覆盖我们需要的设置。(这个模式直接来源于 [Flask 的优秀文档](http://flask.pocoo.org/docs/config/#development-production)。)

我们使用内存中的 [SQLite 数据库](https://realpython.com/python-sqlite-sqlalchemy/)进行测试，以确保我们的测试尽可能快地执行。我们想要享受运行测试的乐趣，只有当它们在合理的时间内执行时，我们才能做到这一点。如果我们真的需要访问测试运行的结果，我们可以用计算出的到`tests.db`的路径覆盖`:memory:`设置。(那是我们`TestConfiguration`里注释掉的`+ join(_cwd, 'testing.db')`)。

我们还在配置中添加了一个`HASH_ROUNDS`键，以控制用户密码在存储之前应该散列多少次。我们可以改变`flask_tracking.users.models.User`的`_hash_password`方法来使用这个键:

```py
from flask import current_app

# ... snip ...

def _hash_password(self, password):
    # ... snip ...
    rounds = current_app.config.get("HASH_ROUNDS", 100000)
    buff = pbkdf2_hmac("sha512", pwd, salt, iterations=rounds)
```

这确保了我们的单元测试将快速运行——否则，每次我们需要创建或登录用户时，我们将不得不等待 100，000 轮 sha512 完成，然后才能继续我们的测试。

最后，我们需要在`flask_tracking/__init__.py`中更新我们的`app.from_object`调用。之前，我们使用`app.from_object('config')`加载配置。现在我们在配置模块中有了两个配置，我们想把它改为`app.from_object('config.BaseConfiguration')`。

现在我们准备测试我们的应用程序。

## 测试

我们将从`users`包开始。

从上次[的](https://realpython.com/python-web-applications-with-flask-part-ii/)我们知道`users`包负责:

*   注册，
*   登录，以及
*   注销。

因此，我们需要编写测试，涵盖用户注册、用户登录和用户注销。让我们从一个简单的例子开始——一个现有用户试图登录:

```py
# flask_tracking/users/tests.py
from flask import url_for

from flask_tracking.test_base import BaseTestCase
from .models import User

class UserViewsTests(BaseTestCase):
    def test_users_can_login(self):
        User.create(name='Joe', email='joe@joes.com', password='12345')

        response = self.client.post(url_for('users.login'),
                                    data={'email': 'joe@joes.com', 'password': '12345'})

        self.assert_redirects(response, url_for('tracking.index'))
```

因为每个测试用例都是从一个完全干净的数据库开始的，所以我们必须首先创建一个现有的用户。然后，我们可以提交与用户(Joe)尝试登录时提交的请求相同的请求。我们希望确保如果 Joe 成功登录，他将被重定向回主页。

我们可以使用 Python 内置的 [`unittest`](http://docs.python.org/2/library/unittest.html) 测试运行器来运行我们的测试。从项目的根目录运行以下命令:

```py
$ python -m unittest discover
```

这会产生以下输出:

```py
----------------------------------------------------------------------
Ran 1 test in 0.045s

OK
```

万岁！我们现在有一个通过测试！让我们测试一下我们与 Flask-Login 的集成是否正常。`current_user`应该是乔，所以我们应该能做到以下几点:

```py
from flask.ext.login import current_user

# And then inside of our test_users_can_login function:
self.assertTrue(current_user.name == 'Joe')
self.assertFalse(current_user.is_anonymous())
```

但是，如果我们尝试这样做，我们会得到以下错误:

>>>

```py
AttributeError: 'AnonymousUserMixin' object has no attribute 'name'
```

`current_user`需要在请求的上下文中被访问(它是一个线程本地对象，就像`flask.request`)。当`self.client.post`完成请求并且每个线程本地对象被拆除时。我们需要保留请求上下文，以便测试我们与 Flask-Login 的集成。幸运的是，`Flask`的 [`test_client`](http://flask.pocoo.org/docs/api/#flask.Flask.test_client) 是一个[上下文管理器](http://flask.pocoo.org/docs/testing/#keeping-the-context-around)，这意味着我们可以在 [`with`语句](https://realpython.com/python-with-statement/)中使用它，只要我们需要它，它就会保留上下文:

```py
with self.client:
    response = self.client.post(url_for('users.login'),
                                data={'email': 'joe@joes.com', 'password': '12345'})

    self.assert_redirects(response, url_for('index'))
    self.assertTrue(current_user.name == 'Joe')
    self.assertFalse(current_user.is_anonymous())
```

现在，当我们再次运行测试时，我们通过了！

```py
----------------------------------------------------------------------
Ran 1 test in 0.053s
```

让我们确保当 Joe 登录时，他可以注销:

```py
def test_users_can_logout(self):
    User.create(name="Joe", email="joe@joes.com", password="12345")

    with self.client:
        self.client.post(url_for("users.login"),
                         data={"email": "joe@joes.com",
                               "password": "12345"})
        self.client.get(url_for("users.logout"))

        self.assertTrue(current_user.is_anonymous())
```

我们再一次创建了 Joe(记住，数据库在每次测试结束时都会被重置)。然后我们让他登录(我们知道这是可行的，因为我们的第一个测试通过了)。最后，我们通过`self.client.get(url_for("users.logout"))`请求注销页面让他注销，并确保我们拥有的用户再次匿名。再次运行测试，享受两次通过测试的满足感。

我们还需要检查其他一些东西:

*   用户可以注册然后登录应用程序吗？
*   当用户注销时，他们会被重定向回索引页面吗？

这些测试可以在[的`flask-tracking`库](https://github.com/mjhea0/flask-tracking)中找到，如果你想复习的话。由于它们与我们已经写过的内容相似，我们将在这里跳过它们。

[*Remove ads*](/account/join/)

## 模拟和集成测试

不过，我们的应用程序有一个部分与其他部分略有不同——我们在`tracking`包中的`add_visit`端点不仅与数据库和用户交互——它还与第三方服务 [Free GeoIP](http://freegeoip.net/) 交互。由于这是一个潜在的破损源，我们将希望彻底测试它。由于免费的 GeoIP 是一个第三方服务(我们可能并不总是能够获得)，它也给了我们一个很好的机会来谈论单元测试和集成测试之间的区别。

### 单元测试与集成测试

到目前为止，我们所写的一切都属于单元测试的范畴。一个**单元测试**是对*我们的*代码的最小可能功能块的测试——对一段不可分割的代码(通常是一个函数或方法)的测试。

**集成测试**，另一方面，测试我们的应用程序的边界——我们的应用程序是否与其他应用程序(很可能是我们编写的)正确交互？测试我们的应用程序是否正确调用 Free GeoIP 并与之交互是一个集成测试。这类测试非常重要，因为它们让我们知道我们所依赖的特性仍然按照我们期望的方式工作。(是的，当我们在生产环境中运行我们的应用程序时，如果 Free GeoIP 更改其合同(API)或完全关闭，这对我们没有帮助，但这正是日志记录的目的——我们稍后将对此进行介绍。)

然而，集成测试的问题是它们通常比单元测试慢一个数量级以上。大量的集成测试会使我们的测试套件变慢，以至于需要一分钟以上的时间来运行——一旦越过这个界限，我们的测试套件就开始成为障碍而不是助手。现在花时间运行我们的测试会打断我们的注意力，而不是简单地验证我们是否在正确的轨道上。此外，对于像 Free GeoIP 这样的分布式服务，这意味着如果我们离线或 Free GeoIP 宕机，我们实际上无法运行我们的测试套件。

这让我们陷入了两难的境地——一方面，集成测试非常重要，另一方面，运行集成测试可能会中断我们的工作流程。

解决方案很简单——我们可以为我们调用的服务创建一个基本的本地实现(在测试术语中称为模拟),并使用这个模拟运行我们的单元测试。我们可以将集成测试分离到一个单独的文件中，并在提交代码更改之前运行这些测试。这样，我们获得了好的单元测试的速度，并保留了集成测试提供的确定性。

### 嘲讽免费地理信息

如果你还记得《T4》第二部分的话，我们在`tracking`包中添加了一个`geodata`模块，实现了一个单一的功能`get_geodata`。我们在我们的`tracking.add_visit`视图中使用这个函数:

```py
ip_address = request.access_route[0] or request.remote_addr
geodata = get_geodata(ip_address)
```

在我们的*单元*测试中，我们想要做的是确保当`get_geodata`按预期工作时，我们将在数据库中正确地记录访问。然而，我们不想调用免费的 GeoIP(否则，我们的测试将比我们的其他测试慢，并且我们将无法在离线时运行测试。)我们需要用另一个函数(一个 mock)替换`get_geodata`。

首先，让我们安装[一个模仿库](http://mock.readthedocs.org/en/latest/)来简化这个过程。将 [`mock==1.0.1`](http://mock.readthedocs.org/en/latest/) 添加到 requirements.txt，再次添加`pip install -r requirements.txt`。(如果您使用的是 Python 3.3 或更高版本，那么您已经将 mock 安装为 [`unittest.mock`](https://realpython.com/python-mock-library/) 。)

现在我们可以编写我们的单元测试了:

```py
# flask_tracking/tracking/tests.py
from decimal import Decimal

from flask import url_for
from mock import Mock, patch
from werkzeug.datastructures import Headers

from flask_tracking.test_base import BaseTestCase
from flask_tracking.users.models import User
from .models import Site, Visit
from ..tracking import views

class TrackingViewsTests(BaseTestCase):
    def test_visitors_location_is_derived_from_ip(self):
        user = User.create(name='Joe', email='joe@joe.com', password='12345')
        site = Site.create(user_id=user.id)

        mock_geodata = Mock(name='get_geodata')
        mock_geodata.return_value = {
            'city': 'Los Angeles',
            'zipcode': '90001',
            'latitude': '34.05',
            'longitude': '-118.25'
        }

        url = url_for('tracking.add_visit', site_id=site.id)
        wsgi_environment = {'REMOTE_ADDR': '1.2.3.4'}
        headers = Headers([('Referer', '/some/url')])

        with patch.object(views, 'get_geodata', mock_geodata):
            with self.client:
                self.client.get(url, environ_overrides=wsgi_environment,
                                headers=headers)

                visits = Visit.query.all()

                mock_geodata.assert_called_once_with('1.2.3.4')
                self.assertEqual(1, len(visits))

                first_visit = visits[0]
                self.assertEqual("/some/url", first_visit.url)
                self.assertEqual('Los Angeles, 90001', first_visit.location)
                self.assertEqual(34.05, first_visit.latitude)
                self.assertEqual(-118.25, first_visit.longitude)
```

不要担心——测试这类集成的痛苦会因为这样一个事实而减轻，即应用程序中的集成通常比代码单元要少。让我们一节一节地浏览这段代码，并把它分成易于理解的几个部分。

### 设置测试数据和模拟

首先，我们设置一个用户和一个站点，因为每次测试开始时数据库都是空的:

```py
def test_visitors_location_is_derived_from_ip(self):
    user = User.create(name='Joe', email='joe@joe.com', password='12345')
    site = Site.create(user_id=user.id)
```

然后，我们创建一个 mock 函数，并指定它应该在每次被调用时返回一个包含洛杉矶坐标的字典(我们可以简单地创建一个总是返回字典的简单函数，但是 mock 还提供了 [`patch.*`](http://mock.readthedocs.org/en/latest/patch.html) 上下文管理器，这非常有用，所以我们将使用这个库):

```py
mock_geodata = Mock(name='get_geodata')
mock_geodata.return_value = {
    'city': 'Los Angeles',
    'zipcode': '90001',
    'latitude': '34.05',
    'longitude': '-118.25'
}
```

最后，我们设置我们将要访问的 URL 和我们需要让`tracking.add_visit`工作的 WSGI 环境的部分(在本例中，它只是我们的假终端用户的访问者的 [IP 地址](https://realpython.com/python-ipaddress-module/)和他们应该来自的 URL):

```py
url = url_for('tracking.add_visit', site_id=site.id)
wsgi_environment = {'REMOTE_ADDR': '1.2.3.4'}
headers = Headers([('Referer', '/some/url')])
```

[*Remove ads*](/account/join/)

### 将模拟补丁插入我们的跟踪模块

我们显式地将`flask_tracking.tracking.views`模块导入到我们的`tests`模块中:

```py
from ..tracking import views
```

现在我们修补模块的`get_views`名，指向我们的`mock_geodata`对象，而不是`flask_tracking.tracking.geodata.get_geodata`函数:

```py
with patch.object(views, 'get_geodata', mock_geodata):
```

通过使用`patch.object`作为上下文管理器，我们确保在我们退出这个`with`块后`flask_tracking.tracking.views.get_geodata`将再次指向`flask_tracking.tracking.geodata.get_geodata`。我们也可以使用`patch.object`作为装饰:

```py
mock_geodata = Mock(name='get_geodata')
# ... snip return setup ...

class TrackingViewsTests(BaseTestCase):
    @patch.object(views, 'get_geodata', mock_geodata)
    def test_visitors_location_is_derived_from_ip(self):
```

或者甚至是一个班级装饰者:

```py
@patch.object(views, 'get_geodata', mock_geodata)
class TrackingViewsTests(BaseTestCase):
```

唯一的区别是补丁的范围。函数装饰版本确保只要我们在`test_visitors_location_is_derived_from_ip`内部，函数`get_geodata`就指向我们的模拟，而类装饰版本确保每个在`TrackingViewsTests`内部以`test`开头的函数都将看到`get_geodata`的模拟版本。

就我个人而言，我更喜欢尽可能限制我的模仿范围。这有助于确保我记住我的测试范围，并避免我在期望访问真实对象并必须对其进行修补时出现意外。

### 运行测试

设置好我们需要的一切后，我们现在可以提出我们的请求:

```py
with self.client:
    self.client.get(url, environ_overrides=wsgi_environment,
                    headers=headers)
```

我们通过自己创建的`wsgi_environment`字典(`wsgi_environment = {'REMOTE_ADDR': '1.2.3.4'}`)向控制器提供查看者的 IP 地址。Flask 的测试客户端是 Werkzeug 的测试客户端的一个实例——它支持你可以传递给 [`EnvironmentBuilder`](http://werkzeug.pocoo.org/docs/test/#werkzeug.test.EnvironBuilder) 的所有参数。

### 断言一切正常

最后，我们从`tracking_visit`表中获取所有的访问:

```py
visits = Visit.query.all()
```

并验证:

*   我们使用用户的 IP 地址来查找他的地理数据:

```py
mock_geodata.assert_called_once_with('1.2.3.4')
```

*   该请求只引发了一次访问:

```py
self.assertEqual(1, len(visits))
```

*   位置数据被正确保存:

```py
first_visit = visits[0]
self.assertEqual("/some/url", first_visit.url)
self.assertEqual('Los Angeles, 90001', first_visit.location)
self.assertEqual(Decimal("34.05"), first_visit.latitude)
self.assertEqual(Decimal("-118.25"), first_visit.longitude)
```

当我们运行`python -m unittest discover`时，我们得到以下输出:

```py
F.....
======================================================================
FAIL: test_visitors_location_is_derived_from_ip (flask_tracking.tracking.tests.TrackingViewsTests)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "~/dev/flask-tracking/flask_tracking/tracking/tests.py", line 41, in test_visitors_location_is_derived_from_ip
 self.assertEqual('Los Angeles, 90001', first_visit.location)
AssertionError: 'Los Angeles, 90001' != None

----------------------------------------------------------------------
Ran 6 tests in 0.147s

FAILED (failures=1)
```

啊，失败了！显然，我们没有正确地映射位置，因为`Visit`的位置没有被持久存储在数据库中。检查我们的视图代码发现，当我们构造我们的`VisitForm`时，我们确实在设置`location`…但是我们实际上并没有*为我们的`VisitForm`设置*字段！还好我们在直播前就发现了。(这种重复的字段会产生问题，应该会给大家带来一些启示——当它出现时，我建议大家看一看 [`wtforms-alchemy`](http://wtforms-alchemy.readthedocs.org/en/latest/) 。)

一旦我们将`location`、`latitude`和`longitude`字段添加到我们的`VisitForm`中，我们应该能够运行我们的测试并得到:

```py
.....
----------------------------------------------------------------------
Ran 5 tests in 0.150s

OK
```

这就完成了我们的第一次模拟测试。

[*Remove ads*](/account/join/)

## 调试

我们的单元测试非常有用——但是当我们试图测试一些东西，而测试代码并没有做我们期望它做的事情时，会发生什么呢？或者更糟糕的是，当用户打电话给我们，抱怨他遇到了一个错误？如果这是一个系统范围的问题，运行我们的单元测试可能会揭示这个问题…但是这只有在我们没有运行我们的测试就签入*并部署*代码的情况下才会发生(我们永远不会这样做，不是吗？)

假设我们总是在提交和部署之前运行测试，那么当生产中出现问题时，我们的单元测试就无法帮助我们。相反，我们将需要要求用户为我们提供一个完整的例子，以便我们可以在本地调试它。

假设我们对我们的登录表单进行了一点点重构-

```py
# If you can see what's broken already, give yourself a prize
# and write a test to ensure it never happens again :-)

class LoginForm(Form):
    email = fields.StringField(validators=[InputRequired(), Email()])
    password = fields.StringField(validators=[InputRequired()])

    def validate_login(form, field):
        try:
            user = User.query.filter(User.email == form.email.data).one()
        except (MultipleResultsFound, NoResultFound):
            raise ValidationError("Invalid user")
        if user is None:
            raise ValidationError("Invalid user")
        if not user.is_valid_password(form.password.data):
            raise ValidationError("Invalid password")

        # Make the current user available
        # to calling code.
        form.user = user
```

-当我们将其推向生产时，我们的第一个用户向我们发送了一封电子邮件，告诉我们他输入了错误的密码，并且仍然登录到系统中。我们在现场验证了这一点。哇，耐莉，这是完全不能接受的！因此，我们迅速关闭登录页面，代之以一条消息，说我们正在进行维护，我们会尽快回来(*SaaS 法则# 0——永远以你希望被对待的方式对待你的客户*)。

从本地来看，我们看不出用户*应该*能够不用密码登录的任何理由。然而，我们还没有编写任何测试来测试错误输入的密码会被错误消息拒绝，所以我们不能 100%确定这不是我们代码中的错误。因此，让我们编写一个测试用例，看看会发生什么:

```py
def test_invalid_password_is_rejected(self):
    User.create(name="Joe", email="joe@joes.com", password="12345")

    with self.client:
        response = self.client.post(url_for("users.login"),
                                    data={"email": "joe@joes.com",
                                          "password": "*****"})

        self.assertTrue(current_user.is_anonymous())
        self.assert_200(response)
        self.assertIn("Invalid password", response.data)
```

运行测试会导致失败:

```py
.F....
======================================================================
FAIL: test_invalid_password_is_rejected (app.users.tests.UserViewsTests)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "~/dev/flask-tracking/flask_tracking/users/tests.py", line 34, in test_invalid_password_is_rejected
 self.assertTrue(current_user.is_anonymous())
AssertionError: False is not true
```

好吧，我们可以在本地复制它。我们有一个测试用例对我们大喊大叫，直到我们解决问题。很好。我们在路上了！

有几种方法可以调试这个问题:

*   我们可以将`print`语句分散在整个应用程序中，直到找到错误的来源。
*   我们可以在代码中故意生成错误，并使用 Flask 的内置调试器查看现有环境。
*   我们可以使用调试器来单步调试代码。

我们将使用这三种技术。首先，让我们给我们的`app.users.models.LoginForm#validate_login`方法添加一个简单的`print`语句:

```py
def validate_login(self, field):
    print 'Validating login'
```

当我们再次运行测试时，我们根本看不到“验证登录”消息。这告诉我们我们的方法没有被调用。让我们在视图中添加一个故意的错误，并利用 Flask 的内部调试器来验证世界的状态。首先，我们将为调试创建一个新的配置:

```py
# config.py
class DebugConfiguration(BaseConfiguration):
    DEBUG = True
```

然后我们将更新`flask_tracking.__init__`以使用新的调试配置:

```py
app.config.from_object('config.DebugConfiguration')
```

最后，我们将在我们的`login_view`方法中添加一个算术错误:

```py
def login_view():
    form = LoginForm(request.form)
    1 / 0  # KABOOM!
```

现在，如果我们跑:

```py
$ python run.py
```

导航到登录页面，我们会看到[一个很好的追溯](https://realpython.com/python-traceback/)。点击回溯(`1 / 0`)最后一行右边的外壳图标，将得到一个交互式 REPL，我们可以用它来测试我们的功能:

>>>

```py
>>> form.validate_login(field=None)  # We don't use the field argument
```

这导致:

```py
Traceback (most recent call last):
    File "<debugger>", line 1, in <module>
    form.validate_login(None)
    File "~/dev/flask-tracking/flask_tracking/users/forms.py", line 15, in validate_login
    raise validators.ValidationError('Invalid user')
    ValidationError: Invalid user
```

所以现在我们知道我们的验证函数*起作用了*——它只是没有被调用。让我们从登录视图中删除这个被零除的错误，代之以对 Python 调试器`pdb` 的调用[。](https://realpython.com/python-debugging-pdb/)

```py
def login_view():
    form = LoginForm(request.form)
    import pdb; pdb.set_trace()
```

现在，当我们再次运行测试时，我们会得到一个调试器:

```py
python -m unittest discover .
.> ~/dev/flask-tracking/app/users/views.py(18)login_view()
-> if form.validate_on_submit():
(Pdb)
```

我们可以通过键入“s”表示“step”来进入`validate_on_submit`方法，并用“n”表示“next”来跳过我们不感兴趣的调用(对 PDB 的完整介绍超出了本教程的范围——有关 PDB 的更多信息，请参见它的[文档](http://docs.python.org/2/library/pdb.html)，或者在`pdb`中键入“h”):

```py
(Pdb) s
--Call--
> ~/.virtualenvs/realpython/lib/python2.7/site-packages/flask_wtf/form.py(120)validate_on_submit()
-> def validate_on_submit(self):
```

我不会带你经历整个调试过程，但是不用说，问题出在我们的代码上。WTForms 允许形式为`validate_[fieldname]`的内联验证器。我们的`validate_login`方法从未被调用，因为我们的表单中没有名为`login`的字段。让我们从控制器中移除`set_trace`调用，并将我们的`flask_tracking.users.forms.LoginForm.validate_login`方法重新命名为`LoginForm.validate_password`，这样 WTForms 就会将它作为我们的`password`字段的内联验证器。这确保了只有在 name 和 password 字段都被验证为包含用户提供的数据之后，才会调用它。

现在，当我们再次运行我们的单元测试时，它们应该会通过。本地测试表明我们确实解决了这个问题。我们现在可以安全地部署和记录我们的维护消息。

[*Remove ads*](/account/join/)

## 错误处理

正如我们已经发现的，一个测试套件并不能保证我们的应用程序没有错误。用户仍有可能在生产中遇到错误。例如，如果我们简单地盲目访问我们的一个控制器中的`request.args['some_optional_key']`,并且我们只使用请求中的可选键集来编写测试，最终用户将默认从 Flask 得到一个`400 Bad Request`响应。在这种情况下，我们希望向用户显示一条*有用的*错误消息。我们还希望避免向用户显示没有品牌或过期的页面，而没有太多的帮助去哪里，或者下一步做什么。

我们可以向 Flask 注册错误处理程序来明确处理这类问题。让我们为最常见的错误注册一个——输入错误或不再存在的链接:

```py
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
```

我们可能还希望显式处理其他类型的错误，例如 Flask 为丢失的键生成的 400 个错误请求错误，以及为未捕获的异常生成的 500 个内部服务器错误:

```py
@app.errorhandler(400)
def key_error(e):
    return render_template('400.html'), 400

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('generic.html'), 500
```

除了我们可能不得不处理的 HTTP 错误之外，Flask 还允许我们在一个未捕获的异常出现时显示不同的错误页面。现在，让我们只为所有未捕获的异常注册一个通用的错误处理程序(但是稍后我们可能希望为我们无能为力的更常见的错误情况注册特定的错误处理程序):

```py
@app.errorhandler(Exception)
def unhandled_exception(e):
    return render_template('generic.html'), 500
```

现在，所有最常见的错误都应该由我们的应用程序优雅地处理了。

然而，我们可以做得更好——让我们确保每个错误都有很好的样式，我们可以为每个可能的错误条件注册相同的错误处理程序:

```py
# flask_tracking/errors.py
from flask import current_app, Markup, render_template, request
from werkzeug.exceptions import default_exceptions, HTTPException

def error_handler(error):
    msg = "Request resulted in {}".format(error)
    current_app.logger.warning(msg, exc_info=error)

    if isinstance(error, HTTPException):
        description = error.get_description(request.environ)
        code = error.code
        name = error.name
    else:
        description = ("We encountered an error "
                       "while trying to fulfill your request")
        code = 500
        name = 'Internal Server Error'

    # Flask supports looking up multiple templates and rendering the first
    # one it finds.  This will let us create specific error pages
    # for errors where we can provide the user some additional help.
    # (Like a 404, for example).
    templates_to_try = ['errors/{}.html'.format(code), 'errors/generic.html']
    return render_template(templates_to_try,
                           code=code,
                           name=Markup(name),
                           description=Markup(description),
                           error=error)

def init_app(app):
    for exception in default_exceptions:
        app.register_error_handler(exception, error_handler)

    app.register_error_handler(Exception, error_handler)

# This can be used in __init__ with a
# import .errors
# errors.init_app(app)
```

这确保了 Flask 知道如何处理的所有 HTTP 错误条件(4XX 和 5XX 级错误)都将`error_handler`函数注册为它们的处理程序。再加上一个`app.register_error_handler(Exception, error_handler)`，这将涵盖我们的应用程序可能抛出的几乎每一个错误。(有一些例外，比如`SystemExit`不会以这种方式被捕获，C 级 segfaults 或 OS 级事件显然不会以这种方式被捕获和处理，但是这些灾难性事件应该是不可抗力事件，而不是我们的应用程序需要半定期准备处理的事情)。

## 记录日志

最后说一下[日志](https://realpython.com/python-logging/)。用户并不总是有足够的时间向我们提交一份完整的 bug 报告(更不用说，坏人会积极寻找利用我们的方法)。)我们需要一种方法来确保我们可以及时回顾过去，看看发生了什么，什么时候发生的。

幸运的是，Python 和 Flask 都有日志功能，所以我们也不需要重新发明轮子。在`app.logger`的 Flask 对象上有一个标准的 Python `logging`记录器。

我们可以使用日志记录的第一个地方是在我们的错误处理程序中。我们不需要记录 404，因为如果设置正确，代理服务器会为我们做这件事，但是我们希望记录其他异常(400、500 和异常)的原因。让我们继续向这些处理程序添加一些更详细的日志记录。因为我们对所有的错误都使用相同的处理程序，所以这很简单:

```py
def error_handler(error):
    error_name = error.__name__ if error else "Unknown-Error"
    app.logger.warning('Request resulted in {}'.format(error_name), exc_info=error)
    # ... etc. ...
```

Python 关于日志模块的文档对各种可用的[日志级别](http://docs.python.org/2/howto/logging.html#when-to-use-logging)以及它们最适合的用途进行了很好的分类。

当我们无法访问`app`(比如说，在我们的`view`模块内部)时，我们可以像使用`app`一样使用线程本地`current_app`。举个例子，让我们在登录和注销处理程序中添加一些日志记录:

```py
from flask import current_app

@users.route('/logout/')
def logout_view():
    current_app.debug('Attempting to log out the current user')
    logout_user()
    current_app.debug('Successfully logged out the current user')
    return redirect(url_for('tracking.index'))
```

这段代码很好地展示了我们在日志记录中可能遇到的一个问题——日志记录太多和太少一样糟糕。在这种情况下，我们的调试代码和应用程序代码一样多，很难再跟踪代码的流程。我们将继续删除这个特定的日志记录代码，因为除了我们在代理服务器的访问日志中看到的内容之外，它没有添加任何内容。

如果我们需要记录每个控制器的进入和退出，我们可以为 [`app.before_request`](http://flask.pocoo.org/docs/api/#flask.Flask.before_request) 和 [`app.teardown_request`](http://flask.pocoo.org/docs/api/#flask.Flask.teardown_request) 添加处理程序。只是为了好玩，下面是我们如何记录对应用程序的每次访问:

```py
@app.before_request
def log_entry():
    context = {
        'url': request.path,
        'method': request.method,
        'ip': request.environ.get("REMOTE_ADDR")
    }
    app.logger.debug("Handling %(method)s request from %(ip)s for %(url)s", context)
```

如果我们在调试模式下运行我们的应用程序并访问我们的主页，那么我们将看到:

```py
--------------------------------------------------------------------------------
DEBUG in __init__ [~/dev/flask-tracking/flask_tracking/__init__.py:68]:
Handling GET request from 127.0.0.1 for /
--------------------------------------------------------------------------------
```

如上所述，在生产日志中，这类信息会复制我们的代理服务器(Apache with mod_wsgi，ngnix with uwsgi，等等)的日志。)将会生成。只有当我们为每个请求生成一个我们绝对需要跟踪的唯一值时，我们才应该这样做。

[*Remove ads*](/account/join/)

### 向我们的日志添加上下文和格式

然而，在我们的异常处理程序中有来自我们的`log_entry`处理程序的上下文就更好了。让我们继续向记录器添加一个 [`Filter`](http://docs.python.org/2/library/logging.html#filter-objects) 实例，以便向所有感兴趣的记录器提供 url、方法、IP 地址和用户 id(这被称为[“上下文日志记录”](http://docs.python.org/2/howto/logging-cookbook.html#filters-contextual):

```py
# flask_tracking/logs.py
import logging

class ContextualFilter(logging.Filter):
    def filter(self, log_record):
        log_record.url = request.path
        log_record.method = request.method
        log_record.ip = request.environ.get("REMOTE_ADDR")
        log_record.user_id = -1 if current_user.is_anonymous() else current_user.get_id()

        return True
```

这个过滤器实际上并不过滤我们的任何消息——相反，它提供了一些我们可以在日志中使用的附加信息。以下是我们如何使用此过滤器的示例:

```py
# Create the filter and add it to the base application logger
context_provider = ContextualFilter()
app.logger.addFilter(context_provider)

# Optionally, remove Flask's default debug handler
# del app.logger.handlers[:]

# Create a new handler for log messages that will send them to standard error
handler = logging.StreamHandler()

# Add a formatter that makes use of our new contextual information
log_format = "%(asctime)s\t%(levelname)s\t%(user_id)s\t%(ip)s\t%(method)s\t%(url)s\t%(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)

# Finally, attach the handler to our logger
app.logger.addHandler(handler)
```

日志消息可能是这样的:

```py
2013-10-12 09:22:52,764    DEBUG   1   127.0.0.1   GET / Some additional message
```

需要注意的一点是，我们传递给`app.logger.[LOGLEVEL]`的消息没有用上下文中的值来扩展。因此，如果我们保留我们的`before_request`日志调用，并将我们的 before request 日志调用更改为

```py
# Note the missing context argument
app.logger.debug("Handling %(method)s request from %(ip)s for %(url)s")
```

-格式字符串将原封不动地通过。但是既然我们的 [`Formatter`](http://docs.python.org/2/library/logging.html#formatter-objects) 中有它们，我们就可以把它们从我们的个人消息中去掉，只留下:

```py
@app.before_request
def log_entry():
    app.logger.debug("Handling request")
```

这就是上下文日志记录的优势——我们可以在所有日志条目中包含重要信息，而不需要在每个日志调用的站点手动收集这些信息。

### 将原木导向不同的地方

我们将记录的大部分信息不会立即付诸行动。然而，我们想立即了解某些类型的错误。例如，一连串的 500 个错误可能意味着我们的应用程序出了问题。我们不能 24/7 粘在我们的日志上，所以我们需要将严重的错误发送给我们。

幸运的是，向我们的应用程序日志记录器添加新的处理程序很容易——因为每个处理程序都可以过滤日志条目，只过滤它感兴趣的条目，所以我们可以避免被日志淹没，但当某些东西严重损坏时，我们仍然会收到警报。

举例来说，让我们添加另一个处理程序，将错误和关键日志消息记录到一个特殊的文件中。这不会给我们想要的提醒，但是[电子邮件](https://realpython.com/python-send-email/)或短信设置取决于您的主机(我们将在 Heroku 的后续文章中进行这样的设置)。为了激起你的兴趣，请看 [Flask 的记录文档](http://flask.pocoo.org/docs/errorhandling/#error-mails)或[中如何记录电子邮件的例子，这些食谱](http://stackoverflow.com/q/8616617/135978):

```py
from logging import ERROR
from logging.handlers import TimedRotatingFileHandler

# Only set up a file handler if we know where to put the logs
if app.config.get("ERROR_LOG_PATH"):

    # Create one file for each day. Delete logs over 7 days old.
    file_handler = TimedRotatingFileHandler(app.config["ERROR_LOG_PATH"], when="D", backupCount=7)

    # Use a multi-line format for this logger, for easier scanning
    file_formatter = logging.Formatter('''
 Time: %(asctime)s Level: %(levelname)s Method: %(method)s Path: %(url)s IP: %(ip)s User ID: %(user_id)s Message: %(message)s ---------------------''')

    # Filter out all log messages that are lower than Error.
    file_handler.setLevel(ERROR)

    file_handler.addFormatter(file_formatter)
    app.logger.addHandler(file_handler)
```

如果我们使用此设置，错误和关键日志消息将同时出现在控制台和配置中指定的文件中。

## 总结

我们在这篇文章中已经讨论了很多。

1.  从单元测试开始，我们讨论了什么是测试以及为什么我们需要测试。
2.  我们继续编写测试，包括有模拟和没有模拟。
3.  我们简要介绍了本地调试错误的三种方法(`print`、触发 Flask 调试器的故意错误和`pdb`。)
4.  我们讨论了错误处理，并确保我们的最终用户只能看到有风格的错误页面。
5.  最后，我们讨论了日志设置。

在第四部分中，我们将进行一些测试驱动的开发，使我们的应用程序能够接受付款并显示简单的报告。

在第五部分中，我们将编写一个 RESTful JSON API 供其他人使用。

在第六部分中，我们将介绍使用 Fabric 和基本 A/B 特性测试的自动化部署(在 Heroku 上)。

最后，在第七部分中，我们将介绍如何用文档、代码覆盖率和质量度量工具来保护您的应用程序。

和往常一样，代码可以从[库](https://github.com/mjhea0/flask-tracking)中获得。期待与您一起继续这一旅程。******
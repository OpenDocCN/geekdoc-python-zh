# 如何写出优美的代码

# 好的结构是关键

# 如何测试 Python 应用

# 测试

对于一个开源项目来说，文档和测试都是必不可少的组成部分，没有足够测试和文档覆盖率的 “开源项目”就是一坨垃圾！当然，对于某些能够做到自文档的大神来说， 文档可以是不必要的，但测试依旧是代码质量的保证。

优秀的测试通常遵循一下基本规章：

1.  每个测试单元应该关注于一个功能，并保证其正确性。

1.  测试单元之间应该尽可能独立，也就是说可以独立运行，与顺序无关。

1.  测试的速度应该尽可能快，过慢的测试速度会成为开发的瓶颈。对于耗费时间很长的重型测试，应该将其独立出来。

1.  在集中编程前后都应该完整地运行一遍测试，以保证不会造成意外的破坏。

1.  在编程过程中，如果需要中断工作，那么编写一个不能运行的测试对于恢复工作非常有帮助。

1.  debug 的第一步就是写一个针对性的单元测试，虽然这做起来并不一定容易，但却非常有价值。

1.  虽然 PEP8 提倡简短的命名，但在测试函数名称应该长而有意义。比如，编程中你可能使用 `square()` 甚至 `sqr()` 这样的函数名称，但是在测试中你应该写成：`test_square_of_number_2()`, `test_square_negative_number()`。

1.  对于新成员来说，阅读测试代码可能是他们了解系统的最快途径之一，热点、难点、边界情况都会一目了然。 因此，加入新功能的第一步应该是编写一个对应的单元测试。

## 基本概念

### 单元测试

[单元测试](https://zh.wikipedia.org/wiki/%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95)是针对程序最小模块单位 进行正确性检验的测试工作。最小单位通常是函数或者方法。理想情况下，每一个单元测试应该独立于其它用例。 单元测试通常由软件开发人员编写，用于确保他们��写的代码符合软件需求和遵循开发目标。

在自动化测试时，为了实现隔离的效果，测试将脱离待测程序单元（或代码主体）本身固有的运行环境之外， 即脱离产品环境或其本身被创建和调用的上下文环境，而在测试框架中运行。 以隔离方式运行有利于充分显露待测试代码与其它程序单元或者产品数据空间的依赖关系。 这些依赖关系在单元测试中可以被消除。隔离模块经常会使用 stubs、mock 或 fake 等测试马甲程序。

### 集成测试

整合测试又称组装测试，即对程序模块采用一次性或增殖方式组装起来，对系统的接口进行正确性检验的测试工作。 整合测试一般在单元测试之后、系统测试之前进行。实践表明，有时模块虽然可以单独工作， 但是并不能保证组装起来也可以同时工作。

### 系统测试

系统测试是将需测试的软件，作为整个基于计算机系统的一个元素， 与计算机硬件、外设、某些支持软件、数据和人员等其他系统元素及环境结合在一起测试。 在实际运行(使用)环境下，对计算机系统进行一系列的组装测试和确认测试。 系统测试的目的在于通过与系统的需求定义作比较，发现软件与系统定义不符合或与之矛盾的地方。

## 基本工具

### doctest

Python 还提供了一个叫做 [doctest](https://docs.python.org/2/library/doctest.html) 的工具，写法如下：

```
"""
一个最简单的 doctest 写法，我这种缩进是为了照顾 Sphinx 文档自动生成工具::

    >>> factorial(5)
    120
"""

def factorial(n):
    """依旧是 doctest，不过更加复杂::

        >>> [factorial(n) for n in range(6)]
        [1, 1, 2, 6, 24, 120]
        >>> [factorial(long(n)) for n in range(6)]
        [1, 1, 2, 6, 24, 120]
        >>> factorial(30)
        265252859812191058636308480000000L
        >>> factorial(30L)
        265252859812191058636308480000000L
        >>> factorial(-1)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 0

        Factorials of floats are OK, but the float must be an exact integer:
        >>> factorial(30.1)
        Traceback (most recent call last):
            ...
        ValueError: n must be exact integer
        >>> factorial(30.0)
        265252859812191058636308480000000L

        It must also not be ridiculously large:
        >>> factorial(1e100)
        Traceback (most recent call last):
            ...
        OverflowError: n too large
    """
    import math

    if not n >= 0:
        raise ValueError("n must be >= 0")
    if math.floor(n) != n:
        raise ValueError("n must be exact integer")
    if n+1 == n:  # catch a value like 1e300
        raise OverflowError("n too large")
    result = 1
    factor = 2
    while factor <= n:
        result *= factor
        factor += 1
    return result

if __name__ == "__main__":
    import doctest
    doctest.testmod() 
```

如果不在代码中显式 `import doctest` 也可以在运行文件的时候输入这样的命令： `python -m doctest -v filename.py`。

从上面的示例代码中也可以看出，doctest 并便于不提供完整的边界数据测试的支持，因此并不能完全替代单元测试。

### unittest 和 unittest2

Python 自带了 [unittest](https://docs.python.org/2/library/unittest.html) 库， 是 Java JUnit 库的 Python 实现，虽然很好用，但我还是想在这里吐槽一下驼峰式命名的方法。 在 Python 2.7 版本以后，unittest.TestCase 类自带了 `assertListEquel()` 等方法， 非常便利，也是我不愿意兼容 Python 2.6 的重要原因。

附即将弃用的方法对照表：

| 方法名 | 即将弃用的方法名 |
| --- | --- |
| assertEqual() | failUnlessEqual, assertEquals |
| assertNotEqual() | failIfEqual |
| assertTrue() | failUnless, assert_ |
| assertFalse() | failIf |
| assertRaises() | failUnlessRaises |
| assertAlmostEqual() | failUnlessAlmostEqual |
| assertNotAlmostEqual() | failIfAlmostEqual |

[unittest2](http://www.voidspace.org.uk/python/articles/unittest2.shtml) 是 unittest 的增强版本，几乎完全兼容 unittest 的接口，升级时只需要将 `import unittest` 替换为 `import unittest2` 即可，提供的新方法更强大也更严谨。

### py.test

[pytest](http://pytest.org/latest/) 是一个成熟的全功能测试框架。

## web 相关

对于 web 功能的测试，最简单的可以使用 `urllib2.get(url)`，然后测试输出的 HTML 结果是否符合预期。 当然针对每一个功能都这样写未免太过低效，因此知名 web 框架大多有专门的测试库提供测试：

+   Django 内置了 [django.test](https://docs.djangoproject.com/en/1.8/topics/testing/overview/)

+   Tornado 内置了 [tornado.testing](http://tornado.readthedocs.org/en/latest/testing.html)

+   Flask 可以使用 [werkzeug.test](http://werkzeug.pocoo.org/docs/0.10/test/#werkzeug.test.Client) 和第三方的 [Flask-Testing](https://pythonhosted.org/Flask-Testing/)

### Django

Django 的启动互相之间的依赖严重，大部分文件都不能单独执行，测试时建议使用封装后的工具， 如： `django.test`、`django_nose` 等等。

### Flask

Flask 在写测试的时候需要主要 `app_context` 和 `request_context` 中的[陷阱](http://flask.pocoo.org/docs/0.10/appcontext/)。

### Tornado

Tornado 的 testing 库很简陋，主要是针对自身异步特性封装了一些工具。

### 浏览器

浏览器端的测试自动化最常用的还是 [Selenium](http://www.seleniumhq.org/)，Python 版本的 [文档](https://selenium-python.readthedocs.org/)并不复杂。示例代码：

```
import unittest
from selenium import webdriver

class TestOne(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Firefox()          # 初始化浏览器，也可以选择 Chrome 或者 PhanatomJS
        self.driver.set_window_size(1280, 550)

    def test_url(self):
        self.driver.get("http://duckduckgo.com/")
        self.driver.find_element_by_id(
            'search_form_input_homepage').send_keys("realpython")
        self.driver.find_element_by_id("search_button_homepage").click()
        self.assertIn(
            "https://duckduckgo.com/?q=realpython", self.driver.current_url
        )

    def tearDown(self):
        self.driver.quit()

if __name__ == '__main__':
    unittest.main() 
```

上面的代码会启动浏览器（这里设置的是 Firefox），并触发浏览器事件模拟用户输入。对于没有浏览器的机器， 比如服务器或，可以配置远程 Selenium server 或者使用 Headless 的 PhantomJS 代替。使用 Headless 浏览器 因为减��了打开和关闭浏览器的时间，因此在测试效率上也更高一些。

使用之后感触最深的是错误提示不够丰富，基本上只能断定页面结果并不符合预期，结果反馈跟 `unittest.TestCase` 简直天壤之别。

## 其它工具

### nose

### tox

发布独立库的时候通常要考虑不同版本间兼容性的问题，虽然可以通过 virtualenv 实现环境的模拟， 但毕竟很不方便，[tox](http://tox.readthedocs.org/) 正是解决这一问题的工具。

tox 简化了 virtualenv 的管理，提供了简便的配置。我常用的配置是这样的：

```
# 文件 tox.ini 的内容，需要和 setup.py 置于同一目录
[tox]
envlist = py26,py27
[testenv]
deps=               # 测试依赖
commands=make test         # 执行测试的命令 
```

### mock

[mock](http://www.voidspace.org.uk/python/mock/) 是一个测试库，提供模拟对象供测试用例使用。 [Python 3 以后](https://docs.python.org/3/library/unittest.mock.html#module-unittest.mock)， 已将 mock 已经加入标准库，调用方法是 `from unittest import mock`。

## Code Coverage

对于任何充分覆盖测试的代码，其 Code Coverage 程度肯定是 100%，任何覆盖率没能达到 100% 的代码都有隐藏 bug 的可能。在 Python 社区，代码覆盖计算工具的标准是 `coverage.py` ， 当然，在计算覆盖率时要记得配合 tox ，以保证你针对不同环境的代码都被运行过。

coverage.py 的工作流程请参阅：[coverage.py 的工作原理](http://coverage.readthedocs.org/en/latest/howitworks.html)； 详细文档请参阅：[文档](http://coverage.readthedocs.org/)。

coverage.py 也有 nose 插件，可以配合使用。

## CI

+   [travis CI](https://travis-ci.org/) 对于开源项目免费

+   [GitLab CI](https://ci.gitlab.com) 免费，提供本地部署支持

## 如何提高测试速度

如果测试很耗费时间，很容易引起开发人员的不满，因而怠于编写测试，所以说提高测试速度对于落实测试来说十分重要。 总结了一些提升测试效率的方法：

1.  合理使用 `setUpClass` 和 `tearDownClass` 方法。作为类方法，在拥有多个测试方法时也只会在一个测试用例中执行一次。

1.  数据库很慢，避免使用数据库。一定需要的话，请使用内存数据库（比如 SQLite）。

1.  使用 [mock](http://mock.readthedocs.org/en/latest/getting-started.html)，避免使用 model。

1.  如果测试写起来很困难，说明需要重构了。

1.  Celery 可以使用如下配置:

    ```
    CELERY_ALWAYS_EAGER = True
    CELERY_EAGER_PROPAGATES_EXCEPTIONS = True
    BROKER_BACKEND = 'memory' 
    ```

1.  django.test.utils.override_settings

1.  关闭调试和日志

1.  删除不必要的中间件和 app

这部分建议对于 Python 项目基本上也适用。

## 参考链接

+   [Python/Django 测试入门](http://django-testing-docs.readthedocs.org/)

+   [DjangoCon 2013 - 如何在 Django 中编写快速高效的单元测试](http://www.slideshare.net/cordiskinsey/djangocon-2013-how-to-write-fast-and-efficient-unit-tests-in-django)

+   [测试和 Django](http://carljm.github.io/django-testing-slides)

# 如何测试 Django 应用

# 如何测试 Django 应用？

Django 的启动互相之间的依赖严重，很多参数和依赖都需要在运行的时候导入，导致大部分文件都不能单独执行。 不过 Django 的社区非常活跃，对于知名的测试框架都有进行封装，如： `django.test`、`django_nose` 等等， 以配合自身的测试命令使用。

## doctest

在 Flask 中测试一个文件的 doctest 只需要运行：`python filename.py`，然而这在 Django 中行不通。 在 Django 中依赖自身的 test 命令：`python manage.py test[ app_name]`，其中 `app_name` 若为空 默认测试所有应用。[在 1.6 及以后版本中](https://docs.djangoproject.com/en/1.6/releases/1.6/#new-test-runner)， 需要首先在 `settings.py` 中指定 `TEST_RUNNER`：

```
INSTALLED_APPS = (
    ...
    'django_nose',
)
TEST_RUNNER = 'django_nose.NoseTestSuiteRunner'
NOSE_ARGS = ['--with-doctest'] 
```

## TestCase

Django 的 `TestCase` 类是 `unittest.TestCase` 的子类，使用起来非常相似。

## Fixture

Fixture 是 unittest 提供读取测试数据的一种方式，在 Django 的 TestCase 中也可以直接使用，使用前需要导出数据：

`python manage.py dumpdata --format=yaml --indent=4 > fixtures_dir/filename.yaml`

支持的数据格式包括 YAML、JSON 等等，YAML 可读性较高，不过需要安装额外的依赖。

配合 testserver 命令启动：

```
python manage.py testserver fixtures_dir/filename.yaml 
```

在测试用例中指定：

```
from django.test import TestCase
from django.contrib.auth import authenticate

class LoginTest(TestCase):
    fixtures = ['mysite.yaml']

    def setUp(self):
        # 导入 fixture 中用户数据，省去创建用户的流程，也免去了清除用户数据的流程。

    def test_has_user(self):
        # 如果已导入 fixture 中数据，则可以使用其中的账号登录。
        self.assertIsNotNone(authenticate(username='windrunner', password='password')) 
```

## Client

Client 提供了用户代理的模拟，其使用类似于 requests 库，不过使用前需要先初始化：`client = Client()`， Client 默认会提供 CSRF 认证，如果需要手动验证 CSRF，需要这样初始化： `csrf_enabled_client = Client(enforce_csrf_checks=True)`。

```
import unittest
from django.test.client import Client

class PageTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_home(self):
        res = self.client.get('/')
        self.assertEqual(200, res.status_code)

    def test_login(self):
        """普通测试。client 实例会自动解决 csrf 问题。"""
        res = self.client.get('/login/')

        self.assertEqual(200, res.status_code)
        self.assertIn('Username', res.content)

        res_post = self.client.post('/login/', {'username': 'windrunner', 'password': 'password', })

        self.assertEqual(200, res_post.status_code)
        self.assertIn('windrunner', res_post.content)

    def test_login_csrf(self):
        """强制 csrf 检查"""
        self.client = Client(enforce_csrf_checks=True)          # 使用检查 CSRF 的 Client 示例代替默认实例
        res = self.client.get('/login/')
        csrf_token = '%s' % res.context['csrf_token']             # 获取 csrf_token

        res_fail = self.client.post('/login/', {'user': 'windrunner', 'pass': 'password', })
        self.assertEqual(403, res_fail.status_code)             # 没有处理 CSRF token 会返回 403 错误代码

        res_csrf = self.client.post('/login/', {'user': 'windrunner', 'pass': 'password', 'csrfmiddlewaretoken': csrf_token, })
        self.assertIn('windrunner', res_csrf.content)

    def test_logout(self):
        res = self.client.post('/logout/')
        self.assertEqual(302, res.status_code) 
```

## testserver

testserver 是 Django 提供的启动测试服务器的方法，会创建一个测试数据库来替代默认数据库， 通常会在启动时导入相应 fixture。命令如下：

```
python manage.py testserver --addrport 7000 fixture1 fixture2 
```

## Selenium

因为 Selenium 是控制浏览器测试 web 服务，因此并不会受到 Django 的干扰，这里有一段示例代码：

```
import unittest
from selenium import webdriver
from django.contrib.auth import get_user_model, authenticate

class LoginTest(unittest.TestCase):
    def setUp(self):
        self.browser = webdriver.Firefox()          # 初始化浏览器，也可以选择 Chrome 或者 PhanatomJS

    def tearDown(self):
        self.browser.quit()                         # 测试结束后关闭浏览器

    def _login(self):
        # 这个方法没有以 ``test`` 开始，因此并不会单独被执行。
        self.browser.get('http://localhost:8000/login')         # 发送 GET 请求并打开页面

        # 使用浏览器的选择权选中 HTML 元素，并发送浏览器事件，复杂的元素选择可以借助 XPath
        self.browser.find_element_by_id('username').send_keys('windrunner')
        self.browser.find_element_by_id('password').send_keys('password')
        self.browser.find_element_by_id('submit').click()   # 触发点击事件

    def test_login(self):
        self._login()
        self.assertIn('windrunner', self.browser.page_source)   # 断言登录后的页面内容

    def test_logout(self):
        self._login()
        self.assertIn('windrunner', self.browser.page_source)
        self.browser.get('http://localhost:8000/logout')
        self.assertIn('nobody', self.browser.page_source)
        self.assertNotIn('windrunner', self.browser.page_source) 
```

# 面向接口测试——Python mock 库

# mock

mock 测试就是在测试过程中，对于某些不容易构建或获取的对象，用虚拟的对象来代替 以便于测试的测试方法。

## 一些基本概念

double 可以理解为置换，它是所有模拟测试对象的统称，我们也可以称它为替身。 一般来说，当你创建任意一种测试置换对象时，它将被用来替代某个指定类的对象。

stub 可以理解为测试桩，它能实现当特定的方法被调用时，返回一个指定的模拟值。 如果你的测试用例需要一个伴生对象来提供一些数据，可以使用 stub 来取代数据源， 在测试设置时可以指定返回每次一致的模拟数据。

spy 可以理解为侦查，它负��汇报情况，持续追踪什么方法被调用了，以及调用过程中传递了哪些参数。 你能用它来实现测试断言，比如一个特定的方法是否被调用或者是否使用正确的参数调用。 当你需要测试两个对象间的某些协议或者关系时会非常有用。

mock 与 spy 类似，但在使用上有些许不同。spy 追踪所有的方法调用，并在事后让你写断言， 而 mock 通常需要你事先设定期望。你告诉它你期望发生什么，然后执行测试代码并验证最后的结果与事先定义的 期望是否一致。

fake 是一个具备完整功能实现和行为的对象，行为上来说它和这个类型的真实对象上一样， 但不同于它所模拟的类，它使测试变得更加容易。一个典型的例子是使用内存中的数据库来生成一个 数据持久化对象，而不是去访问一个真正的生产环境的数据库。

实践中，这些术语常常用起来不同于它们的定义，甚至可以互换，因此不必太过于陷入这些词汇的细节。 这些定义更多的是为了在高层次上区分这些概念， 它也对考虑不同类型测试对象的行为会有帮助。

## 为什么需要 mock

对于为什么需要 mock，或者什么时候需要使用 mock，Tim Mackinnon 提出了一些建议：

+   真实对象具有不可确定的行为（产生不可预测的结果，如股票行情）

+   真实对象很难被创建

+   真实对象的某些行为很难触发（比如网络错误）

+   真实情况令程序的运行速度很慢

+   真实对象有用户界面

+   测试需要询问真实对象它是如何被调用的（比如测试可能需要验证某个回调函数是否被调用了）

+   真实对象实际上并不存在（当需要和其他开发小组，或者新的硬件系统打交道时）

## Mock

`Mock` 类是 `ClallableMinxin` 和 `NonCallableMock` 的子类，实际上 `Mock` 中并没有其它定义： `class Mock(CallableMixin, NonCallableMock):pass`。

### 参数

+   `spec`：这个参数是用来指定 Mock 实例的行为，那些方法是存在的，那些不存在。 其值可以是一个类或实例，也可以是一个字符串列表。

+   `spec_set`：比起 `spec` 更加严格。

+   `side_effect`：在 Mock 实例被调用时被调用的方法，对应 `side_effect`属性，可以用来返回动态值或者 异常。其参数和 mock 相同，如果返回值不为 `DEFAULT` 则用作 Mock 实例的返回值。

    如果 `side_effect` 是一个迭代器，则每次调用的时候返回其中下一个元素。如果迭代器中的元素是异常 则将其抛出而非返回。

+   `return_value`：mock 对象被调用时的返回值，默认是一个新的 `Mock` 实例。

+   `wraps`: Item for the mock object to wrap. If `wraps` is not None then calling the Mock will pass the call through to the wrapped object (returning the real result). Attribute access on the mock will return a Mock object that wraps the corresponding attribute of the wrapped object (so attempting to access an attribute that doesn't exist will raise an `AttributeError`).

    If the mock has an explicity `return_value` set then calls are not passed to the wrapped object and the `return_value` is returned instead. 如果 Mock 实例存在 `return_value` ，不会调用被封装的对象。

+   `name`：Mock 对象在 repr 时的名字，调试时会很有帮助。该参数会传递给子 mock。

Mocks can also be called with arbitrary keyword arguments. These will be used to set attributes on the mock after it is created.

### 属性

+   `call_args`

+   `call_args_list`

+   `call_count`

+   `called`

+   `return_value`

+   `side_effect`

### 方法

+   `attach_mock` Attach a mock as an attribute of this one, replacing its name and parent. Calls to the attached mock will be recorded in the `method_calls` and `mock_calls` attributes of this one.

    +   Set attributes on the mock through keyword arguments. Attributes plus return values and side effects can be set on child mocks using standard dot notation and unpacking a dictionary in the method call:

        ```
        >>> attrs = {'method.return_value': 3, 'other.side_effect': KeyError}
        >>> mock.configure_mock(**attrs) 
        ```

+   `mock_add_spec` Add a spec to a mock. `spec` can either be an object or a list of strings. Only attributes on the `spec` can be fetched as attributes from the mock. If `spec_set` is True then only attributes on the spec can be set.

+   `reset_mock`

## MogicMock

`MagicMock`是`Mock`的子类，与`Mock`的不同之处在于`MagicMock`默认已经模拟了对象的魔术方法（magic method）。推荐使用`MagicMock`。

## patch

`patch( target, new=DEFAULT, spec=None, create=False, spec_set=None, autospec=None, new_callable=None, **kwargs)`有两种主要用法：

装饰器：

```
@patch.object(SomeClass, 'attribute', sentinel.attribute)
def test():
    assert SomeClass.attribute == sentinel.attribute 
```

和上下文管理器：

```
with patch('__builtin__.open', mock):
    handle = open('filename', 'r') 
```

`target`在两种情况下都应仅在函数体内或`with`表达式中被`new`所替代。当函数执行完或退出`with`环境时，`target`将恢复。

`new`默认是一个`MagicMock`对象。如果`patch`作为装饰器使用且省略了`new`，则创建的模拟将作为额外参数传递给装饰函数。如果`patch`作为上下文管理器使用，则上下文管理器将返回创建的模拟。

`target`应为形式为`'package.module.ClassName'`的字符串。导入`target`并用`new`对象替换指定对象，因此`target`必须可以从您调用`patch`的环境中导入。目标在执行装饰函数时导入，而不是在装饰时。

如果 patch 为你创建一个`MagicMock`，则`spec`和`spec_set`关键字参数将传递给`MagicMock`。

此外，您可以传递`spec=True`或`spec_set=True`，这会导致 patch 将被模拟的对象作为规范/spec_set 对象传递。

`new_callable` 允许你指定一个不同的类，或者可调用对象，用来创建`new`对象。默认情况下使用`MagicMock`。

更强大的`spec`形式是`autospec`。如果设置`autospec=True`，则将使用被替换对象的规范创建模拟。模拟的所有属性也将具有被替换对象对应属性的规范。被模拟的方法和函数将检查其参数，并在使用错误签名调用时引发`TypeError`。对于替换类的模拟，它们的返回值（'instance'）将具有与类相同的规范。

而不是`autospec=True`，您可以传递`autospec=some_object`以使用任意对象作为规范，而不是被替换的对象。

默认情况下，`patch`将无法替换不存在的属性。如果传入`create=True`，并且属性不存在，当调用修补的函数时，`patch`将为您创建属性，并在之后删除它。这对编写针对运行时创建属性的生产代码的测试非常有用。默认情况下它是关闭的，因为它可能很危险。启用它后，您可以针对实际不存在的 API 编写通过的测试！

Patch 可以作为`TestCase`类的装饰器使用。它通过装饰类中的每个测试方法来工作。当测试方法共享常见的修补设置时，这减少了样板代码。`patch`通过查找以`patch.TEST_PREFIX`开头的方法名称来查找测试。默认情况下，这是`test`，与`unittest`查找测试的方式相匹配。您可以通过设置`patch.TEST_PREFIX`来指定替代前缀。

`patch` 可以作为上下文管理器使用，使用 `with` 语句。在 `with` 语句后的缩进块中应用补丁。如果使用 "as"，那么修补后的对象将绑定到 "as" 后面的名称；如果 `patch` 为您创建了一个模拟对象，则非常有用。

`patch` 接受任意关键字参数。这些参数将在构造时传递给 `Mock`（或 `new_callable`）。

`patch.dict(...)`、`patch.multiple(...)` 和 `patch.object(...)` 可用于替代用例。

### 猴子补丁（monkey patch）

`patch` 本质上是一个函数，但是在实现的时候通过 MP 添加了很多属性：

```
patch.object = _patch_object
patch.dict = _patch_dict
patch.multiple = _patch_multiple
patch.stopall = _patch_stopall
patch.TEST_PREFIX = 'test' 
```

## 例子

[IPython notebook 在线示例](https://github.com/kxxoling/Python-One-to-Million/blob/ipynb/testing/mock.ipynb)

## 参考文章

+   [使用模拟对象（Mock Object）技术进行测试驱动开发](https://www.ibm.com/developerworks/cn/java/j-lo-mockobject/)

+   [置换测试: Mock, Stub 和其他](http://objccn.io/issue-15-5/)

+   [PyPI - mock](https://pypi.python.org/pypi/mock)

+   [mock - getting started](http://www.voidspace.org.uk/python/mock/getting-started.html)

+   [Mocks/Doubles/Fake/Dummy/Stub 的术语](http://martinfowler.com/articles/mocksArentStubs.html)

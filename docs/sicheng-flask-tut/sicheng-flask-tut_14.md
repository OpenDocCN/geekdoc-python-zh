# Flask 进阶系列(九)–测试

测试是每个应用系统发布前必须经历的步骤，自动化测试对测试效率的提高也是毋庸置疑的。对于 Flask 应用来说，当然可以使用 Web 自动化测试工具，比如 Selenium 等来测。Flask 官方推荐的自动化测试方法是一种白盒测试，它依赖于 Werkzeug 的 Client 对象来模拟客户端。使用这个方法的好处是你不需要真的运行一个应用实例，也不依赖于任何浏览器。而测试框架就使用 Python 中的 unittest 包，对于大家上手也方便。

### 系列文章

*   Flask 进阶系列(一)–上下文环境
*   Flask 进阶系列(二)–信号
*   Flask 进阶系列(三)–Jinja2 模板引擎
*   Flask 进阶系列(四)–视图
*   Flask 进阶系列(五)–文件和流
*   Flask 进阶系列(六)–蓝图(Blueprint)
*   Flask 进阶系列(七)–应用最佳实践
*   Flask 进阶系列(八)–部署和分发
*   Flask 进阶系列(九)–测试

本篇的范例中，我们将针对入门系列第六篇的基于数据库的用户登录登出应用做测试，要运行范例代码前，记得先取回应用代码。

### Set Up 和 Tear Down 方法

熟悉自动化测试的朋友们知道，几乎每个测试框架都有 Set Up 和 Tear Down 方法。Set Up 方法会在每个测试用例执行前被调用，一般用来初始化测试用例的运行环境，而 Tear Down 方法会在每个测试用例执行完后被调用，一般用来销毁该测试用例的运行环境。这样做，就可以保证测试用例之间互相不影响。让我们先创建一个测试代码文件，并写入测试类，及 Set Up 和 Tear Down 方法：

```py
import os
import unittest
import tempfile
import sqlite3
from contextlib import closing
from flask6 import app

class SampleTestCase(unittest.TestCase):

    def setUp(self):
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        self.init_db(app.config['DATABASE'])

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])

    def init_db(self, db_file):
        with closing(sqlite3.connect(db_file)) as db:
            with app.open_resource('init.sql', mode='r') as f:
                db.cursor().executescript(f.read())
            db.commit()

if __name__ == '__main__':
    unittest.main()

```

上述代码中，”flask6″就是我们要测试的应用。测试类”SampleTestCase”继承了”unittest.TestCase”类。我们在其”setUp()”方法中，创建了一个临时的 sqlite3 数据库文件，并将其初始化，同时我们将应用配置项”TESTING”设置为 True，表示使用测试模式。在测试类的”tearDown()”方法中，我们销毁了之前创建的临时数据库文件，因此它不会影响下一个测试用例。执行测试的方法就是”unittest.main()”。

现在我们的测试框架已经建立好了，接下来，让我们开始写个测试用例吧。

### 创建测试用例

下面是一个测试用户成功登录的测试用例，”setUp()”和”tearDown()”方法我们就略去了：

```py
from flask import session

class SampleTestCase(unittest.TestCase):
    # setUp() and tearDown() here...

    # Test case
    def test_valid_login(self):
        with app.test_client() as client:
            response = client.post('/login', data=dict(
                user='admin',
                passwd='123'
            ), follow_redirects=True)
            assert 'Login successfully' in response.data
            assert session['user'] == 'admin'

```

测试用例函数都是以”test”开头，这样 unittest 包会自动识别其为一个测试用例。在测试用例函数中，我们先使用”app.test_client()”来获取一个”werkzeug.test.Client”类型的对象来模拟客户端。此后我们就可以通过”client.get(url)”或”client.post(url)”来模拟发送 GET 或 POST 请求了。”get()”或”post()”方法的”data”参数可以传入请求所需要的参数，它是一个字典；”follow_redirects”参数为 True 时，请求函数（即视图函数）内的”redirect()”重定向才有效。因为我们的 login 成功后会 redirect 到 index 页面，所以这个参数必须设为 True。请求返回 Response 对象，你可以使用 response.data 获取响应体的内容。

大家运行下这个测试，假设测试的代码写在了”sample_test.py”，我们可以执行 python 命令：

```py
$ python sample_test.py
```

然后应该可以在命令行看到如下结果：

```py
.
----------------------------------------------------------------------
Ran 1 test in 0.035s

OK

```

整个函数都在”with app.test_client() as client”语句体内。朋友们可能会问，为什么不把这个 client 的初始化写在”setUp()”方法里，这样省去每个测试函数都要写一遍的麻烦。其实这个 with 语句有一个作用，就是语句块内可以访问请求上下文，所以上例中可以获取 session 对象的内容。离开 with 语句，request 和 session 对象都无法获取，朋友们可以试试。

### 修改会话 session

上面的测试用例中，我们成功将 admin 用户登录，所以 session 中的 user 值为 admin。如果这时我们想修改这个值，怎么做？笨办法是先登出，再换个用户登录。Flask 其实提供了方法让我们修改当前测试用例中的 session 值。我们在上面的测试用例函数中，加上下一段代码：

```py
            # Modify session
            with client.session_transaction() as sess:
                sess['user'] = 'guest'

            # Request home page
            response = client.get('/', follow_redirects=True)
            assert 'Hello guest' in response.data
            assert session['user'] == 'guest'

```

我们通过”client.session_transaction()”方法来获取可以被修改的 session，然后将其”user”字段改为”guest”。此后的请求中 session 中的”user”值就变为了”guest”。

### 构建请求上下文

更进一步，有时候我们不想模拟客户端访问一个已有的 URL 来创建请求。我们想创建一个虚拟的请求，并且建立虚拟的请求上下文环境，看看下面的例子：

```py
from flask import make_response, render_template, request

class SampleTestCase(unittest.TestCase):
    # setUp() and tearDown() here...

    # Test case
    def test_home_with_context(self):
        with app.test_request_context('/?user=admin'):
            assert request.path == '/'
            assert request.args['user'] == 'admin'
            response = make_response(render_template('hello.html',
                                     name=request.args['user']))
            assert 'Hello admin' in response.data

```

这里，我们使用”with app.test_request_context()”语句构建了一个虚拟的”/?user=admin”请求上下文环境，因此我们可以在其中访问到请求对象 request。在使用”with app.test_request_context()”时，离开 with 语句会调用上下文 Hook 函数”teardown_request()”，但是”before_request()”和”after_request()”的 Hook 函数都不会被调用。你必须使用”app.preprocess_request()”和”app.process_response()”来显式地调用它们，比如基于上面的例子，我们可以这样调用”before_request()”和”after_request()”的上下文 Hook 函数：

```py
    def test_home_with_context(self):
        with app.test_request_context('/?user=admin'):
            # All before_request hooks will be called here
            app.preprocess_request()
            assert request.path == '/'
            assert request.args['user'] == 'admin'
            response = make_response(render_template('hello.html',
                                     name=request.args['user']))
            assert 'Hello admin' in response.data
            # All after_request hooks will be called here
            response = app.process_response(response)

```

关于请求上下文 Hook 函数的内容可以参阅本系列第一篇。

### 设置应用上下文

上面我们介绍了如何自己构建一个请求上下文，如果我们想往应用上下文添加或修改内容呢？方法是定义你要修改应用上下文的函数，并将它作为订阅”appcontext_pushed”信号的回调函数。这样，函数会在应用上下文压入栈时被执行。

```py
from contextlib import contextmanager
from flask import appcontext_pushed, g

@contextmanager
def name_set(app, name):
    def handler(sender, **kwargs):
        g.app_name = name
    with appcontext_pushed.connected_to(handler, app):
        yield

```

“@contextmanager”装饰器表明可以针对”name_set()”函数使用 with 语句来限制其上下文作用域，即离开了”with name_set()”语句块后，”appcontext_pushed”信号的订阅就无效了。现在我们可以在测试用例中这样使用这个”name_set()”函数：

```py
from contextlib import contextmanager
from flask import appcontext_pushed, g

class SampleTestCase(unittest.TestCase):
    # setUp() and tearDown() here...

    # Test case
    def test_update_app_context(self):
         with name_set(app, 'Sample'):
            with app.test_client() as client:
                response = client.get('/app')
                assert 'Sample' in response.data

```

上例中，在请求”/app”的视图函数里，”g.app_name”的值即被设为”Sample”.

关于应用上下文的详细信息可参阅本系列第一篇，关于信号的内容可参阅本系列第二篇。

本篇中的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad9.html)
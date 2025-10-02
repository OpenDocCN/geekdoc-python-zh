# Flask 扩展系列–自定义扩展

介绍了那么多 Flask 扩展，该讲下如何写自己的扩展了。你可以写个扩展给自己的项目用，也可以发起审核申请，审核通过的扩展会显示在[官方扩展列表](http://flask.pocoo.org/extensions/)中。本篇中，让我们创建一个为视图访问加日志的扩展 Flask-Logging，并从中了解到写 Flask 扩展的规范。

### 系列文章

*   Flask 扩展系列(一)–Restful
*   Flask 扩展系列(二)–Mail
*   Flask 扩展系列(三)–国际化 I18N 和本地化 L10N
*   Flask 扩展系列(四)–SQLAlchemy
*   Flask 扩展系列(五)–MongoDB
*   Flask 扩展系列(六)–缓存
*   Flask 扩展系列(七)–表单
*   Flask 扩展系列(八)–用户会话管理
*   Flask 扩展系列(九)–HTTP 认证
*   Flask 扩展系列–自定义扩展

### 创建工程

让我们先创建一个工程，目录结构如下：

```py
flask-logging/
  ├ LICENSE           # 授权说明
  ├ README            # 项目介绍
  ├ setup.py          # 打包分发文件
  └ flask_logging/    # 扩展代码包
      └ __init__.py   # 扩展代码

```

根据 Flask 扩展命名规范，扩展名必须为”Flask-Logging”形式，以”Flask-“为前缀，后面的单词首字母大写。扩展的代码必须放在名为”flask_logging”的包下，注意这里是下划线，与扩展名中的横线不同，单词都小写。”LICENSE”和”README”文件都是审核必须的，关于审核部分，我们会在后面介绍。

### 编写分发文件

接下来，我们写”setup.py”文件，示例如下：

```py
"""
Flask-Logging
-------------

Log every request to specific view
"""
from setuptools import setup

setup(
    name='Flask-Logging',
    version='1.0',
    url='http://example.com/flask-logging/',
    license='BSD',
    author='Billy J. Hee',
    author_email='billy@bjhee.com',
    description='Log every request to specific view',
    long_description=__doc__,
    packages=['flask_logging'],
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    install_requires=[
        'Flask'
    ],
    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)

```

这里需要注意几点：

1.  扩展名的格式必须为”Flask-Logging”，上节介绍过
2.  必须指定 url 链接到扩展主页或文档
3.  “zip_safe”必须为 False
4.  “install_requires”必须列出所有依赖的库

### 编写扩展代码

进入主题了，由于我们的扩展相当简单，因此所有代码都放在了”__init__.py”中：

```py
#coding:utf8
from flask import current_app, request
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
import logging
import time

# 指定日志文件名，日志级别，及日志记录格式
entry_log = TimedRotatingFileHandler('entry.log','D')
entry_log.setLevel(logging.DEBUG)
entry_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

class Logging:
    # 构造函数
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    # 初始化应用
    def init_app(self, app):
        app.logger.addHandler(entry_log)

    # 视图装饰器，被装饰的视图将自动记录访问日志
    def log_entry(self, func):
        app = self.app or current_app
        @wraps(func)
        def decorator(*args, **kwargs):
            start = time.time()
            # 记录请求开始
            app.logger.debug('Start request call: %s' % request.url)
            ret = func(*args, **kwargs)
            # 记录请求结束
            app.logger.debug('Finish request call: %s' % request.url)
            duration = time.time() - start
            # 记录请求所耗时长
            app.logger.debug('Request: %s consumed %f s' % (request.url, duration))
            return ret

        return decorator

```

代码逻辑都在写注释里了，这个扩展提供了”log_entry”视图装饰器，来记录视图访问日志。这里同样要注意几个重要的部分：

1.  构造函数”__init__()”和初始化函数”init_app()”是必须的
2.  如果构造函数传入了 app，则调用”init_app()”，这样确保两者功能一致
3.  构造函数里我们设置了”self.app=app”，而”init_app()”没有，这是为什么呢？这是一种规范，或者说习惯。当系统只有一个 app 时，建议使用构造函数初始化扩展对象，这时对象中的 app 就指向这一个应用。而当系统有多个应用同时存在，比如说应用工厂或测试场景下，建议使用”init_app()”来初始化扩展对象，这样扩展对象不会指向任何应用
4.  在视图装饰器里，我们使用了”app = self.app or current_app”来获取当前应用，这分别对应于上一点说的单个应用及多个应用场景
5.  因为视图装饰器是视图访问时被调用，所以此时应用上下文和请求上下文都存在，因此我们可以访问到”current_app”和”request”对象。离开上下文的话，就无效了

扩展写完了，让我们来测试一下，创建一个 Flask 应用：

```py
from flask import Flask
from flask_logging import Logging

app = Flask(__name__)
logging = Logging(app)

@app.route('/')
@logging.log_entry
def index():
    return '<h1>Hello World</h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

```

启动应用，访问”http://localhost:5000/”。查看下代码当前路径，是不是出现了”entry.log”文件，并且记录了 URL 请求日志？

### 关于审核

个人没有发起过审核申请，对于具体流程尚不清楚。不过如果你要将自己的扩展提交官方审核，至少要做到下面几点：

1.  扩展代码在包”flask_myext”下，审核通过后，Flask 会设置一个重定向包”flask.ext.myext”来指向你的包。对于用户来说，官方扩展建议导入”flask.ext.myext”格式的包
2.  必须提供一个”setup.py”分发文件，并在 PyPI 上注册，这样用户就可以通过”pip install”来安装你的扩展
3.  必须提供”LICENSE”文件，并且授权是 BSD, MIT 或 WTFPL
4.  必须提供”README”文件及文档，文档是由[Sphinx](http://www.sphinx-doc.org/en/stable/)生成
5.  必须同时提交单元测试代码
6.  必须支持 Python 2.6 和 2.7 版本

至于其他要求，大家可以参考[官方文档](http://flask.pocoo.org/docs/0.10/extensiondev/)，或者提交一个扩展尝试下。

#### 更多参考资料

[Flask 扩展开发的官方文档](http://flask.pocoo.org/docs/0.10/extensiondev/)
[官方扩展列表](http://flask.pocoo.org/extensions/)中已有扩展的源码

本篇的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ext.html)
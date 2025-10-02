# Flask 进阶系列(八)–部署和分发

到目前为止，我们启动 Flask 应用都是通过”app.run()”方法，在开发环境中，这样固然可行，不过到了生产环境上，势必需要采用一个健壮的，功能强大的 Web 应用服务器来处理各种复杂情形。同时，由于开发过程中，应用变化频繁，手动将每次改动部署到生产环境上很是繁琐，最好有一个自动化的工具来简化持续集成的工作。本篇，我们就会介绍如何将上一篇中 Flask 的应用程序自动打包，分发，并部署到像 Apache, Nginx 等服务器中去。

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

### 使用 setuptools 打包 Flask 应用

首先，你要了解基本的使用 setuptools 打包分发 Python 应用程序的方法。接下来，就让我们开始写一个”setup.py”文件：

```py
from setuptools import setup

setup(
    name='MyApp',
    version='1.0',
    long_description=__doc__,
    packages=['myapp','myapp.main','myapp.admin'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'Flask>=0.10',
        'Flask-Mail>=0.9',
        'Flask-SQLAlchemy>=2.1'
    ]
)

```

把文件放在项目的根目录下。另外，别忘了，还要写一个”MANIFEST.in”文件：

```py
recursive-include myapp/templates *
recursive-include myapp/static *

```

编写完毕后，你可以创建一个干净的虚拟环境，然后运行安装命令试下效果。

```py
$ python setup.py install
```

### 使用 Fabric 远程部署 Flask 应用

同样，你需要先了解如何使用 Fabric 来远程部署 Python 应用。然后，我们来编写”fabfile.py”文件：

```py
from fabric.api import *

env.hosts = ['example1.com', 'example2.com']
env.user = 'bjhee'

def package():
    local('python setup.py sdist --formats=gztar', capture=False)

def deploy():
    dist = local('python setup.py --fullname', capture=True).strip()
    put('dist/%s.tar.gz' % dist, '/tmp/myapp.tar.gz')
    run('mkdir /tmp/myapp')
    with cd('/tmp/myapp'):
        run('tar xzf /tmp/myapp.tar.gz')
        run('/home/bjhee/virtualenv/bin/python setup.py install')
    run('rm -rf /tmp/myapp /tmp/myapp.tar.gz')
    run('touch /var/www/myapp.wsgi')

```

上例中，”package”任务是用来将应用程序打包，而”deploy”任务是用来将 Python 包安装到远程服务器的虚拟环境中，这里假设虚拟环境在”/home/bjhee/virtualenv”下。安装完后，我们将”/var/www/myapp.wsgi”文件的修改时间更新，以通知 WSGI 服务器（如 Apache）重新加载它。对于非 WSGI 服务器，比如 uWSGI，这条语句可以省去。

编写完后，运行部署脚本测试下：

```py
$ fab package deploy
```

### 使用 Apache+mod_wsgi 运行 Flask 应用

Flask 应用是基于 WSGI 规范的，所以它可以运行在任何一个支持 WSGI 协议的 Web 应用服务器中，最常用的就是 Apache+mod_wsgi 的方式。上面的 Fabric 脚本已经完成了将 Flask 应用部署到远程服务器上，接下来要做的就是编写 WSGI 的入口文件”myapp.wsgi”，我们假设将其放在 Apache 的文档根目录在”/var/www”下。

```py
activate_this = '/home/bjhee/virtualenv/bin/activate_this.py'
execfile(activate_this, dict(__file__=activate_this))

import os
os.environ['PYTHON_EGG_CACHE'] = '/home/bjhee/.python-eggs'

import sys;
sys.path.append("/var/www")

from myapp import create_app
import config
application = create_app('config')

```

注意上，你需要预先创建配置文件”config.py”，并将其放在远程服务器的 Python 模块导入路径中。上例中，我们将”/var/www”加入到了 Python 的模块导入路径，因此可以将”config.py”放在其中。另外，记得用 setuptools 打包时不能包括”config.py”，以免在部署过程中将开发环境中的配置覆盖了生产环境。

在 Apache 的”httpd.conf”中加上脚本更新自动重载和 URL 路径映射：

```py
WSGIScriptReloading On
WSGIScriptAlias /myapp /var/www/myapp.wsgi

```

重启 Apache 服务器后，就可以通过”http://example1.com/myapp”来访问应用了。

### 使用 Nginx+uWSGI 运行 Flask 应用

你要先准备好 Nginx+uWSGI 的运行环境，然后编写 uWSGI 的启动文件”myapp.ini”：

```py
[uwsgi]
socket=127.0.0.1:3031
callable=app
mount=/myapp=run.py
manage-script-name=true
master=true
processes=4
threads=2
stats=127.0.0.1:9191
virtualenv=/home/bjhee/virtualenv

```

再修改 Nginx 的配置文件，Linux 上默认是”/etc/nginx/sites-enabled/default”，加上目录配置：

```py
location /myapp {
    include uwsgi_params;
    uwsgi_param SCRIPT_NAME /myapp;
    uwsgi_pass 127.0.0.1:3031;
}

```

重启 Nginx 和 uWSGI 后，就可以通过”http://example1.com/myapp”来访问应用了。

你也可以将我们的应用配置为虚拟服务器，只需要将上述 uWSGI 的配置移到虚拟服务器的配置文件中即可。关于 Nginx 虚拟服务器的配置，可以参考我之前的文章。

### 使用 Tornado 运行 Flask 应用

Tornado 的强大之处在于它是非阻塞式异步 IO 及 Epoll 模型，采用 Tornado 的可以支持数以万计的并发连接，对于高并发的应用有着很好的性能。本文不会展开 Tornado 的介绍，感兴趣的朋友们可以参阅其[官方文档](http://www.tornadoweb.org/en/stable/)。使用 Tornado 来运行 Flask 应用很简单，只要编写下面的运行程序，并执行它即可：

```py
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from myapp import create_app
import config

app = create_app('config')

http_server = HTTPServer(WSGIContainer(app))
http_server.listen(5000)
IOLoop.instance().start()

```

之后你就可以通过”http://example1.com:5000″来访问应用了。

### 使用 Gunicorn 运行 Flask 应用

Gunicorn 是一个 Python 的 WSGI Web 应用服务器，是从 Ruby 的 Unicorn 移植过来的。它基于”pre-fork worker”模型，即预先开启大量的进程，等待并处理收到的请求，每个单独的进程可以同时处理各自的请求，又避免进程启动及销毁的开销。不过 Gunicorn 是基于阻塞式 IO，并发性能无法同 Tornado 比。更多内容可以参阅其[官方网站](http://gunicorn.org/)。另外，Gunicorn 同 uWSGI 一样，一般都是配合着 Nginx 等 Web 服务器一同使用。

让我们先将应用安装到远程服务器上，然后采用 Gunicorn 启动应用，使用下面的命令即可：

```py
$ gunicorn run:app
```

解释下，因为我们的应用使用了工厂方法，所以只在 run.py 文件中创建了应用对象 app，gunicorn 命令的参数必须是应用对象，所以这里是”run:app”。现在你就可以通过”http://example1.com:8000″来访问应用了。默认监听端口是 8000。

假设我们想预先开启 4 个工作进程，并监听本地的 5000 端口，我们可以将启动命令改为：

```py
$ gunicorn -w 4 -b 127.0.0.1:5000 run:app
```

本篇中的示例代码可以在这里下载。

转载请注明出处: [思诚之道](http://www.bjhee.com/flask-ad8.html)
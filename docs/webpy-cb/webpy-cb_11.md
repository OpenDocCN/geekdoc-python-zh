# Deployment 部署

# 通过 Fastcgi 和 lighttpd 部署

如果你对这个主题有任何问题，可以点击下面的链接访问相应的话题:

[`www.mail-archive.com/webpy@googlegroups.com/msg02800.html`](http://www.mail-archive.com/webpy@googlegroups.com/msg02800.html)

下面的代码基于 lighttpd 1.4.18，更高版本也可以工作

## Note:

*   你可以重命名 `code.py`为任何你自己愿意的名字，该例子还是以 code.py 为例。
*   `/path-to/webpy-app` 为包含你的 `code.py`代码的路径。
*   `/path-to/webpy-app/code.py` 应该是你的**python file**的完整路径。

如果你还不确定你的 lighttpd 版本的话，你可以在命令行中使用`lighttpd -v 查看相应的版本信息。`

Note: 较早版本的 lighttpd 可能会按照不同的方式组织.conf 文件，但是它们应该遵循的是相同的原则。

### ligghttpd 在 Debian GNU/Linux 下的配置文件

```py
Files and Directories in /etc/lighttpd:
---------------------------------------

lighttpd.conf:
         main configuration file

conf-available/
        This directory contains a series of .conf files. These files contain
        configuration directives necessary to load and run webserver modules.
        If you want to create your own files they names should be
        build as nn-name.conf where "nn" is two digit number (number
        is used to find order for loading files)

conf-enabled/
        To actually enable a module for lighttpd, it is necessary to create a
        symlink in this directory to the .conf file in conf-available/.

Enabling and disabling modules could be done by provided
/usr/sbin/lighty-enable-mod and /usr/sbin/lighty-disable-mod scripts. 
```

**对于 web py， 你需要允许 mod_fastcgi 模块和 mod_rewrite 模块, 运行: `/usr/sbin/lighty-enable-mod` 启用 `fastcgi` （Mac OS X 可能不需要） (mod_rewrite 模块可能需要启用 `10-fastcgi.conf`文件).**

## 下面是文件的基本结构（Mac OS X 不同）:

*   `/etc/lighttpd/lighttpd.conf`
*   `/etc/lighttpd/conf-available/10-fastcgi.conf`
*   `code.py`

对于 Mac OS X 或任何以 Mac Ports 邓方式安装的 lighttpd，可以直接在路径下编写.conf 文件并用 lighttpd -f xxx.conf 启动 lighttpd，而无需去修改或考虑任何文件结构。

`/etc/lighttpd/lighttpd.conf`

```py
server.modules              = (
            "mod_access",
            "mod_alias",
            "mod_accesslog",
            "mod_compress",
)
server.document-root       = "/path-to/webpy-app" 
```

对我来说，我使用 postgresql，因此需要授予对的数据库权限，可以添加行如下（如果不使用则不需要）。

```py
server.username = "postgres" 
```

`/etc/lighttpd/conf-available/10-fastcgi.conf`

```py
server.modules   += ( "mod_fastcgi" )
server.modules   += ( "mod_rewrite" )

 fastcgi.server = ( "/code.py" =>
 (( "socket" => "/tmp/fastcgi.socket",
    "bin-path" => "/path-to/webpy-app/code.py",
    "max-procs" => 1,
   "bin-environment" => (
     "REAL_SCRIPT_NAME" => ""
   ),
   "check-local" => "disable"
 ))
 )

如果本地的 lighttpd 跑不起来的话，需要设置 check-local 属性为 disable。

 url.rewrite-once = (
   "^/favicon.ico$" => "/static/favicon.ico",
   "^/static/(.*)$" => "/static/$1",
   "^/(.*)$" => "/code.py/$1",
 ) 
```

`/code.py` 在代码头部添加以下代码，让系统环境使用系统环境中当前的 python

```py
#!/usr/bin/env python 
```

最后不要忘记了要对需要执行的 py 代码设置执行权限，否则你可能会遇到“permission denied”错误。

```py
$ chmod 755 /path-to/webpy-app/code.py 
```

# Webpy + Nginx with FastCGI 搭建 Web.py

这一节讲解的是如何使用 Nginx 和 FastCGI 搭建 Web.py 应用

### 环境依赖的软件包

*   Nginx 0.8. *or 0.7.* (需要包含 fastcgi 和 rewrite 模块)。
*   Webpy 0.32
*   Spawn-fcgi 1.6.2
*   Flup

注意：Flup 是最常见的忘记装的软件，需要安装

更老的版本应该也可以工作，但是没有测试过，最新的是可以工作的

### 一些资源

*   [Nginx wiki](http://wiki.nginx.org/NginxInstall)
*   [Spawn-fcgi](http://redmine.lighttpd.net/projects/spawn-fcgi/news)
*   [Flup](http://trac.saddi.com/flup)

### Notes

*   你可以重命名`index.py`为任何你想要的文件名。
*   `/path/to/www` 为代码路径。
*   `/path/to/www/index.py`为 python 代码的完整路径。

## Nginx 配置文件

```py
location / {
    include fastcgi_params;
    fastcgi_param SCRIPT_FILENAME $fastcgi_script_name;  # [1]
    fastcgi_param PATH_INFO $fastcgi_script_name;        # [2]
    fastcgi_pass 127.0.0.1:9002;
} 
```

对于静态文件可以添加如下配置:

```py
location /static/ {
    if (-f $request_filename) {
    rewrite ^/static/(.*)$  /static/$1 break;
    }
} 
```

**注意:** 地址和端口号可能会是不同的。

## Spawn-fcgi

可以通过一下命令启动一个 Spawn-fcgi 进程:

```py
spawn-fcgi -d /path/to/www -f /path/to/www/index.py -a 127.0.0.1 -p 9002 
```

### 启动和关闭的命令

启动:

```py
#!/bin/sh
spawn-fcgi -d /path/to/www -f /path/to/www/index.py -a 127.0.0.1 -p 9002 
```

关闭:

```py
#!/bin/sh
kill `pgrep -f "python /path/to/www/index.py"` 
```

**Note:** 你可以随意填写地址和端口信息，但是一定需要和 Nginx 配置文件相匹配。

## Hello world!

讲下面的代码保存为 index.py（或者任何你喜欢的），注意，使用 Nginx 配置的话，`web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)`这一行代码是必须的。

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import web

urls = ("/.*", "hello")
app = web.application(urls, globals())

class hello:
    def GET(self):
        return 'Hello, world!'

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run() 
```

注意: 同样需要给代码设置权限，代码如下 chmod +x index.py。

## 运行

1.  打开一个 `spawn-fcgi` 进程.
2.  打开 Nginx.

如果需要检查应用程序是否运行，使用`ps aux|grep index.py`可以很容易的查看。

重启 nginx 配置:

```py
/path/to/nginx/sbin/nginx -s reload 
```

停止 nginx:

```py
/path/to/nginx/sbin/nginx -s stop 
```

注意：运行后可访问[`localhost 访问网站，更多信息可以去参考 nginx 官方文档。`](http://localhost 访问网站，更多信息可以去参考 nginx 官方文档。)

# CGI deployment on Apache

Here are the simple steps needed to create and run an web.py application.

*   Install web.py and flups

*   Create the application as documented

    ```py
     if __name__ == "__main__":
          web.run(urls, globals()) 
    ```

For our example, let it be named `app.py`, located in `/www/app` and we need it accessible as `http://server/app/app.py`.

*   Configure Apache (version 2.2 in this example)

    ```py
     ScriptAlias /app "/www/app/"
      &lt;Directory "/www/app/"&gt;
              Options +ExecCGI +FollowSymLinks
              Order allow,deny
              Allow from all
      &lt;/Directory&gt; 
    ```

That's it. Your application is accessible via `http://server/app/app.py/`. Additional URLs handled by the application are added to the end of the URL, for examples `http://server/app/app.py/myurl`.

*   .htaccess configuration

    ```py
     Options +ExecCGI
            AddHandler cgi-script .py
            DirectoryIndex index.py
            &lt;IfModule mod_rewrite.c&gt;
                RewriteEngine on
                RewriteBase /
                RewriteCond %{REQUEST_FILENAME} !-f
                RewriteCond %{REQUEST_FILENAME} !-d
                RewriteCond %{REQUEST_URI} !^/favicon.ico$
                RewriteCond %{REQUEST_URI} !^(/.*)+index.py/
                RewriteRule ^(.*)$ index.py/$1 [PT]
            &lt;/IfModule&gt; 
    ```

Here it is assumed that your application is called index.py. The above htaccess checks if some static file/directory exists failing which it routes the data to your index.py. Change the Rewrite Base to a sub-directory if needed.

# 使用 Apache + mod_wsgi 部署 webpy 应用

下面的步骤在 Apache-2.2.3 (Red Hat Enterprise Linux 5.2, x86_64),mod_wsgi-2.0 中测试通过。（译者注：本人在 Windows2003 + Apache-2.2.15 + mod_wsgi-3.0 也测试通过）

注意：

*   您可以使用您自己的项目名称替换'appname'。
*   您可以使用您自己的文件名称替换'code.py'。
*   /var/www/webpy-app 为包含您的 code.py 的文件夹目录路径。
*   /var/www/webpy-app/code.py 是您的 python 文件的完整路径。

步骤：

*   下载和安装 mod_wsgi 从它的网站：

[`code.google.com/p/modwsgi/`](http://code.google.com/p/modwsgi/). 它将安装一个'.so'的模块到您的 apache 模块文件夹，例如：

```py
 /usr/lib64/httpd/modules/ 
```

*   在 httpd.conf 中配置 Apache 加载 mod_wsgi 模块和您的项目：

    ```py
     LoadModule wsgi_module modules/mod_wsgi.so

      WSGIScriptAlias /appname /var/www/webpy-app/code.py/

      Alias /appname/static /var/www/webpy-app/static/
      AddType text/html .py

      &lt;Directory /var/www/webpy-app/&gt;
          Order deny,allow
          Allow from all
      &lt;/Directory&gt; 
    ```

*   演示文件 'code.py':

    ```py
     import web

      urls = (
          '/.*', 'hello',
          )

      class hello:
          def GET(self):
              return "Hello, world."

      application = web.application(urls, globals()).wsgifunc() 
    ```

*   在您的浏览器地址栏中输入' [`your_server_name/appname`](http://your_server_name/appname)' 来验证它是否可用。

## 注意: mod_wsgi + sessions

如果您需要在 mod_wsgi 中使用 sessions，您可以改变您的代码如下：

```py
app = web.application(urls, globals())

curdir = os.path.dirname(__file__)
session = web.session.Session(app, web.session.DiskStore(curdir + '/' + 'sessions'),)

application = app.wsgifunc() 
```

## mod_wsgi 性能:

有关 mod_wsgi 的性能，请参考 mod_wsgi 的维基页： [`code.google.com/p/modwsgi/wiki/PerformanceEstimates`](http://code.google.com/p/modwsgi/wiki/PerformanceEstimates)

# deploying web.py with nginx and mod_wsgi

It is possible to deploy web.py with nginx using a mod_wsgi similar to the module for Apache.

After compiling and installing nginx with mod_wsgi, you can easily get a web.py app up and running with the following config* (edit the paths and settings with your own):

```py
wsgi_python_executable  /usr/bin/python;

server {
    listen 80;
    server_name www.domain_name.com domain_name.com;
    root /path/to/your/webpy;

    include /etc/nginx/wsgi_vars;
    location / {
        wsgi_pass /path/to/your/webpy/app.py;     
     }
} 
```

*Note: This is a snippet of the relevant information to setup mod_wsgi for your web app and NOT a full config for running nginx.

Helpful links: [nginx website](http://nginx.net/) [wiki page on mod_wsgi](http://wiki.codemongers.com/NginxNgxWSGIModule)

# Webpy + Nginx with FastCGI 搭建 Web.py

这一节讲解的是如何使用 Nginx 和 FastCGI 搭建 Web.py 应用

### 环境依赖的软件包

*   Nginx 0.8. *or 0.7.* (需要包含 fastcgi 和 rewrite 模块)。
*   Webpy 0.32
*   Spawn-fcgi 1.6.2
*   Flup

注意：Flup 是最常见的忘记装的软件，需要安装

更老的版本应该也可以工作，但是没有测试过，最新的是可以工作的

### 一些资源

*   [Nginx wiki](http://wiki.nginx.org/NginxInstall)
*   [Spawn-fcgi](http://redmine.lighttpd.net/projects/spawn-fcgi/news)
*   [Flup](http://trac.saddi.com/flup)

### Notes

*   你可以重命名`index.py`为任何你想要的文件名。
*   `/path/to/www` 为代码路径。
*   `/path/to/www/index.py`为 python 代码的完整路径。

## Nginx 配置文件

```py
location / {
    include fastcgi_params;
    fastcgi_param SCRIPT_FILENAME $fastcgi_script_name;  # [1]
    fastcgi_param PATH_INFO $fastcgi_script_name;        # [2]
    fastcgi_pass 127.0.0.1:9002;
} 
```

对于静态文件可以添加如下配置:

```py
location /static/ {
    if (-f $request_filename) {
    rewrite ^/static/(.*)$  /static/$1 break;
    }
} 
```

**注意:** 地址和端口号可能会是不同的。

## Spawn-fcgi

可以通过一下命令启动一个 Spawn-fcgi 进程:

```py
spawn-fcgi -d /path/to/www -f /path/to/www/index.py -a 127.0.0.1 -p 9002 
```

### 启动和关闭的命令

启动:

```py
#!/bin/sh
spawn-fcgi -d /path/to/www -f /path/to/www/index.py -a 127.0.0.1 -p 9002 
```

关闭:

```py
#!/bin/sh
kill `pgrep -f "python /path/to/www/index.py"` 
```

**Note:** 你可以随意填写地址和端口信息，但是一定需要和 Nginx 配置文件相匹配。

## Hello world!

讲下面的代码保存为 index.py（或者任何你喜欢的），注意，使用 Nginx 配置的话，`web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)`这一行代码是必须的。

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import web

urls = ("/.*", "hello")
app = web.application(urls, globals())

class hello:
    def GET(self):
        return 'Hello, world!'

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run() 
```

注意: 同样需要给代码设置权限，代码如下 chmod +x index.py。

## 运行

1.  打开一个 `spawn-fcgi` 进程.
2.  打开 Nginx.

如果需要检查应用程序是否运行，使用`ps aux|grep index.py`可以很容易的查看。

重启 nginx 配置:

```py
/path/to/nginx/sbin/nginx -s reload 
```

停止 nginx:

```py
/path/to/nginx/sbin/nginx -s stop 
```

注意：运行后可访问[`localhost 访问网站，更多信息可以去参考 nginx 官方文档。`](http://localhost 访问网站，更多信息可以去参考 nginx 官方文档。)
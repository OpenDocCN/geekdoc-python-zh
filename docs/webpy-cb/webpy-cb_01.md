# 安装

## Summary

*   安装
*   开发
*   产品
    *   LightTPD
        *   .. 使用 FastCGI
    *   Apache
        *   .. 使用 CGI
        *   .. 使用 CGI using .htaccess
        *   .. 使用 FastCGI
        *   .. 使用 SCGI
        *   .. 使用 mod_python
        *   .. 使用 mod_wsgi
        *   .. 使用 mod_rewrite

## 安装

安装 web.py, 请先下载：

```py
http://webpy.org/static/web.py-0.37.tar.gz 
```

或者获取最新的开发版：

```py
https://github.com/webpy/webpy/tarball/master 
```

解压并拷贝 *web* 文件夹到你的应用程序目录下。 或者，为了让所有的应用程序都可以使用，运行：

```py
python setup.py install 
```

注意: 在某些类 unix 系统上你可能需要切换到 root 用户或者运行：

```py
sudo python setup.py install 
```

查看 推荐设置.

另外一个选择是使用[Easy Install](http://peak.telecommunity.com/DevCenter/EasyInstall). Easy Install 使用如下：

```py
easy_install web.py 
```

或者 [PIP](http://packages.python.org/distribute/)

```py
sudo pip install web.py 
```

## 开发

web.py 内置了 web 服务器。可以按照 [tutorial](http://webpy.org/tutorial2) 学习如何写一个 Web 应用。 写完后，将你的代码放到 `code.py` 并如下面的方法来启动服务器：

```py
 python code.py 
```

打开你的浏览器输入 [`localhost:8080/`](http://localhost:8080/) 查看页面。 若要制定另外的端口，使用 `python code.py 1234`。

## 产品

现在所运行 web.py 程序的 web 服务器是挺不错的， 但绝大多数网站还是需要更加专业一些的 web 服务器。web.py 实现了 [WSGI](http://www.python.org/dev/peps/pep-0333/) 并能在任何兼容它的服务器上运行。 WSGI 是一个 web 服务器与应用程序之间的通用 API, 就如 Java 的 Servlet 接口。 你需要安装 [flup](http://trac.saddi.com/flup) ([download here](http://www.saddi.com/software/flup/dist/)) 使 web.py 支持 with CGI， FastCGI 或 SCGI， flup 提供了这些 API 的 WSGI 接口。

对于所有的 CGI 变量， 添加以下到你的 `code.py`:

```py
#!/usr/bin/env python 
```

并运行 `chmod +x code.py` 添加可执行属性。

### LightTPD

#### .. 使用 FastCGI

在产品中通过 FastCGI 结合 lighttpd 是 web.py 使用的一种推荐方法。 [reddit.com](http://reddit.com/) 通过该方法来处理百万次的点击。

lighttpd config 设置参考如下：

```py
 server.modules = ("mod_fastcgi", "mod_rewrite")
 server.document-root = "/path/to/root/"     
 fastcgi.server = ( "/code.py" =>     
 (( "socket" => "/tmp/fastcgi.socket",
    "bin-path" => "/path/to/root/code.py",
    "max-procs" => 1
 ))
 )

 url.rewrite-once = (
   "^/favicon.ico$" => "/static/favicon.ico",
   "^/static/(.*)$" => "/static/$1",
   "^/(.*)$" => "/code.py/$1"
 ) 
```

在某些版本的 lighttpd 中， 需要保证 fastcgi.server 选项下的"check-local"属性设置为"false", 特别是当你的 `code.py` 不在文档根目录下。

如果你得到错误显示不能够导入 flup， 请在命令行下输入 "easy_install flup" 来安装。

从修订版本 145 开始， 如果你的代码使用了重定向，还需要在 fastcgi 选项下设置 bin-environment 变量。 如果你的代码重定向到[`domain.com/`](http://domain.com/) 而在 url 栏中你会看到 [`domain.com/code.py/，`](http://domain.com/code.py/，) 你可以通过设置这个环境变量来阻止。 这样你的 fastcgi.server 设置将会如下：

```py
fastcgi.server = ( "/code.py" =>
((
   "socket" => "/tmp/fastcgi.socket",
   "bin-path" => "/path/to/root/code.py",
   "max-procs" => 1,
   "bin-environment" => (
     "REAL_SCRIPT_NAME" => ""
   ),
   "check-local" => "disable"
))
) 
```

### Apache

#### .. 使用 CGI

添加以下到 `httpd.conf` 或 `apache2.conf`。

```py
Alias /foo/static/ /path/to/static
ScriptAlias /foo/ /path/to/code.py 
```

#### .. 使用 CGI .htaccess

CGI 很容易配置， 但不适合高性能网站。 添加以下到你的 `.htaccess`：

```py
Options +ExecCGI
AddHandler cgi-script .py 
```

将你的浏览器指向 `http://example.com/code.py/`。 不要忘记最后的斜杠，否则你将会看到 `not found` 消息 (因为在 `urls` 列表中你输入的没有被匹配到). 为了让其运行的时候不需要添加 `code.py`， 启用 mod_rewrite 法则 (查看如下)。

注意: `web.py` 的实现破坏了 `cgitb` 模块，因为它截取了 `stdout`。 可以通过以下的方法来解决该问题：

```py
import cgitb; cgitb.enable()
import sys

# ... import web etc here...

def cgidebugerror():
    """                                                                         
    """        _wrappedstdout = sys.stdout

    sys.stdout = web._oldstdout
    cgitb.handler()

    sys.stdout = _wrappedstdout

web.internalerror = cgidebugerror 
```

#### .. 使用 FastCGI

FastCGI 很容易配置，运行方式如同 mod_python。

添加以下到 `.htaccess`：

```py
<Files code.py>      SetHandler fastcgi-script
</Files> 
```

不幸的是, 不像 lighttpd, Apache 不能够暗示你的 web.py 脚本以 FastCGI 服务器的形式工作，因此你需要明确的告诉 web.py。 添加以下到 `code.py`的 `if __name__ == "__main__":` 行前：

```py
web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr) 
```

将你的浏览器指向 `http://example.com/code.py/`。 不要忘记最后的斜杠，否则你将会看到 `not found` 消息 (因为在 `urls` 列表中你输入的没有被匹配到). 为了让其运行的时候不需要添加 `code.py`， 启用 mod_rewrite 法则 (查看如下)。

[Walter 还有一些额外建议](http://lemurware.blogspot.com/2006/05/webpy-apache-configuration-and-you.html).

#### .. 使用 SCGI

[`www.mems-exchange.org/software/scgi/`](https://www.mems-exchange.org/software/scgi/) 从 [`www.mems-exchange.org/software/files/mod_scgi/`](http://www.mems-exchange.org/software/files/mod_scgi/) 下载 `mod_scgi` 代码 windows apache 用户： 编辑 httpd.conf：

```py
LoadModule scgi_module Modules/mod_scgi.so
SCGIMount / 127.0.0.1:8080 
```

重启 apache，并在命令行中如下方式启动 code.py：

```py
python code.py 127.0.0.1:8080 scgi 
```

打开你的浏览器，访问 127.0.0.1 It's ok!

#### .. 使用 mod_python

mod_python 运行方式如同 FastCGI， 但不是那么方便配置。

对于 Python 2.5 按照如下：

```py
cd /usr/lib/python2.5/wsgiref
# or in windows: cd /python2.5/lib/wsgiref
wget -O modpython_gateway.py http://projects.amor.org/misc/browser/modpython_gateway.py?format=raw
# or fetch the file from that address using your browser 
```

对于 Python <2.5 按照如下：

```py
cd /usr/lib/python2.4/site-packages
# or in windows: cd /python2.4/lib/site-packages
svn co svn://svn.eby-sarna.com/svnroot/wsgiref/wsgiref
cd wsgiref
wget -O modpython_gateway.py http://projects.amor.org/misc/browser/modpython_gateway.py?format=raw
# or fetch the file from that address using your browser 
```

重命名 `code.py` 为 `codep.py` 或别的名字， 添加：

```py
app = web.application(urls, globals())
main = app.wsgifunc() 
```

在 `.htaccess` 中， 添加：

```py
AddHandler python-program .py
PythonHandler wsgiref.modpython_gateway::handler
PythonOption wsgi.application codep::main 
```

你应该希望添加 `RewriteRule` 将 `/` 指向 `/codep.py/`

确保访问 `/codep.py/` 的时候有添加最后的 `/`。 否则，你将会看到一条错误信息，比如 `A server error occurred. Please contact the administrator.`

#### .. 使用 mod_wsgi

mod_wsgi 是一个新的 Apache 模块 [通常优于 mod_python](http://code.google.com/p/modwsgi/wiki/PerformanceEstimates) 用于架设 WSGI 应用，它非常容易配置。

在 `code.py` 的最后添加：

```py
app = web.application(urls, globals(), autoreload=False)
application = app.wsgifunc() 
```

mod_wsgi 提供 [许多可行方法](http://code.google.com/p/modwsgi/wiki/ConfigurationDirectives) 来实现 WSGI 应用, 但一种简单的方法是添加以下到 .htaccess：

```py
<Files code.py>
    SetHandler wsgi-script
    Options ExecCGI FollowSymLinks
</Files> 
```

如果在 apache 的 error.log 文件中出现 "ImportError: No module named web"， 在导入 web 之前，你可能需要在 code.py 中尝试设置绝对路径：

```py
import sys, os
abspath = os.path.dirname(__file__)
sys.path.append(abspath)
os.chdir(abspath)
import web 
```

同时， 你可能需要查看 [WSGI 应用的常见问题](http://code.google.com/p/modwsgi/wiki/ApplicationIssues)的 "Application Working Directory" 部分。

最终应该可以访问 `http://example.com/code.py/`。

#### mod_rewrite 法则，Apache

如果希望 web.py 能够通过 '[`example.com`](http://example.com)' 访问，代替使用 '[`example.com/code.py/'，`](http://example.com/code.py/'，) 添加以下法则到 `.htaccess` 文件：

```py
<IfModule mod_rewrite.c>      
  RewriteEngine on
  RewriteBase /
  RewriteCond %{REQUEST_URI} !^/icons
  RewriteCond %{REQUEST_URI} !^/favicon.ico$
  RewriteCond %{REQUEST_URI} !^(/.*)+code.py/
  RewriteRule ^(.*)$ code.py/$1 [PT]
</IfModule> 
```

如果 `code.py` 在子目录 `myapp/`中， 调整 RewriteBase 为 `RewriteBase /myapp/`。 如果还有一些静态文件如 CSS 文件和图片文件, 复制这些并改成你需要的地址。
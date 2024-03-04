# 如何使用 Python SimpleHTTPServer

> 原文：<https://www.pythonforbeginners.com/modules-in-python/how-to-use-simplehttpserver>

Python 为我们提供了各种模块来完成不同的任务。如果想用 Python 创建一个简单的 web 服务器来服务文件，可以使用 Python SimpleHTTPServer 模块。在本文中，我们将讨论 Python SimpleHTTPServer 的基础知识及其工作原理。

## 什么是 Python SimpleHTTPServer？

Python 附带的 SimpleHTTPServer 模块是一个简单的 HTTP 服务器，
提供标准的 GET 和 HEAD 请求处理程序。您可以轻松地在 localhost 上设置一个服务器来提供文件服务。您还可以使用 SimpleHTTPServer 模块编写 HTML 文件并在 localhost 上创建一个工作的 web 应用程序。

## 为什么应该使用 Python SimpleHTTPServer？

内置 HTTP 服务器的一个优点是您不需要安装和配置任何东西。你唯一需要的就是安装 Python。当您需要一个快速运行的 web 服务器，并且不想设置 apache 或类似 Ngnix 的服务器时，这是一个完美的选择。

SimpleHTTPServer 是一个简单而高效的工具，它使用 GET 请求和 POST 请求来了解服务器或 web 应用程序是如何工作的。您可以使用它将系统中的任何目录转换成您的 web 服务器目录。

## 如何使用 Python SimpleHTTPServer？

要在端口 8000(默认端口)上启动 HTTP 服务器，只需键入:

```py
python -m SimpleHTTPServer [port] 
```

上述命令适用于 Python 2。要在 Python 3 中运行 SimpleHTTPServer，需要执行以下命令。

```py
python -m http.server [port]
```

执行上述命令后，您可以在 web 浏览器中打开链接 localhost:8000。在那里，您将找到启动 SimpleHTTPServer 的目录中的所有文件。您可以单击任何文件或目录，向服务器发送 GET 请求来访问这些文件。

您也可以将端口更改为其他名称:

```py
$ python -m SimpleHTTPServer 8080 
```

执行上述命令后，Python SimpleHTTPServer 将在端口 8080 上运行，而不是在端口默认端口上运行。

建议阅读:要了解如何使用 SimpleHTTPServer 编写 python 程序来服务带有自定义路径的文件，可以阅读这篇关于[带有默认和自定义路径的 simple http server](https://avidpython.com/python-basics/run-python-simplehttpserver-with-default-and-custom-paths/)的文章。

## 如何共享文件和目录？

使用 SimpleHTTPServer 共享文件和目录。您首先需要移动到您想要共享其内容的目录。为此，您可以打开一个终端和 cd，进入您希望通过浏览器和 HTTP 访问的任何目录。之后，就可以启动服务器了。

```py
cd /var/www/

$ python -m SimpleHTTPServer 
```

按 enter 键后，您应该会看到以下消息:

在 0.0.0.0 端口 8000 上提供 HTTP 服务…

然后，您可以打开您最喜欢的浏览器，输入以下任何一个地址:

```py
http://your_ip_address:8000

http://127.0.0.1:8000 
```

如果目录中没有 index.html 文件，那么所有文件和目录都会列出来。

只要 HTTP 服务器在运行，终端就会随着从 Python web 服务器加载数据而更新。您应该看到标准的 http 日志记录信息(GET 和 PUSH)、404 错误、IP 地址、日期、时间以及所有您希望从标准 HTTP 日志中看到的信息，就好像您正在跟踪 apache access 日志文件一样。

SimpleHTTPServer 是从命令行提供当前目录内容的好方法。虽然有很多 web 服务器软件(Apache、Nginx)，但是使用 Python
内置的 HTTP server 不需要安装和配置。

## 结论

在本文中，我们讨论了 Python SimpleHTTPServer 的基础知识。要了解更多关于 python 编程的知识，您可以阅读这篇关于 python 中的[字符串操作的文章。你可能也会喜欢这篇关于用 Python](https://www.pythonforbeginners.com/basics/string-manipulation-in-python) 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！
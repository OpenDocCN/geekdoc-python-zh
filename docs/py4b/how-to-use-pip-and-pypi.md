# 如何使用 Pip 和 PyPI

> 原文：<https://www.pythonforbeginners.com/basics/how-to-use-pip-and-pypi>

## pip–概述

pip 命令是一个安装和管理 Python 包的工具，比如那些在 Python 包索引中找到的
。

它是 easy_install 的替代品。

## PIP 安装

安装 PIP 很容易，如果你运行的是 Linux，它通常已经安装好了。

如果它没有安装或者当前版本已经过时，你可以使用
包管理器来安装或更新它。

在 Debian 和 Ubuntu 上:

```py
$ sudo apt-get install python-pip 
```

关于 Fedora:

```py
$ sudo yum install python-pip 
```

如果您使用的是 Mac，只需通过 easy_install 安装即可:

```py
sudo easy_install pip 
```

## PyPI——Python 包索引

现在，当 PIP 安装完成后，我们需要找到一个要安装的包。

包通常从 [Python 包索引](https://pypi.python.org/pypi "pypi")开始安装。

Python 包索引是 Python 编程语言
的软件仓库。

## PIP 入门

现在，当我们知道 PIP 是什么并且已经安装在计算机上时，让我们看看如何使用它。

要安装 Python 包索引中的包，只需打开您的终端
并使用 PIP 工具输入搜索查询。

## PIP–命令

只需在您的终端中键入 pip，您就会在
屏幕上看到以下输出:

用法:

```py
pip <command> [options]
```

命令:
安装安装包。
卸载卸载包。
以需求格式冻结输出已安装的软件包。
list 列出已安装的软件包。
显示已安装软件包的信息。
search 搜索 PyPI 找包。
zip 拉链独立包装。
解压解压单个包。
bundle 创建 pybundles。
帮助显示命令的帮助。

pip 最常见的用法是安装、升级或卸载软件包。

## PIP–搜索

要搜索一个包，比如 Flask，请键入以下内容:

```py
pip search Flask 
```

您应该看到一个输出，其中所有包都包含名称“Flask”和一个描述。

Flask-Cache–为您的 Flask 应用程序添加缓存支持。
Flask-Admin–Flask 的简单且可扩展的管理界面框架
Flask-Security–Flask 应用的简单安全性
Flask–基于 Werkzeug、Jinja2 和 good intentions 的微框架

## pip–安装软件包

我们可以看到烧瓶是可用的。

flask——一个基于 Werkzeug、Jinja2 和善意的微框架让我们安装它吧

```py
pip install Flask 
```

## pip–显示信息

Flask 已安装，让我们显示关于我们新安装的软件包的信息。

```py
pip show Flask 
```

—
名称:Flask
版本:0.10.1
位置:/usr/local/lib/python 2.7/dist-packages
要求:Werkzeug、Jinja2、itsdangerous

## pip–卸载软件包

如果你想卸载一个由 PIP 安装的软件包，你也可以这样做。

```py
pip uninstall Flask

Uninstalling Flask:
...
.....

Proceed (y/n)?

Successfully uninstalled Flask 
```

使用 pip 很容易，有了它你可以很容易地从 Pypi 安装软件包。

## 更多阅读

[https://pypi.python.org/pypi](https://pypi.python.org/pypi "pypi")
http://www.pip-installer.org/en/latest/T5[http://flask.pocoo.org/](http://flask.pocoo.org/ "flask")
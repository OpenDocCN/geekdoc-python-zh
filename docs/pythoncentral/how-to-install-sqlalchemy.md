# 如何安装 SQLAlchemy

> 原文：<https://www.pythoncentral.io/how-to-install-sqlalchemy/>

在 Python 的 SQLAlchemy 的系列介绍教程[的上一篇文章中，我们学习了如何使用 SQLAlchemy 的*声明*编写数据库代码。在本文中，我们将学习如何在 Linux、Mac OS X 和 Windows 上安装 SQLAlchemy。](https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/ "Introductory Tutorial to Python's SQLAlchemy")

## 在 Windows 上安装 SQLAlchemy

在 Windows 上安装 SQLAlchemy 之前，您需要使用 Python 的 Windows 安装程序来安装 Python。你可以在 [Python 的发布页面](http://www.python.org/download/releases/ "Python Releases")下载一个 Python 的 Windows MSI 安装程序。双击`.msi`文件即可安装。

在您的 Windows 系统上安装 Python 之后，您可以从 [SQLAlchemy 下载页面](https://www.sqlalchemy.org/download.html "SQLAlchemy Download Page")下载 SQLAlchemy 的源代码，并使用其 setup.py 脚本进行安装。

```py

C:\> C:\Python27\python.exe .\setup.py install

running install

running build

running build_py

   ......

Plain-Python build succeeded.

*********************************

```

## 在 Linux 上安装 SQLAlchemy

建议我们在安装 SQLAlchemy 之前创建一个 virtualenv。所以，让我们开始吧:

```py

$ virtualenv sqlalchemy-workspace

New python executable in sqlalchemy-workspace/bin/python

Installing distribute....................done.

Installing pip...............done

$ cd sqlalchemy-workspace

$ source bin/activate

```

那么，安装 SQLAlchemy 最简单的方法就是使用 Python 的包管理器`pip`:

```py

$ pip install sqlalchemy

Downloading/unpacking sqlalchemy

  Downloading SQLAlchemy-0.8.1.tar.gz (3.8Mb): 3.8Mb downloaded

  Running setup.py egg_info for package sqlalchemy
......
没有找到与' doc/build/output' 
匹配的以前包含的目录成功安装 sqlalchemy 
清理...

```

## 在 Mac OS X 上安装 SQLAlchemy

在 Mac OS X 上安装 SQLAlchemy 相对来说与 Linux 相同。按照与 Linux 相同的步骤创建 Python virtualenv 后，可以使用以下命令安装 SQLAlchemy:

```py

$ virtualenv sqlalchemy-workspace

New python executable in sqlalchemy-workspace/bin/python

Installing setuptools............done.

Installing pip...............done.

$ cd sqlalchemy-workspace

$ source bin/activate

$ pip install sqlalchemy

Downloading/unpacking sqlalchemy

  Downloading SQLAlchemy-0.8.2.tar.gz (3.8MB): 3.8MB downloaded

  Running setup.py egg_info for package sqlalchemy
......
没有找到与' doc/build/output' 
匹配的以前包含的目录成功安装 sqlalchemy 
清理...

```

就是这样。您已经成功地安装了 SQLAlchemy，现在您可以开始使用新的 SQLAlchemy 驱动的 virtualenv 编写*声明性*模型。

## 摘要

在 Linux 或 Mac OS X 下，建议使用 pip 在 virtualenv 中安装 SQLAlchemy，因为这比从源代码安装更方便。而在 Windows 下，您必须使用系统范围的 Python 安装从源代码安装它。
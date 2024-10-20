# 如何在 Windows、Mac 和 Linux 上安装 Django

> 原文：<https://www.pythoncentral.io/how-to-install-python-django-windows-mac-linux/>

在本文中，我们将学习如何在 Windows、Mac 和 Linux 上安装 Django。由于 Mac 和 Linux 都源自 Unix 平台，所以关于在 Mac 和 Linux 上安装 Django 的说明几乎是相同的，我们将在同一节中介绍它们。然而，Windows 与其他两个操作系统是如此的不同，以至于我们需要用一个章节来介绍如何在 Windows 上安装 Django。

## Python 虚拟环境设置

如果你使用的是主流操作系统，如 Windows、Mac OS X 或 Linux，Python 通常已经安装在你的系统上了。然而，如果不是，你可以在官方 Python 网站上下载并安装适合你的操作系统的 Python 版本。假设您的操作系统上安装了 Python 2.7.x，在接下来的章节中，我们将带您完成在 Window、Mac OS X 和 Linux 上创建 Django 开发环境的步骤。

在我们深入创建 Django 开发环境的步骤之前，我们想回顾一下一个叫做`virtualenv`的有用工具。`virtualenv`是一个创建隔离 Python 环境的工具。`virtualenv`不是在一个全局 Python 环境上编写代码，而是允许你创建 Python 环境的孤立“岛”或目录，每个“岛”或目录都是一个独立的 Python 环境，拥有自己的“系统级”包和库。

既然我们可以编写运行在全球 Python 环境之上的代码，为什么我们还需要`virtualenv`？

好吧，让我们想象一种情况，其中`my_library`依赖于另一个版本必须是 1.0.0 的包`dependent_library`。当您将全球 Python 环境从 2.7.3 升级到 2.3.3 时，`dependent_library`也将升级到 1.2.0。现在`my_library`不再工作了，因为它调用了`dependent_library`1 . 0 . 0 中的方法和类。如果你能写`my_library`对抗一个独立的 1.0.0 `dependent_library`以及另一个`upgraded_my_library`对抗 1.2.0 不是很好吗？

或者想象一下，您正在一个共享的托管环境中编程，其中您的用户不能访问根级目录，比如`/usr/lib`，这意味着您不能将全局 Python 环境修改成您喜欢的版本。如果能在自己的主目录下创建一个 Python 环境岂不是很好？

幸运的是，`virtualenv`通过创建一个拥有自己的安装目录的环境解决了上述所有问题，该环境不与其他环境共享任何库。

## 在 Windows 中设置 virtualenv 和 Django

首先，打开浏览器，导航到 [virtualenv](https://pypi.org/project/virtualenv/ "virtualenv") 。点击下载按钮获取最新 virtualenv 的源代码。

其次，打开一个 Powershell 实例，导航到下载了`virtualenv`源代码的目录，并将 tar 文件解压到一个目录中。然后您可以切换到该目录，为您当前的 Python 解释器安装`virtualenv`,该解释器可以从命令行调用。

```py

...> $env:Path = $env:Path + ";C:\Python27"

...> cd virtualenv-x.xx.x

...> python.exe .\setup.py install

Note: without Setuptools installed you will have to use "python -m virtualenv ENV"

running install

running build

......

```

现在您可以在您的主目录中创建一个`virtualenv`实例。

```py

...> python.exe -m virtualenv python2-workspace

New python executable in ...

Installing Setuptools...

Installing Pip...

```

现在我们可以使用`activate`脚本激活新环境。请注意，Windows 的执行策略在默认情况下是受限制的，这意味着不能执行像`activate`这样的脚本。因此，我们需要将执行策略更改为 *AllSigned* ，以便能够激活 virtualenv。

```py

...> Set-ExecutionPolicy AllSigned
执行策略改变
执行策略...
...:Y 
...> cd python2-workspace 
...>。\脚本\激活
 (python-workspace)...>
```

注意，一旦 virtualenv 被激活，您将看到一个字符串“(python2-workspace)”被添加到命令行的 shell 提示符前。现在您可以在新的虚拟环境中安装 Django 了。

```py

...> pip install django

Downloading/unpacking django

......

```

## 在 Mac OS X 和 Linux 中设置 virtualenv 和 Django

在 Mac OS X 和 Linux 上安装 virtualenv 和 Django 类似于 Windows。首先，下载`virtualenv`源代码，解包并使用全局 Python 解释器安装它。

```py

$ curl -O https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.10.1.tar.gz

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current

                                 Dload  Upload   Total   Spent    Left  Speed

100 1294k  100 1294k    0     0   498k      0  0:00:02  0:00:02 --:--:--  508k

$ tar xvf virtualenv-1.10.1.tar.gz

$ cd virtualenv-1.10.1/

$ sudo python setup.py install

Password:

running install

running bdist_egg

running egg_info

writing virtualenv.egg-info/PKG-INFO

writing top-level names to virtualenv.egg-info/top_level.txt

writing dependency_links to virtualenv.egg-info/dependency_links.txt

writing entry points to virtualenv.egg-info/entry_points.txt

reading manifest file 'virtualenv.egg-info/SOURCES.txt'

...

Installed /Library/Python/2.7/site-packages/virtualenv-1.10.1-py2.7.egg

Processing dependencies for virtualenv==1.10.1

Finished processing dependencies for virtualenv==1.10.1

```

然后返回到主目录，并在该目录中创建新的 virtualenv。

```py

$ virtualenv python2-workspace

New python executable in python2-workspace/bin/python

Installing Setuptools..............................................................................................................................................................................................................................done.

Installing Pip.....................................................................................................................................................................................................................................................................................................................................done.

```

一旦创建了环境，就可以激活环境并在其中安装 Django。

```py

$ cd python2-workspace/

$ pip install django

Downloading/unpacking django

  Downloading Django-1.5.4.tar.gz (8.1MB): 8.1MB downloaded

  Running setup.py egg_info for package django
警告:在目录“*
下没有找到匹配“__pycache__”的先前包含的文件警告:没有匹配“*”的先前包含的文件。在目录' *' 
下找到的 py[co]'正在安装收集的包:django 
正在运行 setup.py install for django 
将 build/scripts-2.7/django-admin . py 的模式从 644 更改为 755
警告:在目录“*
下没有找到匹配“__pycache__”的先前包含的文件警告:没有匹配“*”的先前包含的文件。将/private/tmp/python 2-workspace/bin/django-admin . py 的模式更改为 755 
成功安装 django 
清理...

```

## 总结和提示

在这篇文章中，我们学习了如何在 Windows、Mac OS X 和 Linux 上安装`virtualenv`,并使用它的`pip`命令来安装 Django。因为虚拟环境与系统的其余部分是分离的，所以安装的 Django 库只影响在特定环境中执行的文件。与 Mac OS X 和 Linux 相比，在 Windows 中设置`virtualenv`需要一个额外的步骤来改变脚本的执行策略。否则，建立 Django 虚拟环境的步骤在所有平台上几乎是相同的。
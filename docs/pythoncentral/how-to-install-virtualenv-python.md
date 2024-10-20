# 如何安装 Virtualenv (Python)

> 原文：<https://www.pythoncentral.io/how-to-install-virtualenv-python/>

**注:**编辑了关于`--no-site-packages`参数[现在被默认为](https://github.com/pypa/virtualenv/pull/178 "virtualenv --no-site-packages default argument")的评论，感谢 Twitter 上的@dideler。

Python 是一种非常强大的脚本语言。这种语言有许多 Python 包，您可以在项目中安装和使用。有时，您可能有不同的项目需要相同包/模块的不同版本。例如，假设您正在为一个客户开发一个带有最新版本的 Django 的 web 应用程序。同时，你支持一些需要 Django 版本的老项目。在这种情况下，您必须在每次切换项目时更改指向 Django 的路径。如果有一些工具可以让你从一个环境切换到另一个环境而互不影响，那就非常方便了。那个工具叫做**虚拟人**。在这篇文章中，我们将向您展示如何安装 Virtualenv，并开始。

Virtualenv 允许您创建虚拟 Python 环境。您在该环境中安装或删除的所有内容都保持不变，其他环境不受影响。最重要的是，你没有污染你的系统的全局包目录。例如，如果你想测试一个不稳定的包，virtualenv 是最好的选择。如果您测试的不稳定模块在安装/卸载过程中出现错误，您的系统不会受到影响。当你准备移除新的不稳定模块时，你需要做的只是移除你创建的 virtualenv 环境。

## 使用 pip 安装 Virtualenv

在本文中，我们将使用`pip`作为 Python 包管理器。喜欢的话也可以用`easy_install`。首先，让我们设置 **pip** (你可能需要成为 **root** 或者在 Unix 机器上使用`sudo`):

```py

$ easy_install pip

```

下一步是安装 virtualenv 软件包:

```py

$ pip install virtualenv

```

就是这样！要安装 virtualenv，非常简单。

## 用 virtualenv 创建一个环境

下一步是使用 virtualenv 创建环境:

```py

virtualenv my_blog_environment

```

使用前面的命令，我们在目录`my_blog_environment`下创建了一个独立的 Python 环境。注意，在 virtualenv 的当前版本中，默认情况下它使用`--no-site-packages`选项。这告诉 virtualenv 为我们创建一个空的 Python 环境。

另一个选择是让我们的虚拟环境包含全局 Python 目录中的所有包(例如 C:\Python27 或/usr/lib/python2.7)。如果需要这样做，可以使用`--use-site-package`参数。

## 探索虚拟环境

下一步是激活您的虚拟环境:

```py

$ cd my_blog_environment/

$ source bin/activate

```

现在，您应该会看到您的提示发生了一点变化(请参考开头的括号):

```py

user@hostname:~/my_blog_environment

```

现在你在你的虚拟环境中。如果您列出当前目录(virtualenv)的内容，您将看到 3 个目录:

```py

user@hostname:~/my_blog_environment$ ls

bin  include  lib

```

“bin”目录包括我们的 virtualenv 环境的可执行文件。那就是 Python 解释器所在的地方。此外，一些软件包安装的可执行文件也将包含在该目录中。“include”目录包含环境的头文件。最后，“lib”目录包括我们的 virtualenv 系统的已安装模块的 Python 文件。

## 安装 Virtualenv 软件包

下一步是安装一些包并使用我们的环境。正如我们在例子中提到的，让我们安装一个旧版本的 Django，版本 1.0。

```py

pip install Django==1.0

```

现在，我们可以通过检查 Python shell 来检查 Django 是否安装在我们的虚拟环境中。

```py

>>> import django

>>> print django.VERSION

(1, 0, 'final')

>>> exit()

$ deactivate

```

## 虚拟沙盒

virtualenv 的另一个有趣的特性是能够为不同版本的 Python 解释器创建沙箱。使用`-p`标志，您可以创建使用不同版本 Python 解释器的环境。举个例子，你想用 Python 的最新版本 3 创建一个项目。要做到这一点，您首先需要在您的系统上安装想要尝试的 Python 版本。之后，你需要找到解释器的可执行文件的路径。你通常可以在 Linux 的`/usr/bin`(例如`/usr/bin/python3`)下找到它，在 Windows 的`C:\`(例如`C:\Python31`)下找到它。

```py

$ virtualenv --no-site-packages -p /usr/bin/python3 p3_test

$ cd p3_test

$ source bin/activate

$ python

Python 3.2 (r32:88445, Dec  8 2011, 15:26:51)

[GCC 4.5.2] on linux2

Type "help", "copyright", "credits" or "license" for more information.

>>> print('Hello There')

Hello There

```

现在，您已经准备好试验新的 Python 3 环境了。
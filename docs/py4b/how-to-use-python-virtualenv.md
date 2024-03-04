# 如何使用 Python virtualenv

> 原文：<https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv>

## 什么是 Virtualenv？

简而言之，虚拟环境是 Python 的一个独立的工作副本，它允许你在一个特定的项目上工作，而不用担心影响其他项目

它支持多个并行的 Python 安装，每个项目一个。

它实际上并没有安装单独的 Python 副本，但是它确实提供了一种
聪明的方法来隔离不同的项目环境。

## 验证是否安装了 Virtualenv

您的系统上可能已经安装了 virtualenv。

在您的终端中运行以下命令

```py
virtualenv --version 
```

如果您看到一个版本号(在我的例子中是 1.6.1)，那么它已经安装好了。
T2>1 . 6 . 1

## 安装虚拟

有多种方法可以在您的系统上安装 virtualenv。

```py
$ sudo apt-get install python-virtualenv

$ sudo easy_install virtualenv

$ sudo pip install virtualenv 
```

## 设置和使用 Virtualenv

一旦安装了 virtualenv，只需启动一个 shell 并创建自己的
环境。

首先为您新的闪亮的隔离环境创建一个目录

```py
mkdir ~/virtualenvironment 
```

要为您的新应用程序创建一个包含 Python 的干净副本的文件夹，
只需运行:

```py
virtualenv ~/virtualenvironment/my_new_app 
```

(如果您想将您的环境与主站点
软件包目录隔离，请添加–无站点软件包)

要开始使用您的项目，您必须 cd 进入您的目录(project)
并激活虚拟环境。

```py
cd ~/virtualenvironment/my_new_app/bin 
```

最后，激活您的环境:

```py
source activate 
```

请注意您的 shell 提示符是如何改变的，以显示活动环境。

这就是你如何看到你在你的新环境中。

您现在使用 pip 或 easy_install 安装的任何软件包都会被安装到
my _ new _ app/lib/python 2.7/site-packages 中。

要退出 virtualenv，只需输入**“停用”**。

## Virtualenv 做了什么？

这里安装的包不会影响全局 Python 安装。

Virtualenv 不会创建获得全新 python 环境所需的每个文件

它使用全局环境文件的链接，以节省磁盘空间并加速你的虚拟化。

因此，您的
系统上必须已经安装了一个活动的 python 环境。

## 在您的虚拟机中安装软件包

如果您查看 virtualenv 中的 bin 目录，您会看到 easy_install，其中的
已经被修改，可以将 eggs 和包放在 virtualenv 的 site-packages
目录中。

要在虚拟机中安装应用程序，请执行以下操作:

```py
pip install flask 
```

您不必使用 sudo，因为这些文件将全部安装在 virtualenv
/lib/python 2.7/site-packages 目录中，该目录是作为您自己的用户帐户创建的

就这样，我希望你能从这篇文章中学到一些东西

进一步阅读请见:
[http://flask.pocoo.org/docs/installation/#virtualenv](http://flask.pocoo.org/docs/installation/#virtualenv "flask_install_virtualenv")
[http://pypi.python.org/pypi/virtualenv](https://pypi.python.org/pypi/virtualenv "pypi_virtualenv")
[入门-虚拟-隔离-python-environments/](http://mitchfournier.com/2010/06/25/getting-started-with-virtualenv-isolated-python-environments/ "getting_started_virtualenv")
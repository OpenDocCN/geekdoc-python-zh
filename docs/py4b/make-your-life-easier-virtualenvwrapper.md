# 使用 Virtualenvwrapper 让您的生活更轻松

> 原文：<https://www.pythonforbeginners.com/basics/make-your-life-easier-virtualenvwrapper>

当你进行大量的 Python 编程时，你可以用 Pip 把你的系统弄得一团糟。不同的 app 需要不同的需求。一个应用程序需要软件包的 1.2 版本，另一个需要 1.5 版本。然后…你就有麻烦了。

当您想知道 Pip 已经安装了哪些软件包时，请使用:

```py
$ pip freeze
```

结果可能是一个很长的列表。

对于每个项目，无论是简单的应用程序还是巨大的 Django 项目，我都用 virtualenv 创建一个新的虚拟环境。一种简单的方法是使用 virtualenvwrapper。

## Virtualenv 包装器入门

```py
 $ sudo pip install virtualenvwrapper 
```

您可以在[手册中找到一些高级设置。](https://virtualenvwrapper.readthedocs.org/en/latest/index.html "virtualenvwrapper docs")

安装完成后，让我们创建一个名为“testground”的新环境。

```py
 $ mkvirtualenv testground 
```

如果您的 virtualenvwrapper 运行良好，您应该会看到如下内容:

```py
 New python executable in testground/bin/python
Installing Setuptools.........done.
Installing Pip..............done.
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/testground/bin/predeactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/testground/bin/postdeactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/testground/bin/preactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/testground/bin/postactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/testground/bin/get_env_details

(testground)[[email protected]](/cdn-cgi/l/email-protection):~$ 
```

看一下最后一行，现在在提示符前显示(testground)。你的虚拟环境就在那里，你也在其中。

现在，执行一个 pip 冻结，它可能会导致如下结果:

```py
$ pip freeze
argparse==1.2.1
wsgiref==0.1.2 
```

该列表可以比环境之外的列表短得多。

您现在可以安装只影响这种环境的东西。我们试试枕头，2.2.2 版。

```py
 $ pip install Pillow==2.2.2 
```

它现在将尝试安装枕头，完成后，做另一个 pip fip 冻结，并查看它是否已安装

## 在虚拟电视之间切换

让我们创建另一个环境:

```py
$ mkvirtualenv coolapp
```

完成后，它应该会自动激活:

```py
 ...................done.
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/coolapp/bin/predeactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/coolapp/bin/postdeactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/coolapp/bin/preactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/coolapp/bin/postactivate
virtualenvwrapper.user_scripts creating /home/yourname/.virtualenvs/coolapp/bin/get_env_details
(coolapp)[[email protected]](/cdn-cgi/l/email-protection):~$ 
```

现在，让我们先来看看我们用“工作”这个神奇的词创造的虚拟世界:

```py
$ workon testground
(testground)[[email protected]](/cdn-cgi/l/email-protection):~$ 
```

这就对了，就这么简单

## 没有站点包

与 vitualenv 一样，您可以告诉 virtualenvwapper 不要将系统 sitepackages 与–no-site-packages 一起使用:

```py
 $ mkvirtualenv anotherenv --no-site-packages 
```

## 删除虚拟环境

使用 rmvirtualenv 可以很容易地删除虚拟 env:

```py
 $ rmvirtualenv coolapp

Removing coolapp... 
```

## 列出所有虚拟人

```py
 $lsvirtualenv 
```

## 列出你的站点包

```py
 $ lssitepackages

#can result in this:
easy_install.py   Pillow-2.2.2-py2.7.egg-info  pkg_resources.pyc
easy_install.pyc  pip                          setuptools
_markerlib        pip-1.4.1-py2.7.egg-info     setuptools-0.9.8-py2.7.egg-info
PIL               pkg_resources.py 
```
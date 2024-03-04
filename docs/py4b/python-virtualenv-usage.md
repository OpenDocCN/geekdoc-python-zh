# 如何在 Python 中使用 Virtualenv

> 原文：<https://www.pythonforbeginners.com/basics/python-virtualenv-usage>

这篇文章将描述什么是 Virtualenv，以及你如何使用它。

### 什么是 Virtualenv？

```py
 Virtualenv is a tool to create isolated Python environments, it's perhaps the 
easiest way to configure a custom Python environment. 

Virtualenv allows you to add and modify Python modules without access to the
global installation. 
```

### 它是做什么的？

```py
 The basic problem being addressed is one of dependencies and versions, and 
indirectly permissions. 

Imagine you have an application that needs version 1 of LibFoo, but another
application requires version 2\. 

How can you use both these applications? 

If you install everything into /usr/lib/python2.7/site-packages (or whatever your 
platform’s standard location is), it’s easy to end up in a situation where you 
unintentionally upgrade an application that shouldn’t be upgraded.

Or more generally, what if you want to install an application and leave it be? 

If an application works, any change in its libraries or the versions of those
libraries can break the application.

Also, what if you can’t install packages into the global site-packages directory? 

For instance, on a shared host.

In all these cases, virtualenv can help you. 

It creates an environment that has its own installation directories, that doesn’t
share libraries with other virtualenv environments 
```

### 如何安装 Virtualenv？

```py
 There are a few ways to install virtualenv on your machine. 

You can use either the source tarball, pip or by using easy_install. 
```

##### 简单安装

```py
 $ sudo easy_install virtualenv

Searching for virtualenv
Reading http://pypi.python.org/simple/virtualenv/
Reading http://www.virtualenv.org
Reading http://virtualenv.openplans.org
Best match: virtualenv 1.8.2
Downloading http://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.8.2.tar.gz... 
processing virtualenv-1.8.2.tar.gz
.....
....
Processing dependencies for virtualenv
Finished processing dependencies for virtualenv 
```

##### 源球安装

```py
 Get the latest version from here: http://pypi.python.org/packages/source/v/virtualenv/
wget http://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.8.tar.gz
tar xzvf virtualenv-1.8.tar.gz
python virtualenv-1.8/virtualenv.py $HOME/env 
```

##### Pip 安装

```py
 pip install virtualenv 
```

### 使用

```py
 To create virtual environments, you can use the virtualenv command. 

Create an environment called "foobar":
virtualenv foobar

Activate the environment by sourcing its activate script, which is located in the 
environment's bin/ directory:
source foobar/bin/activate

This will change your $PATH so its first entry is the virtualenv’s bin/ directory.

If you install a package in your virtual environment, you'll see that executable 
scripts are placed in foobar/bin/ and eggs in foobar/lib/python2.X/site-packages/
easy_install yolk

Yolk is a small command line tool which can, among other things, list the
currently installed Python packages on your system:
yolk -l

Virtualenv inherits packages from the system's default site-packages directory. 

This is especially useful when relying on certain packages being available,
so you don't have to go through installing them in every environment.

To leave an environment, simply run deactivate:
deactivate

If you execute he yolk command now, you will see that it won't work because the
package was installed only in your virtual environment. 

Once you reactivate your environment it will be available again. 
```

##### 来源

```py
 I used different sources to find information for this article: 
the official virtualenv [website](http://www.virtualenv.org/en/latest/index.html "virtualenv latest"), from Chris Scott "[A Primer on virtualenv](http://iamzed.com/2009/05/07/a-primer-on-virtualenv/ "a primer on virtualenv")" and
from Arthur Koziel '[Working with virtualenv](http://www.arthurkoziel.com/2008/10/22/working-virtualenv/ "working-virtualenv")' 
```
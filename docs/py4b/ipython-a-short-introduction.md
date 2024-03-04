# IPython 简介

> 原文：<https://www.pythonforbeginners.com/basics/ipython-a-short-introduction>

## 概观

本文的目标是写一篇关于 IPython 的简短介绍。虽然 IPython 有两个主要组件(一个交互式 Python shell 和一个交互式并行计算的架构)，但本文将讨论 Python Shell。我将把[并行计算](https://ipython.org/ipython-doc/rel-0.13.1/parallel/index.html#parallel-index "parallel_computing")部分留待下次。将来我还会写关于“[IPython 笔记本](https://ipython.org/notebook.html "notebook_ipython")”的文章，这是一个基于 web 的交互环境，在这里你可以将代码执行、文本、数学、情节和富媒体结合到一个文档中。众所周知，IPython 可以在以下操作系统上工作:Linux、大多数其他类似 Unix 的操作系统(AIX、Solaris、BSD 等)。)Mac OS X Windows (CygWin，XP，Vista 等。)

## IPython 是什么？

IPython 是 Python 编程语言的交互式 shell，它提供了增强的自省、附加的 shell 语法、制表符补全和丰富的历史记录。[来源](https://ipython.org/index.html "ipython")

## 为什么选择 IPython？

默认的交互式 Python shell 有时感觉很简单。有一种替代方法叫做“IPython ”,你可以通过输入 apt-get install IPython 来安装它(参见下面的安装部分)。当它安装好后，你可以通过在终端中输入 IPython 来启动它。IPython 提供了你在基本解释器中可以得到的所有东西，但还有很多额外的东西(行号、高级编辑、更多功能、帮助功能等)

## 安装 IPython？

正如我在上面写的，如果你在 Ubuntu 系统上，你可以通过在你的终端中输入 apt-get install IPython 来安装 IPython。如果您在另一个系统上，请查看这里的让我们来看看在运行 Mac 的系统上安装会是什么样子。

```py
# To see if I've ipython installed, I simply type "ipython" in my terminal. 

$ ipython
-bash: ipython: command not found 
```

所以我的系统上没有安装 IPython。让我们安装它

```py
$ sudo easy_install ipython
Password:
Searching for ipython
Reading http://pypi.python.org/simple/ipython/
Reading http://ipython.scipy.org
Reading http://ipython.scipy.org/dist
Reading http://ipython.org
Reading https://github.com/ipython/ipython/downloads
Reading http://ipython.scipy.org/dist/0.8.4
Reading http://ipython.scipy.org/dist/0.9.1
Reading http://archive.ipython.org/release/0.12.1
Reading http://ipython.scipy.org/dist/old/0.9
Reading http://ipython.scipy.org/dist/0.10
Reading http://archive.ipython.org/release/0.11/
Reading http://archive.ipython.org/release/0.12
Best match: ipython 0.13.1
Downloading http://pypi.python.org/packages/2.7/i/ipython/ipython-0.13.1-py2.7.egg#md5..
Processing ipython-0.13.1-py2.7.egg
creating /Library/Python/2.7/site-packages/ipython-0.13.1-py2.7.egg
Extracting ipython-0.13.1-py2.7.egg to /Library/Python/2.7/site-packages
Adding ipython 0.13.1 to easy-install.pth file
Installing ipcontroller script to /usr/local/bin
Installing iptest script to /usr/local/bin
Installing ipcluster script to /usr/local/bin
Installing ipython script to /usr/local/bin
Installing pycolor script to /usr/local/bin
Installing iplogger script to /usr/local/bin
Installing irunner script to /usr/local/bin
Installing ipengine script to /usr/local/bin
Installed /Library/Python/2.7/site-packages/ipython-0.13.1-py2.7.egg
Processing dependencies for ipython
Finished processing dependencies for ipython 
```

当我这次在终端中键入 IPython 时，它启动了，但是我得到一条错误消息:

```py
$ ipython
/Library/Python/2.7/site-packages/ipython-0.13.1-py2.7.egg/IPython/utils/rlineimpl.py:111: 
RuntimeWarning:
libedit detected - readline will not be well behaved, including but not limited to:
   * crashes on tab completion
   * incorrect history navigation
   * corrupting long-lines
   * failure to wrap or indent lines properly
It is highly recommended that you install readline, which is easy_installable:
     easy_install readline
Note that `pip install readline` generally DOES NOT WORK, because
it installs to site-packages, which come *after* lib-dynload in sys.path,
where readline is located.  It must be `easy_install readline`, or to a custom
location on your PYTHONPATH (even --user comes after lib-dyload). 
```

要解决这个问题，只需输入 easy_install readline(如上所述)

```py
$sudo easy_install readline 
```

```py
Searching for readline
Reading http://pypi.python.org/simple/readline/
Reading http://github.com/ludwigschwardt/python-readline
Reading http://www.python.org/
Best match: readline 6.2.4.1
Downloading http://pypi.python.org/packages/2.7/r/readline/readline-6.2.4.1-py2.7-macosx..
Processing readline-6.2.4.1-py2.7-macosx-10.7-intel.egg
creating /Library/Python/2.7/site-packages/readline-6.2.4.1-py2.7-macosx-10.7-intel.egg
Extracting readline-6.2.4.1-py2.7-macosx-10.7-intel.egg to /Library/Python/2.7/site-packages
Adding readline 6.2.4.1 to easy-install.pth file

Installed /Library/Python/2.7/site-packages/readline-6.2.4.1-py2.7-macosx-10.7-intel.egg
Processing dependencies for readline
Finished processing dependencies for readline 
```

安装了 readline，一切都应该没问题了。

```py
$ ipython
Python 2.7.2 (default, Jun 20 2012, 16:23:33)
Type "copyright", "credits" or "license" for more information.

IPython 0.13.1 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: 
```

IPython 现已安装在您的系统上。

## 启动 IPython

通过在终端中键入“ **ipython** ”来启动 IPython。

```py
$ ipython
Python 2.7.2 (default, Jun 20 2012, 16:23:33) 
Type "copyright", "credits" or "license" for more information.

IPython 0.13.1 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details. 
```

## 从文件运行 Python 代码

IPython 的基本工作流程是使用文本编辑器编辑代码。保存文件并将其加载到 IPython 中。如果您想测试代码的交互性，请使用%run -i，否则，您可以只使用%run。如果出现问题，只需返回文本编辑器，修复错误，保存并退出。然后返回 IPython，再次运行该文件。要运行保存到文件中的 python 代码(例如 hello.py)，可以使用命令 **%run** (在我们的例子中是%run hello.py)。IPython 将在当前文件夹中查找该文件。您可以使用 ls 命令列出当前文件夹中文件的内容，hello.py 中的代码将会运行，但是其中的函数将不可用于交互式调用。如果您想测试代码的交互性，那么在使用%run 时，您必须添加-i 开关。运行它交互的命令是 **%run -i hello.py**

## 制表符结束

制表符补全，尤其是对于属性来说，是一种探索您正在处理的任何对象的结构的便捷方式。 [source](https://ipython.org/ipython-doc/stable/interactive/tutorial.html#tab-completion "tutorial_tabs_ipython") 要使用补全，请键入您希望 shell 匹配的模式，然后按 Tab 键。只需键入 object_name。除了 Python 对象和关键字之外，要查看对象的属性，制表符补全也适用于文件名和目录名。

```py
In [1]: from sys import std
stderr  stdin   stdout

In [1]: from urllib2 import url
url2pathname  urlopen       urlparse 
```

## 宏指令

IPython 宏非常适合反复执行相同的代码。宏允许用户将一个名称与 Python 代码的一部分相关联，这样以后就可以通过引用该名称来运行代码。它们可以通过“%edit”魔法命令进行编辑

## 使用 Python 调试器(pdb)

Python 调试器(pdb)是一个强大的交互式调试器，它允许你单步调试代码、设置断点、观察变量等。启用自动 pdb 调用后，当 Python 遇到未处理的异常时，Python 调试器将自动启动。调试器中的当前行将是发生异常的代码行。如果您使用–pdb 选项启动 IPython，那么您可以在每次代码触发未捕获的异常时调用 Python pdb 调试器。也可以随时使用%pdb magic 命令切换此功能。[来源](https://ipython.org/ipython-doc/stable/interactive/reference.html#using-the-python-debugger-pdb "pdb_ipython")

## 轮廓

概要文件是包含配置和运行时文件的目录，比如日志、并行应用程序的连接信息和 IPython 命令历史。概要文件使得为特定项目保存单独的配置文件、日志和历史变得容易。通过下面的命令可以很容易地创建概要文件。**$ ipython profile create profile_name**这会将名为 profile _ name 的目录添加到您的 IPython 目录中。然后，您可以通过在命令行选项中添加–profile =来加载这个概要文件。所有 IPython 应用程序都支持概要文件。这个命令应该创建并打印配置文件的安装路径。要使用概要文件，只需将概要文件指定为 ipython 的一个参数。**$ IPython–profile = profile _ name**IPython 在 IPython/config/profile 中附带了一些样例概要文件。

## 启动文件

在 profile_/startup 目录中，您可以将任何 python(.py)或 IPython(。ipy)文件，您希望 IPython 一启动就运行这些文件。当前在我的 profile_default/startup 目录中唯一的东西是一个自述文件。该文件的内容应该类似于“这是 IPython 启动目录。py 和。每当您加载此配置文件时，此目录中的 ipy 文件将在通过 exec_lines 或 exec_files 可配置文件指定的任何代码或文件之前运行。文件将按字典顺序运行，因此您可以使用前缀控制文件的执行顺序，例如:00-first . py 50-middle . py 99-last . ipy " OK。太好了，让我们试试这个。在编辑器中创建新文件:

```py
$ vim calc.py

# This is a small python script calculating numbers

a = 1 + 1

b = 2 * 2

c = 10 / 2

d = 10 - 5

print a,b,c,d 
```

退出编辑器并启动 IPython

```py
$ ipython 
Python 2.7.2 (default, Jun 20 2012, 16:23:33) 
Type "copyright", "credits" or "license" for more information.

IPython 0.13.1 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
3

>>> 2 4 5 5

In [1]: %run calc.py
2 4 5 5 
```

这样就可以使用启动文件中的所有内容，而无需在每次重新输入 IPython 时重新键入它们。

## IPython 中的命令

IPython“magic”命令通常以%开头，但是如果标志%automagic 设置为 on(这是默认设置)，那么可以调用前面没有%的 magic 命令。IPython 会检查您输入的命令是否符合它的神奇关键字列表。如果命令是一个神奇的关键字，IPython 知道如何处理它。如果它不是一个神奇的关键字，它会让 Python 知道如何处理它。

##### lxmagic

列出所有内置命令，称为魔术命令。如果变量具有相同的名称，这些变量会以%作为前缀来区分。

```py
#In [57]: lsmagic

Available line magics:
%alias  %alias_magic  %autocall  %autoindent  %automagic  %bookmark  %cd  %colors  %config
%cpaste  %debug  %dhist  %dirs  %doctest_mode  %ed  %edit  %env  %gui  %hist  %history
%install_default_config  %install_ext  %install_profiles  %killbgscripts  %load  %load_ext
%loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %lsmagic  %macro  %magic
%notebook  %page  %paste  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %popd
%pprint  %precision %profile  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab
%quickref  %recall  %rehashx %reload_ext  %rep  %rerun  %reset  %reset_selective  %run
%save  %sc  %store  %sx  %system  %tb %time  %timeit  %unalias  %unload_ext  %who  %who_ls 
%whos  %xdel  %xmode

Available cell magics:
%%!  %%bash  %%capture  %%file  %%perl  %%prun  %%ruby  %%script  %%sh  %%sx  %%system
%%timeit

Automagic is ON, % prefix IS NOT needed for line magics. 
```

##### 帮助命令

**%quickref** 显示了 IPython 中可用的“神奇”命令。如果您在 IPython 中键入%quickref，您将看到快速参考卡，其中包含许多有用的帮助。IPython —增强的交互式 Python —快速参考卡= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = " "很有用。如果你输入？在镜头后？，您将看到关于函数 len 的文档。打字？在一个名称之后，将会给出与该名称相关的对象的信息，

```py
>>>len?

Type:       builtin_function_or_method
String Form:
Namespace:  Python builtin
Docstring:
len(object) -> integer

Return the number of items of a sequence or mapping.

>>>str?

Type:       type
String Form:<type 'str'="">
Namespace:  Python builtin
Docstring:
str(object) -> string

Return a nice string representation of the object.

If the argument is a string, the return value is the same object. 
```

##### 更多命令

**%reset** 重置交互环境 **%hist** 允许您通过键入[55]:hist-g math 19:import math 55:hist-g math**% paste**使用剪贴板中的文本，例如，如果您使用 Ctrl+C 复制了代码，则可以查看您的输入历史的任何部分**% hist-g somestring**Search(‘grep’)。该命令清除某些字符，并尝试找出代码应该如何格式化。**%edit**% edit 命令(及其别名% ed)将调用您环境中的编辑器集作为编辑器。 **%who** 此函数列出对象、函数等。已添加到当前命名空间中的模块，以及已导入的模块。In [50]: who 交互式命名空间为空。

```py
In [51]: import sys

In [52]: import os

In [53]: who
os  sys 
```

## 系统外壳访问

任何以！开头的输入行！字符被逐字传递(减去！)到底层操作系统。比如打字！ls 将在当前目录中运行' ls'。要在系统 shell 中运行任何命令:

```py
In [2]: !ping www.google.com

PING www.google.com (173.194.67.104): 56 data bytes
64 bytes from 173.194.67.104: icmp_seq=0 ttl=49 time=6.096 ms
64 bytes from 173.194.67.104: icmp_seq=1 ttl=49 time=5.963 ms
^C 
```

您可以将输出捕获到一个 [Python 列表](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)中，例如:files =！ls。

## 别名

IPython 附带了一些预定义的别名。您的所有$PATH 都已经作为 IPython 别名加载，因此您应该能够键入任何普通的系统命令并执行它。In [1]: %alias 别名总数:16 Out[1]: ('lk '，' ls -F -l %l | grep ^l')，(' ll '，' ls -F -l ')，(' ls '，' ls -F ')，…更全面的文档可以在这里找到:[http://ipython . org/ipython-doc/stable/interactive/tutorial . html](https://ipython.org/ipython-doc/stable/interactive/tutorial.html "ipython_pub")[http://wiki.ipython.org/index.php?title=Cookbook](http://wiki.ipython.org/index.php?title=Cookbook "ipython_cookbook")
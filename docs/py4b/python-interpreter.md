# 使用 Python 解释器

> 原文：<https://www.pythonforbeginners.com/basics/python-interpreter>

## Python 解释器

Python 解释器通常安装在机器
上的/usr/local/bin/python 中；

将/usr/local/bin 放在您的 Unix shell 的搜索路径中，可以通过在 shell 中键入命令:python 来启动它。

当您以交互模式启动 Python 时，它会提示输入下一个命令，
通常是三个大于号(> > >)。

解释器在打印第一个提示之前打印一条欢迎消息，说明其版本号和一个
版权声明，例如:

```py
Python 2.7.3 (default, Apr 20 2012, 22:39:59) 
[GCC 4.6.3] on linux2
Type "help", "copyright", "credits" or "license" for more information.

>>> the_world_is_flat = 1
>>> if the_world_is_flat:
...     print "Be careful not to fall off!"
... 
```

## Python 交互式

当您交互式地使用 Python 时，每次启动解释器时执行一些标准命令
会很方便。

您可以通过将名为 PYTHONSTARTUP 的环境变量设置为包含启动命令的文件的名称
来实现这一点。

这类似于。Unix shells 的概要特性。

为此，请将这一行添加到您的。bashrc 文件:
> >导出 PYTHONSTARTUP=$HOME/。pythonstartup

创建(或修改)的。pythonstartup 文件，并将您的 python 代码放在那里:

```py
import os
os.system('ls -l') 
```

要退出 Python 交互式提示符，我们将按 Ctrl+D(在 Linux 中)

更多信息见 Pythons 官方文档。
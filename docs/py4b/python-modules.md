# Python 模块

> 原文：<https://www.pythonforbeginners.com/modules-in-python/python-modules>

这是 Python 初学者的新系列文章，应该是 Python 完全初学者的起点。把它当做备忘单、参考资料、手册或任何你想要的东西。目的是非常简短地写下 Python 的基础知识。本页将描述如何在 Python 中使用模块。

## Python 模块

Python 模块使得编程更加容易。它基本上是一个由已经编写好代码组成的文件。当 Python 导入一个模块时，它首先检查模块注册表(sys.modules ),看看该模块是否已经被导入。如果是这种情况，Python 将使用现有的模块对象。导入模块有不同的方法。这也是 Python 自带电池的原因

## 导入模块

让我们来看看导入一个模块的不同方法

```py
import sys    		
#access module, after this you can use sys.name to refer to things defined in module sys.

from sys import stdout  
# access module without qualiying name. 
This reads from the module "sys" import "stdout", so that we would be able to 
refer "stdout"in our program.

from sys import *       
# access all functions/classes in the sys module. 
```

## 捕捉错误

我喜欢使用 import 语句来确保所有模块都可以加载到系统中。导入错误基本上意味着您不能使用这个模块，您应该查看回溯来找出原因。

```py
Import sys
try:
    import BeautifulSoup
except ImportError:
    print 'Import not installed'
    sys.exit() 
```

## Python 标准库

URL:[http://docs.python.org/library/index.html](https://docs.python.org/library/index.html "library")系统中已经存在的模块集合，没有必要安装它们。只需导入您想要使用的模块。搜索一个模块:http://docs.python.org/py-modindex.html

## Python 包索引

URL:[http://pypi.python.org/pypi](https://pypi.python.org/pypi "pypi")由社区成员创建。这是一个包含 2400 多个软件包的软件仓库
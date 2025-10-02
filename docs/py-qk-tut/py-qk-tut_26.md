## Python 补充 02 我的 Python 小技巧

[`www.cnblogs.com/vamei/archive/2012/11/06/2755503.html`](http://www.cnblogs.com/vamei/archive/2012/11/06/2755503.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

在这里列举一些我使用 Python 时积累的小技巧。这些技巧是我在使用 Python 过程中经常使用的。之前很零碎的记在笔记本中，现在整理出来，和大家分享，也作为[Python 快速教程](http://www.cnblogs.com/vamei/archive/2012/09/13/2682778.html)的一个补充。 

**import 模块** 

在 Python 经常使用 import 声明，以使用其他模块(也就是其它.py 文件)中定义的对象。 

1) 使用 __name__

当我们编写 Python 库模块的时候，我们往往运行一些测试语句。当这个程序作为库被 import 的时候，我们并不需要运行这些测试语句。一种解决方法是在 import 之前，将模块中的测试语句注释掉。Python 有一种更优美的解决方法，就是使用 __name__。

下面是一个简单的库程序 TestLib.py。当直接运行 TestLib.py 时，__name__ 为"__main__"。如果被 import 的话，__name__ 为"TestLib"。

```py
def lib_func(a): return a + 10 

```

```py
def lib_func_another(b): return b + 20
if __name__ == '__main__':
    test = 101
    print(lib_func(test))

```

 我们在 user.py 中 import 上面的 TestLib。

```py
import TestLib print(TestLib.lib_func(120))

```

 你可以尝试不在 TestLib.py 中使用 if __name__=='__main__'， 并对比运行结果。

2) 更多 import 使用方式

import TestLib as test         # 引用 TestLib 模块，并将它改名为 t

比如: 

```py
import TestLib as t print(t.lib_func(120))

```

from TestLib import lib_func   # 只引用 TestLib 中的 lib_func 对象，并跳过 TestLib 引用字段 

这样的好处是减小所引用模块的内存占用。 

比如： 

```py
from TestLib import lib_func print(lib_func(120))

```

from TestLib import *          # 引用所有 TestLib 中的对象，并跳过 TestLib 引用字段

比如: 

```py
from TestLib import *
print(lib_func(120))

```

1) 查询函数的参数

当我们想要知道某个函数会接收哪些参数的时候，可以使用下面方法查询。

```py
import inspect print(inspect.getargspec(func))

```

2) 查询对象的属性

除了使用 dir()来查询对象的属性之外，我们可以使用下面内置(built-in)函数来确认一个对象是否具有某个属性：

hasattr(obj, attr_name)   # attr_name 是一个字符串

例如：

```py
a = [1,2,3]
print(hasattr(a,'append'))

```

3) 查询父类

我们可以用 __base__ 属性来查询某个类的父类：

cls.__base__

例如：

**使用中文(以及其它非 ASCII 编码)**

在 Python 程序的第一行加入

```py
#coding=utf8
print("你好吗？")

```

也能用以下方式：

```py
#-*- coding: UTF-8 -*-
print("你好吗？")

```

**表示 2 进制，8 进制和 16 进制数字**

在 2.6 以上版本，以如下方式表示 

```py
print(0b1110)     # 二进制，以 0b 开头
print(0o10)       # 八进制，以 0o 开头
print(0x2A)       # 十六进制，以 0x 开头

```

如果是更早版本，可以用如下方式：

```py
print(int("1110", 2))
print(int("10", 8))
print(int("2A", 16))

```

一行内的注释可以以#开始

多行的注释可以以'''开始，以'''结束，比如

```py
''' This is demo '''

def func(): # print something
    print("Hello world!") # use print() function # main
func()

```

注释应该和所在的程序块对齐。

**搜索路径**

当我们 import 的时候，Python 会在搜索路径中查找模块(module)。比如上面 import TestLib，就要求 TestLib.py 在搜索路径中。

我们可以通过下面方法来查看搜索路径：

```py
import sys print(sys.path)

```

我们可以在 Python 运行的时候增加或者删除 sys.path 中的元素。另一方面，我们可以通过在 shell 中增加 PYTHONPATH 环境变量，来为 Python 增加搜索路径。

下面我们增加/home/vamei/mylib 到搜索路径中：

$export PYTHONPATH=$PYTHONPATH:/home/vamei/mylib

你可以将正面的这行命令加入到～/.bashrc 中。这样，我们就长期的改变了搜索路径。

**脚本与命令行结合**

可以使用下面方法运行一个 Python 脚本，在脚本运行结束后，直接进入 Python 命令行。这样做的好处是脚本的对象不会被清空，可以通过命令行直接调用。

$python -i script.py

**安装非标准包**

Python 的标准库随着 Python 一起安装。当我们需要非标准包时，就要先安装。

1) 使用 Linux repository (Linux 环境)

这是安装 Python 附加包的一个好的起点。你可以在 Linux repository 中查找可能存在的 Python 包 (比如在 Ubuntu Software Center 中搜索 matplot)。

2) 使用 pip。pip 是 Python 自带的包管理程序，它连接 Python repository，并查找其中可能存在的包。

比如使用如下方法来安装、卸载或者升级 web.py：

$pip install web.py 

$pip uninstall web.py

$pip install --upgrade web.py

如果你的 Python 安装在一个非标准的路径(使用$which python 来确认 python 可执行文件的路径)中，比如/home/vamei/util/python/bin 中，你可以使用下面方法设置 pip 的安装包的路径:

$pip install --install-option="--prefix=/home/vamei/util/" web.py 

3) 从源码编译

如果上面方法都没法找到你想要的库，你可能需要从源码开始编译。Google 往往是最好的起点。

以后如果有新的收获，会补充到这篇博文中。
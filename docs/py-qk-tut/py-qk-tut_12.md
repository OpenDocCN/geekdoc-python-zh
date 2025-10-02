## Python 进阶 03 模块

[`www.cnblogs.com/vamei/archive/2012/07/03/2574436.html`](http://www.cnblogs.com/vamei/archive/2012/07/03/2574436.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

我们之前看到了函数和对象。从本质上来说，它们都是为了更好的组织已经有的程序，以方便重复利用。

模块(module)也是为了同样的目的。在 Python 中，一个.py 文件就构成一个模块。通过模块，你可以调用其它文件中的程序。

1\. 引入(import)和使用模块

我们先写一个 first.py 文件，内容如下：

```py
def laugh(): print 'HaHaHaHa'

```

再写一个 second.py

```py
import first for i in range(10):
    first.laugh()

```

在 second.py 中，我们并没有定义 laugh 函数，但通过从 first 中引入(import)，我们就可以直接使用 first.py 中的 laugh 函数了。

从上面可以看到，引入模块后，我们可以通过 模块.对象 的方式来调用所想要使用的对象。上面例子中，first 为引入的模块，laugh()是我们所引入的对象。

此外，还有其它的引入方式, import a as b, from a import *， 都是处于方便书写的原因，本质上没有差别。

2\. 搜索路径

Python 会在以下路径中搜索它想要寻找的模块：

1\. 程序所在的文件夹

2\. 标准库的安装路径

3\. 操作系统环境变量 PYTHONPATH 所包含的路径

如果你有自定义的模块，或者下载的模块，可以根据情况放在相应的路径，以便 python 可以找到。

3\. 模块包

可以将功能相似的模块放在同一个文件夹（比如说 dir）中，通过

的方式引入。

注意，该文件夹中必须包含一个 __init__.py 的文件，以便提醒 python 知道该文件夹为一个模块包。__init__.py 可以是一个空文件。

import module

module.object

__init__.py
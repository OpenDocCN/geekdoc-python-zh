# 第八章 模块

**目录表**

*   简介
*   使用 sys 模块
*   字节编译的.pyc 文件
*   from..import 语句
*   模块的**name**
*   使用模块的**name**
*   制造你自己的模块
*   创建你自己的模块
*   from..import
*   dir()函数
*   使用 dir 函数
*   概括

你已经学习了如何在你的程序中定义一次函数而重用代码。如果你想要在其他程序中重用很多函数，那么你该如何编写程序呢？你可能已经猜到了，答案是使用模块。模块基本上就是一个包含了所有你定义的函数和变量的文件。为了在其他程序中重用模块，模块的文件名**必须**以`.py`为扩展名。

模块可以从其他程序 输入 以便利用它的功能。这也是我们使用 Python 标准库的方法。首先，我们将学习如何使用标准库模块。

```py
#!/usr/bin/python
# Filename: using_sys.py

import sys

print 'The command line arguments are:'
for i in sys.argv:
    print i

print '\n\nThe PYTHONPATH is', sys.path, '\n' 
```

（源文件：code/using_sys.py）

## 输出

```py
$ python using_sys.py we are arguments
The command line arguments are:
using_sys.py
we
are
arguments

The PYTHONPATH is ['/home/swaroop/byte/code', '/usr/lib/python23.zip',
'/usr/lib/python2.3', '/usr/lib/python2.3/plat-linux2',
'/usr/lib/python2.3/lib-tk', '/usr/lib/python2.3/lib-dynload',
'/usr/lib/python2.3/site-packages', '/usr/lib/python2.3/site-packages/gtk-2.0'] 
```

## 它如何工作

首先，我们利用`import`语句 输入 `sys`模块。基本上，这句语句告诉 Python，我们想要使用这个模块。`sys`模块包含了与 Python 解释器和它的环境有关的函数。

当 Python 执行`import sys`语句的时候，它在`sys.path`变量中所列目录中寻找`sys.py`模块。如果找到了这个文件，这个模块的主块中的语句将被运行，然后这个模块将能够被你 使用 。注意，初始化过程仅在我们 第一次 输入模块的时候进行。另外，“sys”是“system”的缩写。

`sys`模块中的`argv`变量通过使用点号指明——`sys.argv`——这种方法的一个优势是这个名称不会与任何在你的程序中使用的`argv`变量冲突。另外，它也清晰地表明了这个名称是`sys`模块的一部分。

`sys.argv`变量是一个字符串的 列表 （列表会在后面的章节详细解释）。特别地，`sys.argv`包含了 命令行参数 的列表，即使用命令行传递给你的程序的参数。

如果你使用 IDE 编写运行这些程序，请在菜单里寻找一个指定程序的命令行参数的方法。

这里，当我们执行`python using_sys.py we are arguments`的时候，我们使用**python**命令运行`using_sys.py`模块，后面跟着的内容被作为参数传递给程序。Python 为我们把它存储在`sys.argv`变量中。

记住，脚本的名称总是`sys.argv`列表的第一个参数。所以，在这里，`'using_sys.py'`是`sys.argv[0]`、`'we'`是`sys.argv[1]`、`'are'`是`sys.argv[2]`以及`'arguments'`是`sys.argv[3]`。注意，Python 从 0 开始计数，而非从 1 开始。

`sys.path`包含输入模块的目录名列表。我们可以观察到`sys.path`的第一个字符串是空的——这个空的字符串表示当前目录也是`sys.path`的一部分，这与`PYTHONPATH`环境变量是相同的。这意味着你可以直接输入位于当前目录的模块。否则，你得把你的模块放在`sys.path`所列的目录之一。

# 字节编译的.pyc 文件

输入一个模块相对来说是一个比较费时的事情，所以 Python 做了一些技巧，以便使输入模块更加快一些。一种方法是创建 字节编译的文件 ，这些文件以`.pyc`作为扩展名。字节编译的文件与 Python 变换程序的中间状态有关（是否还记得 Python 如何工作的介绍？）。当你在下次从别的程序输入这个模块的时候，`.pyc`文件是十分有用的——它会快得多，因为一部分输入模块所需的处理已经完成了。另外，这些字节编译的文件也是与平台无关的。所以，现在你知道了那些`.pyc`文件事实上是什么了。

# from..import 语句

如果你想要直接输入`argv`变量到你的程序中（避免在每次使用它时打`sys.`），那么你可以使用`from sys import argv`语句。如果你想要输入所有`sys`模块使用的名字，那么你可以使用`from sys import *`语句。这对于所有模块都适用。一般说来，应该避免使用`from..import`而使用`import`语句，因为这样可以使你的程序更加易读，也可以避免名称的冲突。

# 模块的 __name__

# 模块的**name**

每个模块都有一个名称，在模块中可以通过语句来找出模块的名称。这在一个场合特别有用——就如前面所提到的，当一个模块被第一次输入的时候，这个模块的主块将被运行。假如我们只想在程序本身被使用的时候运行主块，而在它被别的模块输入的时候不运行主块，我们该怎么做呢？这可以通过模块的**name**属性完成。

```py
#!/usr/bin/python
# Filename: using_name.py

if __name__ == '__main__':
    print 'This program is being run by itself'
else:
    print 'I am being imported from another module' 
```

（源文件：code/using_name.py）

## 输出

```py
$ python using_name.py
This program is being run by itself
$ python
>>> import using_name
I am being imported from another module
>>> 
```

## 它如何工作

每个 Python 模块都有它的`__name__`，如果它是`'__main__'`，这说明这个模块被用户单独运行，我们可以进行相应的恰当操作。

# 制造你自己的模块

创建你自己的模块是十分简单的，你一直在这样做！每个 Python 程序也是一个模块。你已经确保它具有`.py`扩展名了。下面这个例子将会使它更加清晰。

```py
#!/usr/bin/python
# Filename: mymodule.py

def sayhi():
    print 'Hi, this is mymodule speaking.'

version = '0.1'

# End of mymodule.py 
```

（源文件：code/mymodule.py）

上面是一个 模块 的例子。你已经看到，它与我们普通的 Python 程序相比并没有什么特别之处。我们接下来将看看如何在我们别的 Python 程序中使用这个模块。

记住这个模块应该被放置在我们输入它的程序的同一个目录中，或者在`sys.path`所列目录之一。

```py
#!/usr/bin/python
# Filename: mymodule_demo.py

import mymodule

mymodule.sayhi()
print 'Version', mymodule.version 
```

（源文件：code/mymodule_demo.py）

## 输出

```py
$ python mymodule_demo.py
Hi, this is mymodule speaking.
Version 0.1 
```

## 它如何工作

注意我们使用了相同的点号来使用模块的成员。Python 很好地重用了相同的记号来，使我们这些 Python 程序员不需要不断地学习新的方法。

下面是一个使用`from..import`语法的版本。

```py
#!/usr/bin/python
# Filename: mymodule_demo2.py

from mymodule import sayhi, version
# Alternative:
# from mymodule import *

sayhi()
print 'Version', version 
```

（源文件：code/mymodule_demo2.py）

`mymodule_demo2.py`的输出与`mymodule_demo.py`完全相同。

# dir()函数

你可以使用内建的`dir`函数来列出模块定义的标识符。标识符有函数、类和变量。

当你为`dir()`提供一个模块名的时候，它返回模块定义的名称列表。如果不提供参数，它返回当前模块中定义的名称列表。

```py
$ python
>>> import sys
>>> dir(sys) # get list of attributes for sys module
['__displayhook__', '__doc__', '__excepthook__', '__name__', '__stderr__',
'__stdin__', '__stdout__', '_getframe', 'api_version', 'argv',
'builtin_module_names', 'byteorder', 'call_tracing', 'callstats',
'copyright', 'displayhook', 'exc_clear', 'exc_info', 'exc_type',
'excepthook', 'exec_prefix', 'executable', 'exit', 'getcheckinterval',
'getdefaultencoding', 'getdlopenflags', 'getfilesystemencoding',
'getrecursionlimit', 'getrefcount', 'hexversion', 'maxint', 'maxunicode',
'meta_path','modules', 'path', 'path_hooks', 'path_importer_cache',
'platform', 'prefix', 'ps1', 'ps2', 'setcheckinterval', 'setdlopenflags',
'setprofile', 'setrecursionlimit', 'settrace', 'stderr', 'stdin', 'stdout',
'version', 'version_info', 'warnoptions']
>>> dir() # get list of attributes for current module
['__builtins__', '__doc__', '__name__', 'sys']
>>>
>>> a = 5 # create a new variable 'a'
>>> dir()
['__builtins__', '__doc__', '__name__', 'a', 'sys']
>>>
>>> del a # delete/remove a name
>>>
>>> dir()
['__builtins__', '__doc__', '__name__', 'sys']
>>> 
```

## 它如何工作

首先，我们来看一下在输入的`sys`模块上使用`dir`。我们看到它包含一个庞大的属性列表。

接下来，我们不给`dir`函数传递参数而使用它——默认地，它返回当前模块的属性列表。注意，输入的模块同样是列表的一部分。

为了观察`dir`的作用，我们定义一个新的变量`a`并且给它赋一个值，然后检验`dir`，我们观察到在列表中增加了以上相同的值。我们使用`del`语句删除当前模块中的变量/属性，这个变化再一次反映在`dir`的输出中。

关于`del`的一点注释——这个语句在运行后被用来 删除 一个变量/名称。在这个例子中，`del a`，你将无法再使用变量`a`——它就好像从来没有存在过一样。

# 概括

模块的用处在于它能为你在别的程序中重用它提供的服务和功能。Python 附带的标准库就是这样一组模块的例子。我们已经学习了如何使用这些模块以及如何创造我们自己的模块。

接下来，我们将学习一些有趣的概念，它们称为数据结构。
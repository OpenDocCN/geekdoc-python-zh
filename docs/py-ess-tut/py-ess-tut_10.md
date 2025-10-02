# 第十章 自带电池

> 来源：[`www.cnblogs.com/Marlowes/p/5459376.html`](http://www.cnblogs.com/Marlowes/p/5459376.html)
> 
> 作者：Marlowes

现在已经介绍了 Python 语言的大部分基础知识。Python 语言的核心非常强大，同时还提供了更多值得一试的工具。Python 的标准安装中还包括一组模块，称为*标准库*(standard library)。之前已经介绍了一些模块(例如`math`和`cmath`，其中包括了用于计算实数和复数的数学函数)，但是标准库还包含其他模块。本章将向读者展示这些模块的工作方式，讨论如何分析它们，学习它们所提供的功能。本章后面的内容会对标准库进行概括，并且着重介绍一部分有用的模块。

## 10.1 模块

现在你已经知道如何创建和执行自己的程序(或脚本)了，也学会了怎么用`import`从外部模块获取函数并且为自己的程序所用：

```py
>>> import math
>>> math.sin(0)
0.0 
```

让我们来看看怎样编写自己的模块。

### 10.1.1 模块是程序

任何 Python 程序都可以作为模块导入。假设你写了一个代码清单 10-1 所示的程序，并且将它保存为`hello.py`文件(名字很重要)。

```py
代码清单 10-1 一个简单的模块
# hello.py
print "Hello, world!" 
```

程序保存的位置也很重要。下一节中你会了解更多这方面的知识，现在假设将它保存在`C:\python`(Windows)或者`~/python`(UNIX/Mac OS X)目录中，接着就可以执行下面的代码，告诉解释器在哪里寻找模块了(以 Windows 目录为例)：

```py
>>> import sys
>>> sys.path.append("c:/python") 
```

*注：在 UNIX 系统中，不能只是简单地将字符串`"~/python"`添加到`sys.path`中，必须使用完整的路径(例如`/home/yourusername/python`)。如果你希望将这个操作自动完成，可以使用`sys.path.expanduser("~/python")`。*

我这里所做的知识告诉解释器：除了从默认的目录中寻找之外，还需要从目录`c:\python`中寻找模块。完成这个步骤之后，就能导入自己的模块了(存储在`c:\python\hello.py`文件中)：

```py
>>> import hello
Hello, world! 
```

*注：在导入模块的时候，你可能会看到有新文件出现——在本例中是`c:\python\hello.pyc`。这个以`.pyc`为扩展名的文件是(平台无关的)经过处理(编译)的，已经转换成 Python 能够更加有效地处理的文件。如果稍后导入同一个模块，Python 会导入`.pyc`文件而不是`.py`文件，除非`.py`文件已改变，在这种情况下，会生成新的`.pyc`文件。删除`.pyc`文件不会损害程序(只要等效的`.py`文件存在即可)——必要的时候系统还会创建新的`.pyc`文件。*

如你所见，在导入模块的时候，其中的代码被执行了。不过，如果再次导入该模块，就什么都不会发生了：

```py
>>> import hello
>>> 
```

为什么这次没用了？因为导入模块并不意味着在导入时执行某些操作(比如打印文本)。它们主要用于*定义*，比如变量、函数和类等。此外，因为只需要定义这些东西一次，导入模块多次和导入一次的效果是一样的。

**为什么只是一次**

这种“只导入一次”(import-only-once)的行为在大多数情况下是一种实质性优化，对于一下情况尤其重要：两个模块互相导入。

在大多数情况下，你可能会编写两个互相访问函数和类的模块以便实现正确的功能。举例来说，假设创建了两个模块——`clientdb`和`billing`——分别包含了用于客户端数据库和计费系统的代码。客户端数据库可能需要调用计费系统的功能(比如每月自动将账单发送给客户)，而计费系统可能也需要访问客户端数据库的功能，以保证计费准确。

如果每个模块都可以导入数次，那么就出问题了。模块`clientdb`会导入`billing`，而`billing`又导入`clientdb`，然后`clientdb`又······你应该能想象到这种情况。这个时候导入就成了无限循环。(无限递归，记得吗？)但是，因为在第二次导入模块的时候什么都不会发生，所以循环会终止。

如果坚持重新载入模块，那么可以使用内建的`reload`函数。它带有一个参数(需要重新载入的模块)，并且返回重新载入的模块。如果你在程序运行的时候更改了模块并且希望将这些更改反应出来，那么这个功能会比较有用。要重新载入`hello`模块(只包含一个`print`语句)，可以像下面这样做：

```py
>>> hello = reload(hello)
Hello, world! 
```

这里假设`hello`已经被导入过(一次)。那么，通过将`reload`函数的返回值赋给`hello`，我们使用重新载入的版本替换了原先的版本。如你所见，问候语已经打印出来了，在此我完成了模块的导入。

如果你已经通过实例化`bar`模块中的`Foo`类创建了一个对象`x`，然后重新载入 bar 模块，那么不管通过什么方式都无法重新创建引用`bar`的对象`x`，`x`仍然是旧版本`Foo`类的实例(源自旧版本的`bar`)。如果需要 x 基于重新载入的模块`bar`中的新`Foo`类进行创建，那么你就得重新创建它了。

注意，Python3.0 已经去掉了`reload`函数。尽管使用`exec`能够实现同样的功能，但是应该尽可能避免重新载入模块。

### 10.1.2 模块用于定义

综上所述，模块在第一次导入到程序中时被执行。这看起来有点用——但并不算很有用。真正的用处在于它们(像类一样)可以保持自己的作用域。这就意味着定义的所有类和函数以及赋值后的变量都成为了模块的特性。这看起来挺复杂的，用起来却很简单。

1\. 在模块中定义函数

假设我们编写了一个类似代码清单 10-2 的模块，并且将它存储为 hello2.py 文件。同时，假设我们将它放置到 Python 解释器能够找到的地方——可以使用前一节中的`sys.path`方法，也可以用 10.1.3 节中的常规方法。

注：如果希望模块能够像程序一样被执行(这里的程序是用于执行的，而不是真正作为模块使用的)，可以对 Python 解释器使用`-m`切换开关来执行程序。如果`progname.py`(注意后缀)文件和其他模块都已被安装(也就是导入了`progname`)，那么运行 python `-m progname args`命令就会运行带命令行参数`args`的`progname`程序。

```py
代码清单 10-2 包含函数的简单模块
# hello2.py
def hello():
    print "Hello, world!" 
```

可以像下面这样导入：

```py
>>> import hello2 
```

模块就会被执行，这意味着`hello`函数在模块的作用域被定义了。因此可以通过以下方式来访问函数：

```py
>>> hello2.hello()
Hello, world! 
```

我们可以通过同样的方法来使用如何在模块的全局作用域中定义的名称。

我们为什么要这样做呢？为什么不在主程序中定义好一切呢？主要原因是*代码重用*(code reuse)。如果把代码放在模块中，就可以在多个程序中使用这些代码了。这意味着如果编写了一个非常棒的客户端数据库，并且将它放在叫做`clientdb`的模块中，那么你就可以在计费的时候、发送垃圾邮件的时候（当然我可不希望你这么做）以及任何需要访问客户数据的程序中使用这个模块了。如果没有将这段代码放在单独的模块中，那么就需要在每个程序中重写这些代码了。因此请记住：为了让代码可重用，请将它模块化！(是的，这当然也关乎抽象)

2\. 在模块中增加测试代码

模块被用来定义函数、类和其他一些内容，但是有些时候(事实上是经常)，在模块中添加一些检查模块本身是否能正常工作的测试代码是很有用的。举例来说，假如想要确保`hello`函数正常工作，你可能会将`hello2`模块重写为新的模块——代码清单 10-3 中定义的`hello3`。

```py
# hello3.py
def hello():
    print "Hello, world!"

# A test
hello() 
```

这看起来是合理的，如果将它作为普通程序运行，会发现它能够正常工作。但如果将它作为模块导入，然后在其他程序中使用 hello 函数，测试代码就会被执行，就像本章实验开头的第一个`hello`模块一样：

```py
>>> import hello3
Hello, world!
>>> hello3.hello.()
Hello, world！ 
```

这个可不是你想要的。避免这种情况关键在于：“告知”模块本身是作为程序运行还是导入到其他程序。为了实现这一点，需要使用`__name__`变量：

```py
>>> __name__
'__main__'
>>> hello3.__name__
'hello3' 
```

如你所见，在“主程序”(包括解释器的交互式提示符在内)中，变量`__name__`的值是`'__main__'`。而在导入的模块中，这个值就被设定为模块的名字。因此，为了让模块的测试代码更加好用，可以将其放置在 if 语句中，如代码清单 10-4 所示。

```py
代码清单 10-4 使用条件测试代码的模块 # hello4.py

def hello():
    print "Hello, world!"

def test():
    hello()
if __name__ = '__main__':
    test() 
```

如果将它作为程序运行，`hello`函数会被执行。而作为模块导入时，它的行为就会像普通模块一样：

```py
>>> import hello4
>>> hello4.hello()
Hello, world! 
```

如你所见，我将测试代码放在了`test`函数中，也可以直接将它们放入`if`语句。但是，将测试代码放入独立的`test`函数会更灵活，这样做即使在把模块导入其他程序之后，仍然可以对其进行测试：

```py
>>> hello4.test()
Hello, world! 
```

*注：如果需要编写更完整的测试代码，将其放置在单独的程序中会更好。关于编写测试代码的更多内容，参见第十六章。*

### 10.1.3 让你的模块可用

前面的例子中，我改变了`sys.path`，其中包含了(字符串组成的)一个目录列表，解释器在该列表中查找模块。然而一般来说，你可能不想这么做。在理想情况下，一开始`sys.path`本身就应该包含正确的目录(包括模块的目录)。有两种方法可以做到这一点：一是将模块放置在合适的位置，另外则是告诉解释器去哪里查找需要的模块。下面几节将探讨这两种方法。

1\. 将模块放置在正确位置

将模块放置在正确位置(或者说某个正确位置，因为会有多种可能性)是很容易的。只需要找出 Python 解释器从哪里查找模块，然后将自己的文件放置在那里即可。

*注：如果机器上面的 Python 解释器是由管理员安装的，而你又没有管理员权限，可能无法将模块存储在 Python 使用的目录中。这种情况下，你需要使用另外一个解决方案：告诉解释器去那里查找。*

你可能记得，那些(成为搜索路径的)目录的列表可以在`sys`模块中的`path`变量中找到：

```py
>>> import sys, pprint
>>> pprint.pprint(sys.path)
['', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/python2.7/dist-packages/ubuntu-sso-client'] 
```

*注：如果你的数据结构过大，不能在一行打印完，可以使用`pprint`模块中的`pprint`函数替代普通的`print`语句。`pprint`是个相当好的打印函数，能够提供更加智能的打印输出。*

这是安装在 elementary OS 上的 Python2.7 的标准路径，不同的系统会有不同的结果。关键在于每个字符串都提供了一个放置模块的目录，解释器可以从这些目录中找到所需的模块。尽管这些目录都可以使用，但`site-packages`目录是最佳的选择，因为它就是用来做这些事情的。查看你自己的`sys.path`，找到`site-packages`目录，将代码清单 10-4 的模块存储在其中，要记得改名，比如改成`another_hello.py`，然后测试：

```py
>>> import another_hello
>>> another_hello.hello()
Hello, world! 
```

只要将模块放入类似`site-packages`这样的目录中，所有程序就都能将其导入了。

2\. 告诉编译器去那里找

“将模块放置在正确的位置”这个解决方案对于以下几种情况可能并不适用：

☑ 不希望将自己的模块填满 Python 解释器的目录；

☑ 没有在 Python 解释器目录中存储文件的权限；

☑ 想将模块放在其他地方。

最后一点是“想将模块放在其他地方”，那么就要告诉解释器去哪里找。你之前已经看到了一种方法，就是编辑`sys.path`，但这不是通用的方法。标准的实现方法是在`PYTHONPATH`环境变量中包含模块所在的目录。

`PYTHONPATH`环境变量的内容会因为使用的操作系统不同而有所差异(参见下面的“环境变量”)，但基本上来说，它与`sys.path`很类似——一个目录列表。

**环境变量**

环境变量并不是 Python 解释器的一部分——它们是操作系统的一部分。基本上，它相当于 Python 变量，不过是在 Python 解释器外设置的。有关设置的方法，你应该参考操作系统文档，这里只给出一些相关提示。

在 UNIX 和 Mac OS X 中，你可以在一些每次登陆都要执行的 shell 文件内设置环境变量。如果你使用类似 bash 的 shell 文件，那么要设置的就是`.bashrc`，你可以在主目录中找到它。将下面的命令添加到这个文件中，从而将`~/python`加入到`PYTHONPATH`：

```py
export PYTHON=$PYTHONPATH:~/python 
```

注意，多个路径以冒号分隔。其他的 shell 可能会有不同的语法，所以你应该参考相关的文档。

对于 Windows 系统，你可以使用控制面板编辑变量(适用于高级版本的 Windows，比如 Windows XP、2000、NT 和 Vista，旧版本的，比如 Windows 98 就不适用了，而需要修改`autoexec.bat`文件，下段会讲到)。依次点击开始菜单→设置→控制面板。进入控制面板后，双击“系统”图标。在打开的对话框中选择“高级”选项卡，点击“环境变量”按钮。这时会弹出一个分为上下两栏的对话框：其中一栏是用户变量，另外一栏就是系统变量，需要修改的是用户变量。如果你看到其中已经有`PYTHONPATH`项，那么选中它，单击“编辑”按钮进行编辑。如果没有，单击“新建”按钮，然后使用`PYTHONPATH`作为“变量名”，输入目录作为“变量值”。注意，多个目录以分号分分隔。

如果上面的方法不行，你可以编辑`autoexec.bat`文件，该文件可以在 C 盘的根目录下找到(假设是以标准模式安装的 Windows)。用记事本(或者 IDLE 编辑器)打开它，增加一行设置`PYTHONPATH`的内容。如果想要增加目录`C:\pyrhon`。可以像下面这样做：

```py
set PYTHONPATH=%PYTHONPATH%;C:\python 
```

注意，你所使用的 IDE 可能会有自身的机制，用于设置环境变量和 Python 路径。

*注：你不需要使用`PYTHONPATH`来更改`sys.path`。路径配置文件提供了一个有用的捷径，可以让 Python 替你完成这些工作。路径配置文件是以`.pth`为扩展名的文件，包括应该添加到`sys.path`中的目录信息。空行和以`#`开头的行都会被忽略。以`import`开头的文件会被执行。为了执行路径配置文件，需要将其放置在可以找到的地方。对于 Windows 来说，使用`sys.prefix`定义的目录名(可能类似于`C:\Python22`)；在 UNIX 和 Mac OS X 中则使用`site-packages`目录(更多信息可以参见 Python 库参考中`site`模块的内容，这个模板在 Python 解释器初始化时会自动导入)。*

3.命名模块

你可能注意到了，包含模块代码的文件的名字要和模块名一样，再加上`.py`扩展名。在 Windows 系统中，你也可以使用`.pyw`扩展名。有关文件扩展名含义的更多信息请参见第十二章。

### 10.1.4 包

为了组织好模块，你可以将它们分组为*包*(package)。包基本上就是另外一个类模块，有趣的地方就是它们能包含其他模块。当模块存储在文件中时(扩展名`.py`)，包就是模块所在的目录。为了让 Python 将其作为包对待，它必须包含一个命名为`__init__.py`的文件(模块)。如果将它作为普通模块导入的话，文件的内容就是包的内容。比如有个名为`constants`的包，文件`constants/__init__.py`包括语句`PI=3.14`，那么你可以像下面这么做：

```py
import constants
print constants.PI 
```

为了将模块放置在包内，直接把模块放在包目录内即可。

比如，如果要建立一个叫做`drawing`的包，其中包括名为`shapes`和`colors`的模块，你就需要创建表 10-1 中所示的文件和目录(UNIX 路径名)。

表 10-1 简单的包布局

```py
~/python/　　　　　　　　　　　　　　　　　PYTHONPATH 中的目录
~/python/drawing/　　　　　　　　　　　　　包目录(drawing 包)
~/python/drawing/__init__.py　　　　　　　 包代码(drawing 模块)
~/python/drawing/colors.py　　　　　　　　 colors 模块
~/python/drawing/shapes.py　　　　　　　　 shapes 模块 
```

对于表 10-1 中的内容，假定你已经将目录`~/python`放置在`PYTHONPATH`。在 Windows 系统中，只要用`C:\python`替换`~/python`，并且将正斜线为反斜线即可。

依照这个设置，下面的语句都是合法的：

```py
import drawing                    # (1) Imports the drawing package
import drawing.colors             # (2) Imports the colors module
from drawing import shapes        # (3) Imports the shapes module 
```

在第 1 条语句`drawing`中`__init__`模块的内容是可用的，但`drawing`和`colors`模块则不可用。在执行第 2 条语句之后，`colors`模块可用了，可以通过短名(也就是仅使用`shapes`)来使用。注意，这些语句只是例子，执行顺序并不是必需的。例如，不用像我一样，在导入包的模块前导入包本身，第 2 条语句可以独立使用，第 3 条语句也一样。我们还可以在包之间进行嵌套。

## 10.2 探究模块

在讲述标准库模块前，先教你如何独立地探究模块。这种技能极有价值，因为作为 Python 程序员，在职业生涯中可能会遇到很多有用的模块，我又不能在这里一一介绍。目前的标准库已经大到可以出本书了(事实上已经有这类书了)，而且它还在增长。每次新的模块发布后，都会添加到标准库，一些模块经常发生一些细微的变化和改进。同时，你还能在网上找到些有用的模块并且可以很快理解(grok)它们，从而让编程轻而易举地称为一种享受。

### 10.2.1 模块中有什么

探究模块最直接的方式就是在 Python 解释器中研究它们。当然，要做的第一件事就是导入它。假设你听说有个叫做`copy`的标准模块：

```py
>>> import copy 
```

没有引发异常，所以它是存在的。但是它能做什么？它又有什么？

1\. 使用 dir

查看模块包含的内容可以使用`dir`函数，它会将对象的所有特性(以及模块的所有函数、类、变量等)列出。如果想要打印出`dir(copy)`的内容，你会看到一长串的名字(试试看)。一些名字以下划线开始，暗示(约定俗成)它们并不是为在模块外部使用而准备的。所以让我们用列表推导式(如果不记得如何使用了，请参见 5.6 节)过滤掉它们：

```py
>>> [n for n in dir(copy) if not n.startswith("_")]
['Error', 'PyStringMap', 'copy', 'deepcopy', 'dispatch_table', 'error', 'name', 't', 'weakref'] 
```

这个列表推导式是个包含`dir(copy)`中所有不以下划线开头的名字的列表。它比完整列表要清楚些。(如果喜欢用`tab`实现，那么应该看看库参考中的`readline`和`rlcompleter`模块。它们在探究模块时很有用)

2\. `__all__`变量

在上一节中，通过列表推导式所做的事情是推测我可能会在`copy`模块章看到什么。但是我们可以直接从列表本身获得正确答案。在完整的`dir(copy)`列表中，你可能注意到了`__all__`这个名字。这个变量包含一个列表，该列表与我之前通过列表推导式创建的列表很类似——除了这个列表在模块本身中已被默认设置。我们来看看它都包含哪些内容：

```py
>>> copy.__all__
['Error', 'copy', 'deepcopy'] 
```

我的猜测还不算太离谱吧。列表推导式得到的列表只是多出了几个我用不到的名字。但是`__all__`列表从哪来，它为什么会在那儿？第一个问题很容易回答。它是在`copy`模块内部被设置的，像下面这样(从`copy.py`直接复制而来的代码)：

```py
__all__ =
["Error", "copy", "deepcopy"] 
```

那么它为什么在那呢？它定义了模块的公有接口(public interface)。更准确地说，它告诉解释器：从模块导入所有名字代表什么含义。因此，如果你使用如下代码：

```py
from copy import * 
```

那么，你就能使用`__all__`变量中的 4 个函数。要导入`PyStringMap`的话，你就得显示地实现，或者导入`copy`然后使用`copy.PyStringMap`，或者使用`from copy import PyStringMap`。

在编写模块的时候，像设置`__all__`这样的技术是相当有用的。因为模块中可能会有一大堆其他程序不需要或不想要的变量、函数和类，`__all__`会“客气地”将它们过滤了出去。如果没有设定`__all__`，用`import *`语句默认将会导入模块中所有不以下划线开头的全局名称。

### 10.2.2 用`help`获取帮助

目前为止，你已经通过自己的创造力和 Python 的多个函数和特殊特性的知识探究了`copy`模块。对于这样的探究工作，交互式解释器是个非常强大的工具，而对该语言的精通程度决定了对模块探究的深度。不过，还有个标准函数能够为你提供日常所需的信息，这个函数叫做`help`。让我们先用`copy`函数试试：

```py
>>> help(copy.copy)
Help on function copy in module copy:

copy(x)
    Shallow copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info. 
```

这些内容告诉你：`copy`带有一个参数 x，并且是“浅复制操作”。但是它还提到了模块的`__doc__`字符串。这是什么呢？你可能记得第六章提到的文档字符串，它就是写在函数开头并且简述函数功能的字符串，这个字符串可以通过函数的`__doc__`特性引用。就像从上面的帮助文本中所理解到的一样，模块也可以有文档字符串(写在模块开头)，类也一样(写在类开头)。

事实上，前面的帮助文本是从`copy`函数的文档字符串中取出的。

```py
>>> print copy.copy.__doc__
Shallow copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info. 
```

使用`help`与直接检查文档字符串相比，它的好处在于会获得更多信息，比如函数签名(也就是所带的参数)。试着调用`help(copy)`(对模块本身)看看得到什么。它会打印出很多信息，包括`copy`和`deepcopy`之间区别的透彻的讨论(从本质来说，`deepcopy(x)`会将存储在`x`中的值作为属性进行复制，而`copy(x)`只是复制 x，将 x 中的值绑定到副本的属性上)。

### 10.2.3 文档

模块信息的自然来源当然是文件。我把对文档的讨论推后在这里，是因为自己先检查模块总是快一些。举例来说，你可能会问“`range`的参数是什么”。不用在 Python 数据或者标准 Python 文档中寻找有关`range`的描述，而是可以直接查看：

```py
>>> print range.__doc__
range(stop) -> list of integers
range(start, stop[, step]) -> list of integers

Return a list containing an arithmetic progression of integers.
range(i, j) returns [i, i+1, i+2, ..., j-1]; start (!) defaults to 0.
When step is given, it specifies the increment (or decrement).
For example, range(4) returns [0, 1, 2, 3].  The end point is omitted!
These are exactly the valid indices for a list of 4 elements. 
```

这样就获得了关于`range`函数的精确描述，因为 Python 解释器可能已经运行了(在编程的时候，经常会像这样怀疑函数的功能)，访问这些信息花不了几秒钟。

但是，并非每个模块和函数都有不错的文档字符串(尽管都应该有)，有些时候可能需要十分透彻地描述这些模块和函数是如何工作的。大多数从网上下载的模块都有相关的文档。在我看来，学习 Python 编程最有用的文档莫过于 Python 库参考，它对所有标准库中的模块都有描述。如果想要查看 Python 的知识。十有八九我都会去查阅它。[库参考](http://python.org/doc/lib)可以在线浏览，并且提供下载，其他一些标准文档(比如 Python 指南或者 Python 语言参考)也是如此。所有这些文档都可以在[Python 网站上](http://python.org/doc)找到。

### 10.2.4 使用源代码

到目前为止，所讨论的探究技术在大多数情况下都已经够用了。但是，对于希望真正理解 Python 语言的人来说，要了解模块，是不能脱离源代码的。阅读源代码，事实上是学习 Python 最好的方式，除了自己编写代码外。

真正的阅读不是问题，但是问题在于源代码在哪里。假设我们希望阅读标准模块`copy`的源代码，去哪里找呢？一种方案是检查`sys.pat`h，然后自己找，就像解释器做的一样。另外一种快捷的方法是检查模块的`__file__`属性：

```py
>>> print copy.__file__
C:\Python27\lib\copy.pyc 
```

*注：如果文件名以`.pyc`结尾，只要查看对应的以`.py`结尾的文件即可。*

就在那！你可以使用代码编辑器打开`copy.py`(比如 IDLE)，然后查看它是如何工作的。

*注：在文本编辑器中打开标准库文件的时候，你也承担着意外修改它的风险。这样做可能会破坏它，所以在关闭文件的时候，你必须确保没有保存任何可能做出的修改。*

注意，一些模块并不包含任何可以阅读的 Python 源代码。它们可能已经融入到解释器内了(比如`sys`模块)，或者可能是使用 C 程序语言写成的(如果模块是使用 C 语言编写的，你也可以查看它的 C 源代码)。(请查看第十七章以获得更多使用 C 语言扩展 Python 的信息)

## 10.3 标准库：一些最爱

有的读者会觉得本章的标题不知所云。“充电时刻”(batteries included)这个短语最开始由 Frank Stajano 创造，用于描述 Python 丰富的标准库。安装 Python 后，你就“免费”获得了很多有用的模块(充电电池)。因为获得这些模块的更多信息的方式很多(在本章的第一部分已经解释过了)，我不会在这里列出完整的参考资料(因为要占去很大篇幅)，但是我会对一些我最喜欢的标准模块进行说明，从而激发你对模块进行探究的兴趣。你会在“项目章节”(第二十章~第二十九章)碰到更多的标准模块。模块的描述并不完全，但是会强调每个模块比较有趣的特征。

### 10.3.1 `sys`

`sys`这个模块让你能够访问与 Python 解释器联系紧密的变量和函数，其中一些在表 10-2 中列出。

表 10-2 `sys`模块中一些重要的函数和变量

```py
argv　　　　　　　　　　　　　　　命令行参数，包括脚本名称
exit([arg])　　　　　　　　　　　 退出当前的程序，可选参数为给定的返回值或者错误信息
modules　　　　　　　　　　　　　 映射模块名字到载入模块的字典
path　　　　　　　　　　　　　　　查找模块所在目录的目录名列表
platform　　　　　　　　　　　　　类似 sunos5 或者 win32 的平台标识符
stdin　　　　　　　　　　　　　　 标准输入流——一个类文件(file-like)对象
stdout　　　　　　　　　　　　　　标准输出流——一个类文件对象
stderr　　　　　　　　　　　　　　标准错误流——一个类文件对象 
```

变量`sys.argv`包含传递到 Python 解释器的参数，包括脚本名称。

函数`sys.exit`可以退出当前程序(如果在`try/finally`块中调用，`finally`子句的内容仍然会被执行，第八章对此进行了探讨)。你可以提供一个整数作为参数，用来标识程序是否成功运行，这是 UNIX 的一个惯例。大多数情况下使用该整数的默认值就可以了(也就是 0，表示成功)。或者你也可以提供字符串参数，用作错误信息，这对于用户找出程序停止运行的原因会很有用。这样，程序就会在退出的时候提供错误信息和标识程序运行失败的代码。

映射`sys.modules`将模块名映射到实际存在的模块上，它只应用于目前导入的模块。

`sys.path`模块变量在本章前面讨论过，它是一个字符串列表，其中的每个字符串都是一个目录名，在`import`语句执行时，解释器就会从这些目录中查找模块。

`sys.platform`模块变量(它是个字符串)是解释器正在其上运行的“平台”名称。它可能是标识操作系统的名字(比如`sunos5`或`win32`)，也可能标识其他种类的平台，如果运行 Jython 的话，就是 Java 的虚拟机(比如`java1.4.0`)。

`sys.stdin`、`sys.stdout`和`sys.stderr`模块变量是类文件流对象。它们表示标准 UNIX 概念中的标准输入、标准输出和标准错误。简单来说，Python 利用`sys.stdin`获得输入(比如用于函数`input`和`raw_input`中的输入)，利用`sys.stdout`输出。第十一章会介绍更多有关于文件(以及这三个流)的知识。

举例来说，我们思考一下反序打印参数的问题。当你通过命令行调用 Python 脚本时，可能会在后面加上一些参数——这就是*命令行参数*(command-line argument)。这些参数会放置在`sys.argv`列表中，脚本的名字为`sys.argv[0]`。反序打印这些参数很简单，如代码清单 10-5 所示。

```py
# 代码清单 10-5 反序打印命令行参数

import sys

args = sys.argv[1:]
args.reverse()
print " ".join(args) 
```

正如你看到的，我对`sys.argv`进行了复制。你可以修改原始的列表，但是这样做通常是不安全的，因为程序的其他部分可能也需要包含原始参数的`sys.argv`。注意，我跳过了`sys.argv`的第一个元素，这是脚本的名字。我使用`args.reverse()`方法对列表进行反向排序，但是不能打印出这个操作结果的，这是个返回`None`的原地修改操作。下面是另外一种做法：

```py
print " ".join(reversed(sys.argv[1:])) 
```

最后，为了保证输出得更好，我使用了字符串方法`join`。让我们试试看结果如何(我使用的是 MS-DOS，在 UNIX Shell 下它也会工作的同样好)：

```py
D:\Workspace\Basic tutorial>python Code10-5.py
this is a test
test a is this 
```

### 10.3.2 `os`

`os`模块提供了访问多个操作系统服务的功能。`os`模块包括的内容很多，表 10-3 中只是其中一些最有用的函数和变量。另外，`os`和它的子模块`os.path`还包括一些用于检查、构造、删除目录和文件的函数，以及一些处理路径的函数(例如，`os.path.split`和`os.path.join`让你在大部分情况下都可以忽略`os.pathsep`)。关于它的更多信息，请参见标准库文档。

表 10-3 `os`模块中一些重要函数和变量

```py
environ　　　　　　　　　　　　　　　　　　　　对环境变量进行映射
system(command)　　　　　　　　　　　　　　    在子 shell 中执行操作系统命令
sep　　　　　　　　　　　　　　　　　　　　　　路径中的分隔符
pathsep　　　　　　　　　　　　　　　　　　　  分隔路径的分隔符
linesep　　　　　　　　　　　　　　　　　　　　行分隔符("\n", "\r", or "\r\n")
urandom(n)　　　　　　　　　　　　　　　　　　 返回 n 字节的加密强随机数据 
```

`os.environ`映射包含本章前面讲述过的环境变量。比如要访问系统变量`PYTHONPATH`，可以使用表达式`os.environ["PYTHONPATH"]`。这个映射也可以用来更改系统环境变量，不过并非所有系统都支持。

`os.system`函数用于运行外部程序。也有一些函数可以执行外部程序。还有`open`，它可以创建与程序连接的类文件。

关于这些函数的更多信息，请参见标准库文档。

*注：当前版本的 Python 中，包括`subprocess`模块，它包括了`os.system`、`execv`和`open`函数的功能。*

`os.sep`模块变量是用于路径名字中的分隔符。UNIX(以及 Mac OS X 中命令行版本的 Python)中的标准分隔符是`"/"`，Windows 中的是`"\\"`(即 Python 针对单个反斜线的语法)，而 Mac OS 中的是`":"`(有些平台上，`os.altsep`包含可选的路径分隔符，比如 Windows 中的`"/"`)。

你可以在组织路径的时候使用`os.pathsep`，就像在`PYTHONPATH`中一样。`pathsep`用于分割路径名：UNIX(以及 Mac OS X 中的命令行版本的 Python)使用`":"`，Windows 使用`";"`，Mac OS 使用`"::"`。

模块变量`os.linesep`用于文本文件的字符串分隔符。UNIX 中(以及 Mac OS X 中命令行版本的 Python)为一个换行符(`\n`)，Mac OS 中为单个回车符(`\r`)，而在 Windows 中则是两者的组合(`\r\n`)。

`urandom`函数使用一个依赖于系统的"真"(至少是足够强度加密的)随机数的源。如果正在使用的平台不支持它，你会得到`NotImplementedError`异常。

例如，有关启动网络浏览器的问题。`system`这个命令可以用来执行外部程序，这在可以通过命令行执行程序(或命令)的环境中很有用。例如在 UNIX 系统中，你可以用它来列出某个目录的内容以及发送 Email，等等。同时，它对在图形用户界面中启动程序也很有用，比如网络浏览器。在 UNIX 中，你可以使用下面的代码(假设`/usr/bin/firefox`路径下有一个浏览器)：

```py
os.system("/usr/bin/firefox") 
```

以下是 Windows 版本的调用代码(也同样假设使用浏览器的安装路径)：

```py
os.system(r"C:\'Program Files'\'Mozilla Firefox'\firefox.exe") 
```

注意，我很仔细地将`Program Files`和`Mozilla Firefox`放入引号中，不然 DOS(它负责处理这个命令)就会在空格处停不下来(对于在`PYTHONPATH`中设定的目录来说，这点也同样重要)。同时，注意必须使用反斜线，因为 DOS 会被正斜线弄糊涂。如果运行程序，你会注意到浏览器会试图打开叫做`Files'\Mozilla...`的网站——也就是在空格后面的命令部分。另一方面，如果试图在 IDLE 中运行该代码，你会看到 DOS 窗口出现了，但是没有启动浏览器并没有出现，除非关闭 DOS 窗口。总之，使用以上代码并不是完美的解决方法。

另外一个可以更好地解决问题的函数是 Windows 特有的函数——`os.startfile`：

```py
os.startfile(r"C:\Program Files\Mozilla Firefox\firefox.exe") 
```

可以看到，`os.startfile`接受一般路径，就算包含空格也没问题(也就是不用像在`os.system`例子中那样将`Program Files`放在引号中)。

注意，在 Windows 中，由`os.system`(或者`os.startfile`)启动了外部程序之后，Python 程序仍然会继续运行，而在 UNIX 中，程序则会中止，等待`os.system`命令完成。

**更好的解决方案：`WEBBROWSER`**

在大多数情况下，`os.system`函数很有用，但是对于启动浏览器这样特定的任务来说，还有更好的解决方案：`webbrowser`模块。它包括`open`函数，它可以自动启动 Web 浏览器访问给定的 URL。例如，如果希望程序使用 Web 浏览器打开 Python 的网站(启动新浏览器或者使用已经运行的浏览器)，那么可以使用以下代码：

```py
import webbrowser
webbrowser.open("http://www.python.org") 
```

### 10.3.3 `fileinput`

第十一章将会介绍很多读写文件的知识，现在先做个简短的介绍。`fileinput`模块让你能够轻松地遍历文本文件的所有行。如果通过以下方式调用脚本(假设在 UNIX 命令行下)：

```py
$ python some_script.py file1.txt file2.txt file3.txt 
```

这样就可以以此对`file1.txt`到`file3.txt`文件中的所有行进行遍历了。你还能对提供给标准输入(`sys.stdin`，记得吗)的文本进行遍历。比如在 UNIX 的管道中，使用标准的 UNIX 命令`cat`：

```py
$ cat file.txt | python some_script.py 
```

如果使用`fileinput`模块，在 UNIX 管道中使用`cat`来调用脚本的效果和将文件名作为命令行参数提供给脚本是一样的。`fileinput`模块最重要的函数如表 10-4 所示。

`fileinput.input`是其中最重要的函数。它会返回能够于`for`循环遍历的对象。如果不想使用默认行为(`fileinput`查找需要循环遍历的文件)，那么可以给函数提供(序列形式的)一个或多个文件名。你还能将`inplace`参数设为其真值(inplace=True)以进行原地处理。对于要访问的每一行，需要打印出替代的内容，以返回到当前的输入文件中。在进行原地处理的时候，可选的`backup`参数将文件名扩展备份到通过原始文件创建的备份文件中。

表 10-4 fileinput 模块中重要的函数

```py
input(files[, inplace[, backup]])　　　　　便于遍历多个输入流中的行
filename()　　　　　　　　　　　　　　　　 返回当前文件的名称
lineno()　　　　　　　　　　　　　　　　　 返回当前(累计)的行数
filelineno()　　　　　　　　　　　　　　　 返回当前文件的行数
isfirstline()　　　　　　　　　　　　　　  检查当前行是否是文件的第一行
isstdin()　　　　　　　　　　　　　　　　　检查最后一行是否来自 sys.stdin
nextfile()　　　　　　　　　　　　　　　　 关闭当前文件，移动到下一个文件
close()　　　　　　　　　　　　　　　　　　关闭序列 
```

`fileinput.filename`函数返回当前正在处理的文件名(也就是包含了当前正在处理的文本行的文件)。

`fileinput.lineno`返回当前行的行数。这个数值是累计的，所以在完成一个文件的处理并且开始处理下一个文件的时候，行数并不会重置。而是将上一个文件的最后行数加 1 作为计数的起始。

`fileinput.filelineno`函数返回当前处理文件的当前行数。每次处理完一个文件并且开始处理下一个文件时，行数都会重置为 1，然后重新开始计数。

`fileinput.isfirstline`函数在当前行是当前文件的第一行时返回真值，反之返回假值。

`fileinput.isstdin`函数在当前文件为`sys.stdin`时返回真值，否则返回假值。

`fileinput.nextfile`函数会关闭当前文件，跳到下一个文件，跳过的行并不计。在你知道当前文件已经处理完的情况下，这个函数就比较有用了——比如每个文件都包含经过排序的单词，而你需要查找某个词。如果已经在排序中找到了这个词的位置，那么你就能放心地跳到下一个文件了。

`fileinput.close`函数关闭整个文件链，结束迭代。

为了演示`fileinput`的使用，我们假设已经编写了一个 Python 脚本，现在想要为其代码进行编号。为了让程序在完成代码行编号之后仍然能够正常运行，我们必须通过在每一行的右侧加上作为注释的行号来完成编号工作。我们可以使用字符串格式化来将代码行和注释排成一行。假设每个程序行最多有 45 个字符，然后把行号注释加在后面。代码清单 10-6 展示了使用`fileinput`以及`inplace`参数来完成这项工作的简单方法：

```py
# 代码清单 10-6 为 Python 脚本添加行号

#!/usr/bin/env python
# coding=utf-8

# numberlines.py

import fileinput for line in fileinput.input(inplace=True):
    line = line.rstrip()
    num = fileinput.lineno()
    print "%-40s # %2i" % (line, num) 
```

如果你像下面这样在程序本身上运行这个程序：

```py
$ python numberline.py numberline.py 
```

程序会变成类似于代码清单 10-7 那样。注意，程序本身已经被更改了，如果这样运行多次，最终会在每一行中添加多个行号。我们可以回忆一下之前的内容：`rstrip`是可以返回字符串副本的字符串方法，右侧的空格都被删除(请参见 3.4 节，以及附录 B 中的表 B-6)。

```py
# 代码清单 10-7 为已编号的行进行编号

#!/usr/bin/env python                         # 1
# coding=utf-8                                # 2
                                              # 3
# numberline.py                               # 4
                                              # 5
import fileinput                              # 6
                                              # 7
for line in fileinput.input(inplace=True):    # 8
    line = line.rstrip()                      # 9
    num = fileinput.lineno()                  # 10
                                              # 11
    print "%-45s # %2i" % (line, num)         # 12 
```

*注：要小心使用`inplace`参数，它很容易破坏文件。你应该在不使用`inplace`设置的情况下仔细测试自己的程序(这样只会打印出错误)，在确保程序工作正常后再修改文件。*

另外一个使用`fileinput`的例子，请参见本章后面的`random`模块部分。

### 10.3.4 集合、堆和双端队列

在程序设计中，我们会遇到很多有用的数据结构，而 Python 支持其中一些相对通用的类型，例如字典(或者说散列表)、列表(或者说动态数组)是语言必不可少的一部分。其他一些数据结构尽管不是那么重要，但有些时候也能派上用场。

1\. 集合

集合(set)在 Python2.3 才引入。`Set`类位于`sets`模块中。尽管可以在现在的代码中创建`Set`实例。但是除非想要兼容以前的程序，否则没有什么必要这样做。在 Python2.3 中，集合通过`set`类型的实例成为了语言的一部分，这意味着不需要导入`sets`模块——直接创建集合即可：

```py
>>> set(range(10))
set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
```

集合是由序列(或者其他可迭代的对象)构建的。它们主要用于检查成员资格，因此副本是被忽略的：

```py
>>> set(["fee", "fie", "foe"])
set(['foe', 'fee', 'fie']) 
```

除了检查成员资格外，还可以使用标准的集合操作(可能你是通过数学了解到的)，比如求并集和交集，可以使用方法，也可以使用对整数进行位操作时使用的操作(参见附录 B)。比如想要找出两个集合的并集，可以使用其中一个集合的`union`方法或者使用按位与(OR)运算符`"|"`：

```py
>>> a = set([1, 2, 3])
>>> b = set([2, 3, 4])
>>> a.union(b)
set([1, 2, 3, 4])
>>> a | b
set([1, 2, 3, 4]) 
```

以下列出了一些其他方法和对应的运算符，方法的名称已经清楚地表明了其用途：

```py
>>> c = a & b
>>> c.issubset(a)
True
>>> c <= a
True
>>> c.issuperset(a)
False
>>> c >= a
False
>>> a.intersection(b)
set([2, 3])
>>> a & b
set([2, 3])
>>> a.difference(b)
set([1])
>>> a - b
set([1])
>>> a.symmetric_difference(b)
set([1, 4])
>>> a ^ b
set([1, 4])
>>> a.copy()
set([1, 2, 3])
>>> a.copy() is a
False 
```

还有一些原地运算符和对应的方法，以及基本方法`add`和`remove`。关于这方面更多的信息，请参看[Python 库参考的 3.7 节](http://python.org/doc/lib/types-set.html)。

*注：如果需要一个函数，用于查找并且打印两个集合的并集，可以使用来自`set`类型的`union`方法的未绑定版本。这种做法很有用，比如结合`reduce`来使用：*

```py
>>> mySets = [] >>> for i in range(10):
...     mySets.append(set(range(i, i + 5)))
...
>>> reduce(set.union, mySets)
set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) 
```

集合是可变的，所以不能用做字典中的键。另外一个问题就是集合本身只能包含不可变(可散列的)值，所以也就不能包含其他集合。在实际当中，集合的集合是很常用的，所以这个就是个问题了。幸好还有个`frozenset`类型，用于代表*不可变*(可散列)的集合：

```py
>>> a = set()
>>> b = set()
>>> a.add(b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  TypeError: unhashable type: 'set'
>>> a.add(frozenset(b))
>>> a
set([frozenset([])]) 
```

`frozenset`构造函数创建给定集合的副本，不管是将集合作为其他集合成员还是字典的键，`frozenset`都很有用。

2\. 堆

另外一个众所周知的数据结构是*堆*(heap)，它是优先队列的一种。使用优先队列能够以任意顺序增加对象，并且能在任何时间(可能增加对象的同时)找到(也可能是移除)最小的元素，也就是说它比用于列表的`min`方法要有效率得多。

事实上，Python 中并没有独立的堆类型，只有一个包含一些堆操作函数的模块，这个模块叫做`heapq`(`q`是`queue`的缩写，即队列)，包括 6 个函数(参见表 10-5)，其中前 4 个直接和堆操作相关。你必须将列表作为堆对象本身。

表 10-5 heapq 模块中重要的函数

```py
heappush(heap, x)　　　　　　　　　　　　　　　　　　　　将 x 入堆
heappop(heap)　　　　　　　　　　　　　　　　　　　　　　将堆中最小的元素弹出
heapify(heap)　　　　　　　　　　　　　　　　　　　　　　将 heap 属性强制应用到任意一个列表
heapreplace(heap, x)　　　　　　　　　　　　　　　　　　 将堆中最小的元素弹出，同时将 x 入堆
nlargest(n, iter)　　　　　　　　　　　　　　　　　　　　返回 iter 中第 n 大的元素
nsmallset(n, iter)　　　　　　　　　　　　　　　　　　　 返回 iter 中第 n 小的元素 
```

`heappush`函数用于增加堆的项。注意，不能将它用于任何之前讲述的列表中，它只能用于通过各种堆函数建立的列表中。原因是元素的顺序很重要(尽管看起来是随意排列，元素并不是进行严格排序的)。

```py
>>> from heapq import *
>>> from random import shuffle
>>> data = range(10)
>>> shuffle(data)
>>> heap = []
>>> for n in data:
...     heappush(heap, n)
...
>>> heap
[0, 2, 1, 6, 5, 3, 4, 9, 7, 8]
>>> heappush(heap, 0.5)
>>> heap
[0, 0.5, 1, 6, 2, 3, 4, 9, 7, 8, 5] 
```

元素的顺序并不像看起来那么随意。它们虽然不是严格排序的，但是也有规则的：位于`i`位置上的元素总比`i//2`位置处的元素大(反过来说就是`i`位置处的元素总比`2*i`以及`2*i+1`位置处的元素小)。这是底层堆算法的基础，而这个特性称为*堆属性*(heap property)。

`heappop`函数弹出最小的元素，一般来说都是在索引 0 处的元素，并且会确保剩余元素中最小的那个占据这个位置(保持刚才提到的堆属性)。一般来说，尽管弹出列表的第一个元素并不是很有效率，但是在这里不是问题，因为`heappop`在“幕后”会做一些精巧的移位操作：

```py
>>> heappop(heap)
0
>>> heappop(heap)
0.5
>>> heappop(heap)
1
>>> heap
[2, 5, 3, 6, 7, 8, 4, 9] 
```

`heapify`函数使用任意列表作为参数，并且通过尽可能少的移位操作，将其转换为合法的堆(事实上是应用了刚才提到的堆属性)。如果没有用`heappush`建立堆，那么在使用`heappush`和`heappop`前应该使用这个函数。

```py
>>> heap = [5, 8, 0, 3, 6, 7, 9, 1, 4, 2]
>>> heapify(heap)
>>> heap
[0, 1, 5, 3, 2, 7, 9, 8, 4, 6] 
```

`heapreplace`函数不像其他函数那么常用。它弹出堆的最小元素，并且将新元素推入。这样做比调用`heappop`之后再调用`heappush`更高效。

```py
>>> heapreplace(heap, 0.5)
0
>>> heap
[0.5, 1, 5, 3, 2, 7, 9, 8, 4, 6]
>>> heapreplace(heap, 10)
0.5
>>> heap
[1, 2, 5, 3, 6, 7, 9, 8, 4, 10] 
```

`heapq`模块中剩下的两个函数`nlargest(n, iter)`和`nsmallest(n, iter)`分别用来寻找任何可迭代对象`iter`中第`n`大或第`n`小的元素。你可以使用排序(比如使用`sorted`函数)和分片来完成这个工作，但是堆算法更快而且更有效第使用内存(还有一个没有提及的有点：更易用)。

3\. 双端队列

双端队列(double-ended queue，或称`deque`)在需要按照元素增加的顺序来移除元素时非常有用，Python2.4 增加了`collection`模块，它包括`deque`类型。

*注：Python2.5 中的`collections`模块只包括`deque`类型和`defaultdict`类型，为不存在的键提供默认值的字典，未来可能会加入二叉树(B-Tree)和斐波那契堆(Fibonacci heap)。*

双端队列通过可迭代对象(比如集合)创建，而且有些非常有用的方法，如下例所示：

```py
>>> from collections import deque
>>> q = deque(range(5))
>>> q.append(5)
>>> q.appendleft(6)
>>> q
deque([6, 0, 1, 2, 3, 4, 5])
>>> q.pop()
5
>>> q.popleft()
6
>>> q.rotate(3)
>>> q
deque([2, 3, 4, 0, 1])
>>> q.rotate(-1)
>>> q
deque([3, 4, 0, 1, 2]) 
```

双端队列好用的原因是它能够有效的在开头(左侧)增加和弹出元素，这是在列表中无法实现的。除此之外，使用双端队列的好处还有：能够有效地旋转(rotate)元素(也就是将它们左移或者右移，使头尾相连)。双端队列对象还有`extend`和`extendleft`方法，`extend`和列表的`extend`方法差不多，`extendleft`则类似于`appendleft`。注意，`extendleft`使用的可迭代对象中的元素会反序出现在双端队列中。

### 10.3.5 `time`

`time`模块所包括的函数能够实现以下功能：获得当前时间、操作时间和日期、从字符串读取时间以及格式化时间为字符串。日期可以用实数(从“新纪元”的 1 月 1 日 0 点开始计算到现在的秒数，新纪元是一个与平台相关的年份，对 UNIX 来说是 1970 年)，或者是包含有 9 个整数的元组。这些整数的意义如表 10-6 所示，比如，元组：

```py
(2008, 1, 21, 12, 2, 56, 0, 21, 0) 
```

表示 2008 年 1 月 21 日 12 时 2 分 56 秒，星期一，并且是当年的第 21 天(无夏令时)。

表 10-6 Python 日期元组的字段含义

```py
0　　　　　　　　　　年　　　　　　　　　　比如 2000,2001 等等
1　　　　　　　　　　月　　　　　　　　　　范围 1~12
2　　　　　　　　　　日　　　　　　　　　　范围 1~31
3　　　　　　　　　　时　　　　　　　　　　范围 0~23
4　　　　　　　　　　分　　　　　　　　　　范围 0~59
5　　　　　　　　　　秒　　　　　　　　　　范围 0~61
6　　　　　　　　　　周　　　　　　　　　　当周一为 0 时，范围 0~6
7　　　　　　　　　　儒历日　　　　　　　　范围 1~366
8　　　　　　　　　　夏令时　　　　　　　　0、1 或-1 
```

秒的范围是 0~61 是为了应付闰秒和双闰秒。夏令时的数字是布尔值(真或假)，但是如果使用了`-1`，`mktime`(该函数将这样的元组转换为时间戳，它包含从新纪元开始以来的秒数)就会工作正常。`time`模块中最重要的函数如表 10-7 所示。

函数`time.asctime`将当前时间格式化为字符串，如下例所示：

```py
>>> time.asctime() 'Fri May 13 17:35:56 2016' 
```

表 10-7 `time`模块中重要的函数

```py
asctime([tuple])　　　　　　　　　　　　　　　　　　　将时间元组转换为字符串
localtime([secs])　　　　　　　　　　　　　　　　　　 将秒数转换为日期元组，以本地时间为准
mktime(tuple)　　　　　　　　　　　　　　　　　　　　 将时间元组转换为本地时间
sleep(secs)　　　　　　　　　　　　　　　　　　　　　 休眠(不做任何事情)secs 秒
strptime(string[, format])　　　　　　　　　　　　　　将字符串解析为时间元组
time()　　　　　　　　　　　　　　　　　　　　　　　　当前时间(新纪元开始后的描述，以 UTC 为准) 
```

如果不需要使用当前时间，还可以提供一个日期元组(比如通过`localtime`创建的)。(为了实现更精细的格式化，你可以使用`strftime`函数，标准文档对此有相应的介绍)

函数`time.localtime`将实数(从新纪元开始计算的秒数)转换为本地时间的日期元组。如果想获得全球统一时间(有关全球统一时间的更多内容，请参见[`en/wikipedia.org/wiki/Universal_time`](http://en/wikipedia.org/wiki/Universal_time))，则可以使用`gtime`。

函数`time.mktime`将日期元组转换为从新纪元开始计算的秒数，它与`localtime`的功能相反。

函数`time.sleep`让解释器等待给定的秒数。

函数`time.strptime`将`asctime`格式化过的字符串转换为日期元组(可选的格式化参数所遵循的规则与`strftime`的一样，详情请参见标准文档)。

函数`time.time`使用自新纪元开始计算的秒数返回当前(全球统一)时间，尽管每个平台的新纪元可能不同，但是你仍然可以通过记录某事件(比如函数调用)发生前后`time`的结果来对该事件计时，然后计算差值。有关这些函数的实例，请参见下一节的`random`模块部分。

表 10-7 列出的函数只是从`time`模块选出的一部分。该模块的大多数函数所执行的操作与本小节介绍的内容相类似或者相关。如果需要这里没有介绍到的函数，请参见[Python 库参考的 14.2 节](http://python.org/doc/lib/module-time.html)，以获得更多详细信息。

此外，Python 还提供了两个和时间密切相关的模块：`datetime`(支持日期和时间的算法)和`timeit`(帮助开发人员对代码段的执行时间进行计时)。你可以从 Python 库参考中找到更多有关它们的信息，第十六章也会对`timeit`进行简短的介绍。

### 10.3.6 `random`

`random`模块包括返回随机数的函数，可以用于模拟或者用于任何产生随机输出的程序。

*注：事实上，所产生的数字都是伪随机数，也就是说它们看起来是完全随机的，但实际上，它们以一个可预测的系统作为基础。不过，由于这个系统模块在伪装随机方面十分优秀，所以也就不必对此过多担心了(除非为了实现强加密的目标，因为在这种情况下，这些数字就显得不够“强”了，无法抵抗某些特定的攻击，但是如果你已经深入到强加密的话，也就不用我来解释这些基础的问题了)。如果需要真的随机数，应该使用 os 模块的`urandom`函数。`random`模块内的`SystemRandom`类也是基于同种功能，可以让数据接近真正的随机性。*

这个模块中的一些重要函数如表 10-8 所示。

表 10-8 random 模块中的一些重要的函数

```py
random()　　　　　　　　　　　　　　　　　　　　　　　　　　返回 0&lt;n≤1 之间的随机实数 n
getrandbits(n)　　　　　　　　　　　　　　　　　　　　　　　以长整型形式返回 n 个随机位
uniform(a, b)　　　　　　　　　　　　　　　　　　　　　　　 返回随机实数 n，其中 a≤n&lt;b
randrange([start, ]stop[, step])　　　　　　　　　　　　　　返回 range(start, stop, step)中的随机数
choice(seq)　　　　　　　　　　　　　　　　　　　　　　　　 从序列 seq 中返回随意元素
shuffle(seq[, random])　　　　　　　　　　　　　　　　　　　原地指定序列 seq
sample(seq, n)　　　　　　　　　　　　　　　　　　　　　　　从序列 seq 中选择 n 个随机且独立的元素 
```

函数`random.random`是最基本的随机函数之一，它只是返回 0~1 的伪随机数`n`。除非这就是你想要的，否则你应该使用其他提供了额外功能的的函数。`random.getrandbits`以长整型形式返回给定的位数(二进制数)。如果处理的是真正的随机事务(比如加密)，这个函数尤为有用。

为函数`random.uniform`提供两个数值参数`a`和`b`，它会返回在 a~b 的随机(平均分布的)实数 n。所以，比如需要随机数的角度值，可以使用`uniform(0, 360`)。

调用函数`range`可以获得一个范围，而使用与之相同的参数来调用标准函数`random.randrange`则能够产生该范围内的随机整数。比如想要获得 1~10(包括 10)的随机数，可以使用`randrange(1, 11)`(或者使用`randrange(10)+1`)，如果想要获得小于 20 的随机正奇数，可以使用`randrange(1, 20, 2)`。

函数`random.choice`从给定序列中(均一地)选择随机元素。

函数`random.shuffle`将给定(可变)序列的元素进行随机移位，每种排列的可能性都是近似相等的。

函数`random.sample`从给定序列中(均一地)选择给定数目的元素，同时确保元素互不相同。

*注：从统计学的角度来说，还有些与`uniform`类似的函数，它们会根据其他各种不同的分布规则进行抽取，从而返回随机数。这些分布包括贝塔分布、指数分布、高斯分布等等。*

下面介绍一些使用`random`模块的例子。这些例子将使用一些前文介绍的`time`模块中的函数。首先获得代表时间间隔(2008 年)限制的实数，这可以通过时间元组的方式来表示日期(使用-1 表示一周中的某天，一年中的某天和夏令时，以便让 Python 自己计算)，并且对这些元组调用`mktime`：

```py
>>> from random import *
>>> from time import *
>>> date1 = (2008, 1, 1, 0, 0, 0, -1, -1, -1)
>>> time1 = mktime(date1)
>>> date2 = (2009, 1, 1, 0, 0, 0, -1, -1, -1)
>>> time2 = mktime(date2) 
```

然后就能在这个范围内均一地生成随机数(不包括上限)：

```py
>>> random_time = uniform(time1, time2) 
```

然后，可以将数字转换为易读的日期形式：

```py
>>> print asctime(localtime(random_time))
Tue Oct 14 04:33:21 2008 
```

在接下来的例子中，我们要求用户选择投掷的骰子数以及每个骰子具有的面数。投骰子机制可以由`randrange`和`for`循环实现：

```py
#!/usr/bin/env python
# coding=utf-8

from random import randrange

num = input("How many dice? ")
sides = input("How many sides per die? ")
result = 0 for i in range(num):
    result += randrange(sides) + 1

print "The result is", result 
```

如果将代码存为脚本文件并且执行，那么会看到下面的交互操作：

```py
How many dice? 3
How many sides per die? 6
The result is 11 
```

接下来假设有一个新建的文本文件，它的每一行文本都代表一种运势，那么我们就可以使用前面介绍的`fileinput`模块将“运势”都存入列表中，再进行随机选择：

```py
# fortunu.py

import fileinput, random

fortunes = list(fileinput.input())
print random.choice(fortunes) 
```

在 UNIX 中，可以对标准字典文件`/usr/dict/words`进行测试，以获得一个随机单词：

```py
$ python Code.py /usr/dict/words
Greyson 
```

最后一个例子，假设你希望程序能够在每次敲击回车的时候都为自己发一张牌，同时还要确保不会获得相同的牌。首先要创建“一副牌”——字符串列表：

```py
>>> values = range(1, 11) + "Jack Queen King".split()
>>> suits = "diamonds clubs hearts spades".split()
>>> deck = ["%s of %s" % (v, s) for v in values for s in suits] 
```

现在创建的牌还不太适合进行游戏，让我们来看看现在的牌：

```py
>>> from pprint import pprint
>>> pprint(deck[:12])
['1 of diamonds', '1 of clubs', '1 of hearts', '1 of spades', '2 of diamonds', '2 of clubs', '2 of hearts', '2 of spades', '3 of diamonds', '3 of clubs', '3 of hearts', '3 of spades'] 
```

太整齐了，对吧？不过，这个问题很容易解决：

```py
>>> from random import shuffle
>>> shuffle(deck)
>>> pprint(deck[:12])
['7 of hearts', 'Queen of hearts', 'Jack of diamonds', '9 of hearts', '2 of diamonds', '7 of spades', '10 of diamonds', '8 of diamonds', 'Jack of spades', '4 of spades', '2 of clubs', 'King of spades'] 
```

注意，为了节省空间，这里只打印了前 12 张牌。你可以自己看看整副牌。

最后，为了让 Python 在每次按回车的时候都给你发一张牌，知道发完为止，那么只需要创建一个小的`while`循环即可。假设将建立牌的代码放在程序文件中，那么只需要在程序的结尾处加入下面这行代码：

```py
while deck:
    raw_input(deck.pop()) 
```

*注：如果在交互式解释器中尝试上面找到的`while`循环，那么你会注意到每次按下回车的时候都会打印出一个空字符串。因为`raw_input`返回了输入的内容(什么都没有)，并且将其打印出来。在一般的程序中，从`raw_input`返回的值都会被忽略掉。为了能够在交互环节“忽略”它，只需要把`raw_input`的值赋给一些你不想再用到的变量即可。同时将这些变量命名为`ignore`这类名字。*

### 10.3.7 `shelve`

下一章将会介绍如何在文件中存储数据，但如果只需要一个简单的存储方案，那么`shelve`模块可以满足你大部分的需要，你所要做的只是为它提供文件名。`shelve`中唯一的有趣的函数是`open`。在调用它的时候(使用文件名作为参数)，它会返回一个`shelf`对象，你 10.3.7 `shalve`可以用它来存储内容。只需要把它当做普通的字典(但是键一定要作为字符串)来操作即可，在完成工作(并且将内容存储到磁盘中)之后，调用它的`close`方法。

1\. 潜在的陷阱

`shelve.open`函数返回的对象并不是普通的映射，这一点尤其要注意，如下面的例子所示：

```py
>>> import shelve
>>> s = shelve.open("/home/marlowes/workspace/pycharm_Python/Basic_tutorial/test.dat")
>>> s["x"] = ["a", "b", "c"]
>>> s["x"].append("d")
>>> s["x"]
['a', 'b', 'c'] 
```

`"d"`去哪了？

很容易解释：当你在`shelf`对象中查找元素的时候，这个对象都会根据已经存储的版本进行重新构建，当你将元素赋给某个键的时候，它就被存储了。上述例子中执行的操作如下：

☑ 列表`["a", "b", "c"]`存储在键 x 下。

☑ 获得存储的表示，并且根据它来创建新的列表，而"d"被添加到这个副本中。修改的版本还没有被保存！

☑ 最终，再次获得原始版本——没有`"d"`。

为了正确地使用`shelve`模块修改存储的对象。必须将临时变量绑定到获得的副本上，并且在它被修改后重新存储这个副本(感谢 Luther Blissett 指出这个问题)：

```py
>>> temp = s["x"]
>>> temp.append("d")
>>> s["x"] = temp
>>> s["x"]
['a', 'b', 'c', 'd'] 
```

Python2.4 之后的版本还有个解决方法：将`open`函数的`writeback`参数设为 true。如果这样做，所有从`shelf`读取或者赋值到`shelf`的数据结构都会保存在内存(缓存)中，并且只有在关闭`shelf`的时候才写回到磁盘中。如果处理的数据不大，并且不想考虑这些问题，那么将`writeback`设为`true`(确保在最后关闭了`shelf`)的方法还是不错的。

2\. 简单的数据库示例

代码清单 10-8 给出了一个简单的使用 shelve 模块的数据库应用程序。

```py
#!/usr/bin/env python
# coding=utf-8

# database.py

import shelve
def store_person(db):
    """ Query user for data and store it in the shelf object. """
    pid = raw_input("Enter unique ID number: ")
    person = {}
    person["name"] = raw_input("Enter name: ")
    person["age"] = raw_input("Enter age: ")
    person["phone"] = raw_input("Enter phone number: ")
    db[pid] = person
def lookup_person(db):
    """ Query user for ID and desired field, and fetch the correspond data from
    the shelf object. """
    pid = raw_input("Enter ID number: ")
    field = raw_input("What would you like to know? (name, age, phone) ")
    field = field.strip().lower() print field.capitalize() + ":", db[pid][field]
def print_help():
    print "The available commands are:"
    print "store   : Store information about a persoon"
    print "lookup  : Looks up a person from ID number"
    print "quit    : Save changes and exit"
    print "?       : Prints this message"

def enter_command():
    cmd = raw_input("Enter command(? for help): ")
    cmd = cmd.strip().lower()
    return cmd
def main():
    # You may want to change this name
    database = shelve.open("/home/marlowes/workspace/pycharm_Python/Basic_tutorial/database.dat")
    try:
        while True:
            cmd = enter_command()
            if cmd == "store":
                store_person(database)
            elif cmd == "lookup":
                lookup_person(database)
            elif cmd == "?":
                print_help() elif cmd == "quit": return
    finally:
        database.close()
if __name__ == '__main__':
    main() 
```

`Database.py`

代码清单 10-8 中的程序有一些很有意思的特征。

☑ 将所有内容都放到函数中会让程序更加结构化(可能的改进是将函数组织为类的方法)。

☑ 主程序放在 main 函数中，只有在`if __name__ == '__main__'`条件成立的时候才被调用。这意味着可以在其他程序中将这个程序作为模块导入，然后调用`main`函数。

☑ 我在`main`函数中打开数据库(`shelf`)，然后将其作为参数传给另外需要它的函数。当然，我也可以使用全局变量，毕竟这个程序很小。不过，在大多数情况下最好避免使用全局变量，除非有充足的理由要使用它。

☑ 在一些值中进行读取之后，对读取的内容调用`strip`和`lower`函数以生成了一个修改后的版本。这么做的原因在于：如果提供的键与数据库存储的键相匹配，那么它们应该完全一样。如果总是对用户的输入使用`strip`和`lower`函数，那么就可以让用户随意输入大小写字母和添加空格了。同时需要注意的是：在打印字段名称的时候，我使用了`capitalize`函数。

☑ 我使用`try/finally`确保数据库能够正确关闭。我们永远不知道什么时候会出错(同时程序会抛出异常)。如果程序在没有正确关闭数据库的情况下终止，那么，数据库文件就有可能被损坏了，这样的数据文件是毫无用处的。使用`try/finally`就可以避免这种情况了。

接下来，我们测试一下这个数据库。下面是一个简单的交互过程：

```py
Enter command(? for help): ?
The available commands are:
store   : Store information about a persoon
lookup  : Looks up a person from ID number
quit    : Save changes and exit
?       : Prints this message
Enter command(? for help): store
Enter unique ID number: 001 Enter name: Greyson
Enter age: 19 Enter phone number: 001-160309 Enter command(? for help): lookup
Enter ID number: 001 What would you like to know? (name, age, phone) phone
Phone: 001-160309 Enter command(? for help): quit 
```

交互的过程并不是十分有趣，使用普通的字典也能获得和`shelf`对象一样的效果。但是，我们现在退出程序，然后再重新启动它，看看发生了什么？也许第二天才重新启动它：

```py
Enter command(? for help): lookup
Enter ID number: 001 What would you like to know? (name, age, phone) name
Name: Greyson
Enter command(? for help): quit 
```

我们可以看到，程序读出了第一次创建的文件，而 Greyson 的资料还在！

你可以随意试验这个程序，看看是否还能扩展它的功能并且提高用户友好度。你是不是想创建一个供自己使用的版本？创建一个唱片集的数据库怎样？或者创建一个数据库，帮助自己记录借书朋友的名单(我想我会用这个版本)。

### 10.3.8 `re`

有些人面临一个问题时回想：“我知道，可以使用正则表达式来解决这个问题。”于是现在他们就有两个问题了。　　　　——Jamie Zawinski（Lisp 黑客，Netscape 早期开发者。关于他的更详细编程生涯，可见人民邮电出版社出版的《编程人生》一书）

`re`模块包含对*正则表达式*(regular expression)的支持。如果你之前听说过正则表达式，那么你可能知道它有多强大了，如果没有，请做好心里准备吧，它一定会令你很惊讶。

但是应该注意，在学习正则表达式之初会有点困难(好吧，其实是很难)。学习它们的关键是一次只学习一点——(在文档中)查找满足特定任务需要的那部分内容，预先将它们全部记住是没必要的。本章将会对`re`模块主要特征和正则表达式进行介绍，以便让你上手。

*注：除了标准文档外，Andrew Kuchling 的["Regular Expression HOWTO"（正则表达式 HOWTO）](http://amk.ca/python/howto/regex/)也是学习在 Python 中使用正则表达式的有用资源。*

1.什么是正则表达式

正则表达式是可以匹配文本片段的模式。最简单的正则表达式就是普通字符串，可以匹配其自身。换句话说，正则表达式"python"可以匹配字符串"python"。你可以用这种匹配行为搜索文本中的模式，并且用计算后的值替换特定模式，或者将文本进行分段。

○ 通配符

正则表达式可以可以匹配多于一个的字符串，你可以使用一些特殊字符串创建这类模式。比如点号(`.`)可以匹配任何字符(除了换行符)，所以正则表达式`".ython"`可以匹配字符串`"python"`和`"jython"`。它还能匹配`"qython"`、`"+ython"`或者`" ython"`(第一个字母是空格)，但是不会匹配`"cpython"`或者`"ython"`这样的字符，因为点号只能匹配一个字母，而不是两个或者零个。

因为它可以匹配“任何字符串”(除换行符外的任何单个字符)，点号就称为*通配符*(wildcard)。

○ 对特殊字符进行转义

你需要知道：在正则表达式中如果将特殊字符作为普通字符使用会遇到问题，这很重要。比如，假设需要匹配字符串`"python.org"`，直接调用`"python.org"`可以么？这么做是可以的，但是这样也会匹配`"pythonzorg"`，这可不是所期望的结果(点号可以匹配除换行符外的任何字符，还记得吧)。为了让特殊字符表现得像普通字符一样，需要对它进行*转义*(escape)，就像我在第一章中对引号进行转义所做的一样——可以在它前面加上反斜线。因此，在本例中可以使用`"python\\.org"`，这样就只会匹配`"python.org"`了。

*注：为了获得`re`模块所需的单个反斜线，我们要在字符串中使用两个反斜线——为了通过解释器进行转义。这样就需要两个级别的转义了：(1)通过解释器转义；(2)通过 re 模块转义(事实上，有些情况下可以使用单个反斜线，让解释器自动进行转义，但是别依赖这种功能)。如果厌烦了使用双斜线，那么可以使用原始字符串，比如`r"python\.org"`。*

○ 字符集

匹配任意字符可能很有用，但有些时候你需要更多的控制权。你可以使用中括号括住字符串来创建*字符集*(character set)。字符集可以匹配它所包括的任意字符，所以`"[pj]ython"`能够匹配`"python"`和`"jython"`，而非其他内容。你可以使用范围，比如`"[a-z]"`能够(按字母顺序)匹配`a`到`z`的任意一个字符，还可以通过一个接一个的方式将范围联合起来使用，比如`"[a-zA-Z0-9]"`能够匹配任意大小写字母和数字(注意字符集只能匹配一个这样的字符)。

为了反转字符集，可以在开头使用^字符，比如`"[^abc]"`可以匹配任何除了`a`、`b`和`c`之外的字符。

**字符集中的特殊字符**

一般来说，如果希望点号、星号和问号等特殊字符在模式中用作文本字符而不是正则表达式运算符，那么需要用反斜线进行转义。在字符集中，对这些字符进行转义通常是没必要的(尽管是完全合法的)。不过，你应该记住下面的规则：

☑ 如果脱字符(`^`)出现在字符集的开头，那么你需要对其进行转义了，除非希望将它用做否定运算符(换句话说，不要将它放在开头，除非你希望那样用)；

☑ 同样，右中括号(`]`)和横线(`-`)应该放在字符集的开头或者用反斜线转义(事实上，如果需要的话，横线也能放在末尾)。

○ 选择符和子模式

在字符串的每个字符都有各不相同的情况下，字符集是很好用的，但如果只想匹配字符串`"python"`和`"perl"`呢？你就不能使用字符集或者通配符来指定某个特定的模式了。取而代之的是用于选择项的特殊字符：管道符号(|)。因此，所需的模式可以写成`"python|perl"`。

但是，有些时候不需要对整个模式使用选择运算符，只是模式的一部分。这时可以使用圆括号括起需要的部分，或称子模式(subparttern)。前例可以写成`"p(ython|erl)"`。(注意，术语*子模式*也是适用于单个字符)

○ 可选项和可重复子模式

在子模式后面加上问号，它就变成了可选项。它可能出现在匹配字符串中，但并非必需的。例如，下面这个(稍微有点难懂)模式：

```py
r"(http://)?(www\.)?python\.org" 
```

只能匹配下列字符串(而不会匹配其他的)：

```py
"http://www.python.org"
"http://python.org"
"www.python.org"
"python.org" 
```

对于上述例子，下面这些内容是值得注意的：

☑ 对点号进行了转义，防止它被作为通配符使用；

☑ 使用原始字符串减少所需反斜线的数量；

☑ 每个可选子模式都用圆括号括起；

☑ 可选子模式出现与否均可，而且互相独立。

问号表示子模式可以出现一次或根本不出现，下面这些运算符允许子模式重复多次：

☑ `(pattern)*`：允许模式重复 0 次或多次；

☑ `(pattern)+`：允许模式重复 1 次或多次；

☑ `(patten){m,n}`：允许模式重复 m~n 次。

例如，`r"w*\.python\.org"`会匹配`"www.python.org"`，也会匹配`".python.org"`、`"ww.python.org"`和`"wwwwww.python.org"`。类似地，`r"w+\.python\.org"`匹配`"w.python.org"`但不匹配`".python.org"`，而`r"w{3,4}\.python\.org"`只匹配`"www.python.org"`和`"wwww.python.org"`。

*注：这里使用术语匹配(match)表示模式匹配整个字符串。而接下来要说到的 match 函数(参见表 10-9)只要求模式匹配字符串的开始。*

○ 字符串的开始和结尾

目前为止，所出现的模式匹配都是针对整个字符串的，但是也能寻找匹配模式的子字符串，比如字符串`"www.python.org"`中的子字符串`"www"`能够匹配模式`"w+"`。在寻找这样的子字符串时，确定子字符串位于整个字符串的开始还是结尾是很有用的。比如，只想在字符串的开头而不是其他位置匹配`"ht+p"`，那么就可以使用脱字符(`^`)标记开始：`"^ht+p"`会匹配`"http://python.org"`(以及`"httttp://python.org"`)，但是不匹配`"www.python.org"`。类似的，字符串结尾用美元符号(`$`)标识。

*注：有关正则表达式运算符的完整列表，请参见 Python 类参考的[4.2.1 节的内容](http://python.org/doc/lib/re-syntax.html)。*

2.`re`模块的内容

如果不知道如何应用，只知道如何书写正则表达式还是不够的。`re`模块包含一些有用的操作正则表达式的函数。其中最重要的一些函数如表 10-9 所示。

表 10-9 `re`模块中一些重要的函数

```py
compile(pattern[, flags])　　　　　　　　　　　　　　　根据包含正则表达式的字符串创建模式对象
search(pattern， string[, flags])　　　　　　　　　　　在字符串中寻找模式
match(pattern, string[, flags])　　　　　　　　　　　　在字符串的开始处匹配模式
split(pattern string[, maxsplit=0])　　　　　　　　　　根据模式的匹配项来分割字符串
findall(pattern, string)　　　　　　　　　　　　　　　 列出字符串中模式的所有匹配项
sub(pat, repl, string[, count=0])　　　　　　　　　　　将字符串中所有 pat 的匹配项用 repl 替换
escape(string)　　　　　　　　　　　　　　　　　　　　 将字符串中所有特性正则表达式字符转义 
```

函数`re.compile`将正则表达式(以字符串书写的)转换成模式对象，可以实现更有效率的匹配。如果在调用`search`或者`match`函数的时候使用字符串表示的正则表达式，它们也会在内部将字符串转换为正则表达式对象。使用`compile`完成一次转换之后，在每次使用模式的时候就不用进行转换。模式对象本身也没有查找/匹配的函数，就像方法一样，所以`re.search(pat, string)`(`pat`是用字符串表示的正则表达式)等价于`pat.search(string)`(`pat`是用`compile`创建的模式对象)。经过`compile`转换的正则表达式对象也能用于普通的`re`函数。

函数`re.search`会在给定字符串中寻找第一个匹配给定正则表达式的子字符串。一旦找到子字符串，函数就会返回`MatchObject`(值为`True`)，否则返回`None`(值为`False`)。因为返回值的性质，所以该函数可以用在条件语句中，如下例所示：

```py
if re.search(pat, string): print "Found it!" 
```

同时，如果需要更多有关匹配的子字符串的信息，那么可以检查返回的`MatchObject`对象(有关`MatchObject`更多的内容，请参见下一节)。

函数`re.match`会在给定字符串的开头匹配正则表达式。因此，`match("p", "python")`返回真(即匹配对象`MatchObject`)，而`re.match("p", "www.python.org")`则返回假(`None`)。

*注：如果模式与字符串的开始部分相匹配，那么`match`函数会给出匹配的结果，而模式并不需要匹配整个字符串。如果要求模式匹配整个字符串，那么可以在模式的结尾加上美元符号。美元符号会对字符串的末尾进行匹配，从而“顺延”了整个匹配。*

函数`re.split`会根据模式的匹配项来分割字符串。它类似于字符串方法`split`，不过是用完整的正则表达式替代了固定的分隔符字符串。比如字符串方法`split`允许用字符串`","`的匹配项来分割字符串，而`re.split`则允许用任意长度的逗号和空格序列来分割字符串：

```py
>>> import re
>>> some_text = "alpha, beta,,,,gamma delta"
>>> re.split("[, ]+", some_text)
['alpha', 'beta', 'gamma', 'delta'] 
```

*注：如果模式包含小括号，那么括起来的字符组合会散布在分割后的子字符串之间。例如，`re.split("o(o)", "foobar")`回生成`["f", "o", "bar"]`。*

从上述例子可以看到，返回值是子字符串的列表。`maxsplit`参数表示字符串最多可以分割的次数：

```py
>>> re.split("[, ]+", some_text, maxsplit=2)
['alpha', 'beta', 'gamma delta']
>>> re.split("[, ]+", some_text, maxsplit=1)
['alpha', 'beta,,,,gamma delta'] 
```

函数`re.findall`以列表形式返回给定模式的所有匹配项。比如，要在字符串中查找所有的单词，可以像下面这么做：

```py
>>> pat = "[a-zA-Z]+"
>>> text = '"Hm... Err -- are you sure?" he said, sounding insecure.'
>>> re.findall(pat, text)
['Hm', 'Err', 'are', 'you', 'sure', 'he', 'said', 'sounding', 'insecure'] 
```

或者查找标点符号：

```py
>>> pat = r'[.?\-",]+'
>>> re.findall(pat, text)
['"', '...', '--', '?"', ',', '.'] 
```

注意，横线(`-`)被转义了，所以 Python 不会将其解释为字符范围的一部分(比如 a~z)。

函数`re.sub`的作用在于：使用给定的替换内容将匹配模式的子字符串(最左端并且非重叠的子字符串)替换掉。请思考下面的例子：

```py
>>> pat = '{name}'
>>> text = 'Dear {name}...'
>>> re.sub(pat, "Mr. Greyson", text) 'Dear Mr. Greyson...' 
```

请参见本章后面“作为替换的组号和函数”部分，该部分会向你介绍如何更有效地使用这个函数。

`re.escape`是一个很实用的函数，它可以对字符串中所有可能被解释为正则运算符的字符进行转义的应用函数。如果字符串很长且包含很多特殊字符，而你又不想输入一大堆反斜线，或者字符串来自于用户(比如通过`raw_input`函数获取的输入内容)，且要用作正则表达式的一部分的时候，可以使用这个函数。下面的例子向你演示了该函数是如何工作的：

```py
>>> re.escape("www.python.org")
'www\\.python\\.org'
>>> re.escape("But where is the ambiguity?")
'But\\ where\\ is\\ the\\ ambiguity\\?' 
```

*注：你可能会注意到，表 10-9 中有些函数包含了一个名为`flags`的可选参数。这个参数用于改变解释正则表达式的方法。有关它的更多信息，请参见[Python 库参考的 4.2 节](http://python.org/doc/lib/module-re.html) 。这个标志在 4.2.3 节中有介绍。*

3.匹配对象和组

对于`re`模块中那些能够对字符串进行模式匹配的函数而言，当能找到匹配项的时候，它们都会返回`MatchObject`对象。这些对象包括匹配模式的子字符串的信息。它们还包含了那个模式匹配了子字符串哪部分的信息——这些“部分”叫做组(group)。

简而言之，组就是放置在圆括号内的子模式。组的序号取决于它左侧的括号数。组 0 就是整个模式，所以在下面的模式中：

```py
"There (was a (wee) (cooper)) who (lived in Fyfe)" 
```

包含下面这些组：

```py
0 There was a wee cooper who lived in Fyfe
1 was a wee cooper
2 wee
3 cooper
4 lived in Fyfe 
```

一般来说，如果组中包含诸如通配符或者重复运算符之类的特殊字符，那么你可能会对是什么与给定组实现了匹配感兴趣，比如在下面的模式中：

```py
r"www\.(.+)\.com$" 
```

组 0 包含整个字符串，而组 1 则包含位于`"www."`和`".com"`之间的所有内容。像这样创建模式的话，就可以取出字符串中感兴趣的部分了。

`re`匹配对象的一些重要方法如表 10-10 所示。

表 10-10 re 匹配对象的重要方法

```py
group([group1, ...])　　　获取给定子模式(组)的匹配项
start([group])　　　　　　返回给定组的匹配项的开始位置
end([group])　　　　　　　返回给定组的匹配项的结束位置(和分片不一样，不包括组的结束位置)
span([group])　　　　　　 返回一个组的开始和结束位置 
```

`group`方法返回模式中与给定组匹配的(子)字符串。如果没有给出组号，默认为组 0。如果给定一个组号(或者只用默认的 0)，会返回单个字符串。否则会将对应给定组数的字符串作为元组返回。

*注：除了整体匹配外(组 0)，我们只能使用 99 个组，范围 1~99。*

`start`方法返回给定组匹配项的开始索引(默认为 0，即整个模式)。

方法`end`类似于`start`，但是返回结果是结束索引加 1。

方法`span`以元组`(start,end)`的形式返回给定组的开始和结束位置的索引(默认为 0，即整个模式)。

请思考以下例子：

```py
>>> m = re.match(r"www\.(.*)\..{3}", "www.python.org")
>>> m.group(1)
'python'
>>> m.start(1)
4
>>> m.end(1)
10
>>> m.span(1)
(4, 10) 
```

4\. 作为替换的组号和函数

在使用`re.sub`的第一个例子中，我只是把一个字符串用其他的内容替换掉了。我用`replace`这个字符串方法(3.4 节对此进行了介绍)能轻松达到同样的效果。当然，正则表达式很有用，因为它们允许以更灵活的方式搜索，同时它们也允许进行功能更强大的替换。

见证`re.sub`强大功能的最简单方式就是在替换字符串中使用组号。在替换内容中以`"\\n"`形式出现的任何转义序列都会被模式中与组 n 匹配的字符串替换掉。例如，假设要把`"*something*"`用`"<em>something</em>"`替换掉，前者是在普通文本文档(比如 Emaill)中进行强调的常见方法，而后者则是相应的 HTML 代码(用于网页)。我们首先建立正则表达式：

```py
>>> emphasis_pattern = r"\*([^\*]+)\*" 
```

注意，正则表达式很容易变得难以理解，所以为了让其他人(包括自己在内)在以后能够读懂代码，使用有意义的变量名(或者加上一两句注释)是很重要的：

注：让正则表达式变得更加易读的方式是在`re`函数中使用`VERBOSE`标志。它允许在模式中添加空白(空白字符、`tab`、换行符，等等)，`re`则会忽略它们，除非将其放在字符类或者用反斜线转义。也可以在冗长的正则式中添加注释。下面的模式对象等价于刚才写的模式，但是使用了`VERBOSE`标志：

```py
>>> emphasis_pattern = re.compile(r'''
...     \*        # Beginning emphasis tag -- an asterisk
...     (         # Begin group for capturing phrase
...     [^\*]+    # Capture anything except asterisks
...     )         # End group
...     \*        # Ending emphasis tag
... ''', re.VERBOSE) 
```

现在模式已经搞定，接下来就可以使用 re.sub 进行替换了：

```py
>>> re.sub(emphasis_pattern, r"<em>\1</em>", "Hello, *world*!")
'Hello, <em>world</em>!' 
```

从上述例子可以看到，普通文本已经成功地转换为 HTML。

将函数作为替换内容可以让替换功能变得更加强大。`MatchObject`将作为函数的唯一参数，返回的字符串将会用做替换内容。换句话说，可以对匹配的子字符串做任何事，并且可以细化处理过程，以生成替换内容。你可能会问，这个功能用在什么地方呢？开始使用正则表达式以后，你肯定会发现这个功能的无数应用。本章后面的“模板系统示例”部分会向你介绍它的一个应用。

**贪婪和非贪婪模式**

重复运算符默认是贪婪(greedy)的，这意味着它会进行尽可能多的匹配。比如，假设我重写了刚才用到的程序，以使用下面的模式：

```py
>>> emphasis_pattern = r"\*(.+)\*" 
```

它会匹配星号加上一个或多个字符，再加上一个星号的字符串。听起来很完美吧？但实际上不是：

```py
>>> re.sub(emphasis_pattern, r"<em>\1</em>", "*This* is *it*!")
'<em>This* is *it</em>!' 
```

模式匹配了从开始星号到结束星号之间的所有内容——包括中间的两个星号！也就意味着它是贪婪的：将尽可能多的东西都据为己有。

在本例中，你当然不希望出现这种贪婪行为。当你知道某个特定字母不合法的时候，前面的解决方案(使用字符集匹配任何不是星号的内容)才是可行的。但是假设另外一种情况：如果使用`"**something**"`表示强调呢？现在在所强调的部分包括单个星号已经不是问题了，但是如何避免过于贪婪？

事实上非常简单，只要使用重复运算符的非贪婪版本即可。所有的重复运算符都可以通过在其后面加上一个问号变成非贪婪版本：

```py
>>> emphasis_pattern = r"\*\*(.+?)\*\*"
>>> re.sub(emphasis_pattern, r"<em>\1</em>", "**This** is **it**!")
'<em>This</em> is <em>it</em>!' 
```

这里用`+?`运算符代替了`+`，意味着模式也会像之前那样队一个或者多个通配符进行匹配，但是它会进行尽可能少的匹配，因为它是非贪婪的。它仅会在到达`"\*\*"`的下一个匹配项之前匹配最少的内容——也就是在模式的结尾进行匹配。我们可以看到，代码工作得很好。

5\. 找出 Email 的发信人

有没有尝试过将 Email 存为文本文件？如果有的话，你会看到文件的头部包含了一大堆与邮件内容无关的信息，如代码清单 10-9 所示。

```py
#代码清单 10-9 一组(虚构的)Email 头部信息
 From foo@bar.baz Thu Dec 20 01:22:50 2008 Return-Path: <foo@bar.baz> Received: from xyzzy42.bar.com (xyzzy.bar.baz [123.456.789.42])
        by frozz.bozz.floop (8.9.3/8.9.3) with ESMTP id BAA25436 for <maguns@bozz.floop>: Thu 20 Dec 2004 01:22:50 +0100 (MET)
Received: from [43.253.124.23] by bar.baz
          [InterMail vM.4.01.03.27 201-229-121-20010626] with ESMTP
          id <20041220002242.ADASD123.bar.baz@[43.253.124.23]>:
          Thu, 20 Dec 2004 00:22:42 +0000 User-Agent: Microsot-Outlook-Express-Macintosh-Edition/5.02.2022 Date: Wed, 19 Dec 2008 17:22:42 -0700 Subject: Re: Spam
From: Foo Fie <foo@bar.baz> To: Magnus Lie Hetland <magnus@bozz.floop> CC: <Mr.Gumby@bar.baz> Message-ID: <B8467D62.84F%foo@baz.com> In-Reply-To: <20041219013308.A2655@bozz.floop> Mime-version: 1.0 Content-type: text/plain: charset="US-ASCII" Content-transfer-encoding: 7bit
Status: RO
Content-Length: 55 Lines: 6 So long, and thanks for all the spam!

Yours.

Foo Fie 
```

我们试着找出这封 Email 是谁发的。如果直接看文本，你肯定可以指出本例中的发信人(特别是查看邮件结尾签名的话，那就更直接了)。但是能找出通用的模式吗？怎么能把发信人的名字取出而不带着 Email 地址呢？或者如何将头部信息中包含的 Email 地址列示出来呢？我们先处理第一个任务。

包含发信人的文本行以字符串`"From:"`作为开始，以放置在尖括号(`<`和`>`)中的 Email 地址作为结束。我们需要的文本就夹在中间。如果使用`fileinput`模块，那么这个需求就很容易实现了。代码清单 10-10 给出了解决这个问题的程序。

*注：这个问题也可以不使用正则表达式解决，可以使用`email`模块。*

```py
# 代码清单 10-10 寻找 Email 发信人的程序

# RegularExpression.py
import fileinput import re

pat = re.compile(r"From: (.*) <.*?>$")
for line in fileinput.input():
    m = pat.match(line) if m: print m.group(1) 
```

可以像下面这样运行程序(假设邮件内容存储在文本文件`message.eml`中)：

```py
$ python RegularExpression.py message.eml
Foo Fie 
```

对于这个程序，应该注意以下几点：

☑ 我用`compile`函数处理了正则表达式，让处理过程更有效率；

☑ 我将需要取出的子模式放在圆括号中作为组；

☑ 我使用非贪婪模式对邮件地址进行匹配，那么只有最后一对尖括号符合要求(当名字包含了尖括号的情况下)；

☑ 我使用了美元符号表明我要匹配正行；

☑ 我使用 if 语句确保在我试图从特定组中取出匹配内容之前，的确进行了匹配。

为了列出头部信息中所有的 Email 地址，需要建立只匹配 Email 地址的正则表达式。然后可以使用`findall`方法寻找每行出现的匹配项。为了避免重复，可以将地址保存在集合中(本章前面介绍过)。最后，取出所有的键，排序，并且打印出来：

```py
import re import fileinput

pat = re.compile(r"[a-z\-\.]+@[a-z\-\.]+", re.IGNORECASE)
addresses = set()
for line in fileinput.input():
    for address in pat.findall(line):
        addresses.add(address)
    for address in sorted(addresses):
        print address 
```

运行程序的时候会输出如下结果(以代码清单 10-9 的邮件信息作为输入)：

```py
Mr.Gumby@bar.baz
foo@bar.baz
foo@baz.com
magnus@bozz.floop 
```

*注：在这里，我并没有严格照着问题规范去做。问题的要求是在头部找出 Email 地址，但是这个程序找出了整个文件中的地址。为了避免这种情况，如果遇到空行就可以调用`fileinput.close()`，因为头部不包含空行，遇到空行就证明工作完成了。此外，你还可以使用`fileinput.nextfile()`开始处理下一个文件——如果文件多于一个的话。*

6\. 模板系统示例

*模板*是一种通过放入具体值从而得到某种已完成文本的文件。比如，你可能会有只需要插入收件人姓名的邮件模板。Python 有一种高级的模板机制：字符串格式化。但是使用正则表达式可以让系统更加高级。假设需要把所有`"[somethings]"`(字段)的匹配项替换为通过 Python 表达式计算出来的`something`结果，所以下面的字符串：

```py
"The sum of 7 and 9 is [7 + 9]." 
```

应该被翻译为如下形式：

```py
"The sum of 7 and 9 is 16." 
```

同时，还可以在字段内进行赋值，所以下面的字符串：

```py
"[name='Mr. Gumby']Hello, [name]" 
```

应该被翻译为如下形式：

```py
"Hello, Mr. Gumby" 
```

看起来像是复杂的工作，但是我们再看一下可用的工具。

☑ 可以使用正则表达式匹配字段，提取内容。

☑ 可以用`eval`计算字符值，提供包含作用域的字典。可以在`try/except`语句内进行这项工作。如果引发了`SyntaxError`异常，可能是某些语句出现了问题(比如赋值)，应该使用`exec`来代替。

☑ 可以用`exce`执行字符串(和其他语句)的赋值操作，在字典中保存模板的作用域。

☑ 可以使用`re.sub`将求值的结果替换为处理后的字符串。

这样看来，这项工作又不再让人寸步难行了，对吧？

*注：如果某项任务令人望而却步，将其分解为小一些的部分总是有用的。同时，要对解决问题所使用的工具进行评估。*

代码清单 10-11 是一个简单的实现。

```py
#!/usr/bin/env python # coding=utf-8

# templates.py

import re import fileinput
# Matching in brackets in the field.
filed_pat = re.compile(r"\[(.+?)\]")
# We will be variable collected here
scope = {}
# Used in the re.sub.
def replacement(math):
    code = math.group(1)
    try:
        # If the field can be evaluated, then return it.
        return str(eval(code, scope))
    except SyntaxError:
        # Otherwise the same scope of assignment statements.
        exec code in scope
        # Return an empty string.
        return ""

# All text in the from of a string.
# There are other ways, see chapter 11.
lines = []
for line in fileinput.input():
    lines.append(line)

text = "".join(lines)
# Replace all field pattern match.
print filed_pat.sub(replacement, text) 
```

Templates.py

简单来说，程序做了下面的事情。

☑ 定义了用于匹配字段的模式。

☑ 创建充当模板作用域的字典。

☑ 定义具有下列功能的替换函数。

　　* 将组 1 从匹配中取出，放入`code`中；

　　* 通过将作用域字典作为命名空间来对`code`进行求值，将结果转换为字符串返回，如果成功的话。字段就是个表达式，一切正常。否则(也就是引发了`SyntaxError`异常)，跳到下一步；

　　* 执行在相同命名空间(作用域字典)内的字段来对表达式求值，返回空字符串(因为赋值语句没有任何内容进行求值)。

☑ 使用`fileinput`读取所有可用的行，将其放入列表，组合成一个大字符串。

☑ 将所有`field_pat`的匹配项用`re.sub`中的替换函数进行替换，并且打印结果。

*注：在之前的 Python 中，将所有行放入列表，最后再联合要比下面这种方法更有效率：*

```py
text = ""
for line in fileinput.input():
    text += line 
```

*尽管看起来很优雅，但是每个赋值语句都要创建新的字符串，由旧的字符串和新增加字符串联结在一起组成，这样就会造成严重的资源浪费，使程序运行缓慢。在旧版本的 Python 中，使用`join`方法和上述做法之间的差异是巨大的。但是在最近的版本中，使用`+=`运算符事实上会更快。如果觉得性能很重要，那么你可以尝试这两种方式。同时，如果需要一种更优雅的方式来读取文件的所有文本，那么请参见第十一章。*

好了，我只用 15 行代码(不包括空行和注释)就创建了一个强大的模板系统。希望读者已经认识到：使用标准库的时候，Python 有多么强大。下面，我们通过测试这个模板系统来结束本例。试着对代码清单 10-12 中的示例文本运行该系统。

```py
# 代码清单 10-12 简单的模板示例
[x = 2]
[y = 3]
The sum of [x] and [y] is [x + y]. 
```

应该会看到如下结果：

```py
The sum of 2 and 3 is 5. 
```

*注：虽然看起来不明显，但是上面的输出包含了 3 个空行——两个在文本上方，一个在下方。尽管前两个字段已经被替换为空字符串，但是随后的空行还留在那里。同时，`print`语句增加了新行，也就是末尾的空行。*

但是等等，它还能更好！因为使用了`fileinput`，我可以轮流处理几个文件。这意味着可以使用一个文件为变量定义值，而另一个文件作为插入这些值的模板。比如，代码清单 10-13 包含了定义文件，名为`magnus.txt`，而代码清单 10-14 则是模板文件，名为`template.txt`。

```py
# 代码清单 10-13 一些模板定义
[name     = "Magnus Lie Hetland"]
[email = "magnus@foo.bar"]
[language = "python"]
# 代码清单 10-14 一个模板
[import time]
Dear [name].

I would like to learn how to program. I hear you use
the [language] language a lot -- is it something I should consider?

And, by the way, is [email] your correct email address?

Fooville, [time.asctime()]

Oscar Frozzbozz 
```

`import time`并不是赋值语句(而是准备处理的语句类型)，但是因为我不是过分挑剔的人，所以只用了`try/except`语句，使得程序支持任何可以配合`eval`或`exec`使用的语句和表达式。可以像下面这样运行程序(在 UNIX 命令行下)：

```py
$ python templates.py magnus.txt template.txt 
```

你将会看到类似以下内容的输出：

```py
Dear Magnus Lie Hetland.

I would like to learn how to program. I hear you use
the python language a lot -- is it something I should consider?

And, by the way, is magnus@foo.bar your correct email address?

Fooville, Wed May 18 20:58:58 2016 Oscar Frozzbozz 
```

尽管这个模板系统可以进行功能非常强大的替换，但它还是有些瑕疵的。比如，如果能够使用更灵活的方式来编写定义文件就更好了。如果使用`execfile`来执行文件，就可以使用正常的 Python 语法了。这样也会解决输出内容中顶部出现空行的问题。

还能想到其他改进的方法吗？对于程序中使用的概念，还能想到其他用途吗？精通任何程序设计语言的最佳方法是实践——测试它的限制，探索它的威力。看看你能不能重写这个程序，让它工作得更好并且更能满足需求。

*注：事实上，在标准库的`string`模块中已经有一个非常完美的模板系统了。例如，你可以了解一下`Template`类。*

### 10.3.9 其他有趣的标准模块

尽管本章内容已经涵盖了很多模块，但是对于整个标准库来说这只是冰山一角。为了引导你进行深入探索，下面会快速介绍一些很酷的库。

☑ `functools`：你可以从这个库找到一些功能，让你能够通过部分参数来使用某个参数(部分求值)，稍后再为剩下的参数提供数值。在 Python3.0 中，`filter`和`reduce`包含在该模块中。

☑ `difflib`:这个库让你可以计算两个序列的相似度。还能让你从一些序列中(可供选择的序列列表)找出提供的原始序列“最像”的那个。`difflib`可以用于创建简单的搜索程序。

☑ `hashlib`：通过这个模块，你可以通过字符串计算小“签名”(数字)。如果为两个不同的字符串计算出了签名，几乎可以确保这两个签名完全不同。该模块可以应用与大文本文件，同时在加密和安全性(另见`md5`和`sha`模块)方面有很多用途。

☑ `csv`：CSV 是逗号分隔值(Comma-Separated Values)的简写，这是一种很多程序(比如很多电子表格和数据库程序)都可以用来存储表格式数据的简单格式。它主要用于在不同程序间交换数据。使用`csv`模块可以轻松读写 CSV 文件，同时以显而易见的方式来处理这种格式的某些很难处理的地方。

☑ `timeit`、`profile`和`trace`：`time`模块(以及它的命令行脚本)是衡量代码片段运行时间的工具。它有很多神秘的功能，你应该用它来代替`time`模块进行性能测试。`profil`e 模块(和伴随模块 pstats)可用于代码片段效率的全面分析。`trace`模块(和程序)可以提供总的分析(也是代码哪部分执行了，哪部分没执行)。这在写测试代码的时候很有用。

☑ `datetime`：如果`time`模块不能满足时间追踪方面的需求，那么`datetime`可能就有用武之地了。它支持特殊的日期和时间对象，让你能够以多种方式对它们进行构建和联合。它的接口在很多方面比`time`的接口要更加直观。

☑ `itertools`：它有很多工具用来创建和联合迭代器(或者其他可迭代对象)，还包括实现以下功能的函数：将可迭代的对象链接起来、创建返回无限连续整数的迭代器(和`range`类似，但是没有上限)，从而通过重复访问可迭代对象进行循环等等。

☑ `logging`：通过简单的`print`语句打印出程序的哪些方面很有用。如果希望对程序进行跟踪但又不想打印出太多调试内容，那么就需要将这些信息写入日志文件中了。这个模块提供了一组标准的工具，以便让开发人员管理一个或多个核心的日志文件，同时还对日志信息提供了多层次的优先级。

☑ `getopt`和`optparse`：在 UNIX 中，命令行程序经常使用不同的*选项*(option)或者*开关*(switches)运行(Python 解释器就是个典型的例子)。这些信息都可以在`sys.argv`中找到，但是自己要正确处理它们就没有这么简单了。针对这个问题，`getopt`库是个切实可行的解决方案，而`optparse`则更新、更强大并且更易用。

☑ `cmd`：使用这个模块可以编写命令行解释器，就像 Python 的交互式解释器一样。你可以自定义命令，以便让用户能够通过提示符来执行。也许你还能将它作为程序的用户界面。

## 10.4 小结

本章讲述了模块的知识：如何创建、如何探究以及如何使用标准 Python 库中的模块。

☑ 模块：从基本上来说，模块就是子程序，它的主函数则用于定义，包括定义函数、类和变量。如果模块包含测试代码，那么应该将这部分代码放置在检查 `__name__ == '__main__'`是否为真的 if 语句中。能够在`PYTHONPATH`中找到的模块都可以导入。语句`import foo`可以导入存储在`foo.py`文件中的模块。

☑ 包：包是包含有其他模块的模块。包是作为包含`__init__.py`文件的目录来实现的。

☑ 探究模块：将模块导入交互式编辑器后，可以用很多方法对其进行探究。比如使用`dir`检查`__all__`变量以及使用`help`函数。文档和源码是获取信息和内部机制的极好来源。

☑ 标准库：Python 包括了一些模块，总称为标准库。本章讲到了其中的很多模块，以下对其中一部分进行回顾。

```py
○ `sys`：通过该模块可以访问到多个和 Python 解释器联系紧密的变量和函数。

○ `os`：通过该模块可以访问到多个和操作系统联系紧密的变量和函数。

○ `fileinput`：通过该模块可以轻松遍历多个文件和流中所有的行。

○ `sets`、`heapq`和`deque`：这 3 个模块提供了 3 个有用的数据结构。集合也以内建的类型`set`存在。

○ `time`：通过该模块可以获取当前时间，并可进行时间日期操作和格式化。

○ `random`：通过该模块中的函数可以产生随机数，从序列中选取随机元素以及打乱列表元素。

○ `shelve`：通过该模块可以创建持续性映射，同时将映射的内容保存在给定文件名的数据库中。

○ `re`：支持正则表达式的模块。 
```

如果想要了解更多模块，再次建议你浏览[Python 类库参考](http://python.org/doc/lib)，读起来真的很有意思。

### 10.4.1 本章的新函数

本章涉及的新函数如表 10-11 所示。

表 10-11 本章的新函数

```py
dir(obj)        返回按字母顺序排序的属性名称列表
help([obj])     提供交互式帮助或关于特定对象的交互式帮助信息
reload(module)  返回已经导入模块的重新载入版本，该函数在 Python3.0 将要被废除 
```

### 10.4.2 接下来学什么

如果读者能够掌握本章某些概念，那么你的 Python 编程水平就会有很大程度的提高。使用手头上的标准库可以让 Python 从强大变得无比强大。以目前学到的知识为基础，读者已经能编写出用于解决很多问题的程序了。下一章将会介绍如何使用 Python 和外部世界——文件以及网络——进行交互，从而让读者能够解决更多问题。
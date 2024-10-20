# Python 101 -用 pdb 调试代码

> 原文：<https://www.blog.pythonlibrary.org/2020/07/07/python-101-debugging-your-code-with-pdb/>

代码中的错误被称为“bug”。你会犯错误。你会犯很多错误，这完全没关系。很多时候会是错别字之类的简单错误。但是由于计算机是非常字面化的，即使是错别字也会阻止你的代码按预期工作。所以它们需要被修复。修复编程错误的过程被称为**调试**。

Python 编程语言自带名为`pdb`的内置调试器。可以在命令行使用`pdb`或者作为模块导入。`pdb`这个名字是“Python 调试器”的简称。

以下是`pdb`完整文档的链接:

*   [https://docs.python.org/3/library/pdb.html](https://docs.python.org/3/library/pdb.html)

在本文中，您将熟悉使用`pdb`的基础知识。具体来说，您将学习以下内容:

*   从 REPL 出发
*   在命令行上启动`pdb`
*   逐句通过代码
*   在`pdb`中添加断点
*   用`set_trace()`创建断点
*   使用内置的`breakpoint()`功能
*   获得帮助

虽然`pdb`很方便，但是大多数 Python 编辑器都有更多特性的调试器。您会发现 PyCharm 或 WingIDE 中的调试器有更多的特性，比如自动完成、语法突出显示和图形化调用堆栈。

调试器将使用调用堆栈来跟踪函数和方法调用。如果可能，您应该使用 Python IDE 附带的调试器，因为它更容易理解。

然而，有时您可能没有 Python IDE，例如当您在服务器上远程调试时。正是在这些时候，你会发现`pdb`特别有用。

我们开始吧！

### 从 REPL 出发

最好的开始方式是准备一些你想运行的代码。请随意使用您自己的代码或本博客上另一篇文章中的代码示例。

或者您可以在名为`debug_code.py`的文件中创建以下代码:

```py
# debug_code.py

def log(number):
    print(f'Processing {number}')
    print(f'Adding 2 to number: {number + 2}')

def looper(number):
    for i in range(number):
        log(i)

if __name__ == '__main__':
    looper(5)
```

有几种方法可以启动`pdb`并在你的代码中使用它。对于这个例子，您需要打开一个终端(如果您是 Windows 用户，则需要打开`cmd.exe`)。然后导航到保存代码的文件夹。

现在在您的终端中启动 Python。这将为您提供 Python REPL，您可以在其中导入代码并运行调试器。方法如下:

```py
>>> import debug_code
>>> import pdb
>>> pdb.run('debug_code.looper(5)')
> <string>(1)<module>()
(Pdb) continue
Processing 0
Adding 2 to number: 2
Processing 1
Adding 2 to number: 3
Processing 2
Adding 2 to number: 4
Processing 3
Adding 2 to number: 5
Processing 4
Adding 2 to number: 6
```

前两行代码导入您的代码和`pdb`。为了对你的代码运行`pdb`，你需要使用`pdb.run()`并告诉它做什么。在本例中，您将`debug_code.looper(5)`作为一个字符串传入。当你这样做的时候，`pdb`模块会将字符串转换成一个实际的函数调用`debug_code.looper(5)`。

下一行以`(Pdb)`为前缀。这意味着您现在处于调试器中。成功！

要在调试器中运行您的代码，键入`continue`或简称`c`。这将运行您的代码，直到发生以下情况之一:

*   代码引发了一个异常
*   您会到达一个断点(本文稍后会解释)
*   代码结束

在这种情况下，没有设置异常或断点，所以代码运行良好并完成了执行！

### 在命令行上启动`pdb`

启动`pdb`的另一种方法是通过命令行。以这种方式启动`pdb`的过程与前面的方法类似。您仍然需要打开终端并导航到保存代码的文件夹。

但是您将运行以下命令，而不是打开 Python:

```py
python -m pdb debug_code.py
```

当您以这种方式运行`pdb`时，输出会略有不同:

```py
> /python101code/chapter26_debugging/debug_code.py(1)<module>()
-> def log(number):
(Pdb) continue
Processing 0
Adding 2 to number: 2
Processing 1
Adding 2 to number: 3
Processing 2
Adding 2 to number: 4
Processing 3
Adding 2 to number: 5
Processing 4
Adding 2 to number: 6
The program finished and will be restarted
> /python101code/chapter26_debugging/debug_code.py(1)<module>()
-> def log(number):
(Pdb) exit
```

上面的第三行输出具有与您在上一节中看到的相同的 **(Pdb)** 提示。当您看到该提示时，您知道您现在正在调试器中运行。要开始调试，输入`continue`命令。

代码将像以前一样成功运行，但是您将看到一条新消息:

```py
The program finished and will be restarted
```

调试器运行完你所有的代码，然后又从头开始！这对于多次运行您的代码非常方便！如果不希望再次运行代码，可以键入`exit`退出调试器。

### 逐句通过代码

单步执行代码是指使用调试器一次运行一行代码。通过使用`step`命令，或者简称为`s`，你可以使用`pdb`来单步调试你的代码。

如果您使用`pdb`单步执行代码，您将看到以下几行输出:

```py
$ python -m pdb debug_code.py 
> /python101code/chapter26_debugging/debug_code.py(3)<module>()
-> def log(number):
(Pdb) step
> /python101code/chapter26_debugging/debug_code.py(8)<module>()
-> def looper(number):
(Pdb) s
> /python101code/chapter26_debugging/debug_code.py(12)<module>()
-> if __name__ == '__main__':
(Pdb) s
> /python101code/chapter26_debugging/debug_code.py(13)<module>()
-> looper(5)
(Pdb)
```

你传递给`pdb`的第一个命令是`step`。然后使用`s`来遍历下面两行。您可以看到这两个命令的作用完全相同，因为“s”是“step”的快捷方式或别名。

您可以使用`next`(或`n`)命令继续执行，直到函数中的下一行。如果你的函数中有函数调用，`next`会**跳过**。这意味着它将调用函数，执行其内容，然后继续到当前函数中的下一个行**。实际上，这就跳过了函数。**

您可以使用`step`和`next`来导航您的代码并高效地运行各部分。

如果要进入`looper()`功能，继续使用`step`。另一方面，如果您不想运行`looper()`函数中的每一行代码，那么您可以使用`next`来代替。

您应该通过呼叫`step`继续您在`pdb`的会话，以便进入`looper()`:

```py
(Pdb) s
--Call--
> /python101code/chapter26_debugging/debug_code.py(8)looper()
-> def looper(number):
(Pdb) args
number = 5

```

当你进入`looper()`，`pdb`会打印出`--Call--`让你知道你调用了这个函数。接下来，使用`args`命令打印出名称空间中的所有当前参数。在这种情况下，`looper()`有一个参数`number`，它显示在上面输出的最后一行。可以用更短的`a`代替`args`。

您应该知道的最后一个命令是`jump`或`j`。您可以使用该命令跳转到代码中的特定行号，方法是键入`jump`，后跟一个空格，然后是您希望跳转到的行号。

现在让我们来学习如何添加断点！

### 在`pdb`中添加断点

断点是代码中您希望调试器停止的位置，以便您可以检查变量状态。这允许你做的是检查**调用栈**，这是一个时髦的术语，指当前在内存中的所有变量和函数参数。

如果你有 PyCharm 或者 WingIDE，那么他们会有一个图形化的方式让你检查调用栈。您可能能够将鼠标悬停在变量上，以查看它们当前的设置。或者他们可能有一个工具，可以在侧边栏中列出所有变量。

让我们在`looper()`函数的最后一行添加一个断点，即**第 10 行**。

下面是您的代码:

```py
# debug_code.py

def log(number):
    print(f'Processing {number}')
    print(f'Adding 2 to number: {number + 2}')

def looper(number):
    for i in range(number):
        log(i)

if __name__ == '__main__':
    looper(5)
```

要在`pdb`调试器中设置断点，您可以使用`break`或`b`命令，后跟您希望中断的行号:

```py
$ python3.8 -m pdb debug_code.py 
> /python101code/chapter26_debugging/debug_code.py(3)<module>()
-> def log(number):
(Pdb) break 10
Breakpoint 1 at /python101code/chapter26_debugging/debug_code.py:10
(Pdb) continue
> /python101code/chapter26_debugging/debug_code.py(10)looper()
-> log(i)
(Pdb)
```

现在您可以在这里使用`args`命令来找出当前的参数设置为什么。您也可以使用`print`(或简称`p`)命令打印出变量值，如`i`的值:

```py
(Pdb) print(i)
0

```

现在让我们看看如何在代码中添加断点！

### 用`set_trace()`创建断点

Python 调试器允许您导入`pbd`模块并直接在代码中添加断点，如下所示:

```py
# debug_code_with_settrace.py

def log(number):
    print(f'Processing {number}')
    print(f'Adding 2 to number: {number + 2}')

def looper(number):
    for i in range(number):
        import pdb; pdb.set_trace()
        log(i)

if __name__ == '__main__':
    looper(5)
```

现在，当您在终端中运行这段代码时，它会在到达`set_trace()`函数调用时自动启动进入`pdb`:

```py
$ python3.8 debug_code_with_settrace.py 
> /python101code/chapter26_debugging/debug_code_with_settrace.py(12)looper()
-> log(i)
(Pdb)
```

这需要您添加大量额外的代码，稍后您需要删除这些代码。如果您忘记在导入和`pdb.set_trace()`调用之间添加分号，您也会遇到问题。

为了让事情变得更简单，Python 核心开发人员添加了`breakpoint()`，这相当于编写`import pdb; pdb.set_trace()`。

接下来让我们来看看如何使用它！

### 使用内置的`breakpoint()`功能

从 **Python 3.7** 开始，`breakpoint()`函数已经被添加到语言中，以使调试更容易。你可以在这里阅读所有关于变化的信息:

*   [https://www.python.org/dev/peps/pep-0553/](https://www.python.org/dev/peps/pep-0553/)

继续更新上一节的代码，改为使用`breakpoint()`:

```py
# debug_code_with_breakpoint.py

def log(number):
    print(f'Processing {number}')
    print(f'Adding 2 to number: {number + 2}')

def looper(number):
    for i in range(number):
        breakpoint()
        log(i)

if __name__ == '__main__':
    looper(5)
```

现在，当您在终端中运行这个程序时，Pdb 将像以前一样启动。

使用`breakpoint()`的另一个好处是许多 Python IDEs 会识别该函数并自动暂停执行。这意味着您可以在此时使用 IDE 的内置调试器来进行调试。如果您使用旧的`set_trace()`方法，情况就不一样了。

### 获得帮助

本章并没有涵盖`pdb`中所有可用的命令。因此，要了解如何使用调试器的更多信息，您可以在`pdb`中使用`help`命令。它将打印出以下内容:

```py
(Pdb) help

Documented commands (type help <topic>):
========================================
EOF    c          d        h         list      q        rv       undisplay
a      cl         debug    help      ll        quit     s        unt      
alias  clear      disable  ignore    longlist  r        source   until    
args   commands   display  interact  n         restart  step     up       
b      condition  down     j         next      return   tbreak   w        
break  cont       enable   jump      p         retval   u        whatis   
bt     continue   exit     l         pp        run      unalias  where    

Miscellaneous help topics:
==========================
exec  pdb
```

如果你想知道一个特定的命令是做什么的，你可以在命令后面输入`help`。

这里有一个例子:

```py
(Pdb) help where
w(here)
        Print a stack trace, with the most recent frame at the bottom.
        An arrow indicates the "current frame", which determines the
        context of most commands.  'bt' is an alias for this command.
```

自己去试试吧！

### 包扎

成功调试代码需要练习。Python 为您提供了一种无需安装任何其他东西就能调试代码的方法，这很棒。您会发现在 IDE 中使用`breakpoint()`来启用断点也非常方便。

在本文中，您了解了以下内容:

*   从 REPL 出发
*   在命令行上启动`pdb`
*   逐句通过代码
*   用`set_trace()`创建断点
*   在`pdb`中添加断点
*   使用内置的`breakpoint()`功能
*   获得帮助

您应该去尝试在自己的代码中使用您在这里学到的知识。将故意的错误添加到代码中，然后通过调试器运行它们，这是了解事情如何工作的好方法！
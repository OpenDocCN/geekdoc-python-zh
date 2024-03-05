# 用 Pdb 调试 Python

> 原文：<https://realpython.com/python-debugging-pdb/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解: [**Python 调试用 pdb**](/courses/python-debugging-pdb/)

调试应用程序有时可能是不受欢迎的活动。你在时间紧迫的情况下忙于工作，你只是想让它工作。然而，在其他时候，您可能正在学习一种新的语言特性或尝试一种新的方法，并且希望更深入地了解某些东西是如何工作的。

不管在什么情况下，调试代码都是必要的，所以在调试器中舒适地工作是一个好主意。在本教程中，我将向您展示使用 pdb(Python 的交互式源代码调试器)的基础知识。

我将带您了解 pdb 的一些常见用法。当你真的需要它的时候，你可以把这个教程加入书签，以便快速参考。pdb 和其他调试器是不可或缺的工具。当你需要一个调试器时，没有替代品。你真的需要它。

本教程结束时，您将知道如何使用调试器来查看应用程序中任何[变量](https://realpython.com/python-variables/)的状态。您还可以随时停止和恢复应用程序的执行流程，这样您就可以确切地看到每一行代码是如何影响其内部状态的。

这对于追踪难以发现的错误非常有用，并且允许您更快、更可靠地修复错误代码。有时，在 pdb 中单步执行代码并看到值是如何变化的，这真的令人大开眼界，并带来“啊哈”时刻，偶尔还会出现“掌脸”。

pdb 是 Python 标准库的一部分，因此它总是存在并可供使用。如果您需要在无法访问您所熟悉的 GUI 调试器的环境中调试代码，这可能是一个救命稻草。

本教程中的示例代码使用 Python 3.6。你可以在 [GitHub](https://github.com/natej/pdb-basics) 上找到这些例子的源代码。

在本教程的最后，有一个关于[基本 pdb 命令](#essential-pdb-commands)的快速参考。

还有一个可打印的 pdb 命令参考，您可以在调试时用作备忘单:

**免费赠品:** [点击此处获取一份可打印的“pdb 命令参考”(PDF)](#) 放在办公桌上，调试时参考。

## 入门:打印变量值

在第一个例子中，我们将看看使用 pdb 的最简单形式:检查变量值。

在要中断调试器的位置插入以下代码:

```py
import pdb; pdb.set_trace()
```

当执行上面的代码行时，Python 会停下来，等待您告诉它下一步该做什么。你会看到一个`(Pdb)`提示。这意味着您现在在交互式调试器中处于暂停状态，可以输入命令了。

从 Python 3.7 开始，[有另一种方式进入调试器](https://realpython.com/python37-new-features/#the-breakpoint-built-in)。 [PEP 553](https://www.python.org/dev/peps/pep-0553) 描述了内置函数`breakpoint()`，使进入调试器变得容易和一致:

```py
breakpoint()
```

默认情况下，`breakpoint()`将[导入](https://realpython.com/absolute-vs-relative-python-imports/) `pdb`并调用`pdb.set_trace()`，如上图。然而，使用`breakpoint()`更加灵活，允许您通过它的 API 和环境变量`PYTHONBREAKPOINT`的使用来控制调试行为。例如，在您的环境中设置`PYTHONBREAKPOINT=0`将完全禁用`breakpoint()`，从而禁用调试。如果你正在使用 Python 3.7 或更高版本，我鼓励你使用`breakpoint()`而不是`pdb.set_trace()`。

您还可以直接从命令行运行 Python 并传递选项`-m pdb`，从而在不修改源代码和使用`pdb.set_trace()`或`breakpoint()`的情况下进入调试器。如果您的应用程序接受[命令行参数](https://realpython.com/python-command-line-arguments/)，像平常一样在文件名后传递它们。例如:

```py
$ python3 -m pdb app.py arg1 arg2
```

有许多 pdb 命令可用。在本教程的最后，有一个[基本 pdb 命令](#essential-pdb-commands)的列表。现在，让我们使用`p`命令打印一个变量的值。在`(Pdb)`提示符下输入`p variable_name`打印其值。

让我们看看例子。下面是`example1.py`来源:

```py
#!/usr/bin/env python3

filename = __file__
import pdb; pdb.set_trace()
print(f'path = {filename}')
```

如果您从 shell 中运行此命令，您应该得到以下输出:

```py
$ ./example1.py 
> /code/example1.py(5)<module>()
-> print(f'path = {filename}')
(Pdb)
```

> 如果你在从命令行运行例子或你自己的代码时有困难，请阅读[我如何使用 Python 制作我自己的命令行命令？](https://dbader.org/blog/how-to-make-command-line-commands-with-python)如果你在 Windows 上，查看 [Python Windows 常见问题](https://docs.python.org/3.6/faq/windows.html)。

现在输入`p filename`。您应该看到:

```py
(Pdb) p filename
'./example1.py'
(Pdb)
```

因为您是在 shell 中使用 CLI(命令行界面)，所以要注意字符和格式。他们会给你你需要的背景:

*   从第一行开始，告诉你你在哪个源文件中。在文件名之后，括号中是当前的行号。接下来是函数的名称。在这个例子中，因为我们没有在函数内部和模块级别暂停，所以我们看到了`<module>()`。
*   `->`从第二行开始，是 Python 暂停的当前源代码行。这一行还没有执行。在本例中，这是来自上面的`>`行的`example1.py`中的`5`行。
*   `(Pdb)`是 pdb 的提示。它在等待命令。

使用命令`q`退出调试并退出。

[*Remove ads*](/account/join/)

## 打印表达式

当使用 print 命令`p`时，您传递的是一个将由 Python 计算的表达式。如果您传递一个变量名，pdb 打印它的当前值。但是，您可以做更多的工作来调查您正在运行的应用程序的状态。

在这个例子中，调用了函数`get_path()`。为了检查这个函数中发生了什么，我插入了一个对`pdb.set_trace()`的调用，以便在它返回之前暂停执行:

```py
#!/usr/bin/env python3

import os

def get_path(filename):
    """Return file's path or empty string if no path."""
    head, tail = os.path.split(filename)
    import pdb; pdb.set_trace()
    return head

filename = __file__
print(f'path = {get_path(filename)}')
```

如果您从 shell 中运行此命令，您应该会得到以下输出:

```py
$ ./example2.py 
> /code/example2.py(10)get_path()
-> return head
(Pdb)
```

我们在哪里？

*   `>`:我们在源文件`example2.py`中，在函数`get_path()`的`10`行。这是`p`命令用来解析变量名的参考框架，即当前范围或上下文。
*   `->`:执行在`return head`暂停。这一行还没有执行。这是功能`get_path()`中`example2.py`的`10`线，来自上面的`>`线。

让我们打印一些表达式来看看应用程序的当前状态。我最初使用命令`ll` (longlist)来列出函数的源代码:

```py
(Pdb) ll
 6     def get_path(filename):
 7         """Return file's path or empty string if no path."""
 8         head, tail = os.path.split(filename)
 9         import pdb; pdb.set_trace()
 10  ->     return head
(Pdb) p filename
'./example2.py'
(Pdb) p head, tail
('.', 'example2.py')
(Pdb) p 'filename: ' + filename
'filename: ./example2.py'
(Pdb) p get_path
<function get_path at 0x100760e18>
(Pdb) p getattr(get_path, '__doc__')
"Return file's path or empty string if no path."
(Pdb) p [os.path.split(p)[1] for p in os.path.sys.path]
['pdb-basics', 'python36.zip', 'python3.6', 'lib-dynload', 'site-packages']
(Pdb)
```

您可以将任何有效的 Python 表达式传递给`p`进行评估。

当您正在调试并希望在运行时直接在应用程序中测试替代实现时，这尤其有用。

你也可以使用命令`pp`(美化打印)来美化表达式。如果您想要打印具有大量输出的变量或表达式，例如[列表](https://realpython.com/python-lists-tuples/)和字典，这很有帮助。如果可能的话，美化打印将对象保持在一行上，或者如果它们不适合允许的宽度，则将它们分成多行。

## 单步执行代码

调试时，您可以使用两个命令来逐句通过代码:

| 命令 | 描述 |
| --- | --- |
| `n`(下一个) | 继续执行，直到到达当前函数的下一行或返回。 |
| `s`(步骤) | 执行当前行，并在第一个可能的时机停止(在调用的函数中或在当前函数中)。 |

> 有第三个命令名为`unt`(直到)。与`n`(下)有关。我们将在本教程后面的[继续执行](#continuing-execution)一节中讨论它。

`n`(下一步)和`s`(下一步)的区别在于 pdb 停止的地方。

使用`n` (next)继续执行，直到下一行，并停留在当前函数内，即如果调用了一个外部函数，则不在该函数内停止。把下一步想成“留在本地”或者“跨过去”。

使用`s`(步骤)执行当前行，如果调用了一个外来函数，则在该函数中停止。把 step 想成“踏入”。如果在另一个功能中停止执行，`s`将打印`--Call--`。

当到达当前函数的末尾时，`n`和`s`都将停止执行，并在`->`之后的下一行末尾打印`--Return--`和返回值。

让我们看一个使用这两个命令的例子。下面是`example3.py`来源:

```py
#!/usr/bin/env python3

import os

def get_path(filename):
    """Return file's path or empty string if no path."""
    head, tail = os.path.split(filename)
    return head

filename = __file__
import pdb; pdb.set_trace()
filename_path = get_path(filename)
print(f'path = {filename_path}')
```

如果您从 shell 中运行并输入`n`，您应该得到输出:

```py
$ ./example3.py 
> /code/example3.py(14)<module>()
-> filename_path = get_path(filename)
(Pdb) n
> /code/example3.py(15)<module>()
-> print(f'path = {filename_path}')
(Pdb)
```

随着`n`(下一条)，我们停在了下一条线`15`。我们在`<module>()`“呆在当地”，并“跳过”了对`get_path()`的呼叫。函数是`<module>()`,因为我们目前在模块级别，没有在另一个函数中暂停。

让我们试试`s`:

```py
$ ./example3.py 
> /code/example3.py(14)<module>()
-> filename_path = get_path(filename)
(Pdb) s
--Call--
> /code/example3.py(6)get_path()
-> def get_path(filename):
(Pdb)
```

使用`s`(步骤)，我们在函数`get_path()`的第`6`行停止，因为它在第`14`行被调用。注意`s`命令后的一行`--Call--`。

方便的是，pdb 会记住您的最后一个命令。如果你正在单步执行大量代码，你可以按下 `Enter` 来重复最后一个命令。

下面是一个使用`s`和`n`单步调试代码的例子。我最初输入`s`是因为我想“进入”功能`get_path()`并停止。然后我输入`n`一次来“停留在本地”或“跳过”任何其他函数调用，并按下 `Enter` 来重复`n`命令，直到我到达最后一个源代码行。

```py
$ ./example3.py 
> /code/example3.py(14)<module>()
-> filename_path = get_path(filename)
(Pdb) s
--Call--
> /code/example3.py(6)get_path()
-> def get_path(filename):
(Pdb) n
> /code/example3.py(8)get_path()
-> head, tail = os.path.split(filename)
(Pdb) 
> /code/example3.py(9)get_path()
-> return head
(Pdb) 
--Return--
> /code/example3.py(9)get_path()->'.'
-> return head
(Pdb) 
> /code/example3.py(15)<module>()
-> print(f'path = {filename_path}')
(Pdb) 
path = .
--Return--
> /code/example3.py(15)<module>()->None
-> print(f'path = {filename_path}')
(Pdb)
```

注意线`--Call--`和`--Return--`。这是 pdb 让你知道为什么执行被停止。`n`(下一步)和`s`(步骤)将在功能返回前停止。这就是为什么你会看到上面的`--Return--`线。

还要注意在上面第一个`--Return--`之后的行尾的`->'.'`:

```py
--Return--
> /code/example3.py(9)get_path()->'.'
-> return head
(Pdb)
```

当 pdb 在函数返回之前停止在函数末尾时，它也会为您打印返回值。在这个例子中是`'.'`。

[*Remove ads*](/account/join/)

### 清单源代码

不要忘记命令`ll` (longlist:列出当前函数或框架的全部源代码)。当您在单步执行不熟悉的代码时，或者您只想查看整个函数的上下文时，这真的很有帮助。

这里有一个例子:

```py
$ ./example3.py 
> /code/example3.py(14)<module>()
-> filename_path = get_path(filename)
(Pdb) s
--Call--
> /code/example3.py(6)get_path()
-> def get_path(filename):
(Pdb) ll
 6  -> def get_path(filename):
 7         """Return file's path or empty string if no path."""
 8         head, tail = os.path.split(filename)
 9         return head
(Pdb)
```

要查看更短的代码片段，使用命令`l` (list)。如果没有参数，它将在当前行周围打印 11 行，或者继续前面的列表。传递参数`.`总是列出当前行周围的 11 行:`l .`

```py
$ ./example3.py 
> /code/example3.py(14)<module>()
-> filename_path = get_path(filename)
(Pdb) l
 9         return head
 10 
 11 
 12     filename = __file__
 13     import pdb; pdb.set_trace()
 14  -> filename_path = get_path(filename)
 15     print(f'path = {filename_path}')
[EOF]
(Pdb) l
[EOF]
(Pdb) l .
 9         return head
 10 
 11 
 12     filename = __file__
 13     import pdb; pdb.set_trace()
 14  -> filename_path = get_path(filename)
 15     print(f'path = {filename_path}')
[EOF]
(Pdb)
```

## 使用断点

断点非常方便，可以节省你很多时间。不要遍历您不感兴趣的几十行，只需在您想要研究的地方创建一个断点。或者，您也可以告诉 pdb 仅在特定条件为真时才中断。

使用命令`b` (break)设置断点。您可以指定停止执行的行号或函数名。

break 的语法是:

```py
b(reak) [ ([filename:]lineno | function) [, condition] ]
```

如果行号`lineno`前没有指定`filename:`，则使用当前源文件。

请注意`b`的可选第二个参数:`condition`。这个很厉害。想象一下，只有在特定条件存在的情况下，您才想要中断。如果您将 Python 表达式作为第二个参数传递，那么当表达式的值为 true 时，pdb 将会中断。我们将在下面的例子中这样做。

在这个例子中，有一个实用模块`util.py`。让我们在函数`get_path()`中设置一个断点来停止执行。

下面是主脚本`example4.py`的源代码:

```py
#!/usr/bin/env python3

import util

filename = __file__
import pdb; pdb.set_trace()
filename_path = util.get_path(filename)
print(f'path = {filename_path}')
```

下面是实用程序模块`util.py`的源代码:

```py
def get_path(filename):
    """Return file's path or empty string if no path."""
    import os
    head, tail = os.path.split(filename)
    return head
```

首先，让我们使用源文件名和行号设置一个断点:

```py
$ ./example4.py 
> /code/example4.py(7)<module>()
-> filename_path = util.get_path(filename)
(Pdb) b util:5
Breakpoint 1 at /code/util.py:5
(Pdb) c
> /code/util.py(5)get_path()
-> return head
(Pdb) p filename, head, tail
('./example4.py', '.', 'example4.py')
(Pdb)
```

命令`c`(继续)继续执行，直到找到断点。

接下来，让我们使用函数名设置一个断点:

```py
$ ./example4.py 
> /code/example4.py(7)<module>()
-> filename_path = util.get_path(filename)
(Pdb) b util.get_path
Breakpoint 1 at /code/util.py:1
(Pdb) c
> /code/util.py(3)get_path()
-> import os
(Pdb) p filename
'./example4.py'
(Pdb)
```

输入不带参数的`b`来查看所有断点的列表:

```py
(Pdb) b
Num Type         Disp Enb   Where
1   breakpoint   keep yes   at /code/util.py:1
(Pdb)
```

您可以使用命令`disable bpnumber`和`enable bpnumber`禁用和重新启用断点。`bpnumber`是断点列表第一列`Num`中的断点号。请注意`Enb`列的值发生了变化:

```py
(Pdb) disable 1
Disabled breakpoint 1 at /code/util.py:1
(Pdb) b
Num Type         Disp Enb   Where
1   breakpoint   keep no    at /code/util.py:1
(Pdb) enable 1
Enabled breakpoint 1 at /code/util.py:1
(Pdb) b
Num Type         Disp Enb   Where
1   breakpoint   keep yes   at /code/util.py:1
(Pdb)
```

要删除断点，使用命令`cl`(清除):

```py
cl(ear) filename:lineno
cl(ear) [bpnumber [bpnumber...]]
```

现在让我们使用一个 Python 表达式来设置一个断点。想象一下这样一种情况，只有当有问题的函数收到某个输入时，您才想要中断。

在这个示例场景中，`get_path()`函数在接收相对路径时失败，即文件的路径不是以`/`开头。在这种情况下，我将创建一个计算结果为 true 的表达式，并将其作为第二个参数传递给`b`:

```py
$ ./example4.py 
> /code/example4.py(7)<module>()
-> filename_path = util.get_path(filename)
(Pdb) b util.get_path, not filename.startswith('/')
Breakpoint 1 at /code/util.py:1
(Pdb) c
> /code/util.py(3)get_path()
-> import os
(Pdb) a
filename = './example4.py'
(Pdb)
```

在创建了上面的断点并输入`c`继续执行之后，当表达式的值为 true 时，pdb 停止。命令`a` (args)打印当前函数的参数列表。

在上面的示例中，当您使用函数名而不是行号设置断点时，请注意，表达式应该只使用在输入函数时可用的函数参数或全局变量。否则，无论表达式的值是什么，断点都将停止在函数中执行。

如果您需要中断使用带有位于函数内部的变量名的表达式，即变量名不在函数的参数列表中，请指定行号:

```py
$ ./example4.py 
> /code/example4.py(7)<module>()
-> filename_path = util.get_path(filename)
(Pdb) b util:5, not head.startswith('/')
Breakpoint 1 at /code/util.py:5
(Pdb) c
> /code/util.py(5)get_path()
-> return head
(Pdb) p head
'.'
(Pdb) a
filename = './example4.py'
(Pdb)
```

您也可以使用命令`tbreak`设置一个临时断点。第一次击中时会自动移除。它使用与`b`相同的参数。

[*Remove ads*](/account/join/)

## 继续执行

到目前为止，我们已经看了用`n`(下一步)和`s`(单步)单步执行代码，以及用`b`(中断)和`c`(继续)使用断点。

还有一个相关的命令:`unt`(直到)。

使用`unt`像`c`一样继续执行，但是在比当前行大的下一行停止。有时候`unt`使用起来更方便快捷，而且正是你想要的。我将在下面用一个例子来说明这一点。

让我们先来看看`unt`的语法和描述:

| 命令 | 句法 | 描述 |
| --- | --- | --- |
| `unt` | unt(il)[line] | 如果没有`lineno`，则继续执行，直到到达编号大于当前编号的行。使用`lineno`，继续执行，直到到达一个编号大于或等于该编号的行。在这两种情况下，当当前帧返回时也停止。 |

根据是否传递行号参数`lineno`，`unt`可以有两种行为方式:

*   如果没有`lineno`，则继续执行，直到到达编号大于当前编号的行。这个类似于`n`(下一个)。这是执行和“单步执行”代码的另一种方式。`n`和`unt`的区别在于`unt`只有在到达比当前行大的行时才会停止。`n`将在下一个逻辑执行行停止。
*   使用`lineno`，继续执行，直到到达一个编号大于或等于该编号的行。这就像带有行号参数的`c` (continue)。

在这两种情况下，`unt`在当前帧(函数)返回时停止，就像`n`(下一步)和`s`(下一步)一样。

使用`unt`要注意的主要行为是，当达到当前或指定行的行号**大于或等于**时，它将停止。

当您想继续执行并在当前源文件中停止时，使用`unt`。你可以把它看作是`n`(下一个)和`b`(中断)的混合体，这取决于你是否传递了一个行号参数。

在下面的例子中，有一个带有循环的函数。这里，您希望继续执行代码并在循环后停止，而不单步执行循环的每个迭代或设置断点:

以下是`example4unt.py`的示例源:

```py
#!/usr/bin/env python3

import os

def get_path(fname):
    """Return file's path or empty string if no path."""
    import pdb; pdb.set_trace()
    head, tail = os.path.split(fname)
    for char in tail:
        pass  # Check filename char
    return head

filename = __file__
filename_path = get_path(filename)
print(f'path = {filename_path}')
```

并且控制台输出使用`unt`:

```py
$ ./example4unt.py 
> /code/example4unt.py(9)get_path()
-> head, tail = os.path.split(fname)
(Pdb) ll
 6     def get_path(fname):
 7         """Return file's path or empty string if no path."""
 8         import pdb; pdb.set_trace()
 9  ->     head, tail = os.path.split(fname)
 10         for char in tail:
 11             pass  # Check filename char
 12         return head
(Pdb) unt
> /code/example4unt.py(10)get_path()
-> for char in tail:
(Pdb) 
> /code/example4unt.py(11)get_path()
-> pass  # Check filename char
(Pdb) 
> /code/example4unt.py(12)get_path()
-> return head
(Pdb) p char, tail
('y', 'example4unt.py')
```

首先使用`ll`命令打印函数的源代码，然后使用`unt`。pdb 记得最后输入的命令，所以我只需按下 `Enter` 来重复`unt`命令。这将继续执行整个代码，直到到达比当前行更长的源代码行。

请注意，在上面的控制台输出中，pdb 仅在第`10`和`11`行停止一次。由于使用了`unt`，执行仅在循环的第一次迭代中停止。然而，循环的每次迭代都被执行。这可以在输出的最后一行中得到验证。`char`变量的值`'y'`等于`tail`值`'example4unt.py'`中的最后一个字符。

## 显示表达式

类似于用`p`和`pp`打印表达式，您可以使用命令`display [expression]`告诉 pdb 在执行停止时自动显示表达式的值，如果它发生了变化。使用命令`undisplay [expression]`清除一个显示表达式。

以下是这两个命令的语法和描述:

| 命令 | 句法 | 描述 |
| --- | --- | --- |
| `display` | 显示[表情] | 每次在当前帧停止执行时，显示`expression`的值(如果它改变了)。如果没有`expression`，列出当前帧的所有显示表达式。 |
| `undisplay` | 不显示[表情] | 在当前帧中不再显示`expression`。没有`expression`，清除当前帧的所有显示表达式。 |

下面是一个例子，`example4display.py`，演示了它在循环中的用法:

```py
$ ./example4display.py 
> /code/example4display.py(9)get_path()
-> head, tail = os.path.split(fname)
(Pdb) ll
 6     def get_path(fname):
 7         """Return file's path or empty string if no path."""
 8         import pdb; pdb.set_trace()
 9  ->     head, tail = os.path.split(fname)
 10         for char in tail:
 11             pass  # Check filename char
 12         return head
(Pdb) b 11
Breakpoint 1 at /code/example4display.py:11
(Pdb) c
> /code/example4display.py(11)get_path()
-> pass  # Check filename char
(Pdb) display char
display char: 'e'
(Pdb) c
> /code/example4display.py(11)get_path()
-> pass  # Check filename char
display char: 'x'  [old: 'e']
(Pdb) 
> /code/example4display.py(11)get_path()
-> pass  # Check filename char
display char: 'a'  [old: 'x']
(Pdb) 
> /code/example4display.py(11)get_path()
-> pass  # Check filename char
display char: 'm'  [old: 'a']
```

在上面的输出中，pdb 自动显示了`char`变量的值，因为每次遇到断点时，它的值都会改变。有时这很有帮助，而且正是您想要的，但是还有另一种使用`display`的方法。

您可以多次输入`display`来建立一个观察表达式列表。这可能比`p`更容易使用。添加完您感兴趣的所有表达式后，只需输入`display`即可查看当前值:

```py
$ ./example4display.py 
> /code/example4display.py(9)get_path()
-> head, tail = os.path.split(fname)
(Pdb) ll
 6     def get_path(fname):
 7         """Return file's path or empty string if no path."""
 8         import pdb; pdb.set_trace()
 9  ->     head, tail = os.path.split(fname)
 10         for char in tail:
 11             pass  # Check filename char
 12         return head
(Pdb) b 11
Breakpoint 1 at /code/example4display.py:11
(Pdb) c
> /code/example4display.py(11)get_path()
-> pass  # Check filename char
(Pdb) display char
display char: 'e'
(Pdb) display fname
display fname: './example4display.py'
(Pdb) display head
display head: '.'
(Pdb) display tail
display tail: 'example4display.py'
(Pdb) c
> /code/example4display.py(11)get_path()
-> pass  # Check filename char
display char: 'x'  [old: 'e']
(Pdb) display
Currently displaying:
char: 'x'
fname: './example4display.py'
head: '.'
tail: 'example4display.py'
```

[*Remove ads*](/account/join/)

## Python 来电显示

在这最后一部分，我们将在目前所学的基础上，以一个不错的回报结束。我用“来电显示”这个名字来指代电话系统的来电显示功能。这正是这个例子所展示的，除了它适用于 Python。

下面是主脚本`example5.py`的源代码:

```py
#!/usr/bin/env python3

import fileutil

def get_file_info(full_fname):
    file_path = fileutil.get_path(full_fname)
    return file_path

filename = __file__
filename_path = get_file_info(filename)
print(f'path = {filename_path}')
```

这是实用模块`fileutil.py`:

```py
def get_path(fname):
    """Return file's path or empty string if no path."""
    import os
    import pdb; pdb.set_trace()
    head, tail = os.path.split(fname)
    return head
```

在这个场景中，假设有一个大型代码库，它在一个实用程序模块`get_path()`中有一个函数，该函数被无效输入调用。然而，它在不同的包中从许多地方被调用。

如何找到打电话的人？

使用命令`w`(其中)打印堆栈跟踪，最新的帧在底部:

```py
$ ./example5.py 
> /code/fileutil.py(5)get_path()
-> head, tail = os.path.split(fname)
(Pdb) w
 /code/example5.py(12)<module>()
-> filename_path = get_file_info(filename)
 /code/example5.py(7)get_file_info()
-> file_path = fileutil.get_path(full_fname)
> /code/fileutil.py(5)get_path()
-> head, tail = os.path.split(fname)
(Pdb)
```

如果这看起来令人困惑，或者如果您不确定什么是堆栈跟踪或帧，请不要担心。我将在下面解释这些术语。这并不像听起来那么难。

因为最近的帧在底部，所以从那里开始，从底部向上读取。查看以`->`开头的行，但是跳过第一个实例，因为在函数`get_path()`中`pdb.set_trace()`被用于输入 pdb。在这个例子中，调用函数`get_path()`的源代码行是:

```py
-> file_path = fileutil.get_path(full_fname)
```

每个`->`上面的行包含文件名、行号(在括号中)和源代码所在的函数名。所以打电话的人是:

```py
 /code/example5.py(7)get_file_info()
-> file_path = fileutil.get_path(full_fname)
```

在这个用于演示的小例子中，这并不奇怪，但是想象一下一个大型应用程序，其中您设置了一个带有条件的断点，以识别错误输入值的来源。

现在我们知道如何找到打电话的人了。

但是这个堆栈跟踪和框架的东西呢？

一个[堆栈跟踪](https://realpython.com/python-traceback/)只是 Python 创建的用来跟踪函数调用的所有帧的列表。框架是 Python 在调用函数时创建的数据结构，在函数返回时删除。堆栈只是在任何时间点的帧或函数调用的有序列表。(函数调用)堆栈在应用程序的整个生命周期中随着函数的调用和返回而增长和收缩。

打印时，这个有序的帧列表，即堆栈，被称为[堆栈跟踪](https://realpython.com/courses/python-traceback/)。您可以通过输入命令`w`随时看到它，就像我们在上面查找调用者一样。

> 详见维基百科上的这篇 [call stack 文章。](https://en.wikipedia.org/wiki/Call_stack)

为了更好地理解和利用 pdb，让我们更仔细地看看对`w`的帮助:

```py
(Pdb) h w
w(here)
 Print a stack trace, with the most recent frame at the bottom.
 An arrow indicates the "current frame", which determines the
 context of most commands. 'bt' is an alias for this command.
```

**pdb 所说的“当前帧”是什么意思？**

将当前帧视为 pdb 停止执行的当前函数。换句话说，当前帧是应用程序当前暂停的地方，并被用作 pdb 命令(如`p` (print))的参考“帧”。

`p`和其他命令将在需要时使用当前帧作为上下文。在`p`的情况下，当前帧将用于查找和打印变量引用。

当 pdb 打印堆栈跟踪时，箭头`>`指示当前帧。

这有什么用？

您可以使用两个命令`u`(向上)和`d`(向下)来改变当前帧。与`p`相结合，这允许你在任何一帧中沿着调用栈的任何一点检查应用程序中的变量和状态。

以下是这两个命令的语法和描述:

| 命令 | 句法 | 描述 |
| --- | --- | --- |
| `u` | 计数 | 在堆栈跟踪中将当前帧`count`(默认为一个)上移一级(到一个更老的帧)。 |
| `d` | d(自己的)[计数] | 在堆栈跟踪中将当前帧`count`(默认为一个)向下移动一级(到一个较新的帧)。 |

让我们看一个使用`u`和`d`命令的例子。在这个场景中，我们想要检查`example5.py`中函数`get_file_info()`的局部变量`full_fname`。为此，我们必须使用命令`u`将当前帧向上改变一级:

```py
$ ./example5.py 
> /code/fileutil.py(5)get_path()
-> head, tail = os.path.split(fname)
(Pdb) w
 /code/example5.py(12)<module>()
-> filename_path = get_file_info(filename)
 /code/example5.py(7)get_file_info()
-> file_path = fileutil.get_path(full_fname)
> /code/fileutil.py(5)get_path()
-> head, tail = os.path.split(fname)
(Pdb) u
> /code/example5.py(7)get_file_info()
-> file_path = fileutil.get_path(full_fname)
(Pdb) p full_fname
'./example5.py'
(Pdb) d
> /code/fileutil.py(5)get_path()
-> head, tail = os.path.split(fname)
(Pdb) p fname
'./example5.py'
(Pdb)
```

对`pdb.set_trace()`的调用在函数`get_path()`的`fileutil.py`中，所以当前帧最初设置在那里。您可以在上面的第一行输出中看到它:

```py
> /code/fileutil.py(5)get_path()
```

为了访问并打印`example5.py`中函数`get_file_info()`中的局部变量`full_fname`，命令`u`被用于上移一级:

```py
(Pdb) u
> /code/example5.py(7)get_file_info()
-> file_path = fileutil.get_path(full_fname)
```

请注意，在上面的`u`输出中，pdb 在第一行的开头打印了箭头`>`。这是 pdb，让您知道帧已被更改，此源位置现在是当前帧。变量`full_fname`现在是可访问的。此外，重要的是要意识到第二行以`->`开始的源代码行已经被执行。自从这个框架在堆栈中上移后，`fileutil.get_path()`就被调用了。使用`u`，我们将堆栈向上移动(从某种意义上说，及时返回)到调用`fileutil.get_path()`的函数`example5.get_file_info()`。

继续这个例子，在`full_fname`被打印后，使用`d`将当前帧移动到其原始位置，并打印`get_path()`中的局部变量`fname`。

如果我们想的话，我们可以通过将`count`参数传递给`u`或`d`来一次移动多个帧。例如，我们可以通过输入`u 2`进入`example5.py`中的模块级别:

```py
$ ./example5.py 
> /code/fileutil.py(5)get_path()
-> head, tail = os.path.split(fname)
(Pdb) u 2
> /code/example5.py(12)<module>()
-> filename_path = get_file_info(filename)
(Pdb) p filename
'./example5.py'
(Pdb)
```

当你在调试和思考许多不同的事情时，很容易忘记你在哪里。请记住，您总是可以使用名副其实的命令`w` (where)来查看执行在哪里暂停以及当前帧是什么。

[*Remove ads*](/account/join/)

## 基本 pdb 命令

一旦你在 pdb 上花了一点时间，你就会意识到一点点知识可以走很长的路。使用`h`命令总是可以获得帮助。

只需输入`h`或`help <topic>`即可获得所有命令的列表或特定命令或主题的帮助。

作为快速参考，这里列出了一些基本命令:

| 命令 | 描述 |
| --- | --- |
| `p` | 打印表达式的值。 |
| `pp` | 漂亮地打印一个表达式的值。 |
| `n` | 继续执行，直到到达当前函数的下一行或返回。 |
| `s` | 执行当前行，并在第一个可能的时机停止(在调用的函数中或在当前函数中)。 |
| `c` | 继续执行，仅在遇到断点时停止。 |
| `unt` | 继续执行，直到到达数字大于当前数字的那一行。使用行号参数，继续执行，直到到达行号大于或等于行号的行。 |
| `l` | 列出当前文件的源代码。如果没有参数，则在当前行周围列出 11 行，或者继续前面的列表。 |
| `ll` | 列出当前函数或框架的全部源代码。 |
| `b` | 不带参数，列出所有断点。使用行号参数，在当前文件的这一行设置一个断点。 |
| `w` | 打印堆栈跟踪，最新的帧在底部。箭头指示当前帧，它决定了大多数命令的上下文。 |
| `u` | 将堆栈跟踪中的当前帧数(默认为 1)向上移动一级(到一个较旧的帧)。 |
| `d` | 将堆栈跟踪中的当前帧计数(默认为 1)向下移动一级(到一个较新的帧)。 |
| `h` | 查看可用命令列表。 |
| `h <topic>` | 显示命令或主题的帮助。 |
| `h pdb` | 展示完整的 pdb 文档。 |
| `q` | 退出调试器并退出。 |

## 用 pdb 调试 Python:结论

在本教程中，我们介绍了 pdb 的一些基本和常见用法:

*   打印表达式
*   用`n`(下一步)和`s`(下一步)单步执行代码
*   使用断点
*   继续执行`unt`(直到)
*   显示表达式
*   查找函数的调用者

希望对你有帮助。如果您想了解更多信息，请参阅:

*   在您附近的 pdb 提示符下显示 pdb 的完整文档:`(Pdb) h pdb`
*   [Python 的 pdb 文档](https://docs.python.org/3/library/pdb.html)

示例中使用的源代码可以在相关的 [GitHub 库](https://github.com/natej/pdb-basics)中找到。请务必查看我们的可打印 pdb 命令参考，您可以在调试时将其用作备忘单:

**免费赠品:** [点击此处获取一份可打印的“pdb 命令参考”(PDF)](#) 放在办公桌上，调试时参考。

另外，如果你想尝试一个基于 GUI 的 Python 调试器，请阅读我们的[Python ide 和编辑器指南](https://realpython.com/python-ides-code-editors-guide/)，看看哪些选项最适合你。快乐的蟒蛇！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和写好的教程一起看，加深理解: [**Python 调试用 pdb**](/courses/python-debugging-pdb/)*******
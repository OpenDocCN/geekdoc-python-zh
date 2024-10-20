# Python 101:Python 调试器介绍

> 原文：<https://www.blog.pythonlibrary.org/2014/03/19/pytho-101-an-introduction-to-pythons-debugger/>

Python 自带了名为 **pdb** 的调试器模块。该模块为您的 Python 程序提供了一个交互式源代码调试器。你可以设置断点，单步调试你的代码，检查堆栈帧等等。我们将了解本模块的以下方面:

*   如何启动调试器
*   单步执行您的代码
*   设置断点

让我们从创建一段快速代码来尝试调试开始。这里有一个愚蠢的例子:

```py

# debug_test.py

#----------------------------------------------------------------------
def doubler(a):
    """"""
    result = a*2
    print(result)
    return result

#----------------------------------------------------------------------
def main():
    """"""
    for i in range(1,10):
        doubler(i)

if __name__ == "__main__":
    main()

```

现在让我们学习如何针对这段代码运行调试器。

* * *

### 如何启动调试器

您可以通过三种不同的方式启动调试器。第一种方法是导入它并将**pdb.set_trace()**插入到您的代码中以启动调试器。您可以在空闲状态下导入调试器，并让它运行您的模块。或者您可以在命令行上调用调试器。在本节中，我们将重点讨论最后两种方法。我们将从在解释器(空闲)中使用它开始。打开一个终端(命令行窗口)并导航到保存上述代码示例的位置。然后启动 Python。现在执行以下操作:

```py

>>> import debug_test
>>> import pdb
>>> pdb.run('debug_test.main()')
> (1)<module>()
(Pdb) continue
2
4
6
8
10
12
14
16
18
>>>
```

在这里，我们导入我们的模块和 pdb。然后我们执行 pdb 的 **run** 方法，并告诉它调用我们模块的 **main** 方法。这将打开调试器的提示。在这里，我们键入**继续**来告诉它继续运行脚本。您也可以键入字母 **c** 作为**继续**的快捷键。当您继续时，调试器将继续执行，直到到达断点或脚本结束。

启动调试器的另一种方法是通过终端会话执行以下命令:

```py

python -m pdb debug_test.py

```

如果您以这种方式运行，您将看到略有不同的结果:

```py

-> def doubler(a):
(Pdb) c
2
4
6
8
10
12
14
16
18
The program finished and will be restarted

```

你会注意到在这个例子中我们使用了 **c** 而不是**继续**。您还会注意到调试器会在最后重新启动。这保留了调试器的状态(如断点)，比让调试器停止更有用。有时候，你需要反复检查代码几次，才能明白哪里出了问题。

让我们再深入一点，学习如何单步执行代码。

* * *

### 单步执行代码

如果您想一次单步执行一行代码，那么您可以使用 **step** (或者简称为“s”)命令。这里有一个片段供你观赏:

```py

C:\Users\mike>cd c:\py101

c:\py101>python -m pdb debug_test.py
> c:\py101\debug_test.py(4)()
-> def doubler(a):
(Pdb) step
> c:\py101\debug_test.py(11)<module>()
-> def main():
(Pdb) s
> c:\py101\debug_test.py(16)<module>()
-> if __name__ == "__main__":
(Pdb) s
> c:\py101\debug_test.py(17)<module>()
-> main()
(Pdb) s
--Call--
> c:\py101\debug_test.py(11)main()
-> def main():
(Pdb) next
> c:\py101\debug_test.py(13)main()
-> for i in range(1,10):
(Pdb) s
> c:\py101\debug_test.py(14)main()
-> doubler(i)
(Pdb)
```

这里我们启动调试器，并告诉它进入代码。它从顶部开始，遍历前两个函数定义。然后，它到达条件，发现它应该执行 **main** 函数。我们进入主函数，然后使用 **next** 命令。**下一个**命令将执行一个被调用的函数，如果它遇到该函数而没有进入该函数。如果你想进入被调用的函数，那么你只需要使用 **step** 命令。

当你看到类似于**>c:\ py 101 \ debug _ test . py(13)main()**这样的一行时，你会想要注意圆括号中的数字。这个数字是代码中的当前行号。

您可以使用 **args** (或 **a** )将当前参数列表打印到屏幕上。另一个方便的命令是**跳转**(或 **j** )，后跟一个空格和您想要“跳转”到的行号。这让你能够跳过一堆单调的步进，到达你想要到达的那一行。这就引导我们学习断点！

* * *

### 设置断点

断点是代码中要暂停执行的一行。您可以通过调用**break**(或**b**)命令，后跟一个空格和要中断的行号来设置断点。还可以在行号前加上文件名和冒号，以在不同的文件中指定断点。break 命令还允许您使用**function**参数设置断点。还有一个**tbreak**命令，该命令将设置一个临时断点，当遇到断点时会自动删除。

这里有一个例子:

```py

c:\py101>python -m pdb debug_test.py
> c:\py101\debug_test.py(4)()
-> def doubler(a):
(Pdb) break 6
Breakpoint 1 at c:\py101\debug_test.py:6
(Pdb) c
> c:\py101\debug_test.py(6)doubler()
-> result = a*2 
```

我们启动调试器，然后告诉它在第 6 行设置一个断点。然后我们继续，它在第 6 行停止，就像它应该的那样。现在是检查参数列表的好时机，看看它是否是你所期望的。现在输入 **args** 试试看。然后做另一个**继续**和另一个 **args** 来看看它是如何变化的。

* * *

### 包扎

您可以在调试器中使用许多其他命令。我建议阅读文档来了解其他的。但是，此时您应该能够有效地使用调试器来调试您自己的代码。

* * *

### 附加阅读

*   关于 [pdb 模块](http://docs.python.org/2.7/library/pdb.html)的 Python 文档
*   PyMOTW - [pdb](http://pymotw.com/2/pdb/)
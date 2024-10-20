# 如何用 timeit 为小块 Python 代码计时

> 原文：<https://www.blog.pythonlibrary.org/2014/01/30/how-to-time-small-pieces-of-python-code-with-timeit/>

有时候当你在编码的时候，你想知道一个特定的函数运行需要多长时间。这个主题称为性能分析或性能调优。Python 的标准库中内置了几个分析器，但是对于小段代码，使用 Python 的 **timeit** 模块会更容易。因此， **timeit** 将是本教程的重点。 **timeit** 模块使用特定于平台的方法来获得最准确的运行时间。基本上， **timeit** 模块将进行一次设置，运行代码 n 次，并返回运行所用的时间。通常它会输出一个“3 分中最好”的分数。奇怪的是，默认运行代码的次数是 1，000，000 次循环。 **timeit** 在 Linux / Mac 上用 time.time()计时，在 Windows 上用 time.clock()计时，以获得最准确的读数，这是大多数人不会想到的。

您可以从命令行或通过导入来运行 **timeit** 模块。我们将查看这两个用例。

* * *

### 在控制台中计时

在命令行上使用 timeit 模块非常简单。这里有几个例子:

```py

python -m timeit -s "[ord(x) for x in 'abcdfghi']"
100000000 loops, best of 3: 0.0115 usec per loop

python -m timeit -s "[chr(int(x)) for x in '123456789']"
100000000 loops, best of 3: 0.0119 usec per loop

```

这是怎么回事？当你在命令行上调用 Python 并给它传递“-m”选项时，你是在告诉它查找一个模块并把它作为主程序使用。“-s”告诉 **timeit** 模块运行一次设置。然后，它将代码运行 n 次循环，并返回 3 次运行的最佳平均值。对于这些愚蠢的例子，你不会看到太大的区别。让我们快速看一下 **timeit's** help，这样我们可以了解更多关于它是如何工作的:

```py

C:\Users\mdriscoll>python -m timeit -h
Tool for measuring execution time of small code snippets.

This module avoids a number of common traps for measuring execution
times.  See also Tim Peters' introduction to the Algorithms chapter in
the Python Cookbook, published by O'Reilly.

Library usage: see the Timer class.

Command line usage:
    python timeit.py [-n N] [-r N] [-s S] [-t] [-c] [-h] [--] [statement]

Options:
  -n/--number N: how many times to execute 'statement' (default: see below)
  -r/--repeat N: how many times to repeat the timer (default 3)
  -s/--setup S: statement to be executed once initially (default 'pass')
  -t/--time: use time.time() (default on Unix)
  -c/--clock: use time.clock() (default on Windows)
  -v/--verbose: print raw timing results; repeat for more digits precision
  -h/--help: print this usage message and exit
  --: separate options from statement, use when statement starts with -
  statement: statement to be timed (default 'pass')

A multi-line statement may be given by specifying each line as a
separate argument; indented lines are possible by enclosing an
argument in quotes and using leading spaces.  Multiple -s options are
treated similarly.

If -n is not given, a suitable number of loops is calculated by trying
successive powers of 10 until the total time is at least 0.2 seconds.

The difference in default timer function is because on Windows,
clock() has microsecond granularity but time()'s granularity is 1/60th
of a second; on Unix, clock() has 1/100th of a second granularity and
time() is much more precise.  On either platform, the default timer
functions measure wall clock time, not the CPU time.  This means that
other processes running on the same computer may interfere with the
timing.  The best thing to do when accurate timing is necessary is to
repeat the timing a few times and use the best time.  The -r option is
good for this; the default of 3 repetitions is probably enough in most
cases.  On Unix, you can use clock() to measure CPU time.

Note: there is a certain baseline overhead associated with executing a
pass statement.  The code here doesn't try to hide it, but you should
be aware of it.  The baseline overhead can be measured by invoking the
program without arguments.

The baseline overhead differs between Python versions!  Also, to
fairly compare older Python versions to Python 2.3, you may want to
use python -O for the older versions to avoid timing SET_LINENO
instructions.

```

这告诉使用所有我们可以通过的奇妙的标志，以及它们做什么。它也告诉我们一些关于 **timeit** 如何在幕后运作的事情。让我们写一个简单的函数，看看我们能否从命令行计时:

```py

# simple_func.py
def my_function():
    try:
        1 / 0
    except ZeroDivisionError:
        pass

```

这个函数所做的只是导致一个被忽略的错误。是的，这是一个愚蠢的例子。为了让 **timeit** 在命令行上运行这段代码，我们需要将代码导入到它的名称空间中，所以请确保您已经将当前的工作目录更改为该脚本所在的文件夹。然后运行以下命令:

```py

python -m timeit "import simple_func; simple_func.my_function()"
1000000 loops, best of 3: 1.77 usec per loop

```

这里导入函数，然后调用它。注意，我们用分号分隔导入和函数调用，Python 代码用引号括起来。现在我们准备学习如何在实际的 Python 脚本中使用 **timeit** 。

* * *

### 导入 timeit 进行测试

在代码中使用 timeit 模块也很容易。我们将使用之前相同的愚蠢脚本，并在下面向您展示如何操作:

```py

def my_function():
    try:
        1 / 0
    except ZeroDivisionError:
        pass

if __name__ == "__main__":
    import timeit
    setup = "from __main__ import my_function"
    print timeit.timeit("my_function()", setup=setup)

```

在这里，我们检查脚本是否正在直接运行(即没有导入)。如果是，那么我们导入 **timeit** ，创建一个设置字符串将函数导入到 **timeit 的**名称空间，然后我们调用 **timeit.timeit** 。您会注意到，我们用引号将对函数的调用传递出去，然后是设置字符串。这真的就是全部了！

* * *

### 包扎

现在你知道如何使用 **timeit** 模块了。它非常擅长计时简单的代码片段。您通常会将它用于您怀疑运行时间过长的代码。如果您想要更详细地了解代码中正在发生的事情，那么您可能希望切换到分析器。开心快乐编码！

* * *

*   关于 timeit 的 Python [文档](http://docs.python.org/2/library/timeit.html)
*   py motw-Time it "[对小部分 Python 代码的执行进行计时。](http://pymotw.com/2/timeit/)
*   在 timeit 上深入研究 Python [部分](http://www.diveintopython.net/performance_tuning/timeit.html)
*   关于 timeit 的梦想代码论坛教程
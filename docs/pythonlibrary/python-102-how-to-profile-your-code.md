# Python 102:如何剖析您的代码

> 原文：<https://www.blog.pythonlibrary.org/2014/03/20/python-102-how-to-profile-your-code/>

代码分析是试图在代码中找到瓶颈。剖析应该是为了找出代码的哪些部分耗时最长。一旦你知道了这一点，你就可以看看你的代码，并试图找到优化它的方法。Python 内置了三个分析器: **cProfile** 、 **profile** 和 **hotshot** 。根据 Python 文档， **hotshot** “不再维护，可能会在 Python 的未来版本中被删除”。 **profile** 模块是一个纯粹的 Python 模块，但是给被分析的程序增加了很多开销。因此，我们将关注于 **cProfile** ，它有一个模拟 Profile 模块的接口。

* * *

### 用概要文件分析代码

用 cProfile 分析代码真的很容易。你需要做的就是导入模块并调用它的 **run** 函数。让我们看一个简单的例子:

```py

>>> import hashlib
>>> import cProfile
>>> cProfile.run("hashlib.md5('abcdefghijkl').digest()")
         4 function calls in 0.000 CPU seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 :1(<module>)
        1    0.000    0.000    0.000    0.000 {_hashlib.openssl_md5}
        1    0.000    0.000    0.000    0.000 {method 'digest' of '_hashlib.HASH' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

这里我们导入了 **hashlib** 模块，并使用 cProfile 来分析 MD5 散列的创建。第一行显示有 4 个函数调用。下一行告诉我们结果是如何排序的。根据文献记载，*标准名称*是指最右边的列。这里有许多列。

*   **ncalls** 是发出的呼叫数。
*   **tottime** 是给定函数花费的总时间。
*   **percall** 指总时间除以 ncalls 的商
*   **累计时间**是在该功能和所有子功能中花费的累计时间。甚至对递归函数也很准确！
*   第二个 **percall 列**是累计时间除以原始调用的商
*   **filename:line no(function)**提供每个函数各自的数据

原始调用不是通过递归引起的。

这不是一个非常有趣的例子，因为没有明显的瓶颈。让我们创建一段带有内置瓶颈的代码，看看分析器是否能检测到它们。

```py

import time

#----------------------------------------------------------------------
def fast():
    """"""
    print("I run fast!")

#----------------------------------------------------------------------
def slow():
    """"""
    time.sleep(3)
    print("I run slow!")

#----------------------------------------------------------------------
def medium():
    """"""
    time.sleep(0.5)
    print("I run a little slowly...")

#----------------------------------------------------------------------
def main():
    """"""
    fast()
    slow()
    medium()

if __name__ == '__main__':
    main()

```

在这个例子中，我们创建了四个函数。前三种以不同的速度运行。**快速**功能将以正常速度运行；**中**功能大约需要半秒钟运行，**慢**功能大约需要 3 秒钟运行。**主**函数调用其他三个。现在让我们对这个愚蠢的小程序运行 cProfile:

```py

>>> import cProfile
>>> import ptest
>>> cProfile.run('ptest.main()')
I run fast!
I run slow!
I run a little slowly...
         8 function calls in 3.500 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.500    3.500 :1(<module>)
        1    0.000    0.000    0.500    0.500 ptest.py:15(medium)
        1    0.000    0.000    3.500    3.500 ptest.py:21(main)
        1    0.000    0.000    0.000    0.000 ptest.py:4(fast)
        1    0.000    0.000    3.000    3.000 ptest.py:9(slow)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    3.499    1.750    3.499    1.750 {time.sleep}
```

这一次，我们看到程序运行了 3.5 秒。如果您检查结果，您将看到 cProfile 已经将**慢速**函数标识为花费 3 秒来运行。那是继**主**功能之后的最大瓶颈。通常，当您发现类似这样的瓶颈时，您会试图找到一种更快的方法来执行您的代码，或者可能决定运行时是可以接受的。在这个例子中，我们知道加速这个函数的最好方法是删除 **time.sleep** 调用或者至少减少睡眠时间。

您也可以在命令行上调用 cProfile，而不是在解释器中使用它。有一种方法可以做到:

```py

python -m cProfile ptest.py

```

这将按照与我们之前相同的方式对您的脚本运行 cProfile。但是，如果您想保存分析器的输出，该怎么办呢？嗯，使用 cProfile 很简单！你需要做的就是给它传递 **-o** 命令，后跟输出文件的名称(或路径)。这里有一个例子:

```py

python -m cProfile -o output.txt ptest.py

```

不幸的是，它输出的文件不完全是人类可读的。如果你想读取这个文件，那么你需要使用 Python 的 **pstats** 模块。您可以使用 pstats 以各种方式格式化输出。下面是一些代码，展示了如何获得一些类似于我们目前所看到的输出:

```py

>>> import pstats
>>> p = pstats.Stats("output.txt")
>>> p.strip_dirs().sort_stats(-1).print_stats()
Thu Mar 20 18:32:16 2014    output.txt

         8 function calls in 3.501 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.501    3.501 ptest.py:1()
        1    0.001    0.001    0.500    0.500 ptest.py:15(medium)
        1    0.000    0.000    3.501    3.501 ptest.py:21(main)
        1    0.001    0.001    0.001    0.001 ptest.py:4(fast)
        1    0.001    0.001    3.000    3.000 ptest.py:9(slow)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    3.499    1.750    3.499    1.750 {time.sleep}

 <pstats.stats instance="" at=""></pstats.stats> 
```

**strip_dirs** 调用将从输出中去除所有到模块的路径，而 **sort_stats** 调用进行我们习惯看到的排序。在 cProfile 文档中有许多非常有趣的例子，展示了使用 pstats 模块提取信息的不同方法。

* * *

### 包扎

此时，您应该能够使用 cProfile 模块来帮助您诊断代码如此缓慢的原因。您可能还想看看 Python 的 **timeit** 模块。如果您不想处理复杂的概要分析，它允许您对代码的小部分进行计时。还有其他几个第三方模块也很适合进行概要分析，比如 [line_profiler](https://pythonhosted.org/line_profiler/) 和 [memory_profiler](https://github.com/fabianp/memory_profiler) 项目。

* * *

### 相关阅读

*   关于[概要模块](http://docs.python.org/2/library/profile.html)的 Python 文档
*   [如何用 timeit 为小块 Python 代码计时](https://www.blog.pythonlibrary.org/2014/01/30/how-to-time-small-pieces-of-python-code-with-timeit/)
*   [Python 性能分析指南](http://www.huyng.com/posts/python-performance-analysis/)
*   PyMOTW: [profile，cProfile，pstats](http://pymotw.com/2/profile/)
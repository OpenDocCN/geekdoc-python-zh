# 速度

CPython 作为最流行的 Python 环境，对于 CPU 密集型任务（CPU bound tasks）较慢，而 [PyPy](http://pypy.org) [http://pypy.org] 则较快。

使用稍作改动的 [David Beazley 的](http://www.dabeaz.com/GIL/gilvis/measure2.py) [http://www.dabeaz.com/GIL/gilvis/measure2.py] CPU 密集测试代码（增加了循环进行多轮测试）， 你可以看到 CPython 与 PyPy 之间的执行差距。

```py
# PyPy
$ ./pypy -V
Python 2.7.1 (7773f8fc4223, Nov 18 2011, 18:47:10)
[PyPy 1.7.0 with GCC 4.4.3]
$ ./pypy measure2.py
0.0683999061584
0.0483210086823
0.0388588905334
0.0440690517426
0.0695300102234 
```

```py
# CPython
$ ./python -V
Python 2.7.1
$ ./python measure2.py
1.06774401665
1.45412397385
1.51485204697
1.54693889618
1.60109114647 
```

## Context

### The GIL

[GIL](http://wiki.python.org/moin/GlobalInterpreterLock) [http://wiki.python.org/moin/GlobalInterpreterLock] (全局解释器锁)是 Python 支持多线程并行操作的方式。Python 的内存管理不是 线程安全的，所以 GIL 被创造出来避免多线程同时运行同一个 Python 代码。

David Beazley 有一个关于 GIL 如何工作的 [指导](http://www.dabeaz.com/python/UnderstandingGIL.pdf) [http://www.dabeaz.com/python/UnderstandingGIL.pdf] 。他也讨论了 Python3.2 中的 [新 GIL](http://www.dabeaz.com/python/NewGIL.pdf) [http://www.dabeaz.com/python/NewGIL.pdf] 他的结论是为了最大化一个 Python 程序的性能，应该对 GIL 工作方式有一个深刻的理解——它如何 影响你的特定程序，你拥有多少核，以及你程序瓶颈在哪。

### C 扩展

### The GIL

当写一个 C 扩展时必须 [特别关注](http://docs.python.org/c-api/init.html#threads) [http://docs.python.org/c-api/init.html#threads] 在解释器中注册你的线程。

## C 扩展

### Cython

[Cython](http://cython.org/) [http://cython.org/] 是 Python 语言的一个超集，对其你可以为 Python 写 C 或 C++模块。Cython 也使得你可以从已编译的 C 库中调用函数。使用 Cython 让你得以发挥 Python 的变量与操作的强类型优势。

这是一个 Cython 中的强类型例子。

```py
def primes(int kmax):
"""有一些 Cython 附加关键字的素数计算 """

    cdef int n, k, i
    cdef int p[1000]
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result 
```

将这个有一些附加关键字的寻找素数算法实现与下面这个纯 Python 实现比较：

```py
def primes(kmax):
"""标准 Python 语法下的素数计算"""

    p= range(1000)
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result 
```

注意，在 Cython 版本，在创建一个 Python 列表时，你声明了会被编译为 C 类型的整型和整型数组。

```py
def primes(int kmax):
    """有一些 Cython 附加关键字的素数计算 """

    cdef int n, k, i
    cdef int p[1000]
    result = [] 
```

```py
def primes(kmax):
    """标准 Python 语法下的素数计算"""

    p= range(1000)
    result = [] 
```

有什么差别呢？在上面的 Cython 版本中，你可以看到变量类型与整型数组像标准 C 一样被声明。 作为例子，第三行的 cdef int n,k,i 这个附加类型声明（整型）使得 Cython 编译器得以产生比 第二个版本更有效率的 C 代码。标准 Python 代码以 *.py 格式保存，而 Cython 以

> *.pyx 格式保存。

速度上有什么差异呢？看看这个！

```py
import time
#启动 pyx 编译器
import pyximport
pyximport.install()
#Cython 的素数算法实现
import primesCy
#Python 的素数算法实现
import primes

print "Cython:"
t1= time.time()
print primesCy.primes(500)
t2= time.time()
print "Cython time: %s" %(t2-t1)
print ""
print "Python"
t1= time.time()
print primes.primes(500)
t2= time.time()
print "Python time: %s" %(t2-t1) 
```

这两行代码需要一些说明：

```py
import pyximport
pyximport.install() 
```

pyximport 使得你可以导入 *.pyx 文件，（像 primesCy.pyx 这样的）。 pyximport.install() 命令使 Python 解释器可以打开 Cython 编译器直接编译出 *.so 格式 的 C 库。Cython 之后可以导入这个库到你的 Python 代码中，简便而有效。使用 time.time() 函数 你可以比较两个不同的在查找 500 个素数的调用长的时间消耗差异。在一个标准笔记本中 （双核 AMD E-450 1.6GHz），测量值是这样的：

```py
Cython time: 0.0054 seconds

Python time: 0.0566 seconds 
```

而这个是嵌入的 [ARM beaglebone](http://beagleboard.org/Products/BeagleBone) [http://beagleboard.org/Products/BeagleBone] 机的输出结果：

```py
Cython time: 0.0196 seconds

Python time: 0.3302 seconds 
```

### Pyrex

### Shedskin?

### Numba

待处理

Write about Numba and the autojit compiler for NumPy

## Threading

### Threading

### Spawning Processes

### Multiprocessing

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.
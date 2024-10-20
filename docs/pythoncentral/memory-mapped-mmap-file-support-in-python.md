# Python 中的内存映射(mmap)文件支持

> 原文：<https://www.pythoncentral.io/memory-mapped-mmap-file-support-in-python/>

## Python 中的内存映射文件是什么

从 Python 的官方文档中，请务必查看 [Python 的 mmap 模块](http://docs.python.org/2/library/mmap.html "mmap"):

> 内存映射文件对象的行为类似于字符串和文件对象。然而，与普通的字符串对象不同，它们是可变的。

基本上，内存映射(使用 Python 的`mmap`模块)文件对象将普通文件对象映射到内存中。这允许您直接在内存中修改文件对象的内容。由于内存映射文件对象的行为也类似于可变字符串对象，因此您可以像修改字符列表的内容一样修改文件对象的内容:

*   `obj[1] = 'a'` -将字符‘a’分配给文件对象内容的第二个字符。
*   `obj[1:4] = 'abc'` -从文件对象内容的第二个字符开始，将字符列表“abc”分配给三个字符。

简而言之，使用 Python 的`mmap`模块对文件进行内存映射，我们使用操作系统的虚拟内存来直接访问文件系统上的数据。内存映射不是通过系统调用如 *open* 、 *read* 和 *lseek* 来操作文件，而是将文件的数据放入内存中，这样就可以直接操作内存中的文件。这极大地提高了 I/O 性能。

## 用 Python 比较内存映射文件和普通文件

假设我们有一个大于 10 MB 的二进制文件 *test.out* ,并且有某种算法要求我们以这样一种方式处理文件数据，这种方式要求我们重复以下过程:

*   从当前位置开始，寻找 64 个字节，*处理*当前位置开始处的数据。
*   从当前位置开始，seek -32 字节并且*处理*当前位置开始处的数据。

数据的实际处理被一个`pass`语句代替，因为它不影响`mmap`和正常文件访问之间的相对性能比较。该算法会一直处理文件的数据，直到数据超过 10MB。使用普通文件对象执行算法的代码在文件 *normal_process.py* 中列出:

```py

import os

import time
f = open('test.out '，' r ')
buffer _ size = 64
retract _ size =-32
start _ time = time . time()
while True:
f . seek(buffer _ size，os。SEEK_CUR) 
 #从当前位置开始处理一些数据
 pass 
 f.seek(retract_size，os。SEEK_CUR) 
 #从当前位置开始处理一些数据
pass
if f . tell()>1024 * 1024 * 10:
break
end _ time = time . time()
f . close()
print('正常经过时间:{0} '。格式(结束时间-开始时间))

```

使用`mmap`处理文件数据的代码在文件 *mmap_process.py* 中列出:

```py

import os

import time

import mmap
f = open('test.out '，' r') 
 m = mmap.mmap(f.fileno()，0，access=mmap。ACCESS _ READ)
buffer _ size = 64
retract _ size =-32
start _ time = time . time()
while True:
m . seek(buffer _ size，os。SEEK_CUR) 
 #从当前位置开始处理一些数据
 pass 
 m.seek(retract_size，os。SEEK_CUR) 
 #处理从当前位置开始的一些数据
pass
if m . tell()>1024 * 1024 * 10:
break
end _ time = time . time()
m . close()
f . close()
print(' mmap 已用时间:{0} '。格式(结束时间-开始时间))

```

现在，您可以在 shell 中的一个简单 for 循环中比较 *normal_process.py* 和 *mmap_process.py* :

```py

for i in {1..3}

do

    python normal_process.py

    python mmap_process.py

done

normal time elapsed: 0.355199098587

mmap time elapsed: 0.296804904938

normal time elapsed: 0.371860027313

mmap time elapsed: 0.290856838226

normal time elapsed: 0.355377197266

mmap time elapsed: 0.305727958679

```

如你所见， *mmap_process.py* 比 *normal_process.py* 平均快 17%,因为`seek`函数调用是直接针对 *mmap_process.py* 中的内存执行的，而它们是使用 *normal_process.py* 中的文件系统调用执行的。

## 用 mmap 修改 Python 中的内存映射文件

在下面的文件 *mmap_write.py* 中，我们修改了文件 *write_test.txt* 的内容，使用`mmap`和*将更改刷新*回磁盘:

```py

import os

import mmap
f = open('write_test.txt '，' a+b') 
 m = mmap.mmap(f.fileno()，0，access=mmap。ACCESS _ WRITE)
m[0]= ' n '
# Flush 对文件的内存副本所做的更改回磁盘
m . Flush()
m . close()
f . close()

```

假设 *write_test.txt* 包含一行文字:
【shell】
$ cat write _ test . txt
mmap 是一个很酷的功能！

我们运行 python mmap_write.py 后，它的内容会变成(注意句子的第一个字符):
【shell】
$ cat write _ test . txt
nmap 是一个很酷的功能！

## mmap 总结和建议

虽然`mmap`是一个很酷的特性，但是请记住`mmap`必须在进程的地址空间中找到一个连续的地址块，这个地址块足够大，可以容纳整个文件对象。假设您在一个没有足够的连续内存区域来容纳这些文件的系统上处理大文件，那么创建一个`mmap`将会失败。另外，`mmap`对某些特殊的文件对象不起作用，比如管道和 tty。

下表总结了何时应该使用`mmap`:

*   在多线程编程过程中，如果你有多个进程以只读方式从同一个文件中访问数据，那么使用`mmap`可以节省大量内存。
*   `mmap`允许操作系统优化分页操作，这使得程序在页面中的内存能够被操作系统有效地重用。
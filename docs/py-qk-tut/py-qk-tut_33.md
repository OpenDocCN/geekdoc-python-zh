## Python 标准库 02 时间与日期 (time, datetime 包)

[`www.cnblogs.com/vamei/archive/2012/09/03/2669426.html`](http://www.cnblogs.com/vamei/archive/2012/09/03/2669426.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

Python 具有良好的时间和日期管理功能。实际上，计算机只会维护一个挂钟时间(wall clock time)，这个时间是从某个固定时间起点到现在的时间间隔。时间起点的选择与计算机相关，但一台计算机的话，这一时间起点是固定的。其它的日期信息都是从这一时间计算得到的。此外，计算机还可以测量 CPU 实际上运行的时间，也就是处理器时间(processor clock time)，以测量计算机性能。当 CPU 处于闲置状态时，处理器时间会暂停。

1\. time 包

time 包基于 C 语言的库函数(library functions)。Python 的解释器通常是用 C 编写的，Python 的一些函数也会直接调用 C 语言的库函数。

```py
import time print(time.time())   # wall clock time, unit: second
print(time.clock())  # processor clock time, unit: second

```

time.sleep()可以将程序置于休眠状态，直到某时间间隔之后再唤醒程序，让程序继续运行。

```py
import time print('start')
time.sleep(10)     # sleep for 10 seconds
print('wake up')

```

当我们需要定时地查看程序运行状态时，就可以利用该方法。

time 包还定义了 struct_time 对象。该对象实际上是将挂钟时间转换为年、月、日、时、分、秒……等日期信息，存储在该对象的各个属性中(tm_year, tm_mon, tm_mday...)。下面方法可以将挂钟时间转换为 struct_time 对象:

```py
st = time.gmtime()      # 返回 struct_time 格式的 UTC 时间
st = time.localtime()   # 返回 struct_time 格式的当地时间, 当地时区根据系统环境决定。 
s  = time.mktime(st)    # 将 struct_time 格式转换成 wall clock time

```

2\. datetime 包

1) 简介

datetime 包是基于 time 包的一个高级包， 为我们提供了多一层的便利。

datetime 可以理解为 date 和 time 两个组成部分。date 是指年月日构成的日期(相当于日历)，time 是指时分秒微秒构成的一天 24 小时中的具体时间(相当于手表)。你可以将这两个分开管理(datetime.date 类，datetime.time 类)，也可以将两者合在一起(datetime.datetime 类)。由于其构造大同小异，我们将只介绍 datetime.datetime 类。

比如说我现在看到的时间，是 2012 年 9 月 3 日 21 时 30 分，我们可以用如下方式表达：

```py
import datetime
t = datetime.datetime(2012,9,3,21,30) print(t)

```

所返回的 t 有如下属性:

hour, minute, second, microsecond

year, month, day, weekday   # weekday 表示周几

2) 运算

datetime 包还定义了时间间隔对象(timedelta)。一个时间点(datetime)加上一个时间间隔(timedelta)可以得到一个新的时间点(datetime)。比如今天的上午 3 点加上 5 个小时得到今天的上午 8 点。同理，两个时间点相减会得到一个时间间隔。

```py
import datetime
t = datetime.datetime(2012,9,3,21,30)
t_next = datetime.datetime(2012,9,5,23,30)
delta1 = datetime.timedelta(seconds = 600)
delta2 = datetime.timedelta(weeks = 3) print(t + delta1) print(t + delta2)

```

```py
print(t_next - t)

```

在给 datetime.timedelta 传递参数（如上的 seconds 和 weeks）的时候，还可以是 days, hours, milliseconds, microseconds。

两个 datetime 对象还可以进行比较。比如使用上面的 t 和 t_next:

3) datetime 对象与字符串转换

假如我们有一个的字符串，我们如何将它转换成为 datetime 对象呢？

一个方法是用上一讲的正则表达式来搜索字符串。但时间信息实际上有很明显的特征，我们可以用格式化读取的方式读取时间信息。

```py
from datetime import datetime
format = "output-%Y-%m-%d-%H%M%S.txt" str = "output-1997-12-23-030000.txt" t = datetime.strptime(str, format)

```

strptime, p = parsing

我们通过 format 来告知 Python 我们的 str 字符串中包含的日期的格式。在 format 中，%Y 表示年所出现的位置, %m 表示月份所出现的位置……。

反过来，我们也可以调用 datetime 对象的 strftime()方法，来将 datetime 对象转换为特定格式的字符串。比如上面所定义的 t_next,

```py
print(t_next.strftime(format))

```

strftime, f = formatting

具体的格式写法可参阅官方文档([`docs.python.org/library/datetime.html`](http://docs.python.org/library/datetime.html)), 另外，如果是 Linux 系统，也可以查阅 date 命令的手册($man date)。这一功能基于 ISO C，而 Linux 的 date 也是基于此，所以两者相通。

总结：

时间，休眠

datetime, timedelta

格式化时间
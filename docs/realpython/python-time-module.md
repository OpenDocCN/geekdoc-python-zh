# Python 时间模块初学者指南

> 原文：<https://realpython.com/python-time-module/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**掌握 Python 内置的时间模块**](/courses/mastering-time-module/)

Python `time`模块提供了许多用代码表示时间的方式，比如对象、[数字](https://realpython.com/python-numbers/)和字符串。除了表示时间之外，它还提供了其他功能，比如在代码执行过程中等待，以及测量代码的效率。

本文将带您了解`time`中最常用的函数和对象。

**本文结束时，你将能够:**

*   **理解**处理日期和时间的核心概念，如纪元、时区和夏令时
*   **用浮点数、元组和`struct_time`表示代码中的**时间
*   **在不同的时间表示之间转换**
*   **暂停**线程执行
*   **使用`perf_counter()`测量**代码性能

您将从学习如何使用浮点数来表示时间开始。

**免费奖励:** ，它向您展示 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## 用秒处理 Python 时间

在应用程序中管理 Python 时间概念的方法之一是使用一个浮点数来表示自一个时代开始(即从某个起点开始)以来经过的秒数。

让我们更深入地了解这意味着什么，为什么它有用，以及如何在应用程序中使用它来实现基于 Python time 的逻辑。

[*Remove ads*](/account/join/)

### 纪元

在上一节中，您了解了可以用一个浮点数来管理 Python 时间，该浮点数表示自纪元开始以来经过的时间。

韦氏词典将一个时代定义为:

*   一个固定的时间点，从该点开始计算一系列的年份
*   以给定的日期为基础计算出来的时间记数法

这里要把握的重要概念是，在处理 Python 时间时，您考虑的是由起点标识的一段时间。在计算中，你称这个起点为**纪元**。

那么，纪元就是你衡量时间流逝的起点。

例如，如果您将纪元定义为 UTC 1970 年 1 月 1 日的午夜 Windows 和大多数 UNIX 系统上定义的纪元——那么您可以将 UTC 1970 年 1 月 2 日的午夜表示为从该纪元开始的`86400`秒。

这是因为一分钟有 60 秒，一小时有 60 分钟，一天有 24 小时。UTC 时间 1970 年 1 月 2 日仅是纪元后的第一天，因此您可以应用基础数学得出结果:

>>>

```py
>>> 60 * 60 * 24
86400
```

同样重要的是要注意，您仍然可以表示纪元之前的时间。秒数将会是负数。

例如，您可以将 UTC 1969 年 12 月 31 日的午夜(使用 1970 年 1 月 1 日的纪元)表示为`-86400`秒。

虽然 UTC 1970 年 1 月 1 日是一个常见的纪元，但它不是计算中使用的唯一纪元。事实上，不同的操作系统、文件系统和 API 有时使用不同的纪元。

正如您之前看到的，UNIX 系统将纪元定义为 1970 年 1 月 1 日。另一方面，Win32 API 将纪元定义为[1601 年 1 月 1 日](https://blogs.msdn.microsoft.com/oldnewthing/20090306-00/?p=18913/)。

您可以使用`time.gmtime()`来确定您系统的纪元:

>>>

```py
>>> import time
>>> time.gmtime(0)
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=0)
```

在本文的整个过程中，您将了解到`gmtime()`和`struct_time`。现在，只要知道你可以使用`time`来发现使用这个功能的纪元。

既然您已经更多地了解了如何使用 epoch 以秒为单位测量时间，那么让我们来看看 Python 的`time`模块，看看它提供了哪些功能来帮助您这样做。

### 以秒为单位的浮点数形式的 Python 时间

首先，`time.time()`返回自纪元以来经过的秒数。返回值是一个浮点数，表示小数秒:

>>>

```py
>>> from time import time
>>> time()
1551143536.9323719
```

你在你的机器上得到的数字可能非常不同，因为被认为是纪元的参考点可能非常不同。

**延伸阅读:** Python 3.7 引入了 [`time_ns()`](https://realpython.com/python37-new-features/#timing-precision) ，返回一个整数值，表示自纪元以来经过的相同时间，但以纳秒而不是秒为单位。

以秒为单位测量时间非常有用，原因有很多:

*   你可以用一个浮点数来计算两个时间点之间的差值。
*   float 很容易被序列化，这意味着它可以被存储用于数据传输，并在另一端完整无缺地输出。

然而，有时您可能希望看到用字符串表示的[当前时间](https://realpython.com/python-get-current-time/)。为此，您可以将从`time()`获得的秒数传递给`time.ctime()`。

[*Remove ads*](/account/join/)

### 以秒为单位的 Python 时间，作为表示本地时间的字符串

正如您之前看到的，您可能想要将 Python 时间(表示为自纪元以来经过的秒数)转换为一个[字符串](https://realpython.com/python-strings/)。您可以使用`ctime()`来完成:

>>>

```py
>>> from time import time, ctime
>>> t = time()
>>> ctime(t)
'Mon Feb 25 19:11:59 2019'
```

这里，您已经以秒为单位将当前时间记录到[变量](https://realpython.com/python-variables/) `t`中，然后将`t`作为参数传递给`ctime()`，后者返回相同时间的字符串表示。

**技术细节:**根据`ctime()`的定义，表示从纪元开始的秒的参数是可选的。如果没有传递参数，那么默认情况下，`ctime()`使用`time()`的返回值。所以，你可以简化上面的例子:

>>>

```py
>>> from time import ctime
>>> ctime()
'Mon Feb 25 19:11:59 2019'
```

由`ctime()`返回的时间的字符串表示，也称为**时间戳**，被格式化为以下结构:

1.  **星期几:** `Mon` ( `Monday`)
2.  **一年中的月份:** `Feb` ( `February`)
3.  **一月中的某一天:** `25`
4.  **小时、分钟和秒，使用 [24 小时制](https://en.wikipedia.org/wiki/24-hour_clock)符号:** `19:11:59`
5.  **年份:** `2019`

前面的示例显示了从美国中南部地区的计算机捕获的特定时刻的时间戳。但是，假设你住在澳大利亚的悉尼，你在同一时刻执行了同样的命令。

您将看到以下内容，而不是上面的输出:

>>>

```py
>>> from time import time, ctime
>>> t = time()
>>> ctime(t)
'Tue Feb 26 12:11:59 2019'
```

注意，时间戳的`day of week`、`day of month`和`hour`部分与第一个例子不同。

这些输出是不同的，因为由`ctime()`返回的时间戳取决于您的地理位置。

**注意:**虽然时区的概念是相对于您的物理位置而言的，但您可以在计算机的设置中修改时区，而无需实际搬迁。

依赖于你的物理位置的时间表示被称为本地时间，并利用了一个被称为 T2 时区的概念。

**注意:**因为本地时间与您的语言环境相关，所以时间戳通常会考虑特定于语言环境的细节，比如字符串中元素的顺序以及日期和月份缩写的翻译。`ctime()`忽略这些细节。

让我们更深入地研究一下时区的概念，以便更好地理解 Python 时间表示。

## 了解时区

时区是世界上符合标准时间的区域。时区是由它们相对于协调世界时(UTC)的偏移量来定义的，并且可能包括夏令时(我们将在本文后面更详细地介绍)。

有趣的事实:如果你的母语是英语，你可能会奇怪为什么“协调世界时”的缩写是 UTC，而不是更明显的 CUT。然而，如果你的母语是法语，你会称之为“Temps Universel Coordonné”，这意味着不同的缩写:TUC。

最终，[国际电信联盟和国际天文学联盟达成妥协，将 UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time#Etymology) 作为官方缩写，这样，无论何种语言，缩写都是相同的。

### UTC 和时区

UTC 是世界上所有时间同步(或协调)的时间标准。它本身不是一个时区，而是一个定义时区的卓越标准。

UTC 时间是利用参考地球自转的[天文时间](https://www.merriam-webster.com/dictionary/astronomical%20time)和[原子钟](https://en.wikipedia.org/wiki/Atomic_clock)精确测量的。

然后，时区由它们与 UTC 的偏移量来定义。例如，在北美和南美，中部时区(CT)比 UTC 晚五六个小时，因此使用 UTC-5:00 或 UTC-6:00 表示法。

另一方面，澳大利亚的悉尼属于澳大利亚东部时区(AET)，比 UTC 早十或十一个小时(UTC+10:00 或 UTC+11:00)。

这种差异(UTC-6:00 到 UTC+10:00)是您在前面的示例中从`ctime()`的两个输出中观察到的差异的原因:

*   **中部时间(CT):** `'Mon Feb 25 19:11:59 2019'`
*   **澳大利亚东部时间(AET):** `'Tue Feb 26 12:11:59 2019'`

这些时间正好相隔 16 个小时，这与上面提到的时区偏移是一致的。

您可能想知道为什么 CT 会比 UTC 晚五或六个小时，或者为什么 AET 会比 UTC 早十或十一个小时。这是因为世界上的一些地区，包括这些时区的部分地区，采用夏令时。

[*Remove ads*](/account/join/)

### 夏令时

夏季通常比冬季有更多的日照时间。因此，一些地区在春季和夏季实行夏令时(DST ),以更好地利用这些时间。

对于实行夏令时的地方，他们的时钟会在春天开始时向前跳一小时(实际上慢了一小时)。然后，在秋季，时钟将被重置为标准时间。

在时区表示法中，字母 S 和 D 代表标准时间和夏令时:

*   中部标准时间
*   澳大利亚东部夏令时(AEDT)

当您用本地时间表示时间戳时，考虑 DST 是否适用总是很重要的。

`ctime()`表示夏令时。因此，前面列出的输出差异更准确，如下所示:

*   **中部标准时间(CST):** `'Mon Feb 25 19:11:59 2019'`
*   **澳大利亚东部夏令时(AEDT):** `'Tue Feb 26 12:11:59 2019'`

## 使用数据结构处理 Python 时间

现在您已经牢牢掌握了许多基本的时间概念，包括纪元、时区和 UTC，让我们来看看使用 Python `time`模块表示时间的更多方法。

### 作为元组的 Python 时间

不使用数字来表示 Python 时间，可以使用另一种原始数据结构:一个[元组](https://realpython.com/python-lists-tuples/)。

通过抽象一些数据并使其更具可读性，tuple 允许您更轻松地管理时间。

当您将时间表示为元组时，元组中的每个元素都对应于一个特定的时间元素:

1.  年
2.  整数形式的月份，范围在 1(1 月)和 12(12 月)之间
3.  一月中的某一天
4.  整数形式的小时，范围从 0(上午 12 点)到 23(晚上 11 点)
5.  分钟
6.  第二
7.  整数形式的星期几，范围在 0(星期一)到 6(星期日)之间
8.  一年中的某一天
9.  以整数表示的夏令时，值如下:
    *   `1`是夏令时。
    *   `0`是标准时间。
    *   `-1`不详。

使用您已经学过的方法，您可以用两种不同的方式表示相同的 Python 时间:

>>>

```py
>>> from time import time, ctime
>>> t = time()
>>> t
1551186415.360564
>>> ctime(t)
'Tue Feb 26 07:06:55 2019'

>>> time_tuple = (2019, 2, 26, 7, 6, 55, 1, 57, 0)
```

在这种情况下，`t`和`time_tuple`都表示相同的时间，但是元组为处理时间组件提供了更易读的接口。

**技术细节:**实际上，如果您以秒为单位查看由`time_tuple`表示的 Python 时间(您将在本文后面看到如何操作)，您会看到它解析为`1551186415.0`而不是`1551186415.360564`。

这是因为元组没有表示小数秒的方法。

虽然 tuple 为使用 Python time 提供了一个更易于管理的接口，但是还有一个更好的对象:`struct_time`。

[*Remove ads*](/account/join/)

### Python 时间作为对象

tuple 结构的问题是它看起来仍然像一串数字，尽管它比单个基于秒的数字组织得更好。

`struct_time`通过利用来自 Python 的`collections`模块的 [`NamedTuple`](https://dbader.org/blog/writing-clean-python-with-namedtuples) 将元组的数字序列与有用的标识符相关联，提供了一个解决方案:

>>>

```py
>>> from time import struct_time
>>> time_tuple = (2019, 2, 26, 7, 6, 55, 1, 57, 0)
>>> time_obj = struct_time(time_tuple)
>>> time_obj
time.struct_time(tm_year=2019, tm_mon=2, tm_mday=26, tm_hour=7, tm_min=6, tm_sec=55, tm_wday=1, tm_yday=57, tm_isdst=0)
```

**技术细节:**如果你来自另一种语言，术语`struct`和`object`可能会相互对立。

在 Python 中，没有叫做`struct`的[数据类型](https://realpython.com/python-data-types/)。相反，一切都是对象。

然而，`struct_time`这个名字来源于[基于 C 的时间库](https://en.cppreference.com/w/c/chrono/tm)，其中的数据类型实际上是一个`struct`。

其实 Python 的`time`模块，也就是用 C 实现的[，通过包含头文件`times.h`直接使用了这个`struct`。](https://github.com/python/cpython/blob/master/Modules/timemodule.c)

现在，您可以使用属性名而不是索引来访问`time_obj`的特定元素:

>>>

```py
>>> day_of_year = time_obj.tm_yday
>>> day_of_year
57
>>> day_of_month = time_obj.tm_mday
>>> day_of_month
26
```

除了`struct_time`的可读性和可用性之外，了解它也很重要，因为它是 Python `time`模块中许多函数的返回类型。

## 将 Python 时间(秒)转换为对象

既然您已经看到了使用 Python 时间的三种主要方式，那么您将学习如何在不同的时间数据类型之间进行转换。

时间数据类型之间的转换取决于时间是 UTC 时间还是本地时间。

### 协调世界时

纪元使用 UTC 而不是时区来定义。因此，自纪元以来经过的秒数不会因您的地理位置而变化。

然而，`struct_time`就不一样了。Python 时间的对象表示可能会也可能不会考虑您的时区。

有两种方法可以将表示秒的浮点数转换成`struct_time`:

1.  协调世界时。亦称 COORDINATED UNIVERSAL TIME
2.  当地时间

为了将 Python 时间浮点转换成基于 UTC 的`struct_time`，Python `time`模块提供了一个名为`gmtime()`的函数。

您已经在本文中看到过一次`gmtime()`:

>>>

```py
>>> import time
>>> time.gmtime(0)
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=0)
```

您使用这个调用来发现您系统的纪元。现在，你有了一个更好的基础来理解这里到底发生了什么。

`gmtime()`将从纪元开始经过的秒数转换为 UTC 中的`struct_time`。在这种情况下，您已经将`0`作为秒数传递，这意味着您正在尝试查找 UTC 中的纪元本身。

**注意:**注意属性`tm_isdst`被设置为`0`。此属性表示时区是否使用夏令时。UTC 从不订阅 DST，因此在使用`gmtime()`时，该标志将始终为`0`。

正如您之前看到的，`struct_time`不能表示小数秒，因此`gmtime()`忽略参数中的小数秒:

>>>

```py
>>> import time
>>> time.gmtime(1.99)
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=1, tm_wday=3, tm_yday=1, tm_isdst=0)
```

请注意，即使你经过的秒数非常接近`2`，但是`.99`的小数秒被简单地忽略了，如`tm_sec=1`所示。

`gmtime()`的`secs`参数是可选的，这意味着您可以不带任何参数调用`gmtime()`。这样做将提供 UTC 的当前时间:

>>>

```py
>>> import time
>>> time.gmtime()
time.struct_time(tm_year=2019, tm_mon=2, tm_mday=28, tm_hour=12, tm_min=57, tm_sec=24, tm_wday=3, tm_yday=59, tm_isdst=0)
```

有趣的是，这个函数在`time`中没有反函数。相反，你必须在 Python 的`calendar`模块中寻找一个名为`timegm()`的函数:

>>>

```py
>>> import calendar
>>> import time
>>> time.gmtime()
time.struct_time(tm_year=2019, tm_mon=2, tm_mday=28, tm_hour=13, tm_min=23, tm_sec=12, tm_wday=3, tm_yday=59, tm_isdst=0)
>>> calendar.timegm(time.gmtime())
1551360204
```

`timegm()`获取一个 tuple(或`struct_time`，因为它是 tuple 的子类)并返回从 epoch 开始的相应秒数。

**历史脉络:**如果你对`timegm()`为什么不在`time`感兴趣，可以查看 [Python 第 6280 期](https://bugs.python.org/issue6280)的讨论。

简而言之，它最初被添加到`calendar`是因为`time`紧跟 C 的时间库(在`time.h`中定义)，其中不包含匹配函数。上述问题提出了将`timegm()`移动或复制到`time`的想法。

然而，随着`datetime`库的进步，`time.timegm()`的补丁实现中的不一致性，以及如何处理`calendar.timegm()`的问题，维护者拒绝了这个补丁，鼓励使用`datetime`来代替。

使用 UTC 在编程中很有价值，因为它是一种标准。您不必担心 DST、时区或地区信息。

也就是说，在很多情况下，您会希望使用当地时间。接下来，您将看到如何将秒转换为本地时间，这样您就可以这样做了。

[*Remove ads*](/account/join/)

### 当地时间

在您的应用程序中，您可能需要使用本地时间而不是 UTC。Python 的`time`模块提供了一个函数，用于从名为`localtime()`的纪元以来经过的秒数中获取本地时间。

`localtime()`的签名类似于`gmtime()`,因为它采用了一个可选的`secs`参数，使用您的本地时区来构建一个`struct_time`:

>>>

```py
>>> import time
>>> time.time()
1551448206.86196
>>> time.localtime(1551448206.86196)
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=1, tm_hour=7, tm_min=50, tm_sec=6, tm_wday=4, tm_yday=60, tm_isdst=0)
```

注意`tm_isdst=0`。由于 DST 与当地时间有关，`tm_isdst`将在`0`和`1`之间变化，这取决于 DST 是否适用于给定时间。由于`tm_isdst=0`，夏令时不适用于 2019 年 3 月 1 日。

2019 年的美国，夏令时从 3 月 10 日开始。因此，为了测试 DST 标志是否会正确更改，您需要向`secs`参数添加 9 天的秒数。

要计算这一点，您需要将一天中的秒数(86，400)乘以 9 天:

>>>

```py
>>> new_secs = 1551448206.86196 + (86400 * 9)
>>> time.localtime(new_secs)
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=10, tm_hour=8, tm_min=50, tm_sec=6, tm_wday=6, tm_yday=69, tm_isdst=1)
```

现在，你会看到`struct_time`显示的日期是 2019 年 3 月 10 日`tm_isdst=1`。另外，请注意，`tm_hour`也提前跳到了`8`，而不是前面示例中的`7`，这是因为采用了夏令时。

从 Python 3.3 开始，`struct_time`还包含了两个在确定`struct_time`时区时有用的属性:

1.  `tm_zone`
2.  `tm_gmtoff`

起初，这些属性依赖于平台，但是从 Python 3.6 开始，它们在所有平台上都可用。

首先，`tm_zone`存储当地时区:

>>>

```py
>>> import time
>>> current_local = time.localtime()
>>> current_local.tm_zone
'CST'
```

在这里，您可以看到`localtime()`返回一个时区设置为`CST`(中部标准时间)的`struct_time`。

正如您之前看到的，您还可以根据两条信息辨别时区，即 UTC 偏移量和 DST(如果适用):

>>>

```py
>>> import time
>>> current_local = time.localtime()
>>> current_local.tm_gmtoff
-21600
>>> current_local.tm_isdst
0
```

在这种情况下，你可以看到`current_local`比代表格林威治标准时间的 GMT 晚了`21600`秒。GMT 是没有 UTC 偏移的时区:UTC 00:00。

`21600`秒除以秒每小时(3600)表示`current_local`时间为`GMT-06:00`(或`UTC-06:00`)。

您可以使用 GMT 偏移量加上 DST 状态来推断出标准时间的`current_local`是`UTC-06:00`，它对应于中部标准时区。

和`gmtime()`一样，调用`localtime()`时可以忽略`secs`参数，它会在一个`struct_time`中返回当前当地时间:

>>>

```py
>>> import time
>>> time.localtime()
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=1, tm_hour=8, tm_min=34, tm_sec=28, tm_wday=4, tm_yday=60, tm_isdst=0)
```

与`gmtime()`不同，`localtime()`的反函数确实存在于 Python `time`模块中。让我们来看看它是如何工作的。

[*Remove ads*](/account/join/)

## 将本地时间对象转换为秒

您已经看到了如何使用`calendar.timegm()`将 UTC 时间对象转换成秒。要将本地时间转换成秒，您将使用`mktime()`。

`mktime()`要求您传递一个名为`t`的参数，该参数采用普通 9 元组或表示本地时间的`struct_time`对象的形式:

>>>

```py
>>> import time

>>> time_tuple = (2019, 3, 10, 8, 50, 6, 6, 69, 1)
>>> time.mktime(time_tuple)
1552225806.0

>>> time_struct = time.struct_time(time_tuple)
>>> time.mktime(time_struct)
1552225806.0
```

记住`t`必须是表示本地时间的元组，而不是 UTC，这一点很重要:

>>>

```py
>>> from time import gmtime, mktime

>>> # 1
>>> current_utc = time.gmtime()
>>> current_utc
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=1, tm_hour=14, tm_min=51, tm_sec=19, tm_wday=4, tm_yday=60, tm_isdst=0)

>>> # 2
>>> current_utc_secs = mktime(current_utc)
>>> current_utc_secs
1551473479.0

>>> # 3
>>> time.gmtime(current_utc_secs)
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=1, tm_hour=20, tm_min=51, tm_sec=19, tm_wday=4, tm_yday=60, tm_isdst=0)
```

**注意:**对于这个例子，假设当地时间是`March 1, 2019 08:51:19 CST`。

这个例子说明了为什么使用本地时间`mktime()`而不是 UTC 很重要:

1.  不带参数的 **`gmtime()`** 使用 UTC 返回一个`struct_time`。`current_utc`显示`March 1, 2019 14:51:19 UTC`。因为`CST is UTC-06:00`很准确，所以 UTC 应该比当地时间早 6 个小时。

2.  **`mktime()`** 试图返回秒数，期望是本地时间，但是您却传递了`current_utc`。所以，它没有理解`current_utc`是 UTC 时间，而是假设你指的是`March 1, 2019 14:51:19 CST`。

3.  然后使用 **`gmtime()`** 将这些秒转换回 UTC，这会导致不一致。现在时间是`March 1, 2019 20:51:19 UTC`。出现这种差异的原因是，`mktime()`预计当地时间。因此，转换回 UTC 会将当地时间*再增加* 6 个小时。

众所周知，使用时区非常困难，因此了解 UTC 和本地时间之间的差异以及处理这两种时间的 Python 时间函数对您的成功非常重要。

## 将 Python 时间对象转换为字符串

虽然使用元组很有趣，但有时最好使用字符串。

时间的字符串表示，也称为时间戳，有助于提高时间的可读性，对于构建直观的用户界面尤其有用。

有两个 Python `time`函数可用于将`time.struct_time`对象转换为字符串:

1.  `asctime()`
2.  `strftime()`

你将从学习`asctime()`开始。

### `asctime()`

您使用`asctime()`将时间元组或`struct_time`转换为时间戳:

>>>

```py
>>> import time
>>> time.asctime(time.gmtime())
'Fri Mar  1 18:42:08 2019'
>>> time.asctime(time.localtime())
'Fri Mar  1 12:42:15 2019'
```

`gmtime()`和`localtime()`分别为 UTC 和本地时间返回`struct_time`实例。

您可以使用`asctime()`将`struct_time`转换成时间戳。`asctime()`的工作方式与`ctime()`类似，您在本文前面已经了解过，只是您传递的不是浮点数，而是一个元组。甚至这两个函数的时间戳格式也是相同的。

与`ctime()`一样，`asctime()`的参数是可选的。如果您没有将时间对象传递给`asctime()`，那么它将使用当前的本地时间:

>>>

```py
>>> import time
>>> time.asctime()
'Fri Mar  1 12:56:07 2019'
```

和`ctime()`一样，它也忽略了语言环境信息。

`asctime()`最大的缺点之一是它的格式不灵活。`strftime()`通过允许你格式化你的时间戳来解决这个问题。

[*Remove ads*](/account/join/)

### `strftime()`

您可能会发现来自`ctime()`和`asctime()`的字符串格式不适合您的应用程序。相反，您可能希望以对用户更有意义的方式格式化字符串。

其中一个例子是，如果您希望在字符串中显示您的时间，并考虑区域设置信息。

要格式化字符串，给定一个`struct_time`或 Python 时间元组，可以使用`strftime()`，它代表“字符串**格式**时间”

`strftime()`需要两个参数:

1.  **`format`** 指定字符串中时间元素的顺序和形式。
2.  **`t`** 是可选的时间元组。

要格式化一个字符串，可以使用**指令**。指令是以指定特定时间元素的`%`开头的字符序列，例如:

*   **`%d` :** 一月中的某一天
*   **`%m` :** 一年中的月份
*   **`%Y` :** 年份

例如，您可以使用 [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) 标准输出当地时间的日期，如下所示:

>>>

```py
>>> import time
>>> time.strftime('%Y-%m-%d', time.localtime())
'2019-03-01'
```

**延伸阅读:**虽然使用 Python time 表示日期是完全有效且可接受的，但您也应该考虑使用 Python 的`datetime`模块，它提供了快捷方式和更健壮的框架来处理日期和时间。

例如，您可以使用`datetime`简化 ISO 8601 格式的日期输出:

>>>

```py
>>> from datetime import date
>>> date(year=2019, month=3, day=1).isoformat()
'2019-03-01'
```

要了解更多关于使用 Python `datetime`模块的信息，请查看使用 Python datetime 处理日期和时间的

正如您之前看到的，使用`strftime()`而不是`asctime()`的一个很大的好处是它能够呈现利用特定于地区的信息的时间戳。

例如，如果您想以一种区域敏感的方式表示日期和时间，您不能使用`asctime()`:

>>>

```py
>>> from time import asctime
>>> asctime()
'Sat Mar  2 15:21:14 2019'

>>> import locale
>>> locale.setlocale(locale.LC_TIME, 'zh_HK')  # Chinese - Hong Kong
'zh_HK'
>>> asctime()
'Sat Mar  2 15:58:49 2019'
```

请注意，即使以编程方式更改了您的语言环境，`asctime()`仍然会以与以前相同的格式返回日期和时间。

**技术细节:** `LC_TIME`是日期和时间格式的区域设置类别。根据您的系统不同，`locale`参数`'zh_HK'`可能会有所不同。

然而，当您使用`strftime()`时，您会发现它考虑了地区:

>>>

```py
>>> from time import strftime, localtime
>>> strftime('%c', localtime())
'Sat Mar  2 15:23:20 2019'

>>> import locale
>>> locale.setlocale(locale.LC_TIME, 'zh_HK')  # Chinese - Hong Kong
'zh_HK'
>>> strftime('%c', localtime())
'六  3/ 2 15:58:12 2019' 2019'
```

这里，您成功地利用了地区信息，因为您使用了`strftime()`。

**注意:** `%c`是适用于地区的日期和时间的指令。

如果时间元组没有传递给参数`t`，那么默认情况下`strftime()`会使用`localtime()`的结果。因此，您可以通过删除可选的第二个参数来简化上面的示例:

>>>

```py
>>> from time import strftime
>>> strftime('The current local datetime is: %c')
'The current local datetime is: Fri Mar  1 23:18:32 2019'
```

这里，您使用了默认时间，而不是将自己的时间作为参数传递。另外，请注意,`format`参数可以由格式化指令以外的文本组成。

**延伸阅读:**查看`strftime()`可获得的指令的完整[列表。](https://docs.python.org/3/library/time.html#time.strftime)

Python `time`模块还包括将时间戳转换回`struct_time`对象的逆向操作。

[*Remove ads*](/account/join/)

## 将 Python 时间字符串转换为对象

当您处理与日期和时间相关的字符串时，将时间戳转换为时间对象是非常有价值的。

要将时间字符串转换成`struct_time`，您可以使用`strptime()`，它代表“字符串**解析**时间”:

>>>

```py
>>> from time import strptime
>>> strptime('2019-03-01', '%Y-%m-%d')
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=4, tm_yday=60, tm_isdst=-1)
```

`strptime()`的第一个参数必须是您想要转换的时间戳。第二个参数是时间戳所在的`format`。

`format`参数是可选的，默认为`'%a %b %d %H:%M:%S %Y'`。因此，如果您有这种格式的时间戳，您不需要将它作为参数传递:

>>>

```py
>>> strptime('Fri Mar 01 23:38:40 2019')
time.struct_time(tm_year=2019, tm_mon=3, tm_mday=1, tm_hour=23, tm_min=38, tm_sec=40, tm_wday=4, tm_yday=60, tm_isdst=-1)
```

因为一个`struct_time`有 9 个关键的日期和时间组件，`strptime()`必须为那些它不能从`string`解析的组件提供合理的默认值。

在前面的例子中，`tm_isdst=-1`。这意味着`strptime()`无法通过时间戳确定它是否代表夏令时。

现在您知道了如何以多种方式使用`time`模块处理 Python 时间和日期。然而，除了简单地创建时间对象、获取 Python 时间字符串和使用自纪元以来经过的秒数之外，`time`还有其他用途。

## 暂停执行

一个非常有用的 Python 时间函数是 [`sleep()`](https://realpython.com/python-sleep/) ，它将线程的执行暂停一段指定的时间。

例如，您可以暂停程序执行 10 秒钟，如下所示:

>>>

```py
>>> from time import sleep, strftime
>>> strftime('%c')
'Fri Mar  1 23:49:26 2019'
>>> sleep(10)
>>> strftime('%c')
'Fri Mar  1 23:49:36 2019'
```

你的程序将[打印](https://realpython.com/python-print/)第一个格式化的`datetime`字符串，然后暂停 10 秒，最后打印第二个格式化的`datetime`字符串。

您也可以将小数秒传递给`sleep()`:

>>>

```py
>>> from time import sleep
>>> sleep(0.5)
```

对于测试或者让你的程序因为任何原因而等待是有用的，但是你必须小心不要停止你的生产代码，除非你有很好的理由这样做。

在 Python 3.5 之前，发送给你的进程的信号可以中断`sleep()`。但是，在 3.5 和更高版本中，`sleep()`将总是至少在指定的时间内暂停执行，即使进程收到信号。

仅仅是一个 Python 时间函数，它可以帮助你测试你的程序并使它们更加健壮。

[*Remove ads*](/account/join/)

## 测量性能

你可以[使用`time`来衡量你的程序](https://realpython.com/python-timer/)的性能。

这样做的方法是使用`perf_counter()`，顾名思义，它提供了一个高分辨率的性能计数器来测量短距离的时间。

要使用`perf_counter()`，您需要在代码开始执行之前以及代码执行完成之后放置一个计数器:

>>>

```py
>>> from time import perf_counter
>>> def longrunning_function():
...     for i in range(1, 11):
...         time.sleep(i / i ** 2)
...
>>> start = perf_counter()
>>> longrunning_function()
>>> end = perf_counter()
>>> execution_time = (end - start)
>>> execution_time
8.201258441999926
```

首先，`start`捕捉调用函数之前的瞬间。`end`捕捉函数返回后的瞬间。该函数的总执行时间为`(end - start)`秒。

**技术细节:** Python 3.7 引入了`perf_counter_ns()`，其工作原理与`perf_counter()`相同，但使用纳秒而非秒。

`perf_counter()`(或`perf_counter_ns()`)是使用一次执行来测量代码性能的最精确的方法。然而，如果你试图精确地测量代码片段的性能，我推荐使用 [Python `timeit`](https://docs.python.org/3/library/timeit.html) 模块。

`timeit`专门多次运行代码，以获得更准确的性能分析，并帮助您避免过于简化您的时间测量以及其他常见的陷阱。

## 结论

恭喜你！现在，您已经为使用 Python 处理日期和时间打下了良好的基础。

现在，您能够:

*   使用一个浮点数来处理时间，表示从纪元开始经过的秒数
*   使用元组和`struct_time`对象管理时间
*   在秒、元组和时间戳字符串之间转换
*   暂停 Python 线程的执行
*   使用`perf_counter()`测量性能

除此之外，您还学习了一些关于日期和时间的基本概念，例如:

*   世
*   协调世界时。亦称 COORDINATED UNIVERSAL TIME
*   时区
*   夏令时

现在，是时候将您新学到的 Python time 知识应用到现实世界的应用程序中了！

## 延伸阅读

如果您想继续学习更多关于在 Python 中使用日期和时间的知识，请查看以下模块:

*   **[`datetime`](https://docs.python.org/3/library/datetime.html):**Python 的标准库中更健壮的日期和时间模块
*   **[`timeit`](https://docs.python.org/3/library/timeit.html) :** 一个用于衡量代码片段性能的模块
*   **[`astropy`](http://docs.astropy.org/en/stable/time/) :** 天文学中使用的更高精度的日期时间

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**掌握 Python 内置的时间模块**](/courses/mastering-time-module/)***********
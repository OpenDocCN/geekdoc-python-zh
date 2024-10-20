# Python -与 Delorean 共度时光

> 原文：<https://www.blog.pythonlibrary.org/2014/09/03/python-taking-time-with-delorean/>

最近我写了关于 [arrow 项目](https://www.blog.pythonlibrary.org/2014/08/05/arrow-a-new-date-time-package-for-python/)的文章，我的一个读者提到了另一个被称为 [Delorean](http://delorean.readthedocs.org/en/latest/) 的与日期时间相关的项目。因此，在本文中，我们将花一些时间来研究 delorean 项目。这将是一篇高水平的文章，因为没有理由重写 delorean 的文档。

* * *

### 入门指南

要安装 Delorean，您需要的只是 pip 和管理员权限。下面是安装软件包的典型方法:

```py

pip install delorean

```

安装时，您会注意到 Delorean 有几个依赖项:pytz 和 python-dateutil。幸运的是，如果您还没有这些软件，pip 也会为您安装。

* * *

### 使用 Delorean

实际上使用 Delorean 非常简单。我们来看几个例子。我们先来看看 Delorean 如何处理时区:

```py

>>> from delorean import Delorean
>>> CST = "US/Central"
>>> d = Delorean(timezone=CST)
>>> d
Delorean(datetime=2014-09-03 08:01:12.112257-05:00, timezone=US/Central)
>>> e = Delorean(timezone=EST)
>>> e
Delorean(datetime=2014-09-03 09:02:00.537070-04:00, timezone=US/Eastern)
>>> d.shift(EST)
Delorean(datetime=2014-09-03 09:01:12.112257-04:00, timezone=US/Eastern)

```

这里我们可以看到 Delorean 如何使用字符串来设置时区，以及创建具有不同时区的对象是多么容易。我们还可以看到如何在时区之间转换。接下来，我们将检查偏移:

```py

>>> d.next_day(1)
Delorean(datetime=2014-09-04 08:01:12.112257-05:00, timezone=US/Central)
>>> d.next_day(-2)
Delorean(datetime=2014-09-01 08:01:12.112257-05:00, timezone=US/Central)

```

如你所见，在时间中前进和后退是非常容易的。你需要做的就是调用 Delorean 的 **next_day()** 方法。如果您需要使用 Python 的 datetime 模块和 Delorean 对象，那么您可能会想看看 Delorean 的 **epoch()** 和 **naive()** 方法:

```py

>>> d.epoch()
1409749272.112257
>>> d.naive()
datetime.datetime(2014, 9, 3, 13, 1, 12, 112257)

```

正如您可能猜到的，epoch 方法返回从 epoch 开始的秒数。另一方面，naive 方法返回一个 **datetime.datetime** 对象。

Delorean 更有趣的特性之一是它能够使用自然语言来获取与您创建的日期对象相关的某些日期:

```py

>>> d.next_tuesday()
Delorean(datetime=2014-09-09 09:01:12.112257-04:00, timezone=US/Eastern)
>>> d.next_friday()
Delorean(datetime=2014-09-05 09:01:12.112257-04:00, timezone=US/Eastern)
>>> d.last_sunday()
Delorean(datetime=2014-08-31 09:01:12.112257-04:00, timezone=US/Eastern)

```

这不是很方便吗？Delorean 的另一个简洁的特性是它的 stops()函数:

```py

>>> from delorean import stops
>>> import delorean
>>> for stop in stops(freq=delorean.HOURLY, count=10):
        print stop

Delorean(datetime=2014-09-03 13:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 14:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 15:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 16:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 17:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 18:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 19:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 20:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 21:18:51+00:00, timezone=UTC)
Delorean(datetime=2014-09-03 22:18:51+00:00, timezone=UTC)

```

您可以使用 Delorean 为不同的秒、分、小时、天、周、月和年创建一组 Delorean 对象。您还可以包括时区。

* * *

### 包扎

Delorean 附带了其他有趣的功能，如截断和解析日期时间字符串。你绝对应该尝试一下这个项目，看看使用起来有多有趣！

* * *

### 相关阅读

*   德罗林[主页](http://delorean.readthedocs.org/en/latest/)
*   [arrow-Python 的新日期/时间包](https://www.blog.pythonlibrary.org/2014/08/05/arrow-a-new-date-time-package-for-python/)
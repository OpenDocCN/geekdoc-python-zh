# 使用 Python datetime 处理日期和时间

> 原文： [https://realpython.com/python-datetime/](https://realpython.com/python-datetime/)

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 的 datetime 模块**](/courses/python-datetime-module/)

处理日期和时间是编程中最大的挑战之一。在处理时区、夏令时和不同的书面日期格式之间，很难记住您所指的日期和时间。幸运的是，内置的 Python **`datetime`** 模块可以帮助您管理复杂的日期和时间。

在本教程中，您将学习:

*   为什么用**日期和时间**编程如此具有挑战性
*   **Python `datetime`** 模块中有哪些函数
*   如何**以特定格式打印或读取日期和时间**
*   如何用日期和时间做**算术**

此外，您将开发一个简洁的应用程序来倒数到下一次 PyCon US 的剩余时间！

**免费奖励:** ，它向您展示 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

我们开始吧！

## 用日期和时间编程

如果您曾经开发过需要跨几个地理区域记录时间的软件，那么您可能会理解为什么用时间编程会如此痛苦。最根本的脱节是计算机程序更喜欢完全有序和有规律的事件，但大多数人使用和指代时间的方式是非常不规则的。

注意:如果你想了解更多为什么时间会如此复杂，那么网络上有很多很好的资源。这里有几个好的起点:

*   电脑爱好者:时间的问题&时区
*   [使用时区:你希望不需要知道的一切](https://www.youtube.com/watch?v=rz3D8VG_2TY)
*   [时间数据编程的复杂性](https://www.mojotech.com/blog/the-complexity-of-time-data-programming/)

这种不规则性的一个很好的例子就是 [**夏令时**](https://en.wikipedia.org/wiki/Daylight_saving_time) 。在美国和加拿大，时钟在三月的第二个星期天拨快一小时，在十一月的第一个星期天拨慢一小时。然而，这只是自 2007 年以来[的情况。在 2007 年之前，时钟在 4 月的第一个星期天向前拨，在 10 月的最后一个星期天向后拨。](https://www.nist.gov/pml/time-and-frequency-division/popular-links/daylight-saving-time-dst)

当你考虑到 [**时区**](https://en.wikipedia.org/wiki/Time_zone) 时，事情变得更加复杂。理想情况下，时区的边界应该完全沿着经度线。然而，由于历史和政治原因，时区线很少是直的。通常，相距很远的区域位于同一时区，而相邻的区域位于不同的时区。有些时区的[有着非常时髦的形状](https://upload.wikimedia.org/wikipedia/commons/8/88/World_Time_Zones_Map.png)。

[*Remove ads*](/account/join/)

### 计算机如何计算时间

几乎所有的计算机都从一个叫做 [**Unix 纪元**](https://en.wikipedia.org/wiki/Unix_time) 的瞬间开始计时。这发生在 1970 年 1 月 1 日 00:00:00 UTC。UTC 代表 [**协调世界时**](https://en.wikipedia.org/wiki/Coordinated_Universal_Time) ，是指经度 0°的时间。UTC 通常也被称为[格林威治标准时间](https://en.wikipedia.org/wiki/Greenwich_Mean_Time)，或 GMT。UTC 不根据夏令时进行调整，所以它始终保持每天 24 小时。

根据定义，Unix 时间以与 UTC 相同的速率流逝，因此 UTC 中的一秒对应于 Unix 时间中的一秒。通过计算自 Unix 纪元以来的秒数，通常可以计算出自 1970 年 1 月 1 日以来任何给定时刻的 UTC 日期和时间，但 [**闰秒**](https://www.youtube.com/watch?v=Uqjg8Kk1HXo) 除外。闰秒偶尔会添加到 UTC 中，以解释地球自转变慢的原因，但不会添加到 Unix 时间中。

注意:有一个关于 Unix 时间的有趣错误。由于许多较老的操作系统是 32 位的，所以它们将 Unix 时间存储在 32 位有符号整数中。

这意味着在 2038 年 1 月 19 日 03:14:07，整数将溢出，导致所谓的[年 2038 问题](https://en.wikipedia.org/wiki/Year_2038_problem)，或 Y2038。与 [Y2K 问题](https://en.wikipedia.org/wiki/Year_2000_problem)类似，2038 年也需要修正，以避免对关键系统造成灾难性后果。

几乎所有的编程语言，包括 [Python](https://docs.python.org/3/library/time.html) ，都包含了 Unix 时间的概念。Python 的标准库包括一个名为`time`的模块，它可以打印自 Unix 纪元以来的秒数:

>>>

```py
>>> import time
>>> time.time()
1579718137.550164
```

在这个例子中，您[导入](https://realpython.com/lessons/import-statement/) [`time`模块](https://realpython.com/python-time-module/)中的并执行 [`time()`](https://docs.python.org/3/library/time.html#time.time) 来打印 Unix 时间，或者从 epoch 开始的秒数(不包括闰秒)。

除了 Unix 时间，计算机还需要一种向用户传达时间信息的方式。正如您在上一个例子中看到的，Unix 时间对于人来说几乎是不可能解析的。相反，Unix 时间通常被转换为 UTC，然后可以使用**时区偏移量**将其转换为本地时间。

互联网数字地址分配机构(IANA) 维护着一个包含所有时区偏移量的[数据库](https://www.iana.org/time-zones)。IANA 还发布定期更新，包括时区偏移的任何变化。该数据库通常包含在您的操作系统中，尽管某些应用程序可能包含更新的副本。

该数据库包含所有指定时区的副本，以及它们与 UTC 相差多少小时和分钟。因此，在冬季，当夏令时无效时，美国东部时区的时差为-05:00，即比 UTC 时间晚 5 个小时。其他地区有不同的偏移量，可能不是整数小时。例如，尼泊尔的 UTC 时差为+05:45，即比 UTC 时差 5 小时 45 分。

### 如何报告标准日期

Unix 时间是计算机计算时间的方式，但是对于人类来说，通过计算任意日期的秒数来确定时间是非常低效的。相反，我们按照年、月、日等等来工作。但是即使有了这些约定，另一层复杂性源于不同的语言和文化有不同的书写日期的方式。

例如，在美国，日期通常以月开始，然后是日，最后是年。这意味着 2020 年 1 月 31 日写成 **01-31-2020** 。这与日期的长形式书面版本非常匹配。

然而，欧洲的大部分地区和许多其他地区都是以日开始写日期，然后是月，然后是年。这意味着 2020 年 1 月 31 日写成 **31-01-2020** 。当跨文化交流时，这些差异会引起各种各样的困惑。

为了帮助避免沟通错误，国际标准化组织(ISO)开发了[](https://en.wikipedia.org/wiki/ISO_8601)**【ISO 8601】。本标准规定，所有日期都应该按照从最重要到最不重要的顺序书写。这意味着格式是年、月、日、小时、分钟和秒:**

```py
YYYY-MM-DD HH:MM:SS
```

在这个例子中，`YYYY`代表四位数的年份，`MM`和`DD`是两位数的月和日，必要时可以从零开始。之后，`HH`、`MM`和`SS`表示两位数的小时、分钟和秒，必要时以零开始。

这种格式的优点是可以清楚地表示日期。如果日期是有效的月份号，那么写为`DD-MM-YYYY`或`MM-DD-YYYY`的日期可能会被误解。稍后在你会看到一点点[你如何在 Python `datetime`中使用 ISO 8601 格式。](#using-strings-to-create-python-datetime-instances)

### 时间应该如何存储在你的程序中

大多数与时间打交道的开发人员都听过将本地时间转换为 UTC 并存储该值以供以后参考的建议。在许多情况下，尤其是当您存储过去的日期时，这些信息足以进行任何必要的运算。

但是，如果程序的用户以当地时间输入未来的日期，就会出现问题。时区和夏令时规则变化相当频繁，正如您之前看到的 2007 年美国和加拿大夏令时的变化。如果用户所在位置的时区规则在他们输入的未来日期之前发生了变化，那么 UTC 将不会提供足够的信息来转换回正确的当地时间。

**注意:**有许多优秀的资源可以帮助您确定在应用程序中存储时间数据的适当方式。这里有几个地方可以开始:

*   [夏令时和时区最佳实践](https://stackoverflow.com/a/2532962)
*   [存储 UTC 不是灵丹妙药](https://codeblog.jonskeet.uk/2019/03/27/storing-utc-is-not-a-silver-bullet/)
*   [如何为未来事件保存日期时间](http://www.creativedeletion.com/2015/03/19/persisting_future_datetimes.html)
*   中使用日期时间的编码最佳实践。NET 框架

在这种情况下，您需要存储用户输入的本地时间，包括时区，以及用户保存时间时有效的 IANA 时区数据库的版本。这样，您总是能够将本地时间转换为 UTC。然而，这种方法并不总是允许您将 UTC 转换为正确的本地时间。

[*Remove ads*](/account/join/)

## 使用 Python `datetime`模块

如您所见，在编程中处理日期和时间可能很复杂。幸运的是，现在你很少需要从头实现复杂的特性，因为有很多开源库可以帮助你。Python 就是这种情况，它在标准库中包括三个独立的模块来处理日期和时间:

1.  **[`calendar`](https://docs.python.org/3/library/calendar.html#module-calendar)** 使用理想化的[公历](https://en.wikipedia.org/wiki/Gregorian_calendar)输出日历并提供功能。
2.  **[`datetime`](https://docs.python.org/3/library/datetime.html)** 提供用于操作日期和时间的类。
3.  **[`time`](https://docs.python.org/3/library/time.html)** 提供不需要日期的时间相关函数。

在本教程中，你将重点使用 Python **`datetime`** 模块。`datetime`的主要目的是降低访问与日期、时间和时区相关的对象属性的复杂性。由于这些对象非常有用，`calendar`也从`datetime`返回类的实例。

[`time`](https://realpython.com/python-time-module/) 不如`datetime`强大，使用起来更复杂。`time`中的许多函数返回一个特殊的 [**`struct_time`**](https://docs.python.org/3/library/time.html#time.struct_time) 实例。这个对象有一个名为 tuple 的[接口，用于访问存储的数据，使其类似于`datetime`的一个实例。然而，它并不支持`datetime`的所有特性，尤其是对时间值执行算术运算的能力。](https://realpython.com/python-namedtuple/)

`datetime`提供了三个类，组成了大多数人都会使用的高级接口:

1.  **[`datetime.date`](https://docs.python.org/3/library/datetime.html#date-objects)** 是一个理想化的日期，假设公历无限延伸到未来和过去。这个对象将`year`、`month`和`day`存储为属性。
2.  **[`datetime.time`](https://docs.python.org/3/library/datetime.html#time-objects)** 是一个理想化的时间，假设每天有 86，400 秒，没有闰秒。这个对象存储了`hour`、`minute`、`second`、`microsecond`和`tzinfo`(时区信息)。
3.  **[`datetime.datetime`](https://docs.python.org/3/library/datetime.html#datetime-objects)** 是一个`date`和一个`time`的组合。它具有这两个类的所有属性。

### 创建 Python `datetime`实例

在`datetime`中代表日期和时间的三个类有相似的 [**初始值设定项**](https://realpython.com/python3-object-oriented-programming/#instance-attributes) 。它们可以通过为每个属性传递关键字参数来实例化，比如`year`、`date`或`hour`。您可以尝试下面的代码来了解每个对象是如何创建的:

>>>

```py
>>> from datetime import date, time, datetime
>>> date(year=2020, month=1, day=31)
datetime.date(2020, 1, 31)
>>> time(hour=13, minute=14, second=31)
datetime.time(13, 14, 31)
>>> datetime(year=2020, month=1, day=31, hour=13, minute=14, second=31)
datetime.datetime(2020, 1, 31, 13, 14, 31)
```

在这段代码中，您[从`datetime`和**导入**](https://realpython.com/absolute-vs-relative-python-imports/)三个主要类，通过向构造函数传递参数来实例化它们中的每一个。您可以看到这段代码有些冗长，如果您没有所需的信息作为[整数](https://realpython.com/lessons/integers/)，这些技术就不能用于创建`datetime`实例。

幸运的是，`datetime`提供了其他几种创建`datetime`实例的方便方法。这些方法不要求您使用整数来指定每个属性，而是允许您使用一些其他信息:

1.  **[`date.today()`](https://docs.python.org/3/library/datetime.html#datetime.date.today)** 用当前本地日期创建一个`datetime.date`实例。
2.  **[`datetime.now()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.now)** 用当前的本地日期和时间创建一个`datetime.datetime`实例。
3.  **[`datetime.combine()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.combine)** 将`datetime.date`和`datetime.time`的实例组合成一个`datetime.datetime`实例。

当您事先不知道需要向基本初始化器传递什么信息时，这三种创建`datetime`实例的方法很有帮助。您可以尝试这段代码，看看替代的初始化器是如何工作的:

>>>

```py
>>> from datetime import date, time, datetime
>>> today = date.today()
>>> today
datetime.date(2020, 1, 24)
>>> now = datetime.now()
>>> now
datetime.datetime(2020, 1, 24, 14, 4, 57, 10015)
>>> current_time = time(now.hour, now.minute, now.second)
>>> datetime.combine(today, current_time)
datetime.datetime(2020, 1, 24, 14, 4, 57)
```

在这段代码中，您使用`date.today()`、`datetime.now()`和`datetime.combine()`来创建`date`、`datetime`和`time`对象的实例。每个实例存储在不同的[变量](https://realpython.com/python-variables/)中:

1.  **`today`** 是一个只有年、月和日的`date`实例。
2.  **`now`** 是具有年、月、日、小时、分钟、秒和微秒的`datetime`实例。
3.  **`current_time`** 是一个`time`实例，它的小时、分钟和秒设置为与`now`相同的值。

在最后一行，您将`today`中的日期信息与`current_time`中的时间信息结合起来，产生一个新的`datetime`实例。

**警告:** `datetime`还提供了`datetime.utcnow()`，在当前 UTC 返回一个`datetime`的实例。然而，Python [文档](https://docs.python.org/3/library/datetime.html#datetime.datetime.utcnow)建议不要使用这种方法，因为它在结果实例中不包含任何时区信息。

当在`datetime`实例之间做算术或比较时，使用`datetime.utcnow()`可能会产生一些[令人惊讶的结果](https://blog.ganssle.io/articles/2019/11/utcnow.html)。在[后面的章节](#working-with-time-zones)中，您将看到如何为`datetime`实例分配时区信息。

### 使用字符串创建 Python `datetime`实例

创建`date`实例的另一种方法是使用 [`.fromisoformat()`](https://docs.python.org/3/library/datetime.html#datetime.date.fromisoformat) 。要使用这个方法，您需要提供一个带有 ISO 8601 格式日期的[字符串](https://realpython.com/python-strings/)，您之前已经了解了[和](#how-standard-dates-can-be-reported)。例如，您可以提供一个指定了年、月和日的字符串:

```py
2020-01-31
```

根据 ISO 8601 格式，此字符串表示日期 2020 年 1 月 31 日。您可以用下面的例子创建一个`date`实例:

>>>

```py
>>> from datetime import date
>>> date.fromisoformat("2020-01-31")
datetime.date(2020, 1, 31)
```

在这段代码中，您使用`date.fromisoformat()`为 2020 年 1 月 31 日创建一个`date`实例。这种方法非常有用，因为它是基于 ISO 8601 标准的。但是，如果您有一个表示日期和时间的字符串，但不是 ISO 8601 格式的，该怎么办呢？

幸运的是，Python `datetime`提供了一个名为 [`.strptime()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.strptime) 的方法来处理这种情况。这个方法使用一种特殊的**小型语言**来告诉 Python 字符串的哪些部分与`datetime`属性相关联。

要使用`.strptime()`从一个字符串构造一个`datetime`，您必须使用迷你语言的格式化代码告诉 Python 字符串的每个部分代表什么。你可以试试这个例子，看看`.strptime()`是如何运作的:

>>>

```py
 1>>> date_string = "01-31-2020 14:45:37"
 2>>> format_string = "%m-%d-%Y %H:%M:%S"
```

在**的第 1 行**，您创建了`date_string`，它表示 2020 年 1 月 31 日下午 2:45:37 的日期和时间。在**的第 2** 行，您创建了`format_string`，它使用迷你语言来指定如何将`date_string`的各个部分转化为`datetime`属性。

在`format_string`中，您包括几个格式代码和所有的破折号(`-`)、冒号(`:`)和空格，就像它们在`date_string`中出现的一样。要处理`date_string`中的日期和时间，需要包含以下格式代码:

| 成分 | 密码 | 价值 |
| --- | --- | --- |
| 年份(四位数整数) | `%Y` | Two thousand and twenty |
| 月份(以零填充的小数形式) | `%m` | 01 |
| 日期(以零填充的小数形式) | `%d` | Thirty-one |
| 小时(以 24 小时制零填充十进制表示) | `%H` | Fourteen |
| 分钟(以零填充的小数形式) | `%M` | Forty-five |
| 秒(以零填充的小数形式) | `%S` | Thirty-seven |

迷你语言中所有选项的完整列表超出了本教程的范围，但是你可以在网上找到一些好的参考资料，包括 Python 的[文档](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior)和一个名为 strftime.org[的网站](https://strftime.org/)。

既然已经定义了`date_string`和`format_string`，您可以使用它们来创建一个`datetime`实例。这里有一个`.strptime()`如何工作的例子:

>>>

```py
 3>>> from datetime import datetime
 4>>> datetime.strptime(date_string, format_string)
 5datetime.datetime(2020, 1, 31, 14, 45, 37)
```

在这段代码中，您在第 3 行的**上导入`datetime`，并在第 4 行**的**上使用带有`date_string`和`format_string`的`datetime.strptime()`。最后，**第 5 行**显示了由`.strptime()`创建的`datetime`实例中的属性值。您可以看到它们与上表中显示的值相匹配。**

**注意:**创建`datetime`实例有更高级的方法，但是它们涉及到使用必须安装的第三方库。一个特别简洁的库叫做 [`dateparser`](https://dateparser.readthedocs.io/en/latest/) ，它允许你提供自然语言的字符串输入。输入甚至支持多种语言:

>>>

```py
 1>>> import dateparser
 2>>> dateparser.parse("yesterday")
 3datetime.datetime(2020, 3, 13, 14, 39, 1, 350918)
 4>>> dateparser.parse("morgen")
 5datetime.datetime(2020, 3, 15, 14, 39, 7, 314754)
```

在这段代码中，您使用`dateparser`通过传递两个不同的时间字符串表示来创建两个`datetime`实例。在**第 1 行**，你导入`dateparser`。然后，在第 2 行的**上，您使用带有参数`"yesterday"`的`.parse()`来创建一个过去 24 小时的`datetime`实例。写这篇文章时，这是 2020 年 3 月 13 日，下午 2 点 39 分。**

在**的第 3 行**，你用`.parse()`和参数`"morgen"`。 *Morgen* 在德语中是明天的意思，所以`dateparser`在未来 24 小时创建一个`datetime`实例。在撰写本文时，这是 3 月 15 日下午 2 点 39 分。

[*Remove ads*](/account/join/)

## 开始你的 PyCon 倒计时

现在你已经有足够的信息开始为明年的 [PyCon US](https://us.pycon.org/) 倒计时钟工作了！PyCon US 2021 将于 2021 年 5 月 12 日在宾夕法尼亚州匹兹堡开幕。随着 2020 年的活动[被取消](https://pycon.blogspot.com/2020/03/pycon-us-2020-in-pittsburgh.html)，许多蟒蛇对明年的聚会格外兴奋。这是记录你需要等待多长时间的好方法，同时还能提升你的`datetime`技能！

首先，创建一个名为`pyconcd.py`的文件，并添加以下代码:

```py
# pyconcd.py

from datetime import datetime

PYCON_DATE = datetime(year=2021, month=5, day=12, hour=8)
countdown = PYCON_DATE - datetime.now()
print(f"Countdown to PyCon US 2021: {countdown}")
```

在这段代码中，您从`datetime`导入`datetime`并定义一个常量`PYCON_DATE`，它存储下一个 PyCon US 的日期。你不希望 PyCon 的日期改变，所以你用大写字母命名变量，以表明它是一个常数。

接下来，计算`datetime.now()`，即[当前时间](https://realpython.com/python-get-current-time/)和`PYCON_DATE`之间的差值。取两个`datetime`实例之间的差返回一个 [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html#timedelta-objects) 实例。

`timedelta`实例表示两个`datetime`实例之间的时间变化。名称中的 **delta** 是对希腊字母 delta 的引用，在科学和工程中用来表示变化。稍后你会学到更多关于如何使用`timedelta`进行更一般的算术运算的[。](#doing-arithmetic-with-python-datetime)

最后，截至 2020 年 4 月 9 日晚上 9:30 之前的打印输出是:

```py
Countdown to PyCon US 2021: 397 days, 10:35:32.139350
```

距离 PyCon US 2021 只有 397 天了！这个输出有点笨拙，所以[稍后](#finishing-your-pycon-countdown)您将看到如何改进格式。如果您在不同的日子运行这个脚本，您将得到不同的输出。如果您在 2021 年 5 月 12 日上午 8:00 运行该脚本，您将获得负剩余时间！

## 使用时区

如前所述，存储日期所在的时区是确保代码正确的一个重要方面。Python `datetime`提供了`tzinfo`，这是一个抽象基类，允许`datetime.datetime`和`datetime.time`包含时区信息，包括夏令时的概念。

然而，`datetime`没有提供与 IANA 时区数据库交互的直接方式。Python `datetime.tzinfo`文档[推荐](https://docs.python.org/3/library/datetime.html#tzinfo-objects)使用名为`dateutil`的第三方包。可以用 [`pip`](https://realpython.com/what-is-pip/) 安装`dateutil`:

```py
$ python -m pip install python-dateutil
```

请注意，您从 PyPI 安装的包的名称`python-dateutil`不同于您用来导入包的名称，后者只是`dateutil`。

### 使用`dateutil`将时区添加到 Python `datetime`

`dateutil`如此有用的一个原因是它包括一个到 IANA 时区数据库的接口。这消除了为您的`datetime`实例分配时区的麻烦。试试这个例子，看看如何设置一个`datetime`实例，使其符合您的本地时区:

>>>

```py
>>> from dateutil import tz
>>> from datetime import datetime
>>> now = datetime.now(tz=tz.tzlocal())
>>> now
datetime.datetime(2020, 1, 26, 0, 55, 3, 372824, tzinfo=tzlocal())
>>> now.tzname()
'Eastern Standard Time'
```

在这个例子中，您[从`dateutil`导入](https://realpython.com/courses/absolute-vs-relative-imports-python/) `tz`，从`datetime`导入`datetime`。然后使用`.now()`创建一个设置为当前时间的`datetime`实例。

您还将关键字`tz`传递给`.now()`，并将`tz`设置为等于`tz.tzlocal()`。在`dateutil`中，`tz.tzlocal()`返回`datetime.tzinfo`的一个具体实例。这意味着它可以表示`datetime`需要的所有必要的时区偏移和夏令时信息。

您还可以使用`.tzname()`打印时区名称，它会打印`'Eastern Standard Time'`。这是 Windows 的输出，但是在 macOS 或 Linux 上，如果您在美国东部时区的冬天，您的输出可能是`'EST'`。

您也可以创建与您的计算机报告的时区不同的时区。为此，您将使用`tz.gettz()`并传递您感兴趣的时区的官方 [IANA 名称](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)。这里有一个如何使用`tz.gettz()`的例子:

>>>

```py
>>> from dateutil import tz
>>> from datetime import datetime
>>> London_tz = tz.gettz("Europe/London")
>>> now = datetime.now(tz=London_tz)
>>> now
datetime.datetime(2020, 1, 26, 6, 14, 53, 513460, tzinfo=tzfile('GB-Eire'))
>>> now.tzname()
'GMT'
```

在本例中，您使用`tz.gettz()`检索英国伦敦的时区信息，并将其存储在`London_tz`中。然后检索当前时间，将时区设置为`London_tz`。

在 Windows 上，这给了属性`tzinfo`值`tzfile('GB-Eire')`。在 macOS 或 Linux 上，`tzinfo`属性看起来类似于`tzfile('/usr/share/zoneinfo/Europe/London)`，但是根据`dateutil`从哪里提取时区数据，它可能会略有不同。

您还使用`tzname()`打印时区的名称，现在是`'GMT'`，意思是格林威治标准时间。这个输出在 Windows、macOS 和 Linux 上是相同的。

在前面的[章节](#creating-python-datetime-instances)中，您了解到不应该使用`.utcnow()`在当前的 UTC 创建一个`datetime`实例。现在您知道了如何使用`dateutil.tz`向`datetime`实例提供时区。这里有一个修改自 Python 文档中的[建议](https://docs.python.org/3/library/datetime.html#datetime.datetime.utcnow)的例子:

>>>

```py
>>> from dateutil import tz
>>> from datetime import datetime
>>> datetime.now(tz=tz.UTC)
datetime.datetime(2020, 3, 14, 19, 1, 20, 228415, tzinfo=tzutc())
```

在这段代码中，您使用 [`tz.UTC`](https://dateutil.readthedocs.io/en/stable/tz.html#dateutil.tz.dateutil.tz.UTC) 将`datetime.now()`的时区设置为 UTC 时区。相比使用`utcnow()`，推荐使用这种方法，因为`utcnow()`返回一个*朴素* `datetime`实例，而这里演示的方法返回一个*感知* `datetime`实例。

接下来，您将绕一小段路来了解**天真的**与**清醒的** `datetime`实例。如果你已经知道了这一切，那么你可以[跳过](#improving-your-pycon-countdown)来用时区信息改进你的 PyCon 倒计时。

[*Remove ads*](/account/join/)

### 比较幼稚和有意识的 Python `datetime`实例

Python `datetime`实例支持两种类型的操作，简单操作和感知操作。它们之间的基本区别是简单实例不包含时区信息，而感知实例包含时区信息。更正式地说，引用 Python 文档:

> 一个有意识的物体代表了一个不可解释的特定时刻。简单对象不包含足够的信息来明确地定位自己相对于其他日期/时间对象的位置。([来源](https://docs.python.org/3/library/datetime.html#id1))

这是使用 Python `datetime`的一个重要区别。一个**感知的** `datetime`实例可以明确地将其自身与其他感知的`datetime`实例进行比较，并且在算术运算中使用时将总是返回正确的时间间隔。

**天真的** `datetime`相反，实例可能是模棱两可的。这种模糊性的一个例子与夏令时有关。实行夏令时的地区在春季将时钟向前拨一小时，在秋季将时钟向后拨一小时。这通常发生在当地时间凌晨 2:00。在春天，从凌晨 2:00 到 2:59 的时间*从不发生*，在秋天，从凌晨 1:00 到 1:59 的时间*发生两次*！

实际上，这些时区与 UTC 的时差在一年中会发生变化。IANA 跟踪这些变化，并将它们编入计算机上安装的不同数据库文件中。使用像`dateutil`这样的库，它在幕后使用 IANA 数据库，是确保您的代码正确处理时间算术的一个好方法。

**注意:**在 Python 中，naive 和 aware `datetime`实例的区别是由`tzinfo`属性决定的。一个 aware `datetime`实例的`tzinfo`属性等同于`datetime.tzinfo`抽象基类的一个子类。

[Python 3.8](https://realpython.com/python38-new-features/) 及以下版本提供了`tzinfo`的一个具体实现，称为`timezone`。然而，`timezone`仅限于表示 UTC 的固定偏移量，一年中不会改变，所以当您需要考虑夏令时之类的变化时，它就没什么用了。

[Python 3.9](https://realpython.com/python39-new-features/) 包含了一个名为 [`zoneinfo`](https://docs.python.org/3.9/library/zoneinfo.html) 的新模块，它提供了跟踪 IANA 数据库的`tzinfo`的具体实现，因此它包含了像夏令时这样的变化。然而，在 Python 3.9 被广泛使用之前，如果你需要支持多个 Python 版本，依靠`dateutil`可能是有意义的。

`dateutil`还提供了您之前使用的`tz`模块中`tzinfo`的几个具体实现。你可以查看 [`dateutil.tz`文档](https://dateutil.readthedocs.io/en/stable/tz.html)了解更多信息。

这并不意味着您总是需要使用 aware `datetime`实例。但是，如果你在相互比较时间，尤其是在比较世界不同地区的时间时，感知实例是至关重要的。

## 提高你的 PyCon 倒计时

既然您已经知道了如何向 Python `datetime`实例添加时区信息，那么您就可以改进您的 PyCon 倒计时代码了。之前，您使用了标准的`datetime`构造函数来传递 PyCon 将启动的年、月、日和小时。您可以更新您的代码来使用 [`dateutil.parser`](https://dateutil.readthedocs.io/en/stable/parser.html) 模块，它为创建`datetime`实例提供了一个更自然的界面:

```py
# pyconcd.py

from dateutil import parser, tz
from datetime import datetime

PYCON_DATE = parser.parse("May 12, 2021 8:00 AM")
PYCON_DATE = PYCON_DATE.replace(tzinfo=tz.gettz("America/New_York"))
now = datetime.now(tz=tz.tzlocal())

countdown = PYCON_DATE - now
print(f"Countdown to PyCon US 2021: {countdown}")
```

在这段代码中，您[从`dateutil`导入](https://realpython.com/python-import/) `parser`和`tz`，从`datetime`导入`datetime`。接下来，使用`parser.parse()`从字符串中读取下一个 PyCon US 的日期。这比普通的`datetime`构造函数可读性更好。

`parser.parse()`返回一个简单的`datetime`实例，所以您使用`.replace()`将`tzinfo`更改为`America/New_York`时区。PyCon US 2021 将在美国东部时区宾夕法尼亚州的匹兹堡举行。该时区的标准名称是`America/New_York`，因为纽约市是该时区最大的城市。

`PYCON_DATE`是一个 aware `datetime`实例，其时区设置为美国东部时间。由于 5 月 12 日是夏令时生效后，时区名称为`'EDT'`，或`'Eastern Daylight Time'`。

接下来，创建`now`来表示当前时刻，并将其作为您的本地时区。最后，找到`PYCON_DATE`和`now`之间的`timedelta`，并打印结果。如果您所在的地区没有根据夏令时调整时钟，那么您可能会看到 PyCon 改变一小时之前的剩余小时数。

## 用 Python `datetime`做算术

Python `datetime`实例支持几种类型的算法。正如您之前看到的，这依赖于使用`timedelta`实例来表示时间间隔。`timedelta`非常有用，因为它内置于 Python 标准库中。这里有一个如何使用`timedelta`的例子:

>>>

```py
>>> from datetime import datetime, timedelta
>>> now = datetime.now()
>>> now
datetime.datetime(2020, 1, 26, 9, 37, 46, 380905)
>>> tomorrow = timedelta(days=+1)
>>> now + tomorrow
datetime.datetime(2020, 1, 27, 9, 37, 46, 380905)
```

在这段代码中，您创建了存储当前时间的`now`和存储`+1`天的`timedelta`的`tomorrow`。接下来，添加`now`和`tomorrow`来产生未来某一天的`datetime`实例。请注意，使用简单的`datetime`实例，就像您在这里一样，意味着`datetime`的`day`属性增加 1，并且不考虑任何重复或跳过的时间间隔。

`timedelta`实例也支持负值作为参数的输入:

>>>

```py
>>> yesterday = timedelta(days=-1)
>>> now + yesterday
datetime.datetime(2020, 1, 25, 9, 37, 46, 380905)
```

在这个例子中，您提供了`-1`作为`timedelta`的输入，所以当您添加`now`和`yesterday`时，结果是在`days`属性中减少 1。

实例支持加法和减法以及所有参数的正整数和负整数。你甚至可以提供正反两面的观点。例如，您可能想加上三天，减去四小时:

>>>

```py
>>> delta = timedelta(days=+3, hours=-4)
>>> now + delta
datetime.datetime(2020, 1, 29, 5, 37, 46, 380905)
```

在本例中，您加上三天，减去四小时，因此新的`datetime`是 1 月 29 日上午 5:37。`timedelta`在这种方式下非常有用，但是它有一定的局限性，因为它不能加减大于一天的时间间隔，比如一个月或一年。幸运的是，`dateutil`提供了一个更强大的替代品叫做 [**`relativedelta`**](https://dateutil.readthedocs.io/en/stable/relativedelta.html) 。

`relativedelta`的基本语法和`timedelta`非常相似。您可以提供产生任意数量的年、月、日、小时、秒或微秒变化的关键字参数。您可以用这段代码重现第一个`timedelta`示例:

>>>

```py
>>> from dateutil.relativedelta import relativedelta
>>> tomorrow = relativedelta(days=+1)
>>> now + tomorrow
datetime.datetime(2020, 1, 27, 9, 37, 46, 380905)
```

在这个例子中，你用`relativedelta`而不是`timedelta`来寻找明天对应的`datetime`。现在你可以试着给`now`加上五年一个月零三天，同时减去四小时三十分钟:

>>>

```py
>>> delta = relativedelta(years=+5, months=+1, days=+3, hours=-4, minutes=-30)
>>> now + delta
datetime.datetime(2025, 3, 1, 5, 7, 46, 380905)
```

请注意，在此示例中，日期结束于 2025 年 3 月 1 日。这是因为给`now`加三天就是 1 月 29 日，再加一个月就是 2 月 29 日，这一天只存在于闰年。因为 2025 年不是闰年，所以日期会滚动到下个月。

您还可以使用`relativedelta`来计算两个`datetime`实例之间的差异。在前面，您使用了减法运算符来查找两个 Python 实例`datetime`、`PYCON_DATE`和`now`之间的差异。使用`relativedelta`，您需要将两个`datetime`实例作为参数传递，而不是使用减法运算符:

>>>

```py
>>> now
datetime.datetime(2020, 1, 26, 9, 37, 46, 380905)
>>> tomorrow = datetime(2020, 1, 27, 9, 37, 46, 380905)
>>> relativedelta(now, tomorrow)
relativedelta(days=-1)
```

在这个例子中，您通过将`days`字段加 1 来为`tomorrow`创建一个新的`datetime`实例。然后，使用`relativedelta`并传递`now`和`tomorrow`作为两个参数。`dateutil`然后取这两个`datetime`实例之间的差，并将结果作为`relativedelta`实例返回。在这种情况下，差异是`-1`天，因为`now`发生在`tomorrow`之前。

物体还有无数其他用途。您可以使用它们来查找复杂的日历信息，例如下一年的 10 月 13 日是星期五，或者当月的最后一个星期五是几号。您甚至可以使用它们来替换一个`datetime`实例的属性，并创建一个`datetime`，例如，在未来一周的上午 10:00。你可以在`dateutil` [文档](https://dateutil.readthedocs.io/en/stable/examples.html#relativedelta-examples)中了解所有这些其他用途。

[*Remove ads*](/account/join/)

## 完成你的 PyCon 倒计时

你现在已经有足够的工具来完成你的 PyCon 2021 倒计时钟，并且提供了一个很好的界面来使用。在本节中，您将使用`relativedelta`来计算离 PyCon 还有多长时间，开发一个函数来以漂亮的格式打印剩余时间，并向用户显示 PyCon 的日期。

### 在您的 PyCon 倒计时中使用`relativedelta`

首先，用`relativedelta`代替普通的减法运算符。使用减法运算符，您的`timedelta`对象无法计算大于一天的时间间隔。但是，`relativedelta`允许您显示剩余的年、月和日:

```py
 1# pyconcd.py
 2
 3from dateutil import parser, tz
 4from dateutil.relativedelta import relativedelta
 5from datetime import datetime
 6
 7PYCON_DATE = parser.parse("May 12, 2021 8:00 AM")
 8PYCON_DATE = PYCON_DATE.replace(tzinfo=tz.gettz("America/New_York"))
 9now = datetime.now(tz=tz.tzlocal())
10
11countdown = relativedelta(PYCON_DATE, now)
12print(f"Countdown to PyCon US 2021: {countdown}")
```

您在这段代码中所做的唯一更改是用`countdown = relativedelta(PYCON_DATE, now)`替换**第 11 行的**。这个脚本的输出应该告诉您 PyCon US 2021 将在大约一年零一个月后发生，这取决于您何时运行该脚本。

然而，输出并不是很漂亮，因为它看起来像是`relativedelta()`的签名。您可以通过用下面的代码替换前面代码中的第 11 行来构建一些更漂亮的输出:

```py
11def time_amount(time_unit: str, countdown: relativedelta) -> str:
12    t = getattr(countdown, time_unit)
13    return f"{t}  {time_unit}" if t != 0 else ""
14
15countdown = relativedelta(PYCON_DATE, now)
16time_units = ["years", "months", "days", "hours", "minutes", "seconds"]
17output = (t for tu in time_units if (t := time_amount(tu, countdown)))
18print("Countdown to PyCon US 2021:", ", ".join(output))
```

这段代码需要 Python 3.8，因为它使用了新的 [**海象运算符**](https://realpython.com/python38-new-features/#the-walrus-in-the-room-assignment-expressions) 。通过使用传统的 [`for`循环](https://realpython.com/python-for-loop/)代替**第 17 行**，可以让这个脚本在旧版本的 Python 上工作。

在这段代码中，您定义了带有两个参数的`time_amount()`，时间单位和应该从中检索时间单位的`relativedelta`实例。如果时间量不等于零，那么`time_amount()`返回一个带有时间量和时间单位的字符串。否则，它返回一个空字符串。

你在**第 17 行**的[理解](https://realpython.com/list-comprehension-python/)中使用`time_amount()`。那一行创建了一个 [**生成器**](https://realpython.com/introduction-to-python-generators/) 来存储从`time_amount()`返回的非空字符串。它使用 [walrus 运算符](https://realpython.com/python38-new-features/#the-walrus-in-the-room-assignment-expressions)将`time_amount()`的返回值赋给`t`，并且只有在`True`时才包含`t`。

最后，**行 18** 使用[发生器](https://realpython.com/courses/python-generators/)上的 [`.join()`](https://realpython.com/python-string-split-concatenate-join/#going-from-a-list-to-a-string-in-python-with-join) 打印最终输出。接下来，您将看到在脚本的输出中包含 PyCon 日期。

### 在 PyCon 倒计时中显示 PyCon 日期

[在前面的](#using-strings-to-create-python-datetime-instances)中，您学习了如何使用`.strptime()`创建`datetime`实例。这个方法使用 Python 中一种特殊的小型语言来指定日期字符串的格式。

Python `datetime`有一个额外的方法叫做`.strftime()`，它允许你将一个`datetime`实例格式化为一个字符串。从某种意义上说，这是使用`.strptime()`解析的逆向操作。你可以通过记住`.strptime()`中的`p`代表**解析**，而`.strftime()`中的`f`代表**格式**来区分这两种方法。

在您的 PyCon 倒计时中，您可以使用`.strftime()`来打印输出，让用户知道 PyCon US 将开始的日期。记住，你可以在[strftime.org](https://strftime.org)上找到你想要使用的格式化代码。现在将这段代码添加到 PyCon 倒计时脚本的第 18 行的**处:**

```py
18pycon_date_str = PYCON_DATE.strftime("%A, %B %d, %Y at %H:%M %p %Z")
19print(f"PyCon US 2021 will start on:", pycon_date_str)
20print("Countdown to PyCon US 2021:", ", ".join(output))
```

在这段代码中，**第 18 行**使用`.strftime()`创建一个表示 PyCon US 2021 开始日期的字符串。输出包括工作日、月、日、年、小时、分钟、上午或下午以及时区:

```py
Wednesday, May 12, 2021 at 08:00 AM EDT
```

在第**行第 19** 处，您打印这个字符串，让用户看到一些解释文本。最后一行打印离 PyCon 开始日期的剩余时间。接下来，您将完成您的脚本，以便其他人可以更容易地重用它。

[*Remove ads*](/account/join/)

### 完成您的 PyCon 倒计时

你需要采取的最后一步是遵循 Python [最佳实践](https://realpython.com/tutorials/best-practices/)，将产生输出的代码放入 [`main()`](https://realpython.com/python-main-function/) 函数中。应用所有这些更改后，您可以签出完整的最终代码:

```py
 1# pyconcd.py
 2
 3from dateutil import parser, tz
 4from dateutil.relativedelta import relativedelta
 5from datetime import datetime
 6
 7PYCON_DATE = parser.parse("May 12, 2021 8:00 AM")
 8PYCON_DATE = PYCON_DATE.replace(tzinfo=tz.gettz("America/New_York"))
 9
10def time_amount(time_unit: str, countdown: relativedelta) -> str:
11    t = getattr(countdown, time_unit)
12    return f"{t}  {time_unit}" if t != 0 else ""
13
14def main():
15    now = datetime.now(tz=tz.tzlocal())
16    countdown = relativedelta(PYCON_DATE, now)
17    time_units = ["years", "months", "days", "hours", "minutes", "seconds"]
18    output = (t for tu in time_units if (t := time_amount(tu, countdown)))
19    pycon_date_str = PYCON_DATE.strftime("%A, %B %d, %Y at %H:%M %p %Z")
20    print(f"PyCon US 2021 will start on:", pycon_date_str)
21    print("Countdown to PyCon US 2021:", ", ".join(output))
22
23if __name__ == "__main__":
24    main()
```

在这段代码中，您将`print()`和用于生成器的代码移动到`main()`中。在**的第 23 行**，你使用**保护子句**来确保`main()`只在这个文件作为[脚本](https://realpython.com/run-python-scripts/)执行时运行。这允许其他人导入您的代码并重用`PYCON_DATE`，例如，如果他们愿意的话。

现在，您可以随意修改这个脚本。一个简单的做法是允许用户通过传递一个[命令行参数](https://realpython.com/python-command-line-arguments/)来改变与`now`相关的时区。你也可以把`PYCON_DATE`改成离家更近的，比如说 [PyCon Africa](https://realpython.com/pycon-africa-2019-recap/) 或者[europhon](https://www.europython-society.org/)。

要对 PyCon 更加兴奋，请在 PyCon US 2019 和[查看](https://realpython.com/pycon-guide/)[真实 Python，了解如何充分利用 PyCon](https://realpython.com/real-python-pycon-us/) ！

## Python 的替代品`datetime`和`dateutil`

当您处理日期和时间时，Python `datetime`和`dateutil`是强大的库组合。在 Python 文档中甚至推荐使用`dateutil`。然而，还有许多其他库可以用来处理 Python 中的日期和时间。其中一些依赖于`datetime`和`dateutil`，而另一些则是完全独立的替代品:

*   **[pytz](https://pypi.org/project/pytz/)** 提供类似`dateutil`的时区信息。它使用了与标准`datetime.tzinfo`稍有不同的接口，所以如果你决定使用它，请注意[潜在的问题](https://blog.ganssle.io/articles/2018/03/pytz-fastest-footgun.html)。
*   **[箭头](https://arrow.readthedocs.io/en/latest/)** 为`datetime`提供了插播替换。它的灵感来自于`moment.js`，所以如果你是 web 开发出身，那么这可能是一个更熟悉的界面。
*   **[摆](https://pendulum.eustace.io/)** 为`datetime`提供了另一个插播替代。它包括一个时区接口和一个改进的`timedelta`实现。
*   **[Maya](https://github.com/timofurrer/maya)** 提供了与`datetime`类似的界面。它依赖于 Pendulum 的部分解析库。
*   **[dateparser](https://dateparser.readthedocs.io/en/latest/)** 提供了从人类可读的文本中生成`datetime`实例的接口。它很灵活，支持多种语言。

此外，如果您大量使用 [NumPy](https://realpython.com/tutorials/numpy/) 、 [Pandas](https://realpython.com/courses/introduction-pandas-and-vincent/) 或其他[数据科学](https://realpython.com/tutorials/data-science/)软件包，那么有几个选项可能对您有用:

*   **[NumPy](https://numpy.org/doc/1.18/reference/arrays.datetime.html)** 提供了与内置 Python `datetime`库类似的 API，但是 NumPy 版本可以在数组中使用。
*   **[熊猫](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)** 通过使用 NumPy `datetime`模块为[数据帧](https://realpython.com/courses/pandas-dataframes-101/)中的时序数据提供支持，通常是基于时间的事件的顺序值。
*   **[cftime](https://unidata.github.io/cftime/api.html)** 支持除[公历](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar)以外的日历，以及其他符合气候和预测(CF)惯例的时间单位。它被 [`xarray`](http://xarray.pydata.org/en/stable/time-series.html) 包用来提供时序支持。

## 延伸阅读

因为用时间编程可能非常复杂，所以网上有很多资源可以帮助你了解更多。幸运的是，这是许多使用每种编程语言的人都考虑过的问题，因此您通常可以找到信息或工具来帮助解决您可能遇到的任何问题。以下是一些精选的文章和视频，我发现它们对撰写本教程很有帮助:

*   [夏令时和时区最佳实践](https://stackoverflow.com/a/2532962)
*   [存储 UTC 不是灵丹妙药](https://codeblog.jonskeet.uk/2019/03/27/storing-utc-is-not-a-silver-bullet/)
*   [如何为未来事件保存日期时间](http://www.creativedeletion.com/2015/03/19/persisting_future_datetimes.html)
*   中使用日期时间的编码最佳实践。NET 框架
*   电脑爱好者:时间的问题&时区
*   [时间数据编程的复杂性](https://www.mojotech.com/blog/the-complexity-of-time-data-programming/)

另外，Paul Ganssle 是 CPython 的核心贡献者，也是目前`dateutil`的维护者。他的文章和视频对 Python 用户来说是一个很好的资源:

*   [使用时区:你希望不需要知道的一切](https://www.youtube.com/watch?v=rz3D8VG_2TY) (PyCon 2019)
*   [pytz:西方最快的手枪](https://blog.ganssle.io/articles/2018/03/pytz-fastest-footgun.html)
*   [停止使用 utcnow 和 utcfromtimestamp](https://blog.ganssle.io/articles/2019/11/utcnow.html)
*   [非传递日期时间比较的一个奇怪例子](https://blog.ganssle.io/articles/2018/02/a-curious-case-datetimes.html)

## 结论

在本教程中，您学习了关于日期和时间的编程，以及为什么它经常导致错误和混乱。您还了解了 Python **`datetime`** 和 **`dateutil`** 模块，以及如何在代码中处理时区。

**现在你可以:**

*   在你的程序中以一种良好的、经得起未来考验的格式存储日期
*   **用格式化字符串创建** Python `datetime`实例
*   **用`dateutil`将**时区信息添加到`datetime`实例中
*   **使用`relativedelta`对`datetime`实例执行**算术运算

最后，您创建了一个脚本，该脚本倒数到下一次 PyCon US 的剩余时间，这样您就可以为最大的 Python 聚会感到兴奋了。日期和时间可能很棘手，但是有了这些 Python 工具，您就可以解决最棘手的问题了！

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和书面教程一起看，加深理解: [**使用 Python 的 datetime 模块**](/courses/python-datetime-module/)**********
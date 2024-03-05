# 如何在 Python 中获取和使用当前时间

> 原文：<https://realpython.com/python-get-current-time/>

对于许多与时间相关的操作来说，用 Python 获取当前时间(T2)是一个很好的起点。一个非常重要的用例是创建[时间戳](https://en.wikipedia.org/wiki/Timestamp)。在本教程中，您将学习如何使用 [`datetime`](https://docs.python.org/3/library/datetime.html) 模块**获取**、**显示**、**格式化**当前时间。

为了在 Python 应用程序中有效地使用当前时间，您将在腰带上添加一些工具。例如，你将学习如何**读取当前时间的属性**，比如年、分或秒。为了使时间更容易阅读，您将探索**打印**它的选项。您还将了解不同的**时间格式**，并学习计算机**如何表示**时间，如何**序列化**时间，以及如何处理**时区**。

**源代码:** [点击这里下载 Python 中获取和使用当前时间的免费源代码](https://realpython.com/bonus/python-get-current-time-code/)。

## 如何用 Python 讲时间

获取和打印当前时间最直接的方法是使用`datetime`模块中`datetime` [类](https://realpython.com/python3-object-oriented-programming/#define-a-class-in-python)的`.now()` [类方法](https://realpython.com/instance-class-and-static-methods-demystified/#class-methods):

>>>

```py
>>> from datetime import datetime
>>> now = datetime.now()

>>> now
datetime(2022, 11, 22, 14, 31, 59, 331225)

>>> print(now)
2022-11-22 14:31:59.331225
```

类方法`.now()`是一个[构造器方法](https://realpython.com/python-multiple-constructors/#providing-multiple-constructors-with-classmethod-in-python)，它返回一个`datetime`对象。当 REPL 对`now`变量求值时，你会得到一个`datetime`对象的[表示](https://dbader.org/blog/python-repr-vs-str)。很难说出每个数字的含义。但是如果您显式打印`now`变量，那么您会得到一个稍微不同的输出，它以熟悉的时间戳格式显示信息。

**注意:**你在这里得到的`datetime`对象是不知道时区的。通常您的操作系统可以正确解析时区，但是`datetime`对象本身目前没有时区信息。在本教程的[部分，您将了解时区感知对象。](#get-time-zoneaware-python-time-and-date-objects)

您可能会认出打印的`datetime`对象的格式。它严格遵循国际标准，的时间和日期格式。你会在很多地方发现这种格式！

不过，Python 使用的格式与 ISO 8601 标准略有不同。标准规定时间戳的日期和小时部分应该用一个`T`字符分隔，但是通过`print()`函数传递的默认`datetime`对象用一个空格分隔它们。

Python 具有可扩展性和可定制性，使您能够定制打印时间戳的格式。打印时，`datetime`类在内部使用它的`.isoformat()`方法。由于`.isoformat()`只是一个[实例方法](https://realpython.com/instance-class-and-static-methods-demystified/#instance-methods)，您可以从任何`datetime`对象中直接调用它来定制 ISO 时间戳:

>>>

```py
>>> datetime.now().isoformat()
'2022-11-22T14:31:59.331225'

>>> datetime.now().isoformat(sep=" ")
'2022-11-22 14:31:59.331225'
```

您会注意到，当您不带任何参数调用`.isoformat()`时，会使用标准的 ISO 8601 分隔符`T`。然而，`datetime`类实现其[特殊实例方法`.__str__()`](https://realpython.com/operator-function-overloading/#printing-your-objects-prettily-using-str) 的方式是用一个空格作为`sep`参数。

能够获得完整的日期和时间是很好的，但是有时您可能会寻找一些特定的内容。例如，也许您只想要月份或日期。在这些情况下，您可以从一系列属性中进行选择:

>>>

```py
>>> from datetime import datetime
>>> now = datetime.now()
>>> print(f"""
... {now.month = } ... {now.day = } ... {now.hour = } ... {now.minute = } ... {now.weekday() = } ... {now.isoweekday() = }"""
... )
now.month = 11
now.day = 22
now.hour = 14
now.minute = 31
now.weekday() = 1
now.isoweekday() = 2
```

在这个代码片段中，您使用一个三重引号 [f 字符串](https://realpython.com/python-f-strings/)和花括号内的`=`符号来输出表达式及其结果。

继续探索不同的属性和方法，用一个`datetime`对象调用 [`dir()`](https://docs.python.org/3/library/functions.html#dir) 函数，列出当前[作用域](https://realpython.com/python-namespaces-scope/)中可用的名称。或者你可以查看`datetime` 的[文档。无论哪种方式，你都会发现大量的选择。](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)

您会注意到上一个示例的结果通常是数字。这可能很适合你，但是将工作日显示为数字可能并不理想。由于`.weekday()`和`.isoweekday()`方法返回不同的数字，这也可能特别令人困惑。

**注:**对于`.weekday()`法，周一为`0`，周日为`6`。对于`.isoweekday()`，周一是`1`，周日是`7`。

ISO 时间戳很好，但是也许您想要比 ISO 时间戳更易读的东西。例如，对于一个人来说，几毫秒可能有点长。在下一节中，您将学习如何以您喜欢的任何方式格式化您的时间戳。

[*Remove ads*](/account/join/)

## 格式化时间戳以提高可读性

为了以一种定制的、人类可读的方式方便地输出时间，`datetime`有一个名为`.strftime()`的方法。`.strftime()`方法以一个[格式代码](https://docs.python.org/3/library/datetime.html?highlight=datetime#strftime-and-strptime-format-codes)作为参数。格式代码是一串特殊的[标记](https://en.wikipedia.org/wiki/Lexical_analysis#Token)，这些标记将被来自`datetime`对象的信息替换。

`.strftime()`方法会给你很多选择，告诉你如何准确地表示你的`datetime`对象。例如，采用以下格式:

>>>

```py
>>> from datetime import datetime

>>> datetime.now().strftime("%A, %B %d")
'Tuesday, November 22'
```

在本例中，您使用了以下格式代码:

*   `%A`:工作日全名
*   `%B`:月份全称
*   `%d`:一个月中的第几天

格式字符串中的逗号和文字空格按原样打印。`.strftime()`方法只替换它识别为代码的内容。`.strftime()`中的格式代码总是以一个百分号(`%`)开始，它遵循一个[旧 C 标准](https://man7.org/linux/man-pages/man3/strftime.3.html)。这些代码类似于旧的 [`printf`字符串格式样式](https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting)，但它们并不相同。

格式代码的[文档有一个漂亮的表格，向您展示了您可以使用的所有不同的格式代码。在 strftime.org 的](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)[网站](https://strftime.org/)上也有一个不错的小抄。去看看他们。

**注意:** Python 的 f 字符串支持与`.strftime()`相同的格式代码。你可以这样使用它们:

>>>

```py
>>> f"{datetime.now():%A, %B %d}"
'Tuesday, November 22'
```

在 f 字符串中，使用冒号(`:`)来分隔表达式和相应的格式代码。

所以现在你可以得到你喜欢的时间和格式。这应该能满足你基本的报时需求，但也许你对计算机内部如何表示和处理时间以及如何将时间存储在文件或数据库中感到好奇。在下一节中，您将深入了解这一点。

## 用 Python 获取当前的 Unix 时间

电脑喜欢数字。但是日期和时间是遵循有趣规则的有趣的人类数字。一天二十四小时？一小时六十分钟？这些聪明的想法是谁的？

为了简化问题，并且考虑到计算机不介意大数，在开发操作系统的时候做出了一个决定。

这个决定是将所有时间表示为自 1970 年 1 月 1 日午夜[UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time)以来经过的秒数。这个时间点也被称为 Unix [时代](https://en.wikipedia.org/wiki/Epoch_(computing))。这个时间系统被称为 [Unix 时间](https://en.wikipedia.org/wiki/Unix_time)。今天的大多数计算机系统——甚至是 Windows——都使用 Unix 时间来表示内部时间。

Unix 时间在 UTC 时间 1970 年 1 月 1 日午夜是零。如果你想知道当前的 Unix 时间，那么你可以使用另一种`datetime`方法:

>>>

```py
>>> from datetime import datetime

>>> datetime.now().timestamp()
1669123919.331225
```

`.timestamp()`方法返回自 Unix 纪元以来的秒数，精度很高。毕竟，在所有的属性和方法之下，每个日期对大多数计算机来说只不过是一个大数字。

**注意:**因为您创建的`datetime`对象不知道时区，所以您生成的时间戳实际上可能不是 UNIX 时间！这可能没问题，只要您的系统正确配置了时间设置。

在大多数情况下，您可以不去管 Unix 时间。这是一种表示时间的方式，对电脑来说效果很好，但对习惯了公历[的人来说就不太好了。不过，Unix 时间戳会在您的日期和时间冒险中突然出现，所以了解它们绝对有好处。](https://en.wikipedia.org/wiki/Gregorian_calendar)

正确生成的 Unix 时间戳的最大好处之一是它明确地捕捉了世界范围内的某个时刻。Unix 纪元总是使用 UTC，所以时区偏移方面没有歧义——也就是说，如果您能够可靠地创建没有 UTC 偏移的时间戳。

但不幸的是，你将不得不经常处理混乱的时区。不过，不要害怕！在下一节中，您将了解时区感知`datetime`对象。

[*Remove ads*](/account/join/)

## 获取支持时区的 Python 时间和日期对象

Unix 时间戳的明确性是有吸引力的，但是一般来说用 ISO 8601 格式序列化时间和日期更好，因为除了计算机容易解析 T3 之外，它还是人类可读的 T5，并且它是一个国际标准。

更重要的是，尽管 Unix 时间戳在某种程度上是可识别的，但它们可能会被误认为代表其他东西。毕竟，它们只是数字。有了 ISO 时间戳，您马上就知道它代表了什么。引用 Python 的[禅，*可读性算*。](https://realpython.com/lessons/zen-of-python/)

如果你想用完全明确的术语表示你的`datetime`对象，那么你首先需要让你的对象**知道时区**。一旦您有了时区感知对象，时区信息就会添加到您的 ISO 时间戳中:

>>>

```py
>>> from datetime import datetime
>>> now = datetime.now()

>>> print(now.tzinfo)
None

>>> now_aware = now.astimezone()

>>> print(now_aware.tzinfo)
Romance Standard Time

>>> now_aware.tzinfo
datetime.timezone(datetime.timedelta(seconds=3600), 'Romance Standard Time')

>>> now_aware.isoformat()
'2022-11-22T14:31:59.331225+01:00'
```

在这个例子中，您首先演示了`now`对象没有任何时区信息，因为它的`.tzinfo`属性返回了`None`。当您在没有任何参数的情况下调用`now`上的 [`.astimezone()`](https://docs.python.org/3/library/datetime.html#datetime.datetime.astimezone) 时，本地系统时区用于用 [`timezone`](https://docs.python.org/3/library/datetime.html#datetime.timezone) 对象填充`.tzinfo`。

一个`timezone`对象本质上只是一个相对于 UTC 时间的偏移量和一个名称。在示例中，当地时区的名称是*罗马标准时间*，偏移量是 3600 秒，即一个小时。

**注意:**时区的名称也将取决于您的操作系统。`datetime`模块经常与操作系统通信，以获取时间和时区信息，以及其他信息，比如您喜欢的语言。

在 [Python 3.9](https://realpython.com/python39-new-features/#proper-time-zone-support) 中增加了 [`zoneinfo`](https://docs.python.org/3/library/zoneinfo.html#module-zoneinfo) 模块，让你可以访问 [IANA 时区数据库](https://www.iana.org/time-zones)。

既然`datetime`对象有了一个`timezone`对象，您可以认为它是时区感知的。因此，当您在时区感知对象上调用`.isoformat()`时，您会注意到`+01:00`被添加到了末尾。这表示从 UTC 时间偏移一小时。

如果您在不同的地方，比如秘鲁的利马，那么您的`.isoformat()`输出可能如下所示:

>>>

```py
>>> now_aware.isoformat()
'2022-11-22T07:31:59.331225-06:00'
```

时间会有所不同，您会看到 UTC 偏移现在是`-06:00`。所以现在你的时间戳看起来很好，并且在它们代表什么时间方面是明确的。

你甚至可以更进一步，像很多人做的那样，用 UTC 时间存储你的时间戳，这样一切都很好地[规范化了](https://en.wikipedia.org/wiki/Normalization):

>>>

```py
>>> from datetime import datetime, timezone
>>> now = datetime.now()

>>> now.isoformat()
'2022-11-22T14:31:59.331225'

>>> now_utc = datetime.now(timezone.utc)
>>> now_utc.isoformat()
'2022-11-22T13:31:59.331225+00:00'
```

将`timezone.utc`时区传递给`.now()`构造函数方法将返回一个 UTC 时间。请注意，在本例中，时间与本地时间有偏差。

ISO 8601 标准也接受用`Z`代替`+00:00`来表示 UTC 时间。这有时被称为*祖鲁*时间，也就是航空业通常所说的时间。

在航空业，你总是使用 UTC 时间。在像航空这样的领域，无论在什么地方，在同一时间运营都是至关重要的。想象一下，空中交通管制必须处理每架飞机，根据它们的出发地报告预计着陆时间。那种情况会导致混乱和灾难！

## 结论

在本教程中，您已经告诉了时间！您已经生成了一个`datetime`对象，并看到了如何挑选对象的不同属性。您还研究了几种以不同格式输出`datetime`对象的方法。

您还了解了 UNIX 时间和 ISO 时间戳，并探索了如何明确地表示您的时间戳。为此，您已经尝试了复杂的时区世界，并让您的`datetime`对象知道时区。

如果你想知道事情需要多长时间，那么看看教程 [Python 计时器函数:三种监控代码的方法](https://realpython.com/python-timer/)。要更深入地研究`datetime`模块，请查看使用 Python datetime 处理日期和时间的。

现在你可以说[时间真的站在你这边](https://en.wikipedia.org/wiki/Time_Is_on_My_Side)！如何使用`datetime`模块？在下面的评论中分享你的想法和战争故事。

**源代码:** [点击这里下载 Python 中获取和使用当前时间的免费源代码](https://realpython.com/bonus/python-get-current-time-code/)。**
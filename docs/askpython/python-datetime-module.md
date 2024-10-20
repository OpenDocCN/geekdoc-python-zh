# Python 日期时间模块——终极指南

> 原文：<https://www.askpython.com/python-modules/python-datetime-module>

嘿，伙计们！希望你们都过得好。在本文中，我们将重点关注 **Python 日期时间模块**。

本文将特别关注 Python 中处理数据结构的方法或技术。那么，让我们开始吧。

* * *

## 什么是 Python 日期时间模块？

`Python Datetime module`使我们能够在 Python 中处理日期类型的数据值。datetime 模块由不同的类组成，用于处理日期和 Datetime 表达式。

众所周知，Python 并没有为日期和时间提供任何特定的数据类型，相反，我们可以导入并使用 datetime 模块来有效地处理日期和时间。

* * *

## 与 Python 日期时间模块相关联的类

Python Datetime 模块包含六(6)个处理日期和时间值的类。

在 Python 中，日期和日期时间表达式实际上被视为[对象](https://www.askpython.com/python/oops/python-classes-objects)。因此，下面提到的这些类可以用来处理这些日期对象:

*   **日期**
*   **时间**
*   **datetime**
*   **时距增量**
*   **tzinfo**
*   **时区**

现在，让我们详细了解上述每个类及其功能。

* * *

## 1.datetime.date 类

datetime 模块的[日期类](https://www.askpython.com/python/examples/current-date-and-time-in-python)以日期格式— YYYY:MM:DD 表示日期对象的信息。

**语法**:

```py
datetime.date(year,month,day)

```

让我们了解一下 Date 类提供的各种功能。

```py
from datetime import date

dte = date(2000,12,27)

print(dte)

```

在上面的例子中，我们使用了`date() function` 将日、月和年的值转换成标准的日期格式。

**输出:**

```py
2000-12-27

```

在本例中，我们使用了`date.today() function`从系统中获取当前日期。此外，我们还打印了当前的日、年、月值，如下所示:

```py
from datetime import date

dte = date.today()
print("Present date: ",dte)

print("Present year: ",dte.year)
print("Present month: ",dte.month)
print("Present day: ",dte.day)

```

**输出:**

```py
Present date:  2020-07-14
Present year:  2020
Present month:  7
Present day:  14

```

* * *

## 2.datetime.time 类

time 类提供与日期表达式或日期对象无关的时间值的具体信息。

因此，我们可以使用 datetime 模块中的 time 类来访问标准或正则化时间戳形式的时间值。

**语法:**

```py
datetime.time(hour,min,sec)

```

**举例:**

```py
from datetime import time

tym = time(12,14,45)
print("Timestamp: ",tym)

```

**输出:**

```py
Timestamp:  12:14:45

```

在本例中，我们使用`time() function`访问了小时、分钟和秒的值，如下所示

```py
from datetime import time

tym = time(12,14,45)
print("Hour value from the timestamp: ",tym.hour)
print("Minute value from the timestamp: ",tym.minute)
print("Second value from the timestamp: ",tym.second)

```

**输出:**

```py
Hour value from the timestamp:  12
Minute value from the timestamp:  14
Second value from the timestamp:  45

```

* * *

## 3\. datetime.datetime class

datetime 类为我们提供了关于日期和时间值的信息。因此，它使用类的函数表示整个日期和时间值。

**语法:**

```py
datetime.datetime(year,month,day,hour,min,sec,microsecond)

```

在下面的例子中，我们已经传递了必要的参数，并实现了 datetime 类的`datetime() function`。

```py
from datetime import datetime

dte_tym = datetime(2020,3,12,5,45,25,243245)
print(dte_tym)

```

**输出:**

```py
2020-03-12 05:45:25.243245

```

* * *

## 4.datetime.timedelta 类

Python DateTime 模块为我们提供了 [timedelta](https://www.askpython.com/python-modules/python-timedelta) 类来处理各种与日期相关的操作。

**语法:**

```py
datetime.timedelta(days,hours,weeks,minutes,seconds)

```

在对数据执行指定操作后， `timedelta() function`返回日期。

```py
from datetime import datetime, timedelta 

dte = datetime(2020,12,9)

updated_tym = dte + timedelta(days = 2,weeks=4) 
print("Updated datetime: ",updated_tym)

```

在上面的例子中，我们已经向预定义的日期值添加了天数和周数。

**输出:**

```py
Updated datetime:  2021-01-08 00:00:00

```

除了加法，我们还可以通过从日期时间表达式中减去特定的日期部分或时间部分来回溯日期值，如下所示:

```py
from datetime import datetime, timedelta 

dte = datetime(2020,12,9)

updated_tym = dte - timedelta(days = 2,weeks=4) 
print("Updated datetime: ",updated_tym)

```

**输出:**

```py
Updated datetime:  2020-11-09 00:00:00

```

* * *

## 结论

这个话题到此结束。如上所述，datetime 模块在以各种方式表示日期和时间方面起着重要的作用。

如果你有任何疑问，欢迎在下面评论。

在那之前，学习愉快！！

* * *

## 参考

*   Python 日期时间模块— JournalDev
*   [Python 日期时间模块—文档](https://docs.python.org/3/library/datetime.html)
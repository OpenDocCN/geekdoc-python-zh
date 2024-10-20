# 如何使用 Python TimeDelta？

> 原文：<https://www.askpython.com/python-modules/python-timedelta>

大家好！在今天的文章中，我们将看看如何使用 Python timedelta 对象。

如果你想用一个对象来表示一个时间实例，这个对象非常有用。

与整数/浮点数相反，这是一个实际的对象。优点是它给了我们更多的灵活性来集成我们的应用程序！

让我们通过一些简单的例子来看看如何使用它！

* * *

## Python timedelta 对象

这个类在`datetime`模块中可用，它是标准库的一部分。

```py
from datetime import timedelta

```

我们可以使用其类的构造函数来创建 timedelta 对象。

```py
from datetime import timedelta

timedelta_object = timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
print(timedelta_object)

# Same as above (Everything set to 0)
zero_obj = timedelta()
print(zero_obj)

```

**输出**

```py
0:00:00
0:00:00

```

请注意，我们可以自定义参数来创建我们自己的 datetime 对象。如果不设置参数值，默认情况下，该值为 0。

参数可以是整数或浮点数，也可以是正数或负数。

## 操作现有 Python timedelta 对象

除了创建我们自己的 timedelta 对象，我们还可以操作现有的 timedelta 对象。

我们可以使用[基本操作符](https://www.askpython.com/python/python-operators)来加、减或除两个 timedelta 对象！

```py
a = timedelta(hours=4, seconds=100)
print(a)
b = timedelta(hours=2, seconds=50)
print(b)
print("Output:")
print(a + b, a - b, a / b, sep='\n')

```

**输出**

```py
4:01:40
2:00:50
Output:
6:02:30
2:00:50
2.0

```

如你所见，我们可以使用简单的操作直接操纵这些对象！

## 获取当前日期时间对象

现在，对于大多数现实生活中的程序，我们希望使用当前时间并将其存储为 datetime 对象。

我们可以使用 [datetime.now()](https://www.askpython.com/python/examples/current-date-and-time-in-python) 函数将当前时间显示为 datetime 对象！

注意，您必须单独从模块名称中导入它！

```py
from datetime import datetime

```

现在让我们举一个例子。

```py
from datetime import datetime

current_datetime = datetime.now()
print(current_datetime)

```

**输出**

```py
datetime.datetime(2020, 6, 27, 22, 45, 54, 267673)

```

输出好像匹配！它显示当前时间是 2020 年 6 月 27 日，我的当地时区(IST)22:45:54。

现在，我们还可以使用基本运算符来获得过去或未来的时间！

```py
from datetime import datetime, timedelta

current_datetime = datetime.now()

# future dates
one_year_future_date = current_datetime + timedelta(days=365)

print('Current Date:', current_datetime)
print('One year from now Date:', one_year_future_date)

# past dates
three_days_before_date = current_datetime - timedelta(days=3)
print('Three days before Date:', three_days_before_date)

```

**输出**

```py
Current Date: 2020-06-27 22:45:54.267673
One year from now Date: 2021-06-27 22:45:54.267673
Three days before Date: 2020-06-24 22:45:54.267673

```

## 将 Python timedelta 用于日期和时间

我们还可以使用 Python timedelta 类和 date 对象，使用加法和减法。

```py
from datetime import datetime, timedelta

current_datetime = datetime.now()

dt = current_datetime.date()
print('Current Date:', dt)
dt_tomorrow = dt + timedelta(days=1)
print('Tomorrow Date:', dt_tomorrow)

```

**输出**

```py
Current Date: 2020-06-27
Tomorrow Date: 2020-06-28

```

这让我们在尝试存储/显示时间戳时更加灵活。如果您想要一个最小格式的时间戳，使用`datetime_object.date()`是一个不错的选择。

## Python 时间增量总计秒数

我们可以使用:
`timedelta_object.total_seconds()`显示任何 timedelta 对象的总秒数。

```py
from datetime import timedelta

print('Seconds in an year:', timedelta(days=365).total_seconds())

```

**输出**

```py
Output: Seconds in an year: 31536000.0

```

* * *

## 对特定时区使用 datetime 对象

在存储/显示日期时间对象时，您可能希望使用不同的时区。Python 为我们提供了一种简便的方法，使用`pytz`模块(Python 时区)。

您可以使用`pip`安装`pytz`，如果您还没有这样做的话。

```py
pip install pytz

```

现在，我们可以使用`pytz.timezone(TIMEZONE)`来选择我们的时区，并将其传递给 timedelta 对象。

这里有一个简单的例子，它打印不同时区的相同时间。

**便捷提示**:您可以使用`pytz.common_timezones`列出常见时区

```py
from datetime import datetime
from pytz import timezone, common_timezones

datetime_object = datetime.now(timezone('Asia/Kolkata'))
print("Current IST:", datetime_object)

```

**输出**

```py
Current IST: 2020-06-27 23:27:49.003020+05:30

```

```py
from datetime import datetime
from pytz import timezone, common_timezones
import random

for _ in range(4):
    zone = random.choice(common_timezones)
    print(f"Using TimeZone: {zone}")
    datetime_object = datetime.now(timezone(zone))
    print(datetime_object)

```

**输出**

```py
Using TimeZone: America/St_Lucia
2020-06-27 13:57:04.804959-04:00
Using TimeZone: Asia/Muscat
2020-06-27 21:57:04.814959+04:00
Using TimeZone: Asia/Urumqi
2020-06-27 23:57:04.825990+06:00
Using TimeZone: Asia/Seoul
2020-06-28 02:57:04.836994+09:00

```

事实上，我们能够获得不同时区的当前时间！

* * *

## 结论

在本文中，我们学习了如何使用 timedelta 对象将时间表示为 Python 对象！

## 参考

*   关于 Python timedelta 的 JournalDev 文章

* * *
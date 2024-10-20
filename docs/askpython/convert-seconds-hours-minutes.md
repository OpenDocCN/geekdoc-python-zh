# 将秒转换为小时和分钟的 Python 程序

> 原文：<https://www.askpython.com/python/examples/convert-seconds-hours-minutes>

有时候我们必须在 Python 中将秒转换成小时和分钟。当我们以时间戳的形式存储数据，并且必须以分钟和秒的格式正确显示数据时，这是最需要的。在本文中，我们将看看将秒转换成小时和分钟的 Python 程序。

* * *

## 不同时间单位的转换规则

每天由 24 小时组成。每小时有 60 分钟，每分钟有 60 秒。所以，一小时有 3600 秒，一天有 86400 秒。

在 Python 中有不同的方法将秒转换成分钟，将分钟转换成小时。

* * *

## 方法 1:定义一个 Python 函数将秒转换成小时和分钟

我们可以编写一个自定义的 [Python 函数](https://www.askpython.com/python/python-functions)来将秒值转换成小时和分钟。

最初，我们根据 24 小时格式转换输入的秒值。

秒=秒% (24*3600)

由于 1 小时相当于 3600 秒，1 分钟相当于 60 秒，我们按照下面的逻辑将秒转换为小时和分钟。

小时=秒//3600

min =秒// 60

**举例:**

```py
def time_conversion(sec):
   sec_value = sec % (24 * 3600)
   hour_value = sec_value // 3600
   sec_value %= 3600
   min = sec_value // 60
   sec_value %= 60
   print("Converted sec value in hour:",hour_value)
   print("Converted sec value in minutes:",min)

sec = 50000
time_conversion(sec)

```

**输出:**

```py
Converted sec value in hour: 13
Converted sec value in minutes: 53

```

* * *

## 方法 2: Python 时间模块将秒转换成分钟和小时

[Python 时间模块](https://www.askpython.com/python-modules/python-time-module#:~:text=The%20Python%20time%20module%20contains,%2C%20hour%2C%20seconds%2C%20etc.)包含时间。 [strftime](https://www.askpython.com/python-modules/python-strftime) ()函数通过将格式代码作为参数传递，以指定格式将时间戳显示为字符串。

time.gmtime()函数用于将传递给该函数的值转换为秒。此外，`time.strftime() function`使用指定的格式代码显示从`time.gmtime() function`传递到小时和分钟的值。

**举例:**

```py
import time
sec = 123455
ty_res = time.gmtime(sec)
res = time.strftime("%H:%M:%S",ty_res)
print(res)

```

**输出:**

```py
10:17:35

```

* * *

## 方法三:天真的方法

**举例:**

```py
sec = 50000
sec_value = sec % (24 * 3600)
hour_value = sec_value // 3600
sec_value %= 3600
min_value = sec_value // 60
sec_value %= 60
print("Converted sec value in hour:",hour_value)
print("Converted sec value in minutes:",min_value)

```

**输出:**

```py
Converted sec value in hour: 13
Converted sec value in minutes: 53

```

* * *

## 方法 4: Python 日期时间模块

[Python datetime 模块](https://www.askpython.com/python-modules/python-datetime-module)有各种内置函数来操作日期和时间。`datetime.timedelta() function`以适当的时间格式处理和表示数据。

**举例**:

```py
import datetime
sec = 123455
res = datetime.timedelta(seconds =sec)
print(res)

```

**输出:**

```py
1 day, 10:17:35

```

* * *

## 摘要

Python 提供了许多将秒转换成分钟和小时的模块。我们可以创建自己的函数或者使用时间和日期时间模块。

## 下一步是什么？

*   [Python 时间模块](https://www.askpython.com/python-modules/python-time-module)
*   [Python 将数字转换成文字](https://www.askpython.com/python/python-convert-number-to-words)
*   [Python 中的模块](https://www.askpython.com/python-modules/python-modules)

* * *

## 参考

*   [Python 时间模块—文档](https://docs.python.org/3/library/time.html)
*   [Python 日期时间模块—文档](https://docs.python.org/3/library/datetime.html)
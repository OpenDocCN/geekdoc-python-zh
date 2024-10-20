# 如何使用 Python 中的 datetime 模块？

> 原文：<https://www.askpython.com/python-modules/datetime-module-examples>

在 Python 中，有一个叫做`datetime`的模块，它允许我们处理日期和时间。它包含年、月和日格式的日期以及小时、分钟和秒格式的时间。本教程将向您展示如何使用这个模块。

* * *

在`datetime`模块中，日期以下列格式表示:

```py
yyyy-mm-dd 
```

时间用以下格式表示:

```py
hh:mm:ss
```

为了比较日期和时间，我们使用常见的比较运算符，如`>=, <=, >, <, ==, !=`。
考虑两个日期:d1 和 d2。

| **操作** | **意为** |
| d1>d2 | 在日历中，d1 在 d2 之后 |
| d1 | 在日历中，d1 在 d2 之前 |
| d1==d2 | d1 与 d2 相同 |

Date Comparison

同样，考虑两个时间:t1 和 t2。

| **操作** | **意为** |
| t1>t2 | 在时钟中，t1 在 t2 之后 |
| t1 | 在时钟中，t1 在 t2 之前 |
| t1==t2 | t1 与 t2 相同 |

Time Comparison

上述所有操作都返回一个布尔值，即“真”或“假”，这取决于是否满足给定的条件。
在进行日期和时间比较之前，我们先来看看一些基本的`datetime`方法。

* * *

## Python 日期时间方法的示例

让我们直接进入使用日期时间模块的例子。

### 1.获取今天的日期

```py
import datetime

print(datetime.date.today())

```

**输出:**

```py
2022-09-28

```

日期以 yyyy-mm-dd 格式表示。

* * *

### 2.获取当前时间

```py
from datetime import datetime 

print(datetime.now().time())

```

**输出:**

```py
12:40:36.221835

```

时间以 hh:mm:ss 格式表示。

* * *

## 日期和时间比较

### 1.检查一个日期是否大于另一个日期

```py
from datetime import datetime

#date in the format yyyy-mm-dd
date1 = datetime(2022, 5, 15)
date2 = datetime(2012, 4, 15)

print("Is date1 greater than to date2?: ", date1>date2)

```

**输出:**

```py
Is date1 greater than to date2?:  True

```

这里，日期 1 是 2022 年 5 月 15 日，日期 2 是 2022 年 4 月 15 日。由于日历中日期 1 在日期 2 之后，所以输出为真。

* * *

### 2.检查一个日期是否小于另一个日期

```py
from datetime import datetime

#date in the format yyyy-mm-dd
date1 = datetime(2022, 5, 15)
date2 = datetime(2022, 11, 16)

print("Is date1 less than to date2?: ", date1<date2)

```

**输出:**

```py
Is date1 less than to date2?:  True

```

在本例中，日期 1 还是 2022 年 5 月 15 日，日期 2 是 2022 年 11 月 16 日。由于在同一日历年中“五月”在“十一月”之前，所以输出为真。

* * *

### 3.检查两个日期是否相等

```py
from datetime import datetime

#date in the format yyyy-mm-dd
date1 = datetime(2022, 5, 15)
date2 = datetime(2022, 4, 15)

print("Is date1 equal to date2?: ", date1==date2)

```

**输出:**

```py
Is date1 equal to date2?:  False

```

这里，2022 年 5 月 15 日和 2022 年 4 月 15 日是不一样的。因此，输出为假。

* * *

在我们刚刚看到的例子中，只给出了日期，而没有给出时间。让我们学习如何只比较日期，或者只比较时间，如果日期和时间都给定的话。

### 4.仅比较日期

```py
from datetime import datetime

#datetime in the format yyyy-mm-dd hh:mm:ss

#datetime1 -> date: 6 August, 2022 | time: 11:00:00 a.m.
datetime1 = datetime(2022, 8, 6, 11, 0, 0)

#datetime2 -> 21 March, 2022, | time: 2:45:31 p.m.
datetime2 = datetime(2022, 3, 21, 14, 45, 31)

#getting only the dates from datetime 
date1 = datetime1.date()
date2 = datetime2.date()

print("Is date1 greater than date2?: ", date1>date2)
print("Is date1 less than date2?: ", date1<date2)
print("Is date1 equal to date2?: ", date1==date2)

```

**输出:**

```py
Is date1 greater than date2?:  True
Is date1 less than date2?:  False
Is date1 equal to date2?:  False

```

这里，我们使用了`date()`方法从给定的日期时间中只提取日期，然后使用不同的比较操作符进行比较，得到输出。

* * *

### 5.只比较时间

```py
from datetime import datetime

#datetime in the format yyyy-mm-dd hh:mm:ss

#datetime1 -> date: 6 August, 2022 | time: 11:00:00 a.m.
datetime1 = datetime(2022, 8, 6, 11, 0, 0)

#datetime2 -> 21 March, 2022, | time: 2:45:31 p.m.
datetime2 = datetime(2022, 3, 21, 14, 45, 31)

#getting only the time from datetime 
time1 = datetime1.time()
time2 = datetime2.time()

print("Is time1 greater than time2?: ", time1>time2)
print("Is time1 less than time2?: ", time1<time2)
print("Is time1 equal to time2?: ", time1==time2)

```

**输出:**

```py
Is time1 greater than time2?:  False
Is time1 less than time2?:  True
Is time1 equal to time2?:  False

```

与示例 5 类似，这里我们使用 datetime 模块中的`time()`方法，只从给定的 datetime 中提取时间。

* * *

## 摘要

仅此而已！我们还学习了如何使用 Python 中的`datetime`模块来处理日期和时间。
如果你想了解更多不同的 Python 概念，查看我们的其他文章[这里](https://www.askpython.com/)！

* * *

## 参考

*   [正式文件](https://docs.python.org/3/library/datetime.html)
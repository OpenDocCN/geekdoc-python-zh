# Python 日期类–要知道的 7 个函数！

> 原文：<https://www.askpython.com/python-modules/python-date-class>

嘿！在本文中，我们将详细关注 Python 日期类。那么，让我们开始吧。

* * *

## 什么是 Python 日期时间模块？

[Python datetime 模块](https://www.askpython.com/python-modules/python-datetime-module)为我们提供了各种类和函数来处理和操作日期和时间值方面的数据值。

我们需要导入 datetime 模块，以便访问其中包含的不同类和方法。

在这里，您可以使用下面的代码片段来导入 date 类，并通过用户定义的参数创建一个 date 对象

```py
from datetime import date

dte = date(2020,12,27)

print("Date: ", dte)

```

**输出**

```py
Date:  2020-12-27

```

现在，让我们在下一节中理解 Python datetime 模块的 date 类。

* * *

## 日期类的实现方法

Python date 类包含各种处理数据值的日期格式的方法。

通过使用 date 类的函数，我们可以很容易地将数据值处理成标准的日期格式 **YYYY-MM-DD** 。

现在让我们详细看看由 date 类函数构成的一些重要函数。

* * *

### 1.日期.今天()函数

`date.today() function`从系统中获取当前日期并表示。

**语法:**

```py
date.today()

```

**举例:**

```py
from datetime import date

dte = date.today()

print("Date: ", dte)

```

**输出:**

```py
Date:  2020-07-25

```

* * *

### 2.date.year()函数

我们可以使用 year 函数从日期表达式中访问和获取年份。`year function`从提供的日期表达式中提取并表示年值。

**语法:**

```py
date.today().year

```

**举例:**

```py
from datetime import datetime,date

year = date.today().year
print(year)

```

从 date.today()函数中，我们得到当前日期，即 2020-07-26。除此之外，我们还使用 year()函数提取了 year 值。

**输出:**

```py
2020

```

* * *

### 3.日期.月份()函数

为了提取和表示月份值，可以使用`month function` 。

**语法:**

```py
date.today().month

```

**举例:**

```py
from datetime import datetime,date

mnth = date.today().month
print(mnth)

```

**输出:**

```py
7

```

* * *

### 4.date.day 函数

使用如下所示的 `day function`,可以很容易地从日期表达式中提取日值

**语法:**

```py
date.day

```

**举例:**

```py
from datetime import datetime,date

day = date.today().day
print(day)

```

**输出:** 25

* * *

### 5.日期替换()函数

有时，当我们想要改变日期表达式的日期部分时，可能会出现这种情况。这个任务可以使用 replace()函数来完成。

`date.replace() function`可用于替换以下日期部分——

*   年
*   月
*   天

**语法:**

```py
date.replace(year,day,month)

```

**举例:**

```py
from datetime import datetime,date

dt = date.today()
print("Current date: ",dt)

res = dt.replace(year=2021,day=20)
print("Modified date: ",res)

```

在上面的例子中，我们已经替换了当前日期(2020-07-26)的年和日值。

**输出**:

```py
Current date:  2020-07-26
Modified date:  2021-07-20

```

* * *

### 6.日期.工作日()函数

使用`weekday function`，我们可以从日期表达式中获取日期值的星期几。

为工作日提供的索引如下:

*   星期一-0
*   星期二-1
*   星期三-2
*   周四至 3 日
*   星期五-4
*   星期六-5
*   周日至 6 日

**举例:**

```py
from datetime import datetime,date

date = date.today().weekday()
print("Weekday: ",date)

```

在上面的例子中，我们已经计算了当前日期的星期几:2020-07-26。

**输出:**

```py
6

```

* * *

### 7\. date.strftime() function

`date.strftime() function`使我们能够提取日期表达式的日期部分，并将值表示为字符串。

要了解 strftime()函数的变体，请访问 [Python strftime()函数](https://www.askpython.com/python-modules/python-strftime)。

**语法:**

```py
date.strftime("%Y-%m-%d")

```

**举例:**

```py
from datetime import datetime,date

date = date.today() 

year = date.strftime("%Y")
print("Current Year:", year)

str_date = date.strftime("%Y-%m-%d")
print("Date value:",str_date)	

```

**输出:**

```py
Current Year: 2020
Date value: 2020-07-26

```

* * *

## 结论

到此，我们就结束了这个话题。如果你有任何疑问，欢迎在下面评论。

在那之前，学习愉快！！

* * *

## 参考

*   [Python 日期类—文档](https://docs.python.org/3/library/datetime.html#date-objects)
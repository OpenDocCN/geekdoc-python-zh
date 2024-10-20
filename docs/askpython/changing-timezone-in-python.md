# 在 Python 中更改时区

> 原文：<https://www.askpython.com/python-modules/changing-timezone-in-python>

有时，产品或基础设施工程师需要在遍布世界各地的基础设施上工作。他们必须与位于美国、亚洲、欧洲和英国等地的机器协作。因此，时区对 Python 来说更加重要。

随着当今编程语言的不断进步，几乎所有的编程语言中都维护着几个模块。Python 有一个名为`pytz`的时区包，支持跨平台的实时时区计算。

* * *

## 安装 pytz 模块

首先，我们将导入`pytz`模块的时区库。该`pip`命令可用于安装该模块。

```py
pip install pytz

```

* * *

## 导入必要的模块

此外，我们需要从 DateTime 模块导入 DateTime。为了保持一致性，我们可以指定日期和时间输出的格式。

```py
from pytz import timezone
from datetime import datetime

```

* * *

## 获取当前日期和时间

在本程序中，我们将指定格式为 YY-毫米-日时:分:秒。

执行代码时，我们将调用 DateTime 库的 now()方法来获取指定格式的当前时间。另一方面，输出时区格式将采用其 DateTime 对象格式。

因此，为了使它更具可读性，我们通过调用 strftime()方法将其转换为字符串时间格式。

```py
time_format = '%Y-%m%d %H:%M:%S %Z%z'
default_now = datetime.now()
formatted_now = datetime.now().strftime(time_format)

print("Date Time in defaut format: ", default_now, '\n')
print("Date Time in string format: ", formatted_now)

```

```py
Date Time in defaut format:  2021-11-22 09:26:40.054185 

Date Time in string format:  2021-1122 09:26:40 

```

* * *

## 将当前日期和时间转换为多个时区

现在我们将创建一个时区列表，并遍历它，将当前时间转换为该时区。我们将包括美国、欧洲、亚洲和标准 UTC 的时区。

```py
timezones = ['US/Central', 'Europe/London', 
             'Asia/Kolkata', 'Australia/Melbourne', 'UTC']
for tz in timezones:
  dateTime = datetime.now(timezone(tz)).strftime(time_format)
  print(f"Date Time in {tz} is {dateTime}")

```

```py
Date Time in US/Central is 2021-1122 03:27:58 CST-0600
Date Time in Europe/London is 2021-1122 09:27:58 GMT+0000
Date Time in Asia/Kolkata is 2021-1122 14:57:58 IST+0530
Date Time in Australia/Melbourne is 2021-1122 20:27:58 AEDT+1100
Date Time in UTC is 2021-1122 09:27:58 UTC+0000

```

接下来，我们将遍历作为 DateTime 库的 now()方法的参数创建的列表中的所有时区，以获取所有时区和每个时区中的当前时间。我们还将把它转换成字符串格式，以便于阅读。

* * *

## 结论

恭喜你！您刚刚学习了如何在 Python 中更改时区。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [Python 日期时间模块——终极指南](https://www.askpython.com/python-modules/python-datetime-module)
2.  [如何使用 Python TimeDelta？](https://www.askpython.com/python-modules/python-timedelta)
3.  [如何在 Python 中等待一个特定的时间？](https://www.askpython.com/python/examples/python-wait-for-a-specific-time)
4.  [使用 Python strptime()](https://www.askpython.com/python/python-strptime) 将字符串转换为日期时间

感谢您抽出时间！希望你学到了新的东西！！😄

* * *
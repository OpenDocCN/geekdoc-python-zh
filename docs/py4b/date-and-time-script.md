# 日期和时间脚本

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/date-and-time-script>

## 概观

这个脚本可以用来解析日期和时间。打开一个空白文件，将其命名为 dateParser.py。

将下面的代码复制并粘贴到文件中(并确保您理解它的作用)。

## dateParser.py

```py
from datetime import datetime

now = datetime.now()

mm = str(now.month)

dd = str(now.day)

yyyy = str(now.year)

hour = str(now.hour)

mi = str(now.minute)

ss = str(now.second)

print mm + "/" + dd + "/" + yyyy + " " + hour + ":" + mi + ":" + ss

```

现在保存并退出该文件，并通过以下方式运行它:

$ python dateParser.py

## 时间.睡眠

在 Python 中，可以使用 time.sleep()在给定的秒数内暂停执行。括号中给出了秒数。

```py
# How to sleep for 5 seconds in python:

import time

time.sleep(5)

# How to sleep for 0.5 seconds in python:

import time

time.sleep(0.5)

```

## 如何获取当前日期和时间

我在这个优秀的网站上找到了这个日期和时间脚本:[http://www . salty crane . com/blog/2008/06/how-to-get-current-date-and-time-in/](http://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/ "date_and_time")

```py
import datetime

now = datetime.datetime.now()

print
print "Current date and time using str method of datetime object:"
print str(now)

print
print "Current date and time using instance attributes:"
print "Current year: %d" % now.year
print "Current month: %d" % now.month
print "Current day: %d" % now.day
print "Current hour: %d" % now.hour
print "Current minute: %d" % now.minute
print "Current second: %d" % now.second
print "Current microsecond: %d" % now.microsecond

print
print "Current date and time using strftime:"
print now.strftime("%Y-%m-%d %H:%M")

```

```py
The result:

Current date and time using str method of datetime object:
2013-02-17 16:02:49.338517

Current date and time using instance attributes:
Current year: 2013
Current month: 2
Current day: 17
Current hour: 16
Current minute: 2
Current second: 49
Current microsecond: 338517

Current date and time using strftime:
2013-02-17 16:02

```
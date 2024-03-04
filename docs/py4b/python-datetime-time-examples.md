# 如何在 Python 中使用日期和时间

> 原文：<https://www.pythonforbeginners.com/basics/python-datetime-time-examples>

## 日期和时间

这篇文章将展示一些使用 Pythons 日期时间和时间模块的例子。

在之前的一篇文章[Python 中的基本日期和时间类型](https://www.pythonforbeginners.com/basics/python-strftime-and-strptime)中，我写道，datetime 和 time 对象都支持 strftime(format)方法来创建一个在显式格式字符串控制下表示时间的字符串。

## 日期和时间示例

让我们看看可以用 Python 中的 datetime 和 time 模块做些什么

```py
import time
import datetime

print "Time in seconds since the epoch: %s" %time.time()
print "Current date and time: " , datetime.datetime.now()
print "Or like this: " ,datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

print "Current year: ", datetime.date.today().strftime("%Y")
print "Month of year: ", datetime.date.today().strftime("%B")
print "Week number of the year: ", datetime.date.today().strftime("%W")
print "Weekday of the week: ", datetime.date.today().strftime("%w")
print "Day of year: ", datetime.date.today().strftime("%j")
print "Day of the month : ", datetime.date.today().strftime("%d")
print "Day of week: ", datetime.date.today().strftime("%A") 
```

## 输出

它会打印出这样的内容:

```py
 Time in seconds since the epoch: 	1349271346.46
Current date and time:  		2012-10-03 15:35:46.461491
Or like this: 				12-10-03-15-35
Current year:  				2012
Month of year:  			October
Week number of the year:  		40
Weekday of the week:  			3
Day of year:  				277
Day of the month :  			03
Day of week:  				Wednesday 
```

获取某个日期的星期几(你的宠物的生日)。

```py
 import datetime

mydate = datetime.date(1943,3, 13)  #year, month, day
print(mydate.strftime("%A")) 
```
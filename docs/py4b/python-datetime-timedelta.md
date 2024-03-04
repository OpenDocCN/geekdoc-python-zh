# Python 中的日期和时间

> 原文：<https://www.pythonforbeginners.com/basics/python-datetime-timedelta>

## Python 中的日期和时间

在我上一篇关于 Python 中的 datetime & time 模块的文章中，我主要使用 strftime(format)方法来打印日期和时间。

在这篇文章中，我将向您展示如何不用它，只使用 datetime.datetime.now()就能做到。

文章的第二部分是关于 timedelta 类，在这个类中，我们可以看到两个日期、时间或日期时间实例之间微秒级分辨率的差异。

## 日期和时间脚本

最后一个例子是一个简短的脚本，用于计算给定日期(本例中是生日)还剩多少天。

```py
 import datetime
now = datetime.datetime.now()
print "-" * 25
print now
print now.year
print now.month
print now.day
print now.hour
print now.minute
print now.second

print "-" * 25
print "1 week ago was it: ", now - datetime.timedelta(weeks=1)
print "100 days ago was: ", now - datetime.timedelta(days=100)
print "1 week from now is it: ",  now + datetime.timedelta(weeks=1)
print "In 1000 days from now is it: ", now + datetime.timedelta(days=1000)

print "-" * 25
birthday = datetime.datetime(2012,11,04)

print "Birthday in ... ", birthday - now
print "-" * 25
```

您应该会看到类似于以下内容的输出:

```py
-------------------------
2012-10-03 16:04:56.703758
2012
10
3
16
4
56
-------------------------
The date and time one week ago from now was:  2012-09-26 16:04:56.703758
100 days ago was:  2012-06-25 16:04:56.703758
One week from now is it:  2012-10-10 16:04:56.703758
In 1000 days from now is it:  2015-06-30 16:04:56.703758
-------------------------
Birthday in ...  31 days, 7:55:03.296242
-------------------------
```

我在 saltycrane.com 发现了这篇我喜欢并想分享的博文:[http://www . salty crane . com/blog/2008/06/how-to-get-current-date-time-in/](http://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/ "How-to-get-current-date-and-time")
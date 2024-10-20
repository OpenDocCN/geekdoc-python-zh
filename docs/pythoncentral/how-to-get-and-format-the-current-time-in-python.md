# 如何在 Python 中获取和格式化当前时间

> 原文：<https://www.pythoncentral.io/how-to-get-and-format-the-current-time-in-python/>

## 时间回顾

我们知道时间由几个部分表示，其中一些是数字，如秒、分和小时，而其他的是字符串，如时区、上午和下午。Python 在处理时间对象和字符串时提供了大量的实用工具。因此，在 Python 中获取和格式化当前时间相对容易。

### **用 Python 获取当前时间**

有两个函数可用于检索系统的当前时间。`time.gmtime([secs])`将自纪元以来以秒表示的时间(参见[本文](https://www.pythoncentral.io/measure-time-in-python-time-time-vs-time-clock/ "Measure Time in Python – time.time() vs time.clock()")以了解纪元概述)转换为 UTC 中的`struct_time`。如果没有提供参数`secs`或`None`，则使用`time.time`返回的当前时间。

类似地，`time.localtime([secs])`将从纪元开始以秒表示的时间转换为本地时区的`struct_time`。如果没有提供参数`secs`或`None`，则使用`time.time`返回的当前时间。

```py

>>> import time

>>> time.gmtime()

time.struct_time(tm_year=2013, tm_mon=2, tm_mday=21, tm_hour=5, tm_min=27,

tm_sec=32, tm_wday=3, tm_yday=52, tm_isdst=0)

>>> time.localtime()

time.struct_time(tm_year=2013, tm_mon=2, tm_mday=20, tm_hour=23, tm_min=27,

tm_sec=36, tm_wday=2, tm_yday=51, tm_isdst=0)

```

那么，什么是`time.struct_time`对象呢？一个`time.struct_time`对象封装了一个特定时间点的所有信息。它充当一个包装器，将某个时间点的所有相关信息包装在一起，这样就可以将它们作为一个单元进行访问。

### **用 Python 格式化当前时间**

一旦我们有了一个`time.struct_time`对象，我们可以把它格式化成一个字符串，这样它就可以被另一个程序进一步处理。我们可以通过使用当前时间对象作为参数`t`来调用`time.strftime(format[, t])`来实现。如果没有提供 **t** 参数或`None`，则使用`time.localtime`返回的`time_struct`对象。

这里有一个例子:

```py

>>> current_time = time.localtime()

>>> time.strftime('%Y-%m-%d %A', current_time)

'2013-02-20 Wednesday'

>>> time.strftime('%Y Week %U Day %w', current_time)

'2013 Week 07 Day 3'

>>> time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time)

'Wed, 20 Feb 2013 23:52:14 GMT'

```

`format`参数接受一系列指令，这些指令在 [Python 关于时间](http://docs.python.org/3/library/time.html#time.time)的文档中有详细说明。请注意，前面代码中的最后一个例子演示了一种将`time_struct`对象转换成可以在 HTTP 头中使用的字符串表示的方法。

### **从字符串中获取 Python 中的时间对象**

到目前为止，我们已经介绍了如何获取当前的`time_struct`对象并将其转换成字符串对象。反过来怎么样？幸运的是，Python 提供了一个函数`time.strptime`，将一个`string`对象转换成一个`time_struct`对象。

```py

>>> time_str = 'Wed, 20 Feb 2013 23:52:14 GMT'

>>> time.strptime(time_str, '%a, %d %b %Y %H:%M:%S GMT')

time.struct_time(tm_year=2013, tm_mon=2, tm_mday=20, tm_hour=23,

tm_min=52, tm_sec=14, tm_wday=2, tm_yday=51, tm_isdst=-1)

```

当然，在`strings`和`time_struct`对象之间来回转换很容易。

```py

>>> time.strftime('%Y-%m-%dT%H:%M:%S', current_time)

# ISO 8601

'2013-02-20T23:52:14'

>>> iso8601 = '2013-02-20T23:52:14'

>>> time.strptime(iso8601, '%Y-%m-%dT%H:%M:%S')

time.struct_time(tm_year=2013, tm_mon=2, tm_mday=20, tm_hour=23,

tm_min=52, tm_sec=14, tm_wday=2, tm_yday=51, tm_isdst=-1)

```

### **使用 Python 的时间模块的技巧和建议**

*   虽然 Python 的`time_struct`对象非常有用，但是它们在 Python 的解释器之外是不可用的。很多时候，您希望将 Python 中的时间对象传递给另一种语言。例如，基于 Python 的服务器端 JSON 响应包括以 ISO 8601 格式表示的`time_struct`对象，可以由客户端 Javascript 应用程序处理，以一种很好的方式向最终用户呈现时间对象。
*   在`time_struct`和`string`对象之间转换的两个函数很容易记住。`time.strftime`可以记为“从时间格式化的字符串”，而`time.strptime`可以记为“从字符串表示的时间”。
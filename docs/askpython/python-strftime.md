# Python strftime()函数是什么？

> 原文：<https://www.askpython.com/python-modules/python-strftime>

嘿，伙计们！在本文中，我们将了解 **Python strftime()函数**及其变体的工作原理。

那么，让我们开始吧。

Python 有各种各样的模块，这些模块有函数集群来实现数据的各种功能。Python 时间模块用于执行关于不同时间戳的操作。

此外，`Python strftime() function`接受各种形式的时间，并返回一个以标准形式表示时间的字符串。

* * *

## 变体 1: Python strftime()获取当前时间

Python strftime()函数可以与 datetime 模块一起使用，根据格式代码以适当的形式获取当前时间戳。

**语法:**

```py
datetime.now().strftime('format codes')

```

因此，格式代码实际上是预定义的代码，用于以适当和标准的方式表示时间戳。在本文中，我们将进一步了解格式代码。

**举例:**

```py
from datetime import datetime

current_timestamp = datetime.now() 
tym = current_timestamp.strftime("%H:%M:%S")
date = current_timestamp.strftime("%d-%m-%Y")
print("Current Time:",tym)
print("Current Date:",date)

```

在上面的例子中，我们使用了`datetime.now() method`来获取当前的时间戳，然后将其传递给 strftime()函数，以标准格式表示时间戳。

我们使用的格式代码具有如下含义:

*   `%H`–以 24 小时制表示“**小时**”。
*   `%M`–用十进制数字表示“**分钟**”。
*   `%S`–代表时间戳的“**秒**部分。

**输出:**

```py
Current Time: 16:28:40
Current Date: 28-04-2020

```

* * *

## 变体 2:带有预定义时间戳的 Python strftime()

有时会发生这种情况，我们需要显示历史数据集的时间戳。使用 Python strftime()函数也可以执行同样的操作。

`datetime.fromtimestamp()`方法用于获取预定义的时间戳。此外，strftime()函数可用于使用各种格式代码以标准形式表示它，如上所述。

**语法:**

```py
datetime.fromtimestamp(timestamp).strftime()

```

**举例:**

```py
from datetime import datetime

given_timestamp = 124579923 
timestamp = datetime.fromtimestamp(given_timestamp)
tym = timestamp.strftime("%H:%M:%S")
date = timestamp.strftime("%d-%m-%Y")
print("Time according to the given timestamp:",tym)
print("Date according to the given timestamp:",date)

```

**输出:**

```py
Time according to the given timestamp: 03:02:03
Date according to the given timestamp: 13-12-1973

```

* * *

## 对 Python strftime()函数使用不同的格式代码

Python strftime()函数使用格式代码以标准且可维护的格式表示时间戳。此外，我们可以使用格式代码将日、小时、周等从时间戳中分离出来并显示出来。

让我们借助一些例子来理解格式代码。

**例 1:** 格式代码—“%**A**显示当地时间的**当前日期**。

```py
from time import strftime

day = strftime("%A") 
print("Current day:", day) 

```

**输出:**

```py
Current day: Tuesday

```

**例 2:** 格式代码— '%c '显示当前本地时间。

格式代码–“% c”用于显示当前本地时间，遵循以下格式:

**日月日时:分:秒年**

```py
from time import strftime

day = strftime("%c") 
print("Current timestamp:", day) 

```

**输出:**

```py
Current timestamp: Tue Apr 28 16:42:22 2020

```

**例 3:** 格式码—“**% R**”以 **24 小时制**表示时间。

```py
from time import strftime

day = strftime("%R") 
print("Current time in a 24-hour format:", day) 

```

**输出:**

```py
Current time in a 24-hour format: 16:44

```

**示例 4** :格式代码— '%r '以 H:M:S 格式显示时间以及描述，即 AM 或 PM。

```py
from time import strftime

day = strftime("%r") 
print("Current time -- hours:mins:seconds", day) 

```

**输出:**

```py
Current time -- hours:mins:seconds 05:05:19 PM

```

**例 5:**

```py
from time import strftime

day = strftime("%x -- %X %p") 
print("Local date and time:", day) 

```

在上面的例子中，我们使用格式代码“%x”以日期表示本地时间戳，使用“%X”以 H:M:S 的形式表示本地时间。使用“%p”格式代码来表示时间戳是属于 AM 还是 PM。

**输出:**

```py
Local date and time: 04/28/20 -- 17:08:42 PM

```

* * *

## 结论

因此，在本文中，我们已经理解了 Python strftime()函数的工作原理以及所使用的格式代码。

为了了解可用格式代码的列表，请找到参考资料中的链接以访问官方文档。

* * *

## 参考

*   [Python strftime()格式代码](https://strftime.org/)
*   Python strftime()函数— JournalDev
# Python 时间模块

> 原文：<https://www.askpython.com/python-modules/python-time-module>

**Python 时间模块**为我们提供了各种功能，通过 Python 脚本将系统时间灌输到我们的应用程序中。

为了开始使用时间模块，我们需要使用下面的语句将其导入到 python 脚本中:

```py
import time

```

* * *

## 理解“时代”的概念

在对时间戳相关数据执行操作时，必须有一个起点，我们可以从该起点开始对其执行操作。

历元基本上是时间的起点，从该点开始测量时间的流逝。

* * *

## Python 时间模块函数

Python 时间模块提供了一系列处理时间戳的函数。

以下是时间模块最常用的功能:

*   **time.time()**
*   **time.sleep()**
*   **time.ctime()**
*   time . local time()
*   **time.mktime()**
*   **time.gmtime()**
*   **time.strptime()**
*   **time.strftime()**
*   **time.asctime()**

* * *

### 1.time.time()方法

Python 时间模块得到了 **`time.time()`** 方法，给出了**当前当地时间**的秒数。

**语法:**

```py
time.time()

```

**举例:**

```py
import time
seconds = time.time()
print("Current time in seconds since epoch =", seconds)	

```

**输出:**

```py
Current time in seconds since epoch = 1582961644.3032079

```

* * *

### 2.time.sleep()方法

**time.sleep()** 方法提供了当前进程或线程执行之间的**延时或延迟**。

**语法:**

```py
time.sleep(value)

```

**举例:**

```py
import time

print("JournalDev!!!!.")
time.sleep(1.2)
print("AskPython.")
time.sleep(3.2)
print("Engineering")

```

在上面的代码片段中，当我们尝试执行上面的代码时，可以很容易地观察到输出语句显示在控制台上时的延迟。

**输出:**

```py
JournalDev!!!!.
AskPython.
Engineering

```

* * *

### 3.time.localtime()方法

Python 时间模块包含 **struct_time 类**，可以使用时间模块的各种函数访问该类。**它帮助我们访问本地时间戳的各个字段，如年、小时、秒等**。

struct_time 类由以下属性组成:

*   **tm_year** :返回特定本地时间的年份。
*   **tm_hour** :返回特定本地时间的小时。
*   **tm_min** :返回特定本地时间的分钟值。
*   **tm_sec** :返回特定本地时间的秒值。
*   **tm_mon** :返回当地时间的月份。
*   **tm_mday** :返回当地时间的月份日期。
*   **tm_wday** :返回工作日的值，即 0-周一到 6-周日。
*   **tm_yday** :返回 1-366 之间的某一天的数字

**`time.localtime()`** 函数在后端运行 time.time()函数，以**当地时间**的 struct_time 类的格式返回当前时间的详细信息。

我们还可以将自**纪元**以来的秒数作为参数传递给函数。

**语法:**

```py
time.localtime(seconds)

```

**举例:**

```py
import time

local_time = time.localtime()
print("Time:",local_time)
print("Current year:", local_time.tm_year)
print("Current hour:", local_time.tm_hour)
print("Current minute:", local_time.tm_min)
print("Current second:", local_time.tm_sec)

```

**输出:**

```py
Time: time.struct_time(tm_year=2020, tm_mon=2, tm_mday=29, tm_hour=14, tm_min=3, tm_sec=23, tm_wday=5, tm_yday=60, tm_isdst=0)
Current year: 2020
Current hour: 14
Current minute: 3
Current second: 23

```

* * *

### 4.time.ctime()方法

**`time.ctime()`** 方法将 epoch 后的秒值或 time()函数的结果作为参数，并返回一个表示当前本地时间的字符串值。

**语法:**

```py
ctime(seconds)

```

**举例:**

```py
from time import time, ctime

current_time = time()
res = ctime(tim)
print("Local_time:",res)

```

**输出:**

```py
Local_time: Sat Feb 29 14:08:26 2020

```

* * *

### 5.time.mktime()方法

**`time.mktime()`** 方法是 time.localtime()方法的逆方法。

它将 struct _ time(struct _ time 类的所有元组)作为一个参数，并以秒为单位返回从 epoch 开始已经过去的时间。

**语法:**

```py
time.mktime()

```

**举例:**

```py
import time

local_time = time.localtime()
sec = time.mktime(local_time)
print(sec)

```

在上面的例子中，我们使用 locatime()方法获取 struct_time 类的元组，并将其传递给 mktime()方法。

**输出:**

```py
1582966721.0

```

* * *

### 6.time.gmtime()方法

**`time.gmtime()`** 函数在后端运行 time.time()函数，以 **UTC** 中 struct_time 类的格式返回当前时间的详细信息。

**语法:**

```py
time.gmtime()

```

**举例:**

```py
import time

local_time = time.gmtime()
print(local_time)

```

**输出:**

```py
time.struct_time(tm_year=2020, tm_mon=2, tm_mday=29, tm_hour=9, tm_min=2, tm_sec=49, tm_wday=5, tm_yday=60, tm_isdst=0)

```

* * *

### 7.time.strptime()方法

**`time.strptime()`** 方法接受一个表示时间的字符串，并以 struct_time 格式返回时间细节。

**语法:**

```py
time.strptime(string, %format code)

```

**格式代码:**

*   % m–月
*   %d 天
*   % M–月
*   % S–秒
*   %H 小时
*   % Y–年

**举例:**

```py
import time

tym = "29 February, 2020"
sys = time.strptime(tym, "%d %B, %Y")
print(sys)

```

**输出:**

```py
time.struct_time(tm_year=2020, tm_mon=2, tm_mday=29, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=5, tm_yday=60, tm_isdst=-1)

```

* * *

### 8.time.strftime()方法

**`time.strftime()`** 方法是 time.strptime()方法的逆方法。它将 struct_time 类的元组作为参数，并根据输入的格式代码返回一个表示时间的字符串。

**语法:**

```py
time.strftime(%format code, struct_time)

```

**举例:**

```py
import time

tym = time.localtime()
opt = time.strftime("%d/%m/%Y, %H:%M:%S",tym)

print(opt)

```

**输出:**

```py
29/02/2020, 15:07:16

```

* * *

### 9.time.asctime()方法

**`time.asctime()`** 方法将 struct_time 类的元组作为参数，它返回一个表示来自 struct_time 类元组的时间输入的字符串。

**举例:**

```py
import time

tym = time.localtime()

opt = time.asctime(tym)
print("TimeStamp:",opt)

```

**输出:**

```py
TimeStamp: Sat Feb 29 15:27:14 2020

```

* * *

## 结论

在本文中，我们已经了解了 Python 时间模块及其提供的各种功能。

* * *

## 参考

**Python 时间模块**
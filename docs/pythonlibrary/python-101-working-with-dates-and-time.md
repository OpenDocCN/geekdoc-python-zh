# Python 101:使用日期和时间

> 原文：<https://www.blog.pythonlibrary.org/2017/06/15/python-101-working-with-dates-and-time/>

Python 为开发人员提供了几个处理日期和时间的工具。在本文中，我们将关注**日期时间**和**时间**模块。我们将学习它们是如何工作的，以及它们的一些常见用途。让我们从**日期时间**模块开始吧！

### 日期时间模块

我们将从**日期时间**模块中学习以下课程:

*   datetime.date
*   datetime.timedelta
*   datetime.datetime

这些将涵盖 Python 中需要使用日期和日期时间对象的大多数情况。还有一个用于处理时区的 **tzinfo** 类，我们将不再介绍。有关该类的更多信息，请随意查看 Python 文档。

#### datetime.date

Python 可以用几种不同的方式表示日期。我们将首先看一下 **datetime.date** 格式，因为它恰好是一个比较简单的日期对象。

```py

>>> datetime.date(2012, 13, 14)
Traceback (most recent call last):
  File "", line 1, in <fragment>builtins.ValueError: month must be in 1..12
>>> datetime.date(2012, 12, 14)
datetime.date(2012, 12, 14)
```

这段代码展示了如何创建一个简单的日期对象。date 类接受三个参数:年、月和日。如果你给它传递一个无效的值，你会看到一个 **ValueError** ，如上图所示。否则你将会看到一个返回的**日期时间.日期**对象。让我们看另一个例子:

```py

>>> import datetime
>>> d = datetime.date(2012, 12, 14)
>>> d.year
2012
>>> d.day
14
>>> d.month
12

```

这里我们将日期对象赋给变量 **d** 。现在我们可以按名称访问各种日期组件，例如 **d 年**或 d 月。现在让我们找出今天是星期几:

```py

>>> datetime.date.today()
datetime.date(2014, 3, 5)

```

每当你需要记录今天是哪一天时，这都会很有帮助。或者，您可能需要根据今天进行基于日期的计算。不过这是一个方便的小方法。

#### datetime.datetime

**datetime.datetime** 对象包含 datetime.date 加上 datetime.time 对象的所有信息。让我们创建几个例子，以便更好地理解这个对象和 datetime.date 对象之间的区别。

```py

>>> datetime.datetime(2014, 3, 5)
datetime.datetime(2014, 3, 5, 0, 0)
>>> datetime.datetime(2014, 3, 5, 12, 30, 10)
datetime.datetime(2014, 3, 5, 12, 30, 10)
>>> d = datetime.datetime(2014, 3, 5, 12, 30, 10)
>>> d.year
2014
>>> d.second
10
>>> d.hour
12

```

这里我们可以看到 **datetime.datetime** 接受几个额外的参数:年、月、日、小时、分钟和秒。它还允许您指定微秒和时区信息。当您使用数据库时，您会发现自己经常使用这些类型的对象。大多数情况下，您需要将 Python 日期或日期时间格式转换为 SQL 日期时间或时间戳格式。您可以使用两种不同的方法通过 datetime.datetime 找出今天是什么日子:

```py

>>> datetime.datetime.today()
datetime.datetime(2014, 3, 5, 17, 56, 10, 737000)
>>> datetime.datetime.now()
datetime.datetime(2014, 3, 5, 17, 56, 15, 418000)

```

datetime 模块有另一个您应该知道的方法，叫做 **strftime** 。该方法允许开发人员创建一个字符串，以更易于阅读的格式表示时间。在 Python 文档 8.1.7 节中，有一个完整的格式选项表，您应该去阅读。我们将通过几个例子向您展示这种方法的威力:

```py

>>> datetime.datetime.today().strftime("%Y%m%d")
'20140305'
>>> today = datetime.datetime.today()
>>> today.strftime("%m/%d/%Y")
'03/05/2014'
>>> today.strftime("%Y-%m-%d-%H.%M.%S")
'2014-03-05-17.59.53'

```

第一个例子有点像黑客。它展示了如何将今天的 datetime 对象转换成符合 **YYYYMMDD** (年、月、日)格式的字符串。第二个例子更好。这里我们将今天的 datetime 对象赋给一个名为 **today** 的变量，然后尝试两种不同的字符串格式化操作。第一个在 datetime 元素之间添加了正斜杠，并重新排列，使其成为月、日、年。最后一个例子创建了一个时间戳，它遵循一种非常典型的格式: **YYYY-MM-DD。HH.MM.SS** 。如果想去两位数的年份，可以把 **%Y** 换成 **%y** 。

#### datetime.timedelta

**datetime.timedelta** 对象表示持续时间。换句话说，它是两个日期或时间之间的差异。让我们看一个简单的例子:

```py

>>> now = datetime.datetime.now()
>>> now
datetime.datetime(2014, 3, 5, 18, 13, 51, 230000)
>>> then = datetime.datetime(2014, 2, 26)
>>> delta = now - then
>>> type(delta)
 >>> delta.days
7
>>> delta.seconds
65631 
```

我们在这里创建两个日期时间对象。一个是今天的，一个是一周前的。然后我们取两者之差。这将返回一个 timedelta 对象，我们可以用它来找出两个日期之间的天数或秒数。如果你需要知道两者之间的小时数或分钟数，你将不得不使用一些数学来计算出来。有一种方法可以做到:

```py

>>> seconds = delta.total_seconds()
>>> hours = seconds // 3600
>>> hours
186.0
>>> minutes = (seconds % 3600) // 60
>>> minutes
13.0

```

这告诉我们一周有 186 小时 13 分钟。注意，我们使用双正斜杠作为除法运算符。这就是所谓的**楼层划分**。

现在我们准备好继续学习关于**时间**模块的知识了！

* * *

### 时间模块

**time** 模块为 Python 开发者提供了对各种时间相关函数的访问。时间模块是基于所谓的**纪元**，时间开始的时间点。对于 Unix 系统来说，这个时代是在 1970 年。要找出您系统上的纪元，请尝试运行以下命令:

```py

>>> import time
>>> time.gmtime(0)
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, 
                 tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=0)

```

我在 Windows 7 上运行这个程序，它似乎也认为时间是从 1970 年开始的。无论如何，在本节中，我们将学习以下与时间相关的函数:

*   time.ctime
*   时间.睡眠
*   time.strftime
*   时间时间

我们开始吧！

#### time.ctime

**time.ctime** 函数将从 epoch 开始以秒为单位的时间转换为表示本地时间的字符串。如果你没有传递任何东西给它，那么返回当前时间。让我们尝试几个例子:

```py

>>> import time
>>> time.ctime()
'Thu Mar 06 07:28:48 2014'
>>> time.ctime(1384112639)
'Sun Nov 10 13:43:59 2013'

```

这里我们展示了调用 **ctime** 的结果，没有任何东西，并且从 epoch 开始有一个相当随机的秒数。我见过有人把日期保存为纪元后的秒，然后想把它转换成人类可以理解的东西。将大整数(或长整数)保存到数据库中，然后将其从日期时间对象格式化为数据库接受的任何日期对象，这要简单一些。当然，这也有一个缺点，那就是你需要将整型或浮点型值转换回字符串。

#### 时间.睡眠

**time.sleep** 函数让开发人员能够在给定的秒数内暂停脚本的执行。这就像给你的程序添加一个暂停。当我需要等待一个文件完成关闭或者一个数据库提交完成时，我发现这非常有用。让我们看一个例子。在空闲状态下打开一个新窗口，并保存以下代码:

```py

import time

for x in range(5):
    time.sleep(2)
    print("Slept for 2 seconds")

```

现在在空闲状态下运行代码。你可以进入**运行**菜单，然后选择**运行模块**菜单项。当你这样做的时候，你会看到它打印出短语*睡眠 2 秒*五次，每次打印之间有两秒钟的停顿。真的那么好用！

#### time.strftime

**时间**模块有一个 **strftime** 函数，其工作方式与 datetime 版本非常相似。区别主要在于它接受的输入:一个元组或一个 **struct_time** 对象，就像调用 **time.gmtime()** 或 **time.localtime()** 时返回的那些对象。这里有一个小例子:

```py

>>> time.strftime("%Y-%m-%d-%H.%M.%S",
                  time.localtime())
'2014-03-06-20.35.56'

```

这段代码与我们在本章的日期时间部分创建的时间戳代码非常相似。我认为 datetime 方法更直观一些，因为您只需创建一个 **datetime.datetime** 对象，然后用您想要的格式调用它的 **strftime** 方法。对于时间模块，必须传递格式和时间元组。哪个对你最有意义，真的是你自己决定的。

#### 时间时间

**time.time** 函数将以浮点数的形式返回从 epoch 开始的时间(以秒为单位)。让我们来看看:

```py

>>> time.time()
1394199262.318

```

这很简单。当您想将当前时间保存到数据库中，但又不想麻烦地将其转换为数据库的 datetime 方法时，可以使用这个方法。您可能还记得， **ctime** 方法接受以秒为单位的时间，因此我们可以使用 **time.time** 来获取传递给 ctime 的秒数，如下所示:

```py

>>> time.ctime(time.time())
'Fri Mar 07 07:36:38 2014'

```

如果您在 time 模块的文档中做了一些研究，或者只是做了一点试验，您可能会发现这个函数的一些其他用途。

* * *

### 包扎

至此，您应该知道如何使用 Python 的标准模块处理日期和时间。在处理日期时，Python 为您提供了强大的功能。如果您需要创建一个跟踪约会或需要在特定日期运行的应用程序，您会发现这些模块非常有用。它们在处理数据库时也很有用。

* * *

### 相关阅读

*   关于[日期时间模块](https://docs.python.org/3/library/datetime.html)的 Python 文档
*   关于[时间模块](https://docs.python.org/3/library/time.html)的 Python 文档
*   本周 Python 模块:[日期时间](http://pymotw.com/2/datetime/)
*   另一个 Python 日期时间替代:[钟摆](https://www.blog.pythonlibrary.org/2016/07/13/yet-another-python-datetime-replacement-pendulum/)
*   [日期搜索包](https://www.blog.pythonlibrary.org/2016/02/04/python-the-datefinder-package/)
*   Python 与 [Delorean](https://www.blog.pythonlibrary.org/2014/09/03/python-taking-time-with-delorean/) 共度时光
*   arrow "[Python 的新日期/时间包](https://www.blog.pythonlibrary.org/2014/08/05/arrow-a-new-date-time-package-for-python/)
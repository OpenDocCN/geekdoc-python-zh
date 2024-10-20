# Python 中的 dateutil 模块

> 原文：<https://www.askpython.com/python-modules/dateutil-module>

在处理需要实时数据的脚本时，我们使用 dateutil 模块，以便在特定时间调度或检索数据，或者只是输入带有检索时间戳的数据。

考虑到您需要对检索到的数据进行大量的更改，使用大量的脚本并尝试使用默认的`datetime`模块操作日期和时间格式可能是一项艰巨的任务。

幸运的是，`dateutil`模块是为了提供使您的生活更轻松的功能而创建的。

`dateutil`模块专门为现有的`datetime`模块提供扩展功能，因此，`datetime`模块的安装是先决条件。

然而，因为它是 Python 标准库的一部分，所以没什么好担心的。

## 在 Python 中安装 dateutil 模块

在我们开始使用`dateutil`模块之前，我们需要先在我们的电脑上安装它。那么，让我们开始安装程序:

```py
# The command to install the module, through the pip manager.
pip install python-dateutil

```

我们已经使用了 [pip 包管理器](https://www.askpython.com/python-modules/python-pip)来完成这里的安装。你也可以使用 [Anaconda](https://www.askpython.com/python-modules/python-anaconda-tutorial) 来完成安装。

## 使用 dateutil 模块

如果您已经成功安装了该模块，我们现在可以开始使用它了！

### 1.0.模块及其子类。

`dateutil`模块被分成几个不同的子类，我们将马上进入它们，这样你就知道你在做什么，

*   复活节
*   句法分析程序
*   相对 delta
*   尺子
*   坦桑尼亚
*   还有几个！

该模块没有太多的子类，但是，在本文中，我们将只深入研究其中一些的功能。

### 1.1 导入所需的方法

我们已经安装了模块，现在只需要将方法付诸实施并获得结果。

那么，让我们从其中的一些开始吧！

等等，在使用`dateutil`模块之前，我们可能有几个步骤，其中之一就是需要[首先导入](https://www.askpython.com/python/python-import-statement)它们。

```py
# We'll need to import methods from the datetime module as a base.
import datetime

# Now, let's import some methods from the dateutil subclasses.
from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.parser import *
from dateutil.rrule import *

```

这些导入允许我们使用本文中需要的许多方法。

### 1.2 日期时间功能

在我们开始使用`dateutil`模块之前，您可能还记得这个模块也依赖于[日期时间模块](https://www.askpython.com/python-modules/python-datetime-module)的事实，对吗？

嗯，完全正确。`dateutil`模块处理*日期时间*对象，这意味着我们需要在处理它们之前创建*日期时间*对象。

因此，这就是`datetime`模块导入的原因。让我们开始使用`dateutil`中的几个模块。

### 1.3.日期时间和相对增量

`relativedelta`子类扩展了`datetime`模块，为我们提供了允许我们处理与检索信息相关的日期和时间的特性。

这意味着我们可以给当前使用的`datetime`对象添加天数、月数甚至年数。它还允许我们用`datetime` 对象来处理时间间隔。

```py
# Creating a few datetime objects to work with
NOW = datetime.now()
print("The datetime right now : ", NOW)
TODAY = date.today()
print("The date today : ", TODAY)

```

现在，让我们使用相对日期来检索信息。

```py
# Next month
print(NOW + relativedelta(months=+1))

# Next month, plus one week
print(NOW + relativedelta(months=+1, weeks=+1))

# Next month, plus one week, at 5 PM
print(NOW + relativedelta(months=+1, weeks=+1, hour=17))

# Next friday
print(TODAY + relativedelta(weekday=FR))

```

这个模块的一个更适用的用途是使用一些小操作来查找信息。

```py
# Finding out the last tuesday in the month
print(TODAY + relativedelta(day=31, weekday=TU(-1)))

# We can also work with datetime objects directly
# Example: Age of Sacra

sacra_birthday = datetime(1939, 4, 5, 12, 0)
print("Sacra's Age : ", relativedelta(NOW, sacra_birthday).years)

```

如果你已经注意到，我们只从`relativedelta`对象中检索了`years`。

这是用于一个干净的输出，但是，如果你想知道 Sacra 实际上有多老，试着自己摆弄一下 relativedelta 对象。😉

### 1.4.日期时间和复活节

`easter`子类用于计算一般复活节日历的日期和时间，允许计算与各种日历相关的日期时间对象。

子类非常小，只有一个参数和三个选项定义了整个模块。

*   儒略历，复活节 _ 儒略历=1。
*   公历，复活节 _ 东正教=2
*   西历，复活节 _ 西方=3

**在代码中使用它们，看起来很像，**

```py
# The Julian Calendar
print("Julian Calendar : ", easter(1324, 1))

# The Gregorian Calendar
print("Gregorian Calendar : ", easter(1324, 2))

# The Western Calendar
print("Western Calendar : ", easter(1324, 3))

```

### 1.5.日期时间和解析器

`parser`子类带来了一个高级的日期/时间字符串解析器，它能够解析多种已知的表示日期或时间的格式。

```py
# The parser subclass
print(parse("Thu Sep 25 10:36:28 BRST 2003"))

# We can also ignore the timezone which is set to default locally
print(parse("Thu Sep 25 10:36:28 BRST 2003", ignoretz=True))

# We can also not provide a timezone, or a year
# This allows for it to return the current year, with no timezone inclusion.
print(parse("Thu Sep 25 10:36:28"))

# We can also provide variables which contain information, as values.
DEFAULT = datetime(2020, 12, 25)
print(parse("10:36", default=DEFAULT))

```

您可以提供许多选项，包括本地或显式时区。

可以使用作为默认参数传递给函数的变量来提取信息以提供时区、年份、时间，您可以在这里查看。

### 1.6.日期时间和规则

`rrule`子类使用输入分别为我们提供关于`datetime`对象和`datetime`对象的递归信息。

```py
# The rrule subclass
# Daily repetition for 20 occurrences
print(list(rrule(DAILY, count=20, dtstart=parse("20201202T090000"))))

# Repeating based on the interval
print(list(rrule(DAILY, interval=10, count=5, dtstart=parse("20201202T090000"))))

# Weekly repetition
print(list(rrule(WEEKLY, count=10, dtstart=parse("20201202T090000"))))

# Monthly repetition
print(list(rrule(MONTHLY, count=10, dtstart=parse("20201202T090000"))))

# Yearly repetition
print(list(rrule(YEARLY, count=10, dtstart=parse("20201202T090000"))))

```

这个子类是`dateutil`模块的一个很好的特性，它可以让你处理很多调度任务和日历存储创新。

模块本身还有更多的内容，如果你想在更深的层次上了解更多的特性和论点，查看一下[文档](https://dateutil.readthedocs.io/en/stable/index.html)是个好主意。

## 结论

如果您已经阅读了这篇文章，那么您现在知道了`dateutil`模块如何允许我们扩展由`datetime`模块提供的信息，以产生您通常需要计算或处理的结果。

如果你知道为了做某件事应该看哪个模块，生活就会简单得多。

也就是说，这里有一些可以帮助你完成与 [Python 熊猫](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)、 [Scipy](https://www.askpython.com/python-modules/python-scipy) 、 [zipfile](https://www.askpython.com/python-modules/zipfile-module) 和 [psutil](https://www.askpython.com/python-modules/psutil-module) 的工作之旅。

## 参考

*   [官方日期文档](https://dateutil.readthedocs.io/en/stable/)
*   [使用 dateutil 的示例](https://dateutil.readthedocs.io/en/stable/examples.html#)
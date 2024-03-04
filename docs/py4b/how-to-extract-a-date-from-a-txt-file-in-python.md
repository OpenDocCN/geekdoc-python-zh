# 如何用 Python 从. txt 文件中提取日期

> 原文：<https://www.pythonforbeginners.com/files/how-to-extract-a-date-from-a-txt-file-in-python>

在本教程中，我们将研究使用 Python 编程从. txt 文件中提取日期的不同方法。Python 是一种通用的语言——您将会发现——这个问题有许多解决方案。

首先，我们将看看如何使用*正则表达式*模式在文本文件中搜索符合预定义格式的日期。我们将学习使用 **re** 库和创建我们自己的正则表达式搜索。

我们还将检查 **datetime** 对象，并使用它们将字符串转换成数据模型。最后，我们将看到 **datefinder** 模块如何简化在文本文件中搜索未格式化日期的过程，就像我们在自然语言内容中可能会发现的那样。

## 使用正则表达式从. txt 文件中提取日期

日期有许多不同的格式。有时人们写月/日/年。其他日期可能包括一天中的时间或一周中的某一天(2021 年 7 月 8 日星期三晚上 8:00)。

在我们从文本文件中提取日期之前，如何格式化日期是一个需要考虑的因素。

例如，如果日期遵循月/日/年格式，我们可以使用正则表达式模式找到它。使用正则表达式，或者简称为 **regex** ，我们可以通过将字符串与预定义的模式进行匹配来搜索文本。

正则表达式的美妙之处在于我们可以使用特殊字符来创建强大的搜索模式。例如，我们可以设计一个模式，在下面的正文中查找所有格式化的日期。

*minutes . txt*
2021 年 10 月 14 日-会见客户。
2021 年 7 月 1 日——讨论营销策略。
2021 年 12 月 23 日——采访新的团队领导。
2018 年 1 月 28 日–变更域名提供商。
2017 年 6 月 11 日–讨论搬到新办公室。

#### 示例:使用正则表达式查找格式化日期

```py
import re

# open the text file and read the data
file = open("minutes.txt",'r')

text = file.read()
# match a regex pattern for formatted dates
matches = re.findall(r'(\d+/\d+/\d+)',text)

print(matches) 
```

**输出**

[’10/14/2021′, ’07/01/2021′, ’12/23/2021′, ’01/28/2018′, ’06/11/2017′]

这里的 regex 模式使用特殊字符来定义我们想要从文本文件中提取的字符串。字符 **d** 和 **+** 告诉 regex 我们正在文本中寻找多个数字。

我们还可以使用 regex 来查找以不同方式格式化的日期。通过改变 regex 模式，我们可以找到使用正斜杠( **\** )或破折号(**–**)作为分隔符的日期。

这是因为正则表达式允许在搜索模式中使用可选字符。我们可以指定任何一个字符——正斜杠或破折号——都是可接受的匹配。

第一台 Apple II 于 1977 年 7 月 10 日售出。最后一批 Apple II
型号于 1994 年 10 月 15 日停产。

#### 示例:用正则表达式模式匹配日期

```py
import re

# open a text file
f = open("apple2.txt", 'r')

# extract the file's content
content = f.read()

# a regular expression pattern to match dates
pattern = "\d{2}[/-]\d{2}[/-]\d{4}"

# find all the strings that match the pattern
dates = re.findall(pattern, content)

for date in dates:
    print(date)

f.close() 
```

**输出**

1977 年 10 月 7 日
1994 年 10 月 15 日

研究 regex 的全部潜力超出了本教程的范围。尝试使用以下一些特殊字符，以了解有关使用正则表达式模式从. txt 文件中提取日期或其他信息的更多信息。

#### 正则表达式中的特殊字符

*   \ s–空格字符
*   \ S–除空格字符以外的任何字符
*   \ d–0 到 9 之间的任何数字
*   \ D–以及除数字以外的任何字符
*   \ w–任何字符或数字的单词[a-zA-Z0-9]
*   \ W–任何非单词字符

## 从. txt 文件中提取日期时间对象

在 Python 中，我们可以使用 **datetime** 库来处理日期和时间。datetime 库预装了 Python，所以不需要安装它。

通过使用 datetime 对象，我们可以更好地控制从文本文件中读取的字符串数据。例如，我们可以使用 datetime 对象来获取计算机当前日期和时间的副本。

```py
import datetime

now = datetime.datetime.now()
print(now) 
```

**输出**

```py
2021-07-04 20:15:49.185380
```

在下面的例子中，我们将从一家公司提取一个日期。提及预定会议的 txt 文件。我们的雇主需要我们扫描一组这样的文件的日期。稍后，我们计划将收集到的信息添加到 SQLite 数据库中。

我们将从定义一个匹配日期格式的正则表达式模式开始。一旦找到匹配，我们将使用它从字符串数据创建一个*日期时间*对象。

*schedule.txt*

这个项目下个月开始。丹尼斯计划于 2021 年 7 月 10 日在大使馆的会议室开会。

#### 示例:从文件数据创建日期时间对象

```py
import re
from datetime import datetime

# open the data file
file = open("schedule.txt", 'r')
text = file.read()

match = re.search(r'\d+-\d+-\d{4}', text)
# create a new datetime object from the regex match
date = datetime.strptime(match.group(), '%d-%m-%Y').date()
print(f"The date of the meeting is on {date}.")
file.close() 
```

**输出**

```py
The date of the meeting is on 2021-07-10.
```

## 用 Datefinder 模块从文本文件中提取日期

Python **datefinder** 模块可以在文本主体中定位日期。使用 *find_dates* ()方法，可以在文本数据中搜索许多不同类型的日期。Datefinder 将以 datetime 对象的形式返回它找到的任何日期。

与我们在本指南中讨论的其他包不同，Python 没有提供 datefinder。安装 datefinder 模块最简单的方法是在命令提示符下使用 **pip** 。

```py
pip install datefinder
```

安装了 datefinder 后，我们就可以打开文件并提取数据了。对于这个例子，我们将使用一个介绍虚构公司项目的文本文档。使用 datefinder，我们将从。txt 文件，并打印它们的 datimeobject 副本。

请随意将文件保存在本地，然后继续操作。

*project_timeline.txt*
项目胡椒

所有团队成员必须在 2021 年 1 月 4 日之前阅读项目总结。

胡椒计划的第一次会议于 2021 年 1 月 15 日召开

上午九点。请在那之前抽出时间阅读下面的链接。
*创建于* 08-12-2021 下午 05:00

此项目文件包含多种格式的日期。日期用破折号和正斜线书写。更糟糕的是，一月份被写出来。如何用 Python 找到所有这些日期？

#### 示例:使用 datefinder 从文件数据中提取日期

```py
import datefinder

# open the project schedule
file = open("project_timeline.txt",'r')

content = file.read()

# datefinder will find the dates for us
matches = list(datefinder.find_dates(content))

if len(matches) > 0:
    for date in matches:
        print(date)
else:
    print("Found no dates.")

file.close() 
```

**产量**
2021-01-04 00:00:00
2021-01-15 09:00:00
2021-08-12 17:00:00

从输出中可以看到，datefinder 能够在文本中找到各种日期格式。该软件包不仅能够识别月份名称，而且如果包含在文本中，它还能识别一天中的时间。

在另一个例子中，我们将使用 datefinder 包从一个. txt 文件中提取一个日期，该文件包含一位流行歌手即将到来的巡演的日期。

*tour _ dates . txt*
2021 年 7 月 25 日星期六下午 07:00 加利福尼亚州英格尔伍德
2021 年 7 月 26 日星期日下午 7:00 加利福尼亚州英格尔伍德
2021 年 9 月 30 日下午 7:30 马萨诸塞州福克斯堡

#### 示例:使用 datefinder 从. txt 文件中提取游览日期和时间

```py
import datefinder

# open the project schedule
file = open("tour_dates.txt",'r')

content = file.read()

# datefinder will find the dates for us
matches = list(datefinder.find_dates(content))

if len(matches) > 0:
    print("TOUR DATES AND TIMES")
    print("--------------------")
    for date in matches:
        # use f string to format the text
        print(f"{date.date()}     {date.time()}")
else:
    print("Found no dates.")
file.close() 
```

**输出**

游览日期和时间
————————
2021-07-25 19:00:00
2021-07-26 19:00:00
2021-09-30 19:30:00

从示例中可以看出，datefinder 可以找到许多不同类型的日期和时间。如果您要查找的日期没有特定的格式，这很有用，自然语言数据中经常会出现这种情况。

## 摘要

在这篇文章中，我们介绍了几种从. txt 文件中提取日期或时间的方法。我们已经看到了正则表达式在字符串数据中查找匹配的强大功能，以及如何将这些数据转换成 Python datetime 对象。

最后，如果你的文本文件中的日期没有指定的格式——大多数包含自然语言内容的文件都是这种情况——试试 **datefinder** 模块。有了这个 Python 包，就可以从一个事先没有方便格式化的文本文件中提取日期和时间。

## 相关职位

如果你喜欢本教程，并渴望了解更多关于 Python 的知识——我们真诚地希望你是这样——请点击这些链接，获取更多 Python 初学者指南。

*   如何使用 Python 串联来连接字符串
*   使用 [Python try catch](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 来减少错误并防止崩溃
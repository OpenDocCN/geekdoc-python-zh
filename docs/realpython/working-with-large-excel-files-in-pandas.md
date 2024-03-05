# 使用 Pandas 在 Python 中读取大型 Excel 文件

> 原文：<https://realpython.com/working-with-large-excel-files-in-pandas/>

在本教程中，你将学习如何在 [Pandas](https://realpython.com/pandas-python-explore-dataset/) 中处理大型 Excel 文件，重点是[读取和分析 xls 文件](https://realpython.com/pandas-read-write-files/#using-pandas-to-write-and-read-excel-files)，然后处理原始数据的子集。

**免费奖励:** 向您展示了如何读取大型 Excel 文件。

本教程利用了 Python(用 64 位版本的 v2.7.9 和 v3.4.3 测试过)， [Pandas](http://pandas.pydata.org/pandas-docs/version/0.16.1/) (v0.16.1)，和 [XlsxWriter](https://xlsxwriter.readthedocs.org/) (v0.7.3)。我们推荐使用 [Anaconda](http://continuum.io/downloads) 发行版来快速入门，因为它预装了所有需要的库。

## 读取文件

我们要处理的第一个文件是从 1979 年到 2004 年英国所有车祸的汇编，提取 2000 年发生在伦敦的所有车祸。

[*Remove ads*](/account/join/)

### Excel

首先从[data.gov.uk](http://data.dft.gov.uk/road-accidents-safety-data/Stats19-Data1979-2004.zip)下载源 ZIP 文件，并提取内容。然后*尝试*在 Excel 中打开*incidents 7904 . CSV*。*小心。如果你没有足够的内存，这很可能会使你的电脑崩溃。*

会发生什么？

您应该会看到一个“文件没有完全加载”的错误，因为 Excel 一次只能处理一百万行。

> 我们在 [LibreOffice](https://www.libreoffice.org/) 中对此也进行了测试，并收到了类似的错误——“数据无法完全加载，因为超出了每张工作表的最大行数。”

为了解决这个问题，我们可以在 Pandas 中打开文件。在我们开始之前，源代码在 [Github](https://github.com/shantnu/PandasLargeFiles) 上。

### 熊猫

在一个新的项目目录中，激活一个 virtualenv，然后安装 Pandas:

```py
$ pip install pandas==0.16.1
```

现在让我们构建脚本。创建一个名为 *pandas_accidents.py* 的文件，并添加以下代码:

```py
import pandas as pd

# Read the file
data = pd.read_csv("Accidents7904.csv", low_memory=False)

# Output the number of rows
print("Total rows: {0}".format(len(data)))

# See which headers are available
print(list(data))
```

在这里，我们导入 Pandas，读入文件——这可能需要一些时间，取决于您的系统有多少内存——并输出文件的总行数以及可用的标题(例如，列标题)。

运行时，您应该看到:

```py
Total rows: 6224198
['\xef\xbb\xbfAccident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR',
 'Longitude', 'Latitude', 'Police_Force', 'Accident_Severity', 'Number_of_Vehicles',
 'Number_of_Casualties', 'Date', 'Day_of_Week', 'Time', 'Local_Authority_(District)',
 'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number', 'Road_Type',
 'Speed_limit', 'Junction_Detail', 'Junction_Control', '2nd_Road_Class',
 '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control',
 'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions', 'Weather_Conditions',
 'Road_Surface_Conditions', 'Special_Conditions_at_Site', 'Carriageway_Hazards',
 'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',
 'LSOA_of_Accident_Location']
```

所以，有超过 600 万行！难怪 Excel 噎着了。请注意标题列表，特别是第一个标题:

```py
'\xef\xbb\xbfAccident_Index',
```

这应该是`Accident_Index`。开头多出来的`\xef\xbb\xbf`是怎么回事？嗯，`\x`其实是指值为[十六进制](http://en.wikipedia.org/wiki/Hexadecimal)，是一个[字节顺序标志](http://stackoverflow.com/a/18664752/1799408)，表示文本为 [Unicode](https://realpython.com/python-encodings-guide/) 。

为什么这对我们很重要？

你不能假定你阅读的文件是干净的。它们可能包含像这样的额外符号，会影响你的脚本。

这个文件是好的，因为它在其他方面是干净的——但是许多文件有丢失的数据、内部格式不一致的数据等等..所以任何时候你有一个文件要分析，你必须做的第一件事就是清理它。打扫了多少？足以让你做一些分析。遵循[吻](http://en.wikipedia.org/wiki/KISS_principle)的原则。

你可能需要什么样的清理工作？

*   确定日期/时间。同一个文件可能有不同格式的日期，比如美国(mm-dd-yy)或欧洲(dd-mm-yy)格式。这些需要纳入一个共同的格式。
*   *删除任何空值。*文件可能有空白的列和/或行，在 Pandas 中这将显示为 *NaN* (不是一个数字)。Pandas 提供了一个简单的方法来删除这些:函数`dropna()`。我们在[上一篇博文](https://realpython.com/analyzing-obesity-in-england-with-python/)中看到了这样的例子。
*   *删除任何进入数据的垃圾值。*这些值没有意义(就像我们之前看到的字节顺序标记)。有时候，也许可以绕过他们。例如，可能有一个数据集将年龄作为浮点数输入(误输入)。然后可以使用`int()`函数来确保所有年龄都是整数格式。

[*Remove ads*](/account/join/)

## 分析

对于熟悉 SQL 的人来说，可以使用带有不同关键字的 SELECT、WHERE 和/或语句来优化搜索。我们可以在熊猫身上做同样的事情，而且是以一种对程序员更友好的方式。

首先，让我们找出星期天发生的所有事故。查看上面的标题，有一个`Day_of_Weeks`字段，我们将使用它。

在您下载的 ZIP 文件中，有一个名为*Road-Accident-Safety-Data-Guide-1979-2004 . xls*的文件，其中包含了所使用代码的额外信息。如果你打开它，你会看到*星期日*有代码`1`。

```py
print("\nAccidents")
print("-----------")

# Accidents which happened on a Sunday
accidents_sunday = data[data.Day_of_Week == 1]
print("Accidents which happened on a Sunday: {0}".format(
    len(accidents_sunday)))
```

就是这么简单。

在这里，我们以`Day_of_Weeks`字段为目标，返回一个[数据帧](https://realpython.com/pandas-dataframe/)，其中包含我们检查的条件- `day of week == 1`。

当尤然应该看到:

```py
Accidents
-----------
Accidents which happened on a Sunday: 693847
```

如你所见，周日发生了 693，847 起事故。

让我们把我们的查询变得更复杂一些:找出发生在星期天的所有事故，涉及 20 多辆汽车:

```py
# Accidents which happened on a Sunday, > 20 cars
accidents_sunday_twenty_cars = data[
    (data.Day_of_Week == 1) & (data.Number_of_Vehicles > 20)]
print("Accidents which happened on a Sunday involving > 20 cars: {0}".format(
    len(accidents_sunday_twenty_cars)))
```

运行脚本。现在我们有 10 起事故:

```py
Accidents
-----------
Accidents which happened on a Sunday: 693847
Accidents which happened on a Sunday involving > 20 cars: 10
```

我们再加一个条件:天气。

打开道路-事故-安全-数据-指南-1979-2004.xls，转到*天气*单。你会看到代码`2`的意思是，“没有大风的雨”。

添加到我们的查询:

```py
# Accidents which happened on a Sunday, > 20 cars, in the rain
accidents_sunday_twenty_cars_rain = data[
    (data.Day_of_Week == 1) & (data.Number_of_Vehicles > 20) &
    (data.Weather_Conditions == 2)]
print("Accidents which happened on a Sunday involving > 20 cars in the rain: {0}".format(
    len(accidents_sunday_twenty_cars_rain)))
```

因此，在一个星期天发生了四起事故，涉及 20 多辆汽车，当时正在下雨:

```py
Accidents
-----------
Accidents which happened on a Sunday: 693847
Accidents which happened on a Sunday involving > 20 cars: 10
Accidents which happened on a Sunday involving > 20 cars in the rain: 4
```

如果需要，我们可以继续把它变得越来越复杂。现在，我们将停止，因为我们的主要兴趣是看看伦敦的事故。

如果你再看*道路-事故-安全-数据-指南-1979-2004.xls* ，有一张表叫*警队*。`1`的代码写着，“伦敦警察厅”。这就是通常所说的*苏格兰场*，也是负责伦敦大部分地区(尽管不是全部)的警察力量。对于我们的例子来说，这已经足够了，我们可以像这样提取信息:

```py
# Accidents in London on a Sunday
london_data = data[data['Police_Force'] == 1 & (data.Day_of_Week == 1)]
print("\nAccidents in London from 1979-2004 on a Sunday: {0}".format(
    len(london_data)))
```

运行脚本。这为“伦敦警察厅”从 1979 年到 2004 年的一个周日处理的事故创建了一个新的数据框架:

```py
Accidents
-----------
Accidents which happened on a Sunday: 693847
Accidents which happened on a Sunday involving > 20 cars: 10
Accidents which happened on a Sunday involving > 20 cars in the rain: 4

Accidents in London from 1979-2004 on a Sunday: 114624
```

如果您想创建一个只包含 2000 年事故的新数据框架，该怎么办？

我们需要做的第一件事是[使用`pd.to_datetime()`](https://realpython.com/python-time-module/) [函数](http://pandas.pydata.org/pandas-docs/version/0.16.1/generated/pandas.to_datetime.html?highlight=to_datetime#pandas.to_datetime)将日期格式转换成 Python 可以理解的格式。它接受任何格式的日期，并将其转换为我们可以理解的格式( *yyyy-mm-dd* )。然后我们可以创建另一个只包含 2000 年事故的数据框架:

```py
# Convert date to Pandas date/time
london_data_2000 = london_data[
    (pd.to_datetime(london_data['Date'], coerce=True) >
        pd.to_datetime('2000-01-01', coerce=True)) &
    (pd.to_datetime(london_data['Date'], coerce=True) <
        pd.to_datetime('2000-12-31', coerce=True))
]
print("Accidents in London in the year 2000 on a Sunday: {0}".format(
    len(london_data_2000)))
```

运行时，您应该看到:

```py
Accidents which happened on a Sunday: 693847
Accidents which happened on a Sunday involving > 20 cars: 10
Accidents which happened on a Sunday involving > 20 cars in the rain: 4

Accidents in London from 1979-2004 on a Sunday: 114624
Accidents in London in the year 2000 on a Sunday: 3889
```

所以，这一开始有点混乱。通常，要过滤一个数组，你只需使用一个带有条件的`for`循环:

```py
for data in array:
    if data > X and data < X:
        # Do something
```

然而，你真的不应该定义自己的循环，因为许多高性能的库，比如 Pandas，都有助手函数。在这种情况下，上面的代码循环遍历所有元素，过滤掉设定日期之外的数据，然后返回日期范围内的数据点。

不错！

[*Remove ads*](/account/join/)

## 转换

很有可能，在使用 Pandas 的同时，您组织中的其他人都在使用 Excel。想要与使用 Excel 的人共享数据框架吗？

首先，我们需要做一些清理工作。还记得我们之前看到的字节顺序标记吗？这在将这些数据写入 Excel 文件时会导致问题——Pandas 抛出一个 *UnicodeDecodeError* 。为什么？因为文本的其余部分被解码为 ASCII，但十六进制值不能用 ASCII 表示。

我们可以把所有东西都写成 Unicode，但是记住这个字节顺序标记是多余的(对我们来说),我们不想要或者不需要。因此，我们将通过重命名列标题来消除它:

```py
london_data_2000.rename(
    columns={'\xef\xbb\xbfAccident_Index': 'Accident_Index'}, 
    inplace=True)
```

这是在熊猫中重命名列的方法；老实说，有点复杂。因为我们想要修改现有的结构，而不是创建一个副本，这是 Pandas 默认做的。

现在我们可以将数据保存到 Excel:

```py
# Save to Excel
writer = pd.ExcelWriter(
    'London_Sundays_2000.xlsx', engine='xlsxwriter')
london_data_2000.to_excel(writer, 'Sheet1')
writer.save()
```

确保在运行前安装 [XlsxWriter](https://xlsxwriter.readthedocs.org/) :

```py
$ pip install XlsxWriter==0.7.3
```

如果一切顺利，这应该会创建一个名为 *London_Sundays_2000.xlsx* 的文件，然后将我们的数据保存到 *Sheet1* 中。在 Excel 或 LibreOffice 中打开该文件，并确认数据是正确的。

## 结论

那么，我们完成了什么？嗯，我们拿了一个 Excel 打不开的很大的文件，用熊猫来-

1.  打开文件。
2.  对数据执行类似 SQL 的查询。
3.  用原始数据的子集创建一个新的 XLSX 文件。

请记住，尽管这个文件将近 800MB，但在大数据时代，它仍然很小。如果你想打开一个 4GB 的文件呢？即使你有 8GB 或更多的内存，那也是不可能的，因为你的大部分内存是为操作系统和其他系统进程保留的。事实上，我的笔记本电脑在第一次读取 800MB 文件时死机了几次。如果我打开一个 4GB 的文件，它会心脏病发作。

**免费奖励:** 向您展示了如何读取大型 Excel 文件。

那么我们该如何进行呢？

诀窍是不要一次打开整个文件。这就是我们将在下一篇博文中关注的内容。在那之前，分析你自己的数据。请在下面留下问题或评论。你可以从 [repo](https://github.com/shantnu/PandasLargeFiles) 中获取本教程的代码。***
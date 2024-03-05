# pandas group by:Python 数据分组指南

> 原文：<https://realpython.com/pandas-groupby/>

无论你是刚刚开始使用[熊猫](https://realpython.com/pandas-python-explore-dataset/)并希望掌握它的核心能力之一，还是你正在寻找填补你对`.groupby()`理解的一些空白，本教程将帮助你从头到尾分解并想象一个**熊猫群通过**运作。

本教程旨在补充官方熊猫文档和 T2 熊猫食谱，在那里你会看到独立的、一口大小的例子。但是，在这里，您将重点关注三个更复杂的使用真实数据集的演练。

**在本教程中，您将学习:**

*   如何通过对**真实世界数据**的操作来使用熊猫**分组**
*   **拆分-应用-合并**操作链是如何工作的
*   如何**将**拆分-应用-合并链分解成步骤
*   如何根据目的和结果按对象对熊猫组的方法进行分类

本教程假设您对 pandas 本身有一些经验，包括如何使用`read_csv()`将 CSV 文件作为 pandas 对象读入内存。如果你需要复习，那就看看[阅读带熊猫的 CSVs】和](https://realpython.com/python-csv/#reading-csv-files-with-pandas)[熊猫:如何读写文件](https://realpython.com/pandas-read-write-files/)。

您可以点击下面的链接下载本教程中所有示例的源代码:

**下载数据集:** [点击这里下载数据集，你将在本教程中使用](https://realpython.com/bonus/pandas-groupby/)来了解熊猫的分组。

## 先决条件

在您继续之前，请确保您在新的[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中拥有最新版本的 pandas:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
PS> python -m venv venv
PS> venv\Scripts\activate
(venv) PS> python -m pip install pandas
```

```py
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ python -m pip install pandas
```

在本教程中，您将关注三个数据集:

1.  [美国国会数据集](https://github.com/unitedstates/congress-legislators)包含国会历史成员的公开信息，并展示了`.groupby()`的几项基本功能。
2.  [空气质量数据集](http://archive.ics.uci.edu/ml/datasets/Air+Quality)包含定期气体传感器读数。这将允许您使用[浮点数](https://realpython.com/python-numbers/#floating-point-numbers)和时间序列数据。
3.  [新闻聚合数据集](http://archive.ics.uci.edu/ml/datasets/News+Aggregator)拥有几十万篇新闻文章的元数据。你将与弦乐一起工作，并与`.groupby()`一起发短信[和](https://en.wikipedia.org/wiki/Data_wrangling)。

您可以点击下面的链接下载本教程中所有示例的源代码:

**下载数据集:** [点击这里下载数据集，你将在本教程中使用](https://realpython.com/bonus/pandas-groupby/)来了解熊猫的分组。

下载完`.zip`文件后，将该文件解压到当前目录下的一个名为`groupby-data/`的文件夹中。在您继续阅读之前，请确保您的[目录树](https://realpython.com/directory-tree-generator-python/)如下所示:

```py
./
│
└── groupby-data/
    │
    ├── legislators-historical.csv
    ├── airqual.csv
    └── news.csv
```

安装了`pandas`,激活了虚拟环境，下载了数据集，您就可以开始了！

[*Remove ads*](/account/join/)

## 示例 1:美国国会数据集

你将通过剖析国会历史成员的[数据集直接进入事物。你可以用`read_csv()`将 CSV 文件读入一个](https://github.com/unitedstates/congress-legislators)[熊猫`DataFrame`T5:](https://realpython.com/pandas-dataframe/)

```py
# pandas_legislators.py

import pandas as pd

dtypes = {
    "first_name": "category",
    "gender": "category",
    "type": "category",
    "state": "category",
    "party": "category",
}
df = pd.read_csv(
    "groupby-data/legislators-historical.csv",
    dtype=dtypes,
    usecols=list(dtypes) + ["birthday", "last_name"],
    parse_dates=["birthday"]
)
```

数据集包含成员的名字和姓氏、生日、性别、类型(`"rep"`代表众议院或`"sen"`代表参议院)、美国州和政党。您可以使用`df.tail()`来查看数据集的最后几行:

>>>

```py
>>> from pandas_legislators import df
>>> df.tail()
 last_name first_name   birthday gender type state       party
11970   Garrett     Thomas 1972-03-27      M  rep    VA  Republican
11971    Handel      Karen 1962-04-18      F  rep    GA  Republican
11972     Jones     Brenda 1959-10-24      F  rep    MI    Democrat
11973    Marino        Tom 1952-08-15      M  rep    PA  Republican
11974     Jones     Walter 1943-02-10      M  rep    NC  Republican
```

`DataFrame`使用分类 **dtypes** 实现[空间效率](https://realpython.com/python-pandas-tricks/#5-use-categorical-data-to-save-on-time-and-space):

>>>

```py
>>> df.dtypes
last_name             object
first_name          category
birthday      datetime64[ns]
gender              category
type                category
state               category
party               category
dtype: object
```

您可以看到数据集的大多数列都具有类型`category`，这减少了机器上的内存负载。

### 熊猫组的`Hello, World!`by

现在您已经熟悉了数据集，您将从 pandas GroupBy 操作的`Hello, World!`开始。在数据集的整个历史中，每个州的国会议员人数是多少？在 [SQL](https://realpython.com/python-mysql/) 中，您可以通过`SELECT`语句找到答案:

```py
SELECT  state,  count(name) FROM  df GROUP  BY  state ORDER  BY  state;
```

这是熊猫的近似情况:

>>>

```py
>>> n_by_state = df.groupby("state")["last_name"].count()
>>> n_by_state.head(10)
state
AK     16
AL    206
AR    117
AS      2
AZ     48
CA    361
CO     90
CT    240
DC      2
DE     97
Name: last_name, dtype: int64
```

您调用`.groupby()`并传递您想要分组的列的名称，即`"state"`。然后，使用`["last_name"]`来指定要执行实际聚合的列。

除了将单个列名作为第一个参数传递给`.groupby()`之外，您还可以传递更多信息。您还可以指定以下任一选项:

*   一个 [`list`](https://realpython.com/python-lists-tuples/) 的多个列名
*   一只 [`dict`](https://realpython.com/python-dicts/) 或者熊猫`Series`
*   一个 [NumPy 数组](https://realpython.com/numpy-array-programming/)或者熊猫`Index`，或者一个类似数组的 iterable

下面是一个对两列进行联合分组的示例，它先按州，然后按性别查找国会议员人数:

>>>

```py
>>> df.groupby(["state", "gender"])["last_name"].count()
state  gender
AK     F           0
 M          16
AL     F           3
 M         203
AR     F           5
 ...
WI     M         196
WV     F           1
 M         119
WY     F           2
 M          38
Name: last_name, Length: 116, dtype: int64
```

类似的 [SQL 查询](http://www.sqlfiddle.com/#!17/f64b0b/1)如下所示:

```py
SELECT  state,  gender,  count(name) FROM  df GROUP  BY  state,  gender ORDER  BY  state,  gender;
```

正如您接下来将会看到的，`.groupby()`和类似的 [SQL](https://realpython.com/python-sql-libraries/) 语句是近亲，但是它们在功能上通常并不相同。

[*Remove ads*](/account/join/)

### 熊猫 GroupBy vs SQL

这是介绍 pandas GroupBy 操作和上面的 SQL 查询之间的一个显著区别的好时机。SQL 查询的结果集包含三列:

1.  `state`
2.  `gender`
3.  `count`

在 pandas 版本中，默认情况下，成组的列被推入结果`Series`的 [`MultiIndex`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html) :

>>>

```py
>>> n_by_state_gender = df.groupby(["state", "gender"])["last_name"].count()
>>> type(n_by_state_gender)
<class 'pandas.core.series.Series'>
>>> n_by_state_gender.index[:5]
MultiIndex([('AK', 'M'),
 ('AL', 'F'),
 ('AL', 'M'),
 ('AR', 'F'),
 ('AR', 'M')],
 names=['state', 'gender'])
```

为了更接近地模拟 SQL 结果并将分组后的列推回到结果中的列，可以使用`as_index=False`:

>>>

```py
>>> df.groupby(["state", "gender"], as_index=False)["last_name"].count()
 state gender  last_name
0      AK      F          0
1      AK      M         16
2      AL      F          3
3      AL      M        203
4      AR      F          5
..    ...    ...        ...
111    WI      M        196
112    WV      F          1
113    WV      M        119
114    WY      F          2
115    WY      M         38

[116 rows x 3 columns]
```

这会产生一个有三列的`DataFrame`和一个 [`RangeIndex`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.RangeIndex.html) ，而不是一个有`MultiIndex`的`Series`。简而言之，使用`as_index=False`将使您的结果更接近类似操作的默认 SQL 输出。

**注**:在`df.groupby(["state", "gender"])["last_name"].count()`中，你也可以用`.size()`代替`.count()`，因为你知道没有`NaN`姓。使用`.count()`排除`NaN`值，而`.size()`包含一切，`NaN`与否。

还要注意，上面的 SQL 查询显式地使用了`ORDER BY`，而`.groupby()`没有。那是因为`.groupby()`默认通过它的参数`sort`来做这件事，除非你另外告诉它，否则这个参数就是`True`:

>>>

```py
>>> # Don't sort results by the sort keys
>>> df.groupby("state", sort=False)["last_name"].count()
state
DE      97
VA     432
SC     251
MD     305
PA    1053
 ...
AK      16
PI      13
VI       4
GU       4
AS       2
Name: last_name, dtype: int64
```

接下来，您将深入研究`.groupby()`实际产生的对象。

### 熊猫小组如何工作

在您深入了解细节之前，先回顾一下`.groupby()`本身:

>>>

```py
>>> by_state = df.groupby("state")
>>> print(by_state)
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x107293278>
```

什么是`DataFrameGroupBy`？[打印函数](https://realpython.com/python-print/)显示的`.__str__()`值并不能给你太多关于它实际上是什么或者如何工作的信息。一个`DataFrameGroupBy`物体让你难以理解的原因是它天生懒惰。除非你告诉它，否则它不会真的做任何操作来产生有用的结果。

**注**:在本教程中，通用术语**熊猫分组对象**既指`DataFrameGroupBy`对象，也指`SeriesGroupBy`对象，它们有很多共同点。

与`.groupby()`一起经常使用的一个术语是**拆分-应用-合并**。这是指一连串的三个步骤:

1.  **将一张桌子分成几组。**
2.  **对每个较小的表应用**一些操作。
3.  **合并**结果。

检查`df.groupby("state")`可能很困难，因为它实际上什么都不做，除非你对结果对象做些什么。pandas GroupBy 对象实际上延迟了拆分-应用-合并过程的每一个部分，直到您对它调用一个方法。

所以，如果你看不到它们中的任何一个孤立地发生，你怎么能在精神上把拆分、应用和合并阶段分开呢？检查 pandas GroupBy 对象并查看拆分操作的一个有用方法是对其进行迭代:

>>>

```py
>>> for state, frame in by_state:
...     print(f"First 2 entries for {state!r}")
...     print("------------------------")
...     print(frame.head(2), end="\n\n")
...
First 2 entries for 'AK'
------------------------
 last_name first_name   birthday gender type state        party
6619    Waskey      Frank 1875-04-20      M  rep    AK     Democrat
6647      Cale     Thomas 1848-09-17      M  rep    AK  Independent

First 2 entries for 'AL'
------------------------
 last_name first_name   birthday gender type state       party
912   Crowell       John 1780-09-18      M  rep    AL  Republican
991    Walker       John 1783-08-12      M  sen    AL  Republican
```

如果您正在处理一个具有挑战性的聚合问题，那么迭代 pandas GroupBy 对象是可视化拆分-应用-合并的**拆分**部分的一个好方法。

还有一些其他的方法和属性可以让您查看单个组及其拆分。属性会给你一个`{group name: group label}`对的字典。例如，`by_state.groups`是一个以州为键的`dict`。下面是`"PA"`键的值:

>>>

```py
>>> by_state.groups["PA"]
Int64Index([    4,    19,    21,    27,    38,    57,    69,    76,    84,
 88,
 ...
 11842, 11866, 11875, 11877, 11887, 11891, 11932, 11945, 11959,
 11973],
 dtype='int64', length=1053)
```

每个值都是属于该特定组的行的索引位置序列。在上面的输出中， *4* 、 *19* 和 *21* 是`df`中状态等于`"PA"`的第一个索引。

您还可以使用`.get_group()`从单个组向下钻取到子表:

>>>

```py
>>> by_state.get_group("PA")
 last_name first_name   birthday gender type state                party
4        Clymer     George 1739-03-16      M  rep    PA                  NaN
19       Maclay    William 1737-07-20      M  sen    PA  Anti-Administration
21       Morris     Robert 1734-01-20      M  sen    PA   Pro-Administration
27      Wynkoop      Henry 1737-03-02      M  rep    PA                  NaN
38       Jacobs     Israel 1726-06-09      M  rep    PA                  NaN
...         ...        ...        ...    ...  ...   ...                  ...
11891     Brady     Robert 1945-04-07      M  rep    PA             Democrat
11932   Shuster       Bill 1961-01-10      M  rep    PA           Republican
11945   Rothfus      Keith 1962-04-25      M  rep    PA           Republican
11959  Costello       Ryan 1976-09-07      M  rep    PA           Republican
11973    Marino        Tom 1952-08-15      M  rep    PA           Republican
```

这实际上相当于使用`.loc[]`。您可以用类似于`df.loc[df["state"] == "PA"]`的东西得到相同的输出。

还值得一提的是，`.groupby()`确实通过为您传递的每个键构建一个`Grouping`类实例来完成一些(但不是全部)拆分工作。然而，保存这些分组的`BaseGrouper`类的许多方法是被延迟调用的，而不是在`.__init__()`调用的，而且许多方法还使用了缓存的属性设计。

接下来，**应用**部分呢？您可以将流程的这一步看作是对拆分阶段生成的每个子表应用相同的操作(或可调用操作)。

从熊猫 GroupBy 对象`by_state`中，你可以抓取最初的美国州和带有`next()`的`DataFrame`。当您迭代一个 pandas GroupBy 对象时，您将得到可以解包成两个变量的对:

>>>

```py
>>> state, frame = next(iter(by_state))  # First tuple from iterator
>>> state
'AK'
>>> frame.head(3)
 last_name first_name   birthday gender type state        party
6619    Waskey      Frank 1875-04-20      M  rep    AK     Democrat
6647      Cale     Thomas 1848-09-17      M  rep    AK  Independent
7442   Grigsby     George 1874-12-02      M  rep    AK          NaN
```

现在，回想一下您最初的完整操作:

>>>

```py
>>> df.groupby("state")["last_name"].count()
state
AK      16
AL     206
AR     117
AS       2
AZ      48
...
```

**应用**阶段，当应用到您的单个子集`DataFrame`时，将如下所示:

>>>

```py
>>> frame["last_name"].count()  # Count for state == 'AK'
16
```

您可以看到结果 16 与组合结果中的值`AK`相匹配。

最后一步， **combine** ，获取所有子表上所有应用操作的结果，并以直观的方式将它们组合在一起。

请继续阅读，了解更多拆分-应用-合并流程的示例。

[*Remove ads*](/account/join/)

## 示例 2:空气质量数据集

[空气质量数据集](http://archive.ics.uci.edu/ml/datasets/Air+Quality)包含来自意大利气体传感器设备的每小时读数。CSV 文件中缺失的值用 *-200* 表示。您可以使用`read_csv()`将两列合并成一个时间戳，同时使用其他列的子集:

```py
# pandas_airqual.py

import pandas as pd

df = pd.read_csv(
    "groupby-data/airqual.csv",
    parse_dates=[["Date", "Time"]],
    na_values=[-200],
    usecols=["Date", "Time", "CO(GT)", "T", "RH", "AH"]
).rename(
    columns={
        "CO(GT)": "co",
        "Date_Time": "tstamp",
        "T": "temp_c",
        "RH": "rel_hum",
        "AH": "abs_hum",
    }
).set_index("tstamp")
```

这会产生一个带有一个`DatetimeIndex`和四个`float`列的`DataFrame`:

>>>

```py
>>> from pandas_airqual import df
>>> df.head()
 co  temp_c  rel_hum  abs_hum
tstamp
2004-03-10 18:00:00  2.6    13.6     48.9    0.758
2004-03-10 19:00:00  2.0    13.3     47.7    0.726
2004-03-10 20:00:00  2.2    11.9     54.0    0.750
2004-03-10 21:00:00  2.2    11.0     60.0    0.787
2004-03-10 22:00:00  1.6    11.2     59.6    0.789
```

这里，`co`是该小时的平均[一氧化碳](https://en.wikipedia.org/wiki/Carbon_monoxide)读数，而`temp_c`、`rel_hum`和`abs_hum`分别是该小时的平均摄氏度温度、[相对湿度](https://en.wikipedia.org/wiki/Relative_humidity)和绝对湿度。观察时间从 2004 年 3 月到 2005 年 4 月:

>>>

```py
>>> df.index.min()
Timestamp('2004-03-10 18:00:00')
>>> df.index.max()
Timestamp('2005-04-04 14:00:00')
```

到目前为止，您已经通过将列名指定为`str`，比如`df.groupby("state")`，对列进行了分组。但是`.groupby()`比这灵活多了！接下来你会看到。

### 基于派生数组的分组

前面您已经看到,`.groupby()`的第一个参数可以接受几个不同的参数:

*   一列或一列
*   一只`dict`或熊猫`Series`
*   一个 NumPy 数组或 pandas `Index`，或一个类似数组的 iterable

您可以利用最后一个选项，按一周中的某一天进行分组。使用索引的`.day_name()`产生一个熊猫`Index`的字符串。以下是前十个观察结果:

>>>

```py
>>> day_names = df.index.day_name()
>>> type(day_names)
<class 'pandas.core.indexes.base.Index'>
>>> day_names[:10]
Index(['Wednesday', 'Wednesday', 'Wednesday', 'Wednesday', 'Wednesday',
 'Wednesday', 'Thursday', 'Thursday', 'Thursday', 'Thursday'],
 dtype='object', name='tstamp')
```

然后，您可以将这个对象用作`.groupby()`键。在熊猫中，`day_names`是**阵列式的**。这是一维的标签序列。

**注意**:对于熊猫`Series`，而不是`Index`，你需要`.dt`访问器来访问像`.day_name()`这样的方法。如果`ser`是你的`Series`，那么你就需要`ser.dt.day_name()`。

现在，将该对象传递给`.groupby()`以查找一周中每一天的平均一氧化碳(`co`)读数:

>>>

```py
>>> df.groupby(day_names)["co"].mean()
tstamp
Friday       2.543
Monday       2.017
Saturday     1.861
Sunday       1.438
Thursday     2.456
Tuesday      2.382
Wednesday    2.401
Name: co, dtype: float64
```

split-apply-combine 过程的行为与之前基本相同，只是这次的拆分是在人工创建的列上完成的。该列不存在于 DataFrame 本身中，而是从它派生出来的。

如果您不仅想按一周中的某一天分组，还想按一天中的某个小时分组，那该怎么办？那个结果应该有`7 * 24 = 168`个观察值。为此，您可以传递一个类似数组的对象列表。在这种情况下，您将传递熊猫`Int64Index`对象:

>>>

```py
>>> hr = df.index.hour
>>> df.groupby([day_names, hr])["co"].mean().rename_axis(["dow", "hr"])
dow        hr
Friday     0     1.936
 1     1.609
 2     1.172
 3     0.887
 4     0.823
 ...
Wednesday  19    4.147
 20    3.845
 21    2.898
 22    2.102
 23    1.938
Name: co, Length: 168, dtype: float64
```

这里还有一个类似的例子，使用 [`.cut()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) 将温度值绑定到离散的区间:

>>>

```py
>>> import pandas as pd
>>> bins = pd.cut(df["temp_c"], bins=3, labels=("cool", "warm", "hot"))
>>> df[["rel_hum", "abs_hum"]].groupby(bins).agg(["mean", "median"])
 rel_hum        abs_hum
 mean median    mean median
temp_c
cool    57.651   59.2   0.666  0.658
warm    49.383   49.3   1.183  1.145P
hot     24.994   24.1   1.293  1.274
```

在这种情况下，`bins`实际上是一个`Series`:

>>>

```py
>>> type(bins)
<class 'pandas.core.series.Series'>
>>> bins.head()
tstamp
2004-03-10 18:00:00    cool
2004-03-10 19:00:00    cool
2004-03-10 20:00:00    cool
2004-03-10 21:00:00    cool
2004-03-10 22:00:00    cool
Name: temp_c, dtype: category
Categories (3, object): [cool < warm < hot]
```

不管是一个`Series`、NumPy 数组还是 list 都没关系。重要的是，`bins`仍然作为一个标签序列，由`cool`、`warm`和`hot`组成。如果你真的想，那么你也可以使用一个`Categorical`数组，甚至是一个普通的`list`:

*   **原生 Python 列表:** `df.groupby(bins.tolist())`
*   **熊猫`Categorical`阵列:**T1】

如你所见，`.groupby()`很聪明，可以处理很多不同的输入类型。其中任何一个都会产生相同的结果，因为它们都作为一个标签序列来执行分组和拆分。

[*Remove ads*](/account/join/)

### 重新采样

您已经将`df`与`df.groupby(day_names)["co"].mean()`按一周中的某一天分组。现在考虑一些不同的东西。如果您想按观察的年份和季度分组，该怎么办？有一种方法可以做到这一点:

>>>

```py
>>> # See an easier alternative below
>>> df.groupby([df.index.year, df.index.quarter])["co"].agg(
...     ["max", "min"]
... ).rename_axis(["year", "quarter"])
 max  min
year quarter
2004 1         8.1  0.3
 2         7.3  0.1
 3         7.5  0.1
 4        11.9  0.1
2005 1         8.7  0.1
 2         5.0  0.3
```

或者，整个操作可以通过**重采样**来表达。重采样的用途之一是作为基于时间的分组依据[。你所需要做的就是传递一个](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling)[频率串](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)，比如`"quarterly"`的`"Q"`，熊猫会做剩下的事情:

>>>

```py
>>> df.resample("Q")["co"].agg(["max", "min"])
 max  min
tstamp
2004-03-31   8.1  0.3
2004-06-30   7.3  0.1
2004-09-30   7.5  0.1
2004-12-31  11.9  0.1
2005-03-31   8.7  0.1
2005-06-30   5.0  0.3
```

通常，当您使用`.resample()`时，您可以用更简洁的方式表达基于时间的分组操作。结果可能与更冗长的`.groupby()`稍有不同，但是你会经常发现`.resample()`给了你想要的东西。

## 示例 3:新闻聚合数据集

现在，您将使用第三个也是最后一个数据集，该数据集包含几十万篇新闻文章的元数据，并将它们分组到主题簇中:

```py
# pandas_news.py

import pandas as pd

def parse_millisecond_timestamp(ts):
    """Convert ms since Unix epoch to UTC datetime instance."""
    return pd.to_datetime(ts, unit="ms")

df = pd.read_csv(
    "groupby-data/news.csv",
    sep="\t",
    header=None,
    index_col=0,
    names=["title", "url", "outlet", "category", "cluster", "host", "tstamp"],
    parse_dates=["tstamp"],
    date_parser=parse_millisecond_timestamp,
    dtype={
        "outlet": "category",
        "category": "category",
        "cluster": "category",
        "host": "category",
    },
)
```

要使用适当的`dtype`将数据读入内存，您需要一个助手函数来解析时间戳列。这是因为它被表示为自 **Unix 纪元**以来的毫秒数，而不是分数秒。如果您想了解更多关于使用 Python 处理时间的信息，请查看使用 Python datetime 处理日期和时间的。

与您之前所做的类似，您可以使用分类`dtype`来有效地编码那些相对于列长度来说唯一值数量相对较少的列。

数据集的每一行都包含标题、URL、发布出口的名称和域，以及发布时间戳。`cluster`是文章所属主题簇的随机 ID。`category`是新闻类别，包含以下选项:

*   `b`因公出差
*   `t`对于科技
*   `e`为了娱乐
*   `m`为了健康

这是第一行:

>>>

```py
>>> from pandas_news import df
>>> df.iloc[0]
title       Fed official says weak data caused by weather,...
url         http://www.latimes.com/business/money/la-fi-mo...
outlet                                      Los Angeles Times
category                                                    b
cluster                         ddUyU0VZz0BRneMioxUPQVP6sIxvM
host                                          www.latimes.com
tstamp                             2014-03-10 16:52:50.698000
Name: 1, dtype: object
```

现在，您已经对数据有了初步的了解，您可以开始询问关于它的更复杂的问题。

### 在`.groupby()`中使用λ函数

这个数据集引发了更多潜在的问题。这里有一个随机但有意义的问题:哪些渠道谈论最多的是[美联储](https://en.wikipedia.org/wiki/Federal_Reserve)？为简单起见，假设这需要搜索对`"Fed"`的区分大小写的提及。请记住，这可能会对像`"Federal government"`这样的术语产生一些误报。

要按 outlet 统计提及次数，您可以在 outlet 上调用`.groupby()`，然后使用 [Python lambda 函数](https://realpython.com/python-lambda/)在每个组上调用`.apply()`函数:

>>>

```py
>>> df.groupby("outlet", sort=False)["title"].apply(
...     lambda ser: ser.str.contains("Fed").sum()
... ).nlargest(10)
outlet
Reuters                         161
NASDAQ                          103
Businessweek                     93
Investing.com                    66
Wall Street Journal \(blog\)     61
MarketWatch                      56
Moneynews                        55
Bloomberg                        53
GlobalPost                       51
Economic Times                   44
Name: title, dtype: int64
```

让我们分解一下，因为有几个连续的方法调用。像以前一样，您可以通过从 pandas GroupBy 迭代器中取出第一个`tuple`来取出第一个组及其对应的 pandas 对象:

>>>

```py
>>> title, ser = next(iter(df.groupby("outlet", sort=False)["title"]))
>>> title
'Los Angeles Times'
>>> ser.head()
1       Fed official says weak data caused by weather,...
486            Stocks fall on discouraging news from Asia
1124    Clues to Genghis Khan's rise, written in the r...
1146    Elephants distinguish human voices by sex, age...
1237    Honda splits Acura into its own division to re...
Name: title, dtype: object
```

在这种情况下，`ser`是一只熊猫`Series`而不是一只`DataFrame`。那是因为你跟`["title"]`跟进了`.groupby()`电话。这有效地从每个子表中选择了单个列。

接下来是`.str.contains("Fed")`。当一篇文章标题在搜索中注册了一个匹配时，这将返回一个[布尔值](https://realpython.com/python-boolean/) `Series`即`True`。果然，第一排从`"Fed official says weak data caused by weather,..."`开始，亮为`True`:

>>>

```py
>>> ser.str.contains("Fed")
1          True
486       False
1124      False
1146      False
1237      False
 ...
421547    False
421584    False
421972    False
422226    False
422905    False
Name: title, Length: 1976, dtype: bool
```

接下来就是`.sum()`这个`Series`了。由于`bool`在技术上只是`int`的一种特殊类型，你可以对`True`和`False`的`Series`求和，就像你对`1`和`0`的序列求和一样:

>>>

```py
>>> ser.str.contains("Fed").sum()
17
```

结果是《洛杉矶时报》在数据集中对`"Fed"`的提及次数。同样的惯例也适用于路透社、纳斯达克、商业周刊和其他公司。

[*Remove ads*](/account/join/)

### 提高`.groupby()`的性能

现在再次回溯到`.groupby().apply()`来看看为什么这个模式可能是次优的。要获得一些背景信息，请查看[如何加快你的熊猫项目](https://realpython.com/fast-flexible-pandas/#pandas-apply)。使用`.apply()`可能会发生的事情是，它将有效地对每个组执行 Python 循环。虽然`.groupby().apply()`模式可以提供一些灵活性，但它也可以阻止 pandas 使用其基于 [Cython 的优化](https://realpython.com/python-bindings-overview/#cython)。

也就是说，每当你发现自己在考虑使用`.apply()`时，问问自己是否有办法用[向量化](https://realpython.com/numpy-array-programming/)的方式来表达操作。在这种情况下，您可以利用这样一个事实，即`.groupby()`不仅接受一个或多个列名，还接受许多类似**数组的**结构:

*   一维 NumPy 数组
*   一份名单
*   一只熊猫`Series`还是`Index`

还要注意的是，`.groupby()`对于`Series`来说是一个有效的[实例方法](https://realpython.com/instance-class-and-static-methods-demystified/)，而不仅仅是一个`DataFrame`，所以你可以从本质上颠倒分割逻辑。考虑到这一点，您可以首先构造一个布尔值`Series`，它指示标题是否包含`"Fed"`:

>>>

```py
>>> mentions_fed = df["title"].str.contains("Fed")
>>> type(mentions_fed)
<class 'pandas.core.series.Series'>
```

现在，`.groupby()`也是`Series`的一个方法，所以你可以将一个`Series`分组到另一个上:

>>>

```py
>>> import numpy as np
>>> mentions_fed.groupby(
...     df["outlet"], sort=False
... ).sum().nlargest(10).astype(np.uintc)
outlet
Reuters                         161
NASDAQ                          103
Businessweek                     93
Investing.com                    66
Wall Street Journal \(blog\)     61
MarketWatch                      56
Moneynews                        55
Bloomberg                        53
GlobalPost                       51
Economic Times                   44
Name: title, dtype: uint32
```

这两个`Series`不必是同一个`DataFrame`对象的列。它们只需要有相同的形状:

>>>

```py
>>> mentions_fed.shape
(422419,)
>>> df["outlet"].shape
(422419,)
```

最后，如果您决定尽可能获得最紧凑的结果，可以用`np.uintc`将结果转换回无符号整数。下面是两个版本的直接比较，会产生相同的结果:

```py
# pandas_news_performance.py

import timeit
import numpy as np

from pandas_news import df

def test_apply():
    """Version 1: using `.apply()`"""
    df.groupby("outlet", sort=False)["title"].apply(
        lambda ser: ser.str.contains("Fed").sum()
    ).nlargest(10)

def test_vectorization():
    """Version 2: using vectorization"""
    mentions_fed = df["title"].str.contains("Fed")
    mentions_fed.groupby(
        df["outlet"], sort=False
    ).sum().nlargest(10).astype(np.uintc)

print(f"Version 1: {timeit.timeit(test_apply, number=3)}")
print(f"Version 2: {timeit.timeit(test_vectorization, number=3)}")
```

您使用`timeit`模块来估计两个版本的运行时间。如果你想了解更多关于测试代码性能的知识，那么 [Python Timer Functions:三种监控代码的方法](https://realpython.com/python-timer/)值得一读。

现在，运行脚本，看看两个版本的性能如何:

```py
(venv) $ python pandas_news_performance.py
Version 1: 2.5422707499965327
Version 2: 0.3260433749965159
```

运行三次时，`test_apply()`函数耗时 2.54 秒，而`test_vectorization()`仅需 0.33 秒。对于几十万行来说，这是一个令人印象深刻的 CPU 时间差异。想想当您的数据集增长到几百万行时，这种差异会变得多么显著！

**注意**:为了简单起见，这个例子忽略了数据中的一些细节。也就是说，搜索词`"Fed"`也可能找到类似`"Federal government"`的内容。

如果你想使用一个包含[负前瞻](https://docs.python.org/3/library/re.html#index-21)的表达式，那么`Series.str.contains()`也可以把一个编译过的[正则表达式](https://realpython.com/regex-python/)作为参数。

你可能还想计算的不仅仅是原始的被提及次数，而是被提及次数相对于一家新闻机构发表的所有文章的比例。

## 熊猫小组:把所有的放在一起

如果你在一个 pandas GroupBy 对象上调用`dir()`,那么你会看到足够多的方法让你眼花缭乱！很难跟踪熊猫 GroupBy 对象的所有功能。一种拨开迷雾的方法是将不同的方法划分为它们做什么和它们的行为方式。

概括地说，pandas GroupBy 对象的方法分为几类:

1.  **聚合方法**(也称为**缩减方法**)将许多数据点组合成关于这些数据点的聚合统计。一个例子是取十个数字的和、平均值或中值，结果只是一个数字。

2.  **过滤方法**带着原始`DataFrame`的子集回来给你。这通常意味着使用`.filter()`来删除基于该组及其子表的一些比较统计数据的整个组。在这个定义下包含许多从每个组中排除特定行的方法也是有意义的。

3.  **转换方法**返回一个`DataFrame`，其形状和索引与原始值相同，但值不同。使用聚合和过滤方法，得到的`DataFrame`通常比输入的`DataFrame`小。这对于转换来说是不正确的，它转换单个的值本身，但是保留原始的`DataFrame`的形状。

4.  **元方法**不太关心你调用`.groupby()`的原始对象，更关注于给你高层次的信息，比如组的数量和那些组的索引。

5.  **的剧情方法**模仿了[为一只熊猫`Series`或`DataFrame`](https://realpython.com/pandas-plot-python/) 剧情的 API，但通常会将输出分成多个支线剧情。

官方文档对这些类别有自己的解释。在某种程度上，它们可以有不同的解释，本教程在对哪种方法进行分类时可能会有细微的不同。

有几个熊猫分组的方法不能很好地归入上面的类别。这些方法通常产生一个不是`DataFrame`或`Series`的中间对象。例如，`df.groupby().rolling()`产生了一个`RollingGroupby`对象，然后您可以在其上调用[聚合、过滤或转换方法](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#new-syntax-to-window-and-resample-operations)。

如果您想更深入地研究，那么 [`DataFrame.groupby()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby) 、 [`DataFrame.resample()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html) 和 [`pandas.Grouper`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Grouper.html) 的 API 文档是探索方法和对象的资源。

在 pandas 文档中还有另一个[单独的表](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html)，它有自己的分类方案。选择最适合你并且看起来最直观的！

[*Remove ads*](/account/join/)

## 结论

在本教程中，您已经介绍了大量关于 **`.groupby()`** 的内容，包括它的设计、它的 API，以及如何将方法链接在一起以将数据转化为适合您的目的的结构。

**你已经学会:**

*   如何通过对**真实世界数据**的操作来使用熊猫**分组**
*   **分割-应用-组合**操作链是如何工作的，以及如何将它分解成步骤
*   如何根据它们的意图和结果对熊猫分组的**方法**进行**分类**

`.groupby()`的内容比你在一个教程中能涵盖的要多得多。但是希望这篇教程是进一步探索的良好起点！

您可以点击下面的链接下载本教程中所有示例的源代码:

**下载数据集:** [点击这里下载数据集，你将在本教程中使用](https://realpython.com/bonus/pandas-groupby/)来了解熊猫的分组。*********
# 熊猫小组讲解:熊猫小组使用指南

> 原文：<https://www.pythoncentral.io/pandas-groupby-explained-a-guide-on-using-pandas-groupby/>

Python 是一个优秀的数据分析工具，因为它拥有丰富的以数据为中心的包生态系统。Pandas 是最受欢迎的软件包之一，它简化了数据的导入和分析。

在本指南中，我们将讨论。groupby()方法使用 split-apply-combine 以及如何访问组和转换数据。

## 什么是熊猫分组法？

Pandas 中的 GroupBy 方法旨在模拟 SQL 中 GROUP BY 语句的功能。Pandas 方法的工作方式类似于 SQL 语句，先拆分数据，然后按照指示聚合数据，最后再组合数据。

自。groupby()首先分割数据，Python 用户可以直接处理组。此外，聚合是在拆分后完成的，让用户可以完全控制聚合方式。

## **加载熊猫数据帧**

为了理解如何使用 GroupBy 方法，首先加载一个样本数据集。您可以通过粘贴下面使用。read_csv()方法加载数据:

```py
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/sales.csv', parse_dates=['date'])
```

在了解 GroupBy 方法的对象如何工作之前，让我们使用。头()方法:

```py
print(df.head())
```

你会注意到有一个日期栏显示交易日期。性别和地区列是保存销售人员数据的字符串类型列。sales 列表示各自的销售额。

## **熊猫如何按对象分组**

创建一个 GroupBy 对象就像对 DataFrame 应用该方法一样简单。您可以传递单个列或一组列。这看起来是这样的:

```py
print(df.groupby('region'))

# Output: <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fb78815a4f0>
```

如你所见，它返回了一个 DataFrameGroupBy 对象，由于它是一个对象，我们可以探究它的一些属性。

### **group by 对象的属性**

这些对象带有一个. ngroups 属性，存储分组中的组数。下面是计算对象中的组的方法:

```py
print(df.groupby('region').ngroups)

# Output: 3
```

同样，你可以使用。属性来查找有关组的更多详细信息。属性输出类似字典的对象。对象将组作为键保存，这些键的值是组中行的索引。

要访问对象中的组，可以运行:

```py
print(df.groupby('region').groups) 
```

但是如果你只想找到对象的组名，你可以只返回字典的键，就像这样:

```py
print(df.groupby('region').groups.keys())

# Output: dict_keys(['North-East', 'North-West', 'South'])
```

### **逐组选择熊猫**

group by 方法允许您选择特定组中的所有记录。当您想了解一个组中的数据时，这是非常有用的。您可以通过使用。get_group()方法并传递组名。

例如，运行下面的语句将显示我们的样本数据集中“South”地区的数据:

```py
print(df.groupby('region').get_group('South'))
```

## **通过拆分-应用-组合了解熊猫群体**

GroupBy 方法利用一种称为分割-应用-组合的过程来提供数据帧的有用修改或聚合。这个过程是不言自明的，如下所示:

1.  根据某些标准将数据分成组
2.  功能独立应用于每组
3.  结果被组合成一个数据结构

前面几节将带您了解如何使用？groupby()方法根据传递的标准将数据分组。因此，您已经熟悉了第一步。

分割数据背后的想法是将大数据分析问题分解成更小、更可行的部分。当你需要处理的问题比较小的时候，在把零件重新组装起来之前，对零件进行操作会比较容易。

虽然应用和合并步骤是不同的，并且是分开运行的，但是这个库让这两个步骤看起来像是一次完成的。

### **用 GroupBy 聚合数据**

运行以下代码将聚合数据:

```py
averages = df.groupby('region')['sales'].mean()

print(averages)

# Output:
# region
# North-East    17386.072046
# North-West    15257.732919
# South         24466.864048
# Name: sales, dtype: float64

```

让我们分解这段代码来理解它是如何工作的:

1.  df.groupby('region ')根据 region 列将数据分成组。
2.  ['sales']仅从组中选取区域列。
3.  The。mean()将 mean 方法应用于每个组中的列。
4.  数据然后被组合成最终的数据帧。

### **具有 GroupBy 的其他聚合**

现在您已经了解了拆分-应用-合并流程，下面是各种可用聚合函数的概述:

| **聚合方法** | **描述** |
| 。count() | 非空记录的数量 |
| 。max() | 组的最大值 |
| 。平均值() | 数值的算术平均值 |
| 。中位数() | 数值的中间值 |
| 。min() | 组的最小值 |
| 。模式() | 组中最频繁出现的值 |
| 。std() | 组的标准偏差 |
| 。sum() | 数值总和 |
| 【t0 . var()】 | 组的方差 |

您可以随意使用这些方法来处理数据。例如，如果您想计算每组的标准偏差，下面的代码可以做到:

```py
standard_deviations = df.groupby('region')['sales'].std()

print(standard_deviations)
```

### **应用多个聚合**

Pandas 库最强大的特性之一是它允许你通过。agg()方法。使用这种方法，可以将可调用列表传递给 GroupBy。下面是如何使用？agg()函数:

```py
import numpy as np

aggs = df.groupby('region')['sales'].agg([np.mean, np.std, np.var])

print(aggs)

# Output:
#                     mean          std           var
# region                                             
# North-East  17386.072046  2032.541552  4.131225e+06
# North-West  15257.732919  3621.456493  1.311495e+07
# South       24466.864048  5253.702513  2.760139e+07
```

The。agg()函数使您能够基于不同的组生成汇总统计数据。函数使得处理数据变得很方便，因为不需要使用。groupby()方法三次以获得相同的结果。

## **用分组方式转换数据**

GroupBy 方法也使用户可以很容易地转换数据。简单地说，转换数据意味着执行特定于该组的操作。

这可能包括通过基于组分配值和使用 z 值标准化数据来处理缺失数据。

但是，这种转换与聚合和过滤有什么不同呢？与聚合和过滤过程不同，输出数据帧在转换后将始终具有与原始数据相同的维度。对于聚合和筛选来说，情况并非总是如此。

使用。transform()方法为原始数据集中的每条记录返回一个值。这保证了结果将是相同的大小。

### **利用。变换()**

了解如何。transform()使用一个例子会更容易。假设您想计算一个地区总销售额的百分比。您可以传递“sum”可调用函数，并将该组的总和返回到每一行。然后，您可以将原始销售列除以总和，如下所示:

```py
df['Percent Of Region Sales'] = df['sales'] / df.groupby('region')['sales'].transform('sum')

print(df.head())
```

有趣的是，您还可以在 GroupBy 中转换数据，而无需使用。transform()方法。您可以应用返回单个值而不聚合数据的函数。

例如，如果你应用。rank()方法，它将对每个组中的值进行排序:

```py
df['ranked'] = df.groupby('region')['sales'].rank(ascending=False)

print(df.sort_values(by='sales', ascending=False).head())
```

运行这段代码将返回一个与原始数据帧长度相同的 Pandas 系列。然后，您可以将该系列分配给新列。

## **按分组过滤数据**

过滤数据帧是使用。groupby()方法。但重要的是要知道这种方法不同于常规的筛选，因为 GroupBy 允许您应用基于组值聚合的筛选方法。

例如，您可以过滤您的数据框架，去掉一个组的平均销售价格低于 20，000 的行:

```py
df = df.groupby('region').filter(lambda x: x['sales'].mean() < 20000)

print(df.head())
```

下面是这段代码的工作原理:

1.  首先，代码根据“区域”列对数据进行分组。
2.  接下来，该。filter()方法根据您传递的 lambda 函数过滤数据。
3.  最后，如果“销售”列组中的平均值低于 20，000，lambda 函数就会计算出来。

以这种方式过滤数据，您就不需要在过滤出这些值之前确定每组的平均值。虽然在这个例子中，这种方式的过滤似乎是不必要的，但是在处理较小的组时，这种方式是非常宝贵的。

## **通过多列对数据帧进行分组**

通过按多列对数据进行分组，您可以探索 GroupBy 方法的其他功能。在上面的所有例子中，我们传递一个表示单个列的字符串，并根据该列对 DataFrame 进行分组。

然而，GroupBy 也允许你传递一个字符串列表，每个字符串代表不同的列。这样，您可以进一步拆分数据。

让我们通过计算按“地区”和“性别”分组的所有销售额的总和来探索 GroupBy 的这一功能:

```py
sums = df.groupby(['region', 'gender']).sum()

print(sums.head())
```

更有趣的是，这篇文章中提到的所有方法都可以以这种方式使用。您可以应用。rank()函数再次确定每个地区和性别组合的最高销售额，如下:

```py
df['rank'] = df.groupby(['region', 'gender'])['sales'].rank(ascending=False)

print(df.head())
```

## **通过 GroupBy 使用自定义函数**

GroupBy 方法最好的特性之一是你可以应用你自己的函数来处理数据。您可以选择使用匿名 lambda 函数，但也可以根据您的分析需要定义特定的函数。

让我们通过定义一个自定义函数来理解这是如何工作的，该函数通过计算最小值和最大值之间的差值来返回一个组的范围。以下是在将这样的函数应用到您的。groupby()方法调用:

```py
def group_range(x):
    return x.max() - x.min()

ranges = df.groupby(['region', 'gender'])['sales'].apply(group_range)
print(ranges)

# Output:
# region      gender
# North-East  Female    10881
#             Male      10352
# North-West  Female    20410
#             Male      17469
# South       Female    30835
#             Male      27110
# Name: sales, dtype: int64
```

上面定义的 group_range()函数接受一个参数，在我们的例子中，这个参数是一系列“销售”分组。接下来，代码在返回最小值和最大值之间的差之前先找到这两个值。这样，它有助于我们看到组范围如何不同。

## **结论**

GroupBy 函数简单易用，是最好的数据分析方法之一。使用该方法时需要记住的一些实用技巧包括:

1.  当传递多个组关键字时，只有具有相同组关键字值的行彼此匹配才会被添加到组中。
2.  将排序参数设置为 False 可以提高代码执行时间。
3.  将分组步骤链接起来并应用一个函数可以减少代码行。

现在您已经了解了如何使用 GroupBy，是时候自己测试一下了，并享受快速简单的数据分析带来的好处。
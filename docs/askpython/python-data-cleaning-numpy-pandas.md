# 使用 NumPy 和 Pandas 清理 Python 数据

> 原文：<https://www.askpython.com/python/examples/python-data-cleaning-numpy-pandas>

Python 数据清理是用一些默认值替换空值、删除不需要的列、删除缺失的行等的过程。当处理大量原始数据时，在分析之前清理数据是有意义的，这样我们就可以处理完整的数据集。

## Python 中的数据清理

Python NumPy 和 Pandas 模块为 Python 中的数据清理提供了一些方法。数据清理是指通过更新或删除缺失、不准确、格式不正确、重复或不相关的信息来清理需要传递到数据库或用于数据分析的所有数据的过程。应该定期进行定期数据清理，以避免多年来堆积未清理的数据。

## 为什么我们需要清理 Python 中的数据？

如果数据清理不当，可能会导致巨大的损失，包括降低营销效率。因此，清理数据对于避免主要结果中的所有不准确性变得非常重要。

高效的数据清理意味着更少的错误，从而导致更满意的客户和更少沮丧的员工。它还能提高生产力，做出更好的决策。

## 清理 Python 数据集中的数据的步骤

### 1.数据加载

现在让我们对我从网上下载的一个随机的`csv`文件进行数据清理。数据集的名称是“旧金山建筑许可”。在对数据进行任何处理之前，首先从文件中加载数据。数据加载的代码如下所示:

```py
import numpy as np
import pandas as pd
data = pd.read_csv('Building_Permits.csv',low_memory=False)

```

首先，导入所有需要的模块，然后加载 CSV 文件。我添加了一个名为`low_memory`的额外参数，其目的是确保程序不会因为庞大的数据集而出现任何内存错误。

数据集包含 198900 个许可细节和 43 列。数据集中的列如下:

1.  许可证号码
2.  许可类型
3.  许可类型定义
4.  许可证创建日期
5.  街区
6.  大量
7.  街道号码
8.  街道号码后缀
9.  街道名称
10.  街道后缀
11.  单位
12.  单位后缀
13.  描述
14.  当前状态
15.  当前状态日期
16.  字段日期
17.  签发日期
18.  完成日期
19.  首次施工文件日期
20.  结构通知
21.  现有故事的数量
22.  提议故事的数量
23.  自愿软层改造
24.  只允许用火
25.  许可证到期日期
26.  预算造价
27.  修订成本
28.  现有用途
29.  现有单位
30.  拟议用途
31.  提议的单位
32.  计划集
33.  TIDF 合规
34.  现有建筑类型
35.  现有建筑类型描述
36.  提议的结构类型
37.  建议的结构类型描述
38.  工地许可证
39.  主管区
40.  邻域-分析边界
41.  邮政编码
42.  位置
43.  记录 ID

### 2.删除不必要的列

当我们查看数据集时，我们看到数据集中有如此多的列。但是对于处理，我们可以在处理过程中跳过一些列。

现在，让我们删除一些随机的列，即 TIDF 合规性、仅火灾许可、单位后缀、区块和批次。

```py
columns_to_drop=['TIDF Compliance', 'Fire Only Permit', 'Unit Suffix', 'Block','Lot']
data_dropcol=data.drop(columns_to_drop,axis=1)

```

我们将首先创建一个列表，存储要从数据集中删除的所有列名。

在下一行中，我们使用了 drop 函数，并将创建的列表传递给该函数。我们还传递轴参数，该参数的值可以是 0(按行放置)或 1(按列放置)。

代码执行后，新数据只包含 38 列，而不是 43 列。

### 3.删除缺少的值行

在直接移动到[删除带有缺失值的行](https://www.askpython.com/python/examples/impute-missing-data-values)之前，让我们首先分析数据集中有多少缺失值。出于同样的目的，我们使用下面提到的代码。

```py
no_missing = data_dropcol.isnull().sum()
total_missing=no_missing.sum()

```

在代码执行时，我们发现数据集中有 1670031 个丢失的值。因为丢失的值太多了，所以我们不是删除丢失数据的行，而是删除丢失值最多的列。相同的代码如下所示。

```py
drop_miss_value=data_dropcol.dropna(axis=1)

```

该代码导致删除了最大列数，结果数据集中只剩下 10 列。是的，大部分信息都从数据集中删除了，但至少现在数据集得到了充分的清理。

## 摘要

数据分析是一项资源密集型操作。因此，在分析之前清理原始数据以节省时间和精力是有意义的。数据清理也确保了我们的分析更加准确。Python pandas 和 NumPy 模块最适合 CSV 数据清理。

## 下一步是什么？

*   [蟒蛇熊猫](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)
*   [NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-module)
*   [Python CSV 模块](https://www.askpython.com/python-modules/python-csv-module)
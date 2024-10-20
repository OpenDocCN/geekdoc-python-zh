# 如何在 Python Pandas 中使用 drop_duplicates()函数删除重复项

> 原文：<https://pythonguides.com/python-pandas-drop-duplicates/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在本 [Python 教程](https://pythonguides.com/learn-python/)中，我们将学习如何在 python pandas 中使用 `drop_duplicates()` 函数删除重复项。本博客中使用的数据集要么是自己创建的，要么是从 [kaggle 下载的。](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)此外，我们还将讨论这些话题。

*   蟒蛇熊猫掉落复制品
*   熊猫基于列删除重复项
*   蟒蛇熊猫把复制品放在最后
*   熊猫滴重复多列
*   Python 熊猫丢弃重复子集
*   Python 熊猫删除重复索引
*   Python 熊猫掉落副本不起作用
*   Python 熊猫从列表中删除重复项
*   Python 熊猫删除重复项区分大小写

如果你是 Python 熊猫的新手，可以看看一篇关于 Python 中的熊猫的文章。

目录

[](#)

*   [蟒蛇熊猫掉落复制品](#Python_pandas_drop_duplicates "Python pandas drop duplicates")
*   [Python 熊猫基于列](#Python_Pandas_drop_duplicates_based_on_column "Python Pandas drop duplicates based on column")丢弃副本
*   [蟒蛇熊猫掉落复制品留到最后](#Python_pandas_drop_duplicates_keep_last "Python pandas drop duplicates keep last")
*   [熊猫滴重复多栏](#Pandas_drop_duplicates_multiple_columns "Pandas drop duplicates multiple columns")
*   [Python 熊猫掉落副本子集](#Python_pandas_drop_duplicates_subset "Python pandas drop duplicates subset")
*   [Python 熊猫掉重复索引](#Python_pandas_drop_duplicates_index "Python pandas drop duplicates index")
*   [熊猫从列表中删除重复项](#Pandas_drop_duplicates_from_list "Pandas drop duplicates from list")
*   [Python pandas drop duplicates 区分大小写](#Python_pandas_drop_duplicates_case_sensitive "Python pandas drop duplicates case sensitive")

## 蟒蛇熊猫掉落复制品

*   在这一节中，我们将学习如何在 python pandas 中使用 `drop_duplicates()` 函数删除副本。
*   当使用数据集时，有时情况只要求唯一的条目，这时我们必须从数据集中删除重复的值。
*   删除重复是数据清理的一部分。 `drop_duplicates()` 函数允许我们从整个数据集或特定列中删除重复值

**语法:**

下面是 **drop_duplicates()的语法。**语法被分成几个部分来解释函数的潜力。

从整个数据集中移除重复项

```py
df.drop_duplicates()
```

**子集**用于删除特定列中的重复项

```py
df.drop_duplicates(subset='column_name')
```

在子集中传递列名列表，以删除多个列中的重复项

```py
df.drop_duplicates(subset=['column1', 'column2', 'column3'])
```

**保留**选项设置为'**最后'**,删除重复项，仅保留最后出现的项目

```py
df.drop_duplicates(subset='column_name', keep='last')
```

**保留**选项设置为**‘第一个’**，删除重复项，仅保留第一个出现项

```py
df.drop_duplicates(subset='column_name', keep='first')
```

**保持**选项被设置为**假**以移除重复列的所有出现

```py
df.drop_duplicates(subset='column_name', keep=False)
```

下面是对 `drop_duplicates()` 函数可用选项的详细描述。

| 选择 | 说明 |
| --- | --- |
| **子集** | **列标签或序列标签，可选**
列标签某些列用于标识重复项，默认情况下使用所有的列。 |
| **保持** | **{'first '，' last '，False}，默认为' first'**
确定要保留哪些重复项(如果有)。
**–第一:**删除除第一次出现的重复项
**–最后:**删除除最后一次出现的重复项。
**–False:**删除所有重复项。 |
| **原地** | **bool，默认 False**
是将副本放回原处还是返回副本。 |
| **忽略 _ 索引** | **bool，default false**
如果为真，则生成的轴将被标记为 0，1，2…n-1。 |

也可能你喜欢， [Python 熊猫 CSV 教程](https://pythonguides.com/python-pandas-csv/)。

## Python 熊猫基于列丢弃副本

*   在这一节中，我们将学习如何在 Python Pandas 中删除基于列的副本。
*   要删除重复的列，我们必须在子集中提供列名。

**语法**:

在此语法中，我们从名为' column_name '的单个列中删除重复项

```py
df.drop_duplicates(subset='column_name')
```

下面是 jupyter 笔记本上基于列删除重复项的实现。
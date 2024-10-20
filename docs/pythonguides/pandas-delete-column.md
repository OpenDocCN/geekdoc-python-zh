# 熊猫删除栏

> 原文：<https://pythonguides.com/pandas-delete-column/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 [Python 熊猫教程](https://pythonguides.com/pandas-in-python/)中，我们将讨论关于**熊猫删除列**和**如何使用熊猫在数据帧**中删除列的一切。

*   熊猫删除列数据框
*   熊猫按名字删除栏目
*   熊猫按索引删除列
*   如果存在，熊猫删除列
*   熊猫按条件删除列
*   熊猫删除带 NaN 的列
*   熊猫删除列，如果所有南
*   熊猫按位置删除列
*   熊猫删除没有名字的专栏
*   熊猫删除列标题
*   熊猫删除栏目除外
*   在熊猫数据框架中删除第一列
*   删除熊猫数据框中的最后一列
*   在 Pandas 数据框架中删除多列
*   删除 Pandas 数据框架中的重复列
*   删除熊猫数据框架中的第一列
*   删除 Pandas 数据框架中的列名
*   落柱熊猫系列
*   在熊猫数据框架中删除一列
*   熊猫数据框架中的下拉列表

我们使用了从 Kaggle 下载的[电动汽车数据集](https://www.kaggle.com/geoffnel/evs-one-electric-vehicle-dataset)。

目录

[](#)

*   [熊猫删除列数据框](#Pandas_Delete_Column_DataFrame "Pandas Delete Column DataFrame")
*   [熊猫删除列名](#Pandas_Delete_Column_by_Name "Pandas Delete Column by Name")
*   [熊猫按索引删除列](#Pandas_Delete_Column_by_Index "Pandas Delete Column by Index")
*   [熊猫删除列如果存在](#Pandas_Delete_Column_if_Exists "Pandas Delete Column if Exists")
*   [熊猫按条件删除列](#Pandas_Delete_Column_by_Condition "Pandas Delete Column by Condition")
*   [熊猫删除带 NaN 的栏目](#Pandas_Delete_Columns_with_NaN "Pandas Delete Columns with NaN")
*   [熊猫删除列如果所有楠](#Pandas_Delete_Column_if_all_nan "Pandas Delete Column if all nan")
*   [熊猫按位置删除列](#Pandas_Delete_Column_by_Position "Pandas Delete Column by Position")
*   [熊猫删除没有名字的栏目](#Pandas_Delete_Column_with_no_Name "Pandas Delete Column with no Name")
*   [熊猫删除列标题](#Pandas_Delete_Column_Header "Pandas Delete Column Header")
*   [熊猫删除除](#Pandas_Delete_Columns_Except "Pandas Delete Columns Except")外的栏目
*   [在熊猫数据框中删除列](#Drop_column_in_Pandas_DataFrame "Drop column in Pandas DataFrame")
*   [在熊猫数据框中删除第一列](#Drop_first_column_in_Pandas_DataFrame "Drop first column in Pandas DataFrame")
*   [从数据帧中删除第一列](#Drop_the_first_column_from_DataFrame "Drop the first column from DataFrame")
*   [删除熊猫数据框的最后一列](#Drop_last_column_in_Pandas_DataFrame "Drop last column in Pandas DataFrame")
*   [删除熊猫数据帧的最后一列](#Drop_the_last_column_of_Pandas_DataFrame "Drop the last column of Pandas DataFrame")
*   [在 Pandas 数据框架中删除多列](#Drop_multiple_columns_in_Pandas_DataFrame "Drop multiple columns in Pandas DataFrame")
*   [如何在 Pandas 中放置多列](#How_to_drop_multiple_columns_in_Pandas "How to drop multiple columns in Pandas")
*   [删除熊猫数据框架中的重复列](#Drop_duplicate_columns_in_Pandas_DataFrame "Drop duplicate columns in Pandas DataFrame")
*   [删除熊猫数据框中的第一列](#Remove_first_column_in_Pandas_DataFrame "Remove first column in Pandas DataFrame")
*   [删除熊猫数据框中的列名](#Remove_column_names_in_Pandas_DataFrame "Remove column names in Pandas DataFrame")
*   [降柱熊猫系列](#Drop_column_Pandas_series "Drop column Pandas series")
*   [在熊猫数据框中删除一列](#Drop_one_column_in_Pandas_DataFrame "Drop one column in Pandas DataFrame")
*   [在 Pandas 数据框中下拉列列表](#Drop_list_of_column_in_Pandas_DataFrame "Drop list of column in Pandas DataFrame")

## 熊猫删除列数据框

在这一节中，我们将学习如何使用 Python 从数据帧中删除列。

*   在 Python Pandas 中，有三种方法可以从数据帧中删除列。drop()，delete()，pop()。
*   dop() 是 Python Pandas 中最常用的移除行或列的方法，我们也将使用同样的方法。

**语法:**

这是 Python Pandas 中 drop()方法的语法。

```py
df.drop(
    labels=None,
    axis: 'Axis' = 0,
    index=None,
    columns=None,
    level: 'Level | None' = None,
    inplace: 'bool' = False,
    errors: 'str' = 'raise',
)
```

*   标签–提供行或列的名称
*   axis–1 表示行，0 表示列，如果标签是列名，则提供 axis=1。
*   index–axis 0 =标签，如果使用行，则提供索引
*   列–轴=1，列=标签
*   就地–如果设置为 True，则更改将立即生效。不需要重新分配值。
*   错误–如果设置为“引发”,则当出现问题时会出现错误。

在 jupyter 笔记本的例子中，我们已经演示了所有这些方法。
# 计算熊猫数据框中的行数

> 原文：<https://pythonguides.com/count-rows-in-pandas-dataframe/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 [Python Pandas 教程](https://pythonguides.com/pandas-in-python/)中，我们将学习 Python Pandas 数据帧中的**计数行。此外，我们将涵盖这些主题。**

*   计算熊猫数据框中的行数
*   熊猫数据帧中的计数条目
*   计算 Pandas 数据帧中的重复行
*   计算熊猫数据帧中不同的行
*   计算熊猫数据帧中的唯一行
*   计算满足条件的熊猫数据帧中的行数
*   熊猫 df 中的计数记录
*   熊猫系列中的行数

目录

[](#)

*   [计算熊猫数据帧中的行数](#Count_Rows_in_Pandas_DataFrame "Count Rows in Pandas DataFrame")
*   [计数熊猫数据帧中的条目](#Count_Entries_in_Pandas_DataFrame "Count Entries in Pandas DataFrame")
*   [计算熊猫数据帧中的重复行](#Count_Duplicate_Rows_in_Pandas_DataFrame "Count Duplicate Rows in Pandas DataFrame")
*   [计算熊猫数据帧中不同的行](#Count_Distinct_rows_in_Pandas_DataFrame "Count Distinct rows in Pandas DataFrame")
*   [计算熊猫数据帧中的唯一行](#Count_Unique_Rows_in_Pandas_DataFrame "Count Unique Rows in Pandas DataFrame")
*   [计算满足条件的熊猫数据帧中的行数](#Count_Rows_in_a_Pandas_DataFrame_that_Satisfies_a_Condition "Count Rows in a Pandas DataFrame that Satisfies a Condition")
*   [计算熊猫系列的行数](#Count_Rows_in_Series_Pandas "Count Rows in Series Pandas")

## 计算熊猫数据帧中的行数

在这一节中，我们将学习如何**计算 Pandas 数据帧**中的行数。

*   使用 Python Pandas 中的 count()方法，我们可以对行和列进行计数。
*   计数方法需要轴信息，列的轴=1，行的轴=0。
*   要计算 Python Pandas 中的行数，请键入`df.count(axis=1)`，其中 df 是数据帧，axis=1 表示列。

```py
df.count(axis=1)
```

**在 Jupyter 笔记本上实现**
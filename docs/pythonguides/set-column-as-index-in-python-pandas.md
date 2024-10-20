# 如何在 Python Pandas 中将列设置为索引

> 原文：<https://pythonguides.com/set-column-as-index-in-python-pandas/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 Python 教程中，我们将学习**如何在 Python Pandas** 中将列设置为索引。此外，我们将涵盖这些主题。

*   将列设置为索引熊猫数据框架
*   将第一列设置为索引熊猫
*   将日期列设置为索引熊猫
*   将日期时间列设置为索引熊猫
*   将列名设置为索引熊猫
*   将列设为行索引熊猫
*   将两列设置为索引熊猫

如果你是熊猫的新手，可以看看 Python 中的熊猫。

出于演示目的，我们使用从 [Kaggle](https://www.kaggle.com/jackogozaly/steam-player-data/download) 下载的 **Steam 播放器数据**。

![Set Column as Index in Python Pandas](img/cb41d6c53fe4cde9d009766ec8b06c03.png "Set Column as Index in Python Pandas")

Set Column as Index in Python Pandas

目录

[](#)

*   [将列设置为索引熊猫数据框](#Set_Column_as_Index_Pandas_DataFrame "Set Column as Index Pandas DataFrame")
*   [将第一列设置为索引熊猫](#Set_First_Column_as_Index_Pandas "Set First Column as Index Pandas")
*   [将日期列设置为索引熊猫](#Set_Date_Column_as_Index_Pandas "Set Date Column as Index Pandas")
*   [将日期时间列设置为索引熊猫](#Set_Datetime_column_as_Index_Pandas "Set Datetime column as Index Pandas")
*   [将列名设置为索引熊猫](#Set_Column_names_as_Index_Pandas "Set Column names as Index Pandas")
*   [将列设置为行索引熊猫](#Set_Column_as_Row_Index_Pandas "Set Column as Row Index Pandas")
*   [设置两列为索引熊猫](#Set_Two_Column_as_Index_Pandas "Set Two Column as Index Pandas")

## 将列设置为索引熊猫数据框

在这一节中，我们将学习如何**在 Pandas DataFrame** 中将列设置为索引。

*   Python Pandas 提供了多种处理数据的选项。
*   在这些选项中，有一个选项是 `dataframe.set_index()` 。
*   使用 dataframe.set_index()方法，我们可以将任何列设置为索引。
*   此方法接受要设置为索引的列的名称。
*   在 jupyter 笔记本的例子中，我们将日期设置为索引值。
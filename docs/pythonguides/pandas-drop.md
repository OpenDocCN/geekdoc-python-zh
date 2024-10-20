# 如何在 Python 中使用 Pandas drop()函数【有用教程】

> 原文：<https://pythonguides.com/pandas-drop/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

这个 [Python 教程](https://pythonguides.com/learn-python/)都是关于 **Python 熊猫滴()函数**的。我们将看到如何在 Python 中使用 Pandas drop()函数。此外，我们还将讨论以下主题:

在本教程中，我们将学习如何使用熊猫的**。Drop 是数据科学中使用的一个主要函数&机器学习来清理数据集。此外，我们将涵盖这些主题**

*   熊猫 drop 语法
*   熊猫下降栏
    *   熊猫逐列下降指数
    *   熊猫有条件降柱
    *   如果存在熊猫，则删除列
    *   熊猫和南一起下栏
    *   熊猫用全零删除列
    *   熊猫读 CSV 时掉柱
    *   没有名字的熊猫掉柱
    *   熊猫掉柱除了
    *   熊猫删除非数字列

*   熊猫掉行
    *   熊猫有条件地放弃争吵
    *   熊猫行，南列
    *   熊猫删除带有 nan 的行 + 熊猫删除特定列中带有 nan 的行
    *   熊猫用条件字符串删除行
    *   熊猫会删除任何列中有值的行
    *   熊猫掉落一排排
    *   熊猫在列中删除零行
    *   熊猫下降标题行
    *   熊猫丢弃非整数行
    *   熊猫丢弃非数字行
    *   熊猫放下空行
    *   熊猫丢弃丢失的行

*   在 Pandas 数据框架中删除具有 NaN 值的列
*   删除 Pandas 数据框架中具有 NaN 值的列替换
*   在 Pandas 数据帧中删除具有 NaN 值的列

目录

[](#)

*   [熊猫滴()功能](#Pandas_drop_function "Pandas drop() function")
*   [熊猫滴语法](#Pandas_drop_syntax "Pandas drop syntax")
*   [熊猫降柱](#Pandas_drop_column "Pandas drop column")
    *   [熊猫按指数下降列](#Pandas_drop_column_by_index "Pandas drop column by index")
    *   [熊猫用条件掉列](#Pandas_drop_columns_with_condition "Pandas drop columns with condition")
    *   [熊猫如果存在则删除列](#Pandas_drop_column_if_exists "Pandas drop column if exists")
    *   [熊猫和楠一起掉柱](#Pandas_drop_column_with_nan "Pandas drop column with nan")
    *   [熊猫删除全零列](#Pandas_drop_columns_with_all_zeros "Pandas drop columns with all zeros")
    *   [熊猫读 CSV 时掉柱](#Pandas_drop_column_while_reading_CSV "Pandas drop column while reading CSV")
    *   [没有名字的熊猫掉柱](#Pandas_drop_column_with_no_name "Pandas drop column with no name")
    *   [熊猫掉柱除了](#Pandas_drop_columns_except "Pandas drop columns except")
    *   [熊猫丢弃非数字列](#Pandas_drop_non_numeric_columns "Pandas drop non numeric columns")
*   [熊猫掉行](#Pandas_drop_rows "Pandas drop rows")
    *   [熊猫丢弃带有条件的行](#Pandas_drop_rows_with_condition "Pandas drop rows with condition")
    *   [熊猫丢行，南在列](#Pandas_drop_rows_with_nan_in_column "Pandas drop rows with nan in column")
    *   [熊猫与南闹翻](#Pandas_drop_rows_with_nan "Pandas drop rows with nan")
    *   [熊猫删除特定列中带有 nan 的行](#Pandas_drop_rows_with_nan_in_specific_column "Pandas drop rows with nan in specific column")
    *   [熊猫用条件字符串](#Pandas_drop_rows_with_condition_string "Pandas drop rows with condition string")删除行
    *   [熊猫删除任意列中有值的行](#Pandas_drop_rows_with_value_in_any_column "Pandas drop rows with value in any column")
    *   [熊猫掉落一排排](#Pandas_drop_range_of_rows "Pandas drop range of rows")
    *   [熊猫删除列中带有零的行](#Pandas_drop_rows_with_zero_in_column "Pandas drop rows with zero in column")
    *   [熊猫滴标题行](#Pandas_drop_header_row "Pandas drop header row")
    *   [熊猫丢弃非整数行](#Pandas_drop_non-integer_rows "Pandas drop non-integer rows")
    *   [熊猫丢弃非数字行](#Pandas_drop_non_numeric_rows "Pandas drop non numeric rows")
    *   [熊猫丢空行](#Pandas_drop_blank_rows "Pandas drop blank rows")
    *   [熊猫丢行](#Pandas_drop_missing_rows "Pandas drop missing rows")
*   [在 Pandas 数据帧中删除具有 NaN 值的列](#Drop_Column_with_NaN_values_in_Pandas_DataFrame "Drop Column with NaN values in Pandas DataFrame")
*   [删除 Pandas 数据帧中 NaN 值的列替换](#Drop_Column_with_NaN_Values_in_Pandas_DataFrame_Replace "Drop Column with NaN Values in Pandas DataFrame Replace")
*   [删除 Pandas 数据帧中具有 NaN 值的列，得到最后一个 Non](#Drop_Column_with_NaN_Values_in_Pandas_DataFrame_Get_Last_Non "Drop Column with NaN Values in Pandas DataFrame Get Last Non")

## 熊猫滴()功能

Python 中的 **Pandas drop()函数**用于从行和列中删除指定的标签。Drop 是数据科学中使用的一个主要函数& [机器学习](https://pythonguides.com/machine-learning-using-python/)来清理数据集。

**Pandas Drop()函数**从行或列中删除指定的标签。使用多索引时，可以通过指定级别来删除不同级别上的标签。

刚接触 Python 熊猫？查看一篇关于 Python 中[熊猫的文章。](https://pythonguides.com/pandas-in-python/)

## 熊猫滴语法

下面是 **Pandas drop()函数语法**。

```py
DataFrame.drop(
    labels=None, 
    axis=0, 
    index=None, 
    columns=None, 
    level=None, 
    inplace=False, 
    errors='raise'
)
```

| 选择 | 说明 |
| --- | --- |
| 标签 | 要删除的单个标签或类似列表的
索引或列标签。 |
| 轴 | 拖放将移除提供的轴，轴可以是 0 或 1。
轴= 0 表示行或索引(垂直)
轴= 1 表示列(水平)
默认情况下，轴= 0
 |
| 指数 | 单一标签或列表式。
索引是行(垂直方向)&相当于轴=0 |
| 列 | 单一标签或列表式。
表格视图中的列是水平的&用轴=1 表示。 |
| 水平 | int 或 level name，对于 MultiIndex 可选
，标签将从哪个级别移除。 |
| 适当的 | 接受布尔值(真或假)，默认为假
在原地进行更改，然后&在那里进行更改。不需要给变量赋值。 |
| 错误 | 错误可以是'**忽略了**或'**引发了**。默认为“引发”
如果忽略，则抑制错误，仅删除现有标签
如果引发，则显示错误消息&不允许删除数据。 |

看看吧，Python 熊猫里的 [Groupby。](https://pythonguides.com/groupby-in-python-pandas/)

## 熊猫降柱

让我们看看**如何使用熊猫降柱**。
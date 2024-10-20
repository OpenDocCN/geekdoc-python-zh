# Python 熊猫删除行示例

> 原文：<https://pythonguides.com/python-pandas-drop-rows-example/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在使用 Python 数据集时，工程师会根据项目要求清理数据集。Drop 函数通常用于删除对项目没有用处的行和列。

在本教程中，我们将了解 python 熊猫 drop 行。此外，我们将涵盖这些主题。

*   Python 熊猫掉落功能
*   Python 熊猫按索引删除行
*   Python 熊猫按条件删除行
*   Python 熊猫在特定列中删除带有 nan 的行
*   蟒蛇熊猫和南闹翻了
*   Python 熊猫基于列值删除行
*   Python 熊猫删除包含字符串的行

我们在本教程中使用的数据集是从 [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) 下载的。

目录

[](#)

*   [Python 熊猫掉落功能](#Python_Pandas_Drop_Function "Python Pandas Drop Function")
*   [Python 熊猫按索引丢行](#Python_pandas_drop_rows_by_index "Python pandas drop rows by index")
*   [Python 熊猫按条件丢行](#Python_pandas_drop_rows_by_condition "Python pandas drop rows by condition")
*   [Python 熊猫删除特定列中带有 nan 的行](#Python_pandas_drop_rows_with_nan_in_specific_column "Python pandas drop rows with nan in specific column")
*   [蟒蛇熊猫与南掉行](#Python_pandas_drop_rows_with_nan "Python pandas drop rows with nan")
*   [Python pandas 基于列值删除行](#Python_pandas_drop_rows_based_on_column_value "Python pandas drop rows based on column value")
*   [Python 熊猫丢弃包含字符串的行](#Python_pandas_drop_rows_containing_string "Python pandas drop rows containing string")

## Python 熊猫掉落功能

`Pandas drop` 是 [Python pandas](https://pythonguides.com/pandas-in-python/) 中的一个函数，用于放下数据集的行或列。该功能常用于数据清理。**轴= 0** 表示行，**轴= 1** 表示列。

**语法:**

下面是实现 pandas `drop()` 的语法

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
默认情况下，轴= 0 |
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

还看，[如何在 Python 中使用 Pandas drop()函数](https://pythonguides.com/pandas-drop/)

## Python 熊猫按索引丢行

*   在这一节中，我们将学习如何在 Python Pandas 中通过索引删除行。要按索引删除行，我们所要做的就是传递索引号或索引号列表，以防多次删除。
*   要按索引删除行，只需使用下面的代码:``df.drop(index)``。这里 df 是您正在处理的数据帧，代替 index type 的是索引号或名称。
*   这是 jupyter 笔记本上的代码实现，请仔细阅读评论和 markdown，一步一步的解释。
# 如何将熊猫数据帧转换成字典

> 原文：<https://pythonguides.com/how-to-convert-pandas-dataframe-to-a-dictionary/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 Python 教程中，我们将学习如何将 [Pandas](https://pythonguides.com/pandas-in-python/) DataFrame 转换成 Python 中的字典。此外，我们将涵盖这些主题。

*   将熊猫数据帧转换为字典
*   将 Pandas 数据帧转换为不带索引的字典
*   将熊猫数据帧转换为字典列表
*   将熊猫数据帧转换为嵌套字典
*   将熊猫系列转换为字典
*   将熊猫列转换为词典
*   Python Pandas 将数据帧转换为具有多个值的字典。
*   将 Pandas 数据帧转换为以一列为键的字典
*   将熊猫行转换为字典

我们已经用 [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) 上的[美国房屋销售](https://www.kaggle.com/harlfoxem/housesalesprediction)预测数据集展示了例子。

目录

[](#)

*   将熊猫数据帧转换成字典
*   [将熊猫数据帧转换成不带索引的字典](#Convert_Pandas_DataFrame_to_Dictionary_Without_Index "Convert Pandas DataFrame to Dictionary Without Index")
*   [将熊猫数据帧转换为字典列表](#Convert_Pandas_DataFrame_to_Dictionary_List "Convert Pandas DataFrame to Dictionary List")
*   [将熊猫数据帧转换为嵌套字典](#Convert_Pandas_DataFrame_to_Nested_Dictionary "Convert Pandas DataFrame to Nested Dictionary")
*   [将熊猫系列转换成字典](#Convert_Pandas_Series_to_Dictionary "Convert Pandas Series to Dictionary")
*   [将熊猫列转换为字典](#Convert_Pandas_Column_to_Dictionary "Convert Pandas Column to Dictionary")
*   Python 熊猫将数据帧转换成具有多个值的字典。
*   [将熊猫数据帧转换为字典，其中一列为关键字](#Convert_Pandas_DataFrame_to_Dictionary_with_One_Column_as_Key "Convert Pandas DataFrame to Dictionary with One Column as Key")
*   [将熊猫行转换成字典](#Convert_Pandas_Row_To_Dictionary "Convert Pandas Row To Dictionary")

## 将熊猫数据帧转换成字典

在这一节中，我们将学习如何用 Python 将熊猫数据帧转换成字典。

*   DataFrame 是表格形式的数据表示，即行和列。
*   字典是键值对。每个键都是唯一的，值可以是任何数据类型。
*   字典的另一个名字是字典，所以如果你正在寻找如何将熊猫数据帧转换成字典。然后，您可以遵循本节中介绍的解决方案。
*   在 Python Pandas 中使用`*data frame . to _ dict()*`我们可以将数据帧转换成字典。2
*   在我们的例子中，我们使用了从 kaggle 下载的美国房屋销售预测数据集。
*   在这段代码中，我们只将数据帧的前 5 行转换为字典。
*   所有的数据都被转换成键值对，它们的数据类型现在是 dictionary。
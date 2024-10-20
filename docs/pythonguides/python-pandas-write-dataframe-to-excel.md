# Python 熊猫写数据帧到 Excel

> 原文：<https://pythonguides.com/python-pandas-write-dataframe-to-excel/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在本教程中，我们将学习 Python 和将数据帧写入 Excel。此外，我们将涵盖这些主题。

*   Python 熊猫写数据帧到 Excel
*   Python 熊猫写数据框到 Excel 没有索引
*   Python 熊猫将数据帧写入 CSV
*   Python 熊猫写数据帧到 CSV 没有索引
*   Python 熊猫将数据帧写入 CSV 示例
*   Python 熊猫将数据帧写入现有的 Excel
*   Python 熊猫将多个数据帧导出到 Excel

目录

[](#)

*   [Python 熊猫写数据帧到 Excel](#Python_Pandas_Write_DataFrame_to_Excel "Python Pandas Write DataFrame to Excel")
*   [Python 熊猫写数据帧到 Excel 没有索引](#Python_Pandas_Write_DataFrame_to_Excel_Without_Index "Python Pandas Write DataFrame to Excel Without Index")
*   [Python 熊猫将数据帧写入 CSV](#Python_Pandas_Write_DataFrame_to_CSV "Python Pandas Write DataFrame to CSV")
*   [Python 熊猫将数据帧写入不带索引的 CSV](#Python_Pandas_Write_DataFrame_to_CSV_without_Index "Python Pandas Write DataFrame to CSV without Index")
*   [Python 熊猫将数据帧写入 CSV 示例](#Python_Pandas_Write_DataFrame_to_CSV_Example "Python Pandas Write DataFrame to CSV Example")
*   [Python 熊猫将数据帧写入现有 Excel](#Python_Pandas_Write_DataFrame_to_Existing_Excel "Python Pandas Write DataFrame to Existing Excel")
*   [Python 熊猫将多个数据帧导出到 Excel](#Python_Pandas_Export_Multiple_DataFrames_to_Excel "Python Pandas Export Multiple DataFrames to Excel")

## Python 熊猫写数据帧到 Excel

在这一节中，我们将学习 Python [Pandas](https://pythonguides.com/pandas-in-python/) 将数据帧写入 Excel。

*   使用 **`.to_excel()`** 我们可以在 Pyhton Pandas 中将数据帧转换成 Excel 文件。
*   没有名为“openpyxl”的模块，如果出现此错误，意味着您需要在您的系统上安装 openpyxl 包。
*   如果您正在使用 anaconda 环境，请使用`conda install openpyxl`。
*   如果使用 pip，使用`pip install openpyxl`。
*   在我们的例子中，我们已经创建了一个汽车的数据帧，并使用 Python Pandas 将该数据帧写入 excel 文件
# Python 熊猫 CSV 教程

> 原文：<https://pythonguides.com/python-pandas-csv/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在这个 [Python 教程](https://pythonguides.com/learn-python/)中，让我们来讨论一下 **Python 熊猫 CSV** 。我们将学习**如何在 Python Pandas** 中读取 CSV 文件，以及如何在 Python Pandas 中保存 CSV 文件，并将涵盖以下主题:

*   熊猫中的 CSV 文件是什么
*   如何在熊猫中读取 CSV 文件
*   如何在熊猫中保存 CSV 文件
*   如何在没有标题的熊猫中读取 CSV 文件
*   如何在熊猫中读取带标题的 CSV 文件
*   如何在 pandas 中使用附加 CSV 文件
*   什么是 CSV 熊猫南
*   熊猫 CSV 到数据帧
*   熊猫 CSV 到 JSON
*   熊猫 CSV 超越
*   熊猫 CSV 到字典

目录

[](#)

*   [熊猫 Python 中的 CSV 文件](#CSV_file_in_Pandas_Python "CSV file in Pandas Python")
*   [读取熊猫的 CSV 文件](#Read_CSV_File_in_Pandas "Read CSV File in Pandas")
*   [在熊猫中保存 CSV 文件](#Save_CSV_File_in_Pandas "Save CSV File in Pandas")
*   [在没有标题的 Pandas 中读取 CSV 文件](#Read_CSV_File_in_Pandas_Without_Header "Read CSV File in Pandas Without Header")
*   [在 Pandas 中读取带有标题](#Read_CSV_File_in_Pandas_With_a_Header "Read CSV File in Pandas With a Header")的 CSV 文件
*   [在熊猫中追加 CSV 文件](#Append_CSV_File_in_Pandas "Append CSV File in Pandas")
*   CSV 熊猫男
*   [用熊猫 Python 写 CSV 文件](#Write_CSV_file_in_Pandas_Python "Write CSV file in Pandas Python")
*   [熊猫 CSV 到数据帧](#Pandas_CSV_to_DataFrame "Pandas CSV to DataFrame")
*   [熊猫 CSV 转 JSON](#Pandas_CSV_to_JSON "Pandas CSV to JSON")
*   [熊猫 CSV 到 excel](#Pandas_CSV_to_excel "Pandas CSV to excel")
*   [熊猫 CSV 到字典](#Pandas_CSV_to_the_dictionary "Pandas CSV to the dictionary")
*   [Python 熊猫 CSV 转 HTML](#Python_Pandas_CSV_to_HTML "Python Pandas CSV to HTML")

## 熊猫 Python 中的 CSV 文件

*   在这一节，我们将学习如何使用熊猫&读取 CSV 文件如何使用熊猫**导出 CSV 文件**。
*   CSV(逗号分隔值)文件是一种具有特定格式的文本文件，允许以表格结构格式保存数据。
*   CSV 被认为是最适合与熊猫合作的工具，因为它们简单易行
*   在使用 CSV 时，我们执行两项主要任务
    *   **读取 CSV 或导入 CSV**
    *   **写入 CSV 或导出 CSV**
*   我们将在接下来的章节中详细讨论这两个问题。

你可能喜欢 [Python 串联列表与示例](https://pythonguides.com/python-concatenate-list/)

## 读取熊猫的 CSV 文件

*   在本节中，我们将学习如何在 pandas 中读取 CSV 文件。读取 CSV 文件也意味着在熊猫中导入 CSV 文件。
*   在我们处理 CSV 格式的数据集之前，我们需要导入该 CSV。

**语法:**

```py
import pandas as pd

pd.read_csv('file_path_name.csv')
```

**实施:**
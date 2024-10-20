# 如何将熊猫数据帧转换成 Excel 文件

> 原文：<https://www.askpython.com/python-modules/pandas/convert-pandas-dataframe-to-excel>

将数据导出到 Excel 文件通常是任何用户阅读和解释给定数据集的最首选和最方便的方式。可以使用 python 代码将您的网络抓取或其他收集的数据导出到 Excel 文件中，这也可以通过使用 Pandas 库以非常简单的步骤完成。

## 将 Pandas 数据框架转换为 Excel 的步骤

按照下面的一步一步的教程，学习如何将熊猫数据帧写入 Excel 文件。

### 第一步:安装 pandas 和 openpyxl

由于您需要导出 pandas 数据帧，显然您必须已经安装了 pandas 包。如果没有，运行下面的 [pip 命令](https://www.askpython.com/python-modules/python-pip)在您的计算机上安装 Pandas python 包。

```py
pip install openpyxl

```

现在，要在 Python 中使用 Excel 文件函数，您需要使用下面的 **pip** 命令安装 openpyxl 模块。

```py
pip install openpyxl

```

您可以将 DataFrame 写入 Excel 文件，而无需提及任何工作表名称。下面给出了一步一步的过程:

### 第二步:制作数据框架

*   在 python 代码/脚本文件中导入 Pandas 包。
*   创建要导出的数据的数据框架，并用行和列的值初始化数据框架。

Python 代码:

```py
#import pandas package
import pandas as pd

# creating pandas dataframe
df_cars = pd.DataFrame({'Company': ['BMW', 'Mercedes', 'Range Rover', 'Audi'],
     'Model': ['X7', 'GLS', 'Velar', 'Q7'],
     'Power(BHP)': [394.26, 549.81, 201.15, 241.4],
     'Engine': ['3.0 L 6-cylinder', '4.0 L V8', '2.0 L 4-cylinder', '4.0 L V-8']})

```

### 步骤 3:创建 Writer 对象并导出到 Excel 文件

*   使用 pandas 包的:ExcelWriter()方法创建一个 Excel Writer 对象
*   输入输出 excel 文件的名称，您要将我们的数据框架和扩展名写入该文件。(在我们的示例中，我们将输出 excel 文件命名为“converted-to-excel.xlsx”)

```py
# creating excel writer object

writer = pd.ExcelWriter('converted-to-excel.xlsx')

```

*   在数据帧上调用 _excel()函数，将 excel Writer 作为参数传递，将数据导出到具有给定名称和扩展名的 Excel 文件中。
*   保存 writer 对象以保存 Excel 文件

```py
# write dataframe to excel

df_cars.to_excel(writer)

# save the excel
writer.save()
print("DataFrame is exported successfully to 'converted-to-excel.xlsx' Excel File.")﻿

```

## 替代-直接方法

一种直接的方法是将数据框直接导出到 Excel 文件，而不使用 ExcelWriter 对象，如以下代码示例所示:

```py
import pandas as pd

# creating pandas dataframe from dictionary of data
df_cars = pd.DataFrame({'Company': ['BMW', 'Mercedes', 'Range Rover', 'Audi'],
     'Model': ['X7', 'GLS', 'Velar', 'Q7'],
     'Power(BHP)': [394.26, 549.81, 201.15, 241.4],
     'Engine': ['3.0 L 6-cylinder', '4.0 L V8', '2.0 L 4-cylinder', '4.0 L V-8']})

#Exporting dataframe to Excel file
df_cars.to_excel("converted-to-excel.xlsx")

```

**输出 Excel 文件**

打开 excel 文件，您将看到写入文件的索引、列标签和行数据。

## **奖励提示**

您不仅只能控制 excel 文件名，而不能将 python 数据帧导出到 Excel 文件，而且您还可以在 pandas 包中定制许多功能。

您可以更改 excel 文件的工作表名称

```py
df.to_excel("output.xlsx", sheet_name='Sheet_name_1')

```

使用 Excel writer 向现有 Excel 文件追加内容

```py
pd.ExcelWriter('output.xlsx', mode='a')

```

其他选项包括渲染引擎、开始行、标题、索引、合并单元格、编码和许多其他选项。

在[熊猫官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html)了解更多可用选项。

## 结论

我希望你现在明白了如何使用手头的不同库将熊猫数据帧导出到 Excel。请关注 AskPython，获取更多有趣的教程。
# 用 Python 重命名数据帧中的列

> 原文：<https://www.pythonforbeginners.com/basics/rename-columns-in-a-dataframe-in-python>

Pandas 数据帧是 python 中处理表格数据的最有效的数据结构之一。当我们将表格数据从 csv 文件导入数据帧时，我们通常需要重命名数据帧中的列。在本文中，我们将讨论如何在 python 中重命名数据帧中的列。

## 使用 Rename()方法重命名 DataFrame 列

pandas 模块为我们提供了`rename()`方法来重命名数据帧中的列。在 dataframe 上调用`rename()`方法时，该方法将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python)作为其第一个输入参数。字典中的键应该由要重命名的列的原始名称组成。与键相关联的值应该是新的列名。执行后，`rename()`方法返回一个新的带有修改名称的数据帧。例如，我们可以使用`rename()`方法重命名给定数据帧的`‘Roll’`列，如下例所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The original column names are:")
print(df.columns.values)
nameDict={"Roll":"Roll No."}
df=df.rename(columns=nameDict)
print("The modified column names are:")
print(df.columns.values)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The original column names are:
['Name' 'Roll' 'Language']
The modified column names are:
['Name' 'Roll No.' 'Language']
```

如果您想要重命名数据帧中的多个列，您可以在字典中传递旧的列名和新的列名，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The original column names are:")
print(df.columns.values)
nameDict={"Name":"Person","Roll":"Roll No."}
df=df.rename(columns=nameDict)
print("The modified column names are:")
print(df.columns.values)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The original column names are:
['Name' 'Roll' 'Language']
The modified column names are:
['Person' 'Roll No.' 'Language']
```

在上面的例子中，原始列中的列名没有被修改。相反，我们得到一个新的数据帧，其中包含修改后的列名。

您还可以重命名原始数据框架的列。为此，我们将使用`rename()`方法的`‘inplace’` 参数。`‘inplace’`参数采用一个可选的输入参数，它有默认值`False`。因此，原始数据帧中的列名不会被修改。您可以将`‘inplace’`参数设置为值`True`来修改原始数据帧的列名，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The original column names are:")
print(df.columns.values)
nameDict={"Name":"Person","Roll":"Roll No."}
df.rename(columns=nameDict,inplace=True)
print("The modified column names are:")
print(df.columns.values)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The original column names are:
['Name' 'Roll' 'Language']
The modified column names are:
['Person' 'Roll No.' 'Language']
```

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用列名列表重命名数据框架列

如果必须一次重命名数据帧的所有列，可以使用 python 列表来完成。为此，我们只需将包含新数据帧名称的列表分配给数据帧的`‘columns’` 属性，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The original column names are:")
print(df.columns.values)
df.columns=['Person', 'Roll No.', 'Language']
print("The modified column names are:")
print(df.columns.values)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The original column names are:
['Name' 'Roll' 'Language']
The modified column names are:
['Person' 'Roll No.' 'Language']
```

## 结论

在本文中，我们讨论了如何用 python 重命名数据帧中的列。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。
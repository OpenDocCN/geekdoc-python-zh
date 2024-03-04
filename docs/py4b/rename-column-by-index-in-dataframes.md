# 在数据帧中按索引重命名列

> 原文：<https://www.pythonforbeginners.com/basics/rename-column-by-index-in-dataframes>

在 python 中，数据帧用于处理表格数据。在本文中，我们将讨论如何通过 python 中的[数据帧中的索引来重命名列。](https://www.pythonforbeginners.com/basics/select-row-from-a-dataframe-in-python)

## 使用索引号更改列名

我们可以使用“columns”属性访问数据帧中的列名。数据帧的 columns 属性包含一个`Index`对象。`Index`对象包含一个列名列表，如下例所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The column object is:")
print(df.columns)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The column object is:
Index(['Name', 'Roll', 'Language'], dtype='object')
```

您可以使用`Index`对象的 `‘values’`属性来访问列名数组，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The column object is:")
print(df.columns)
print("The columns are:")
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
The column object is:
Index(['Name', 'Roll', 'Language'], dtype='object')
The columns are:
['Name' 'Roll' 'Language']
```

要在数据帧中通过索引重命名列，我们可以在 values 属性中修改数组。例如，您可以使用值数组的索引 0 来更改第一列的名称，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The column object is:")
print(df.columns)
print("The columns are:")
print(df.columns.values)
df.columns.values[0]="First Name"
print("The modified column object is:")
print(df.columns)
print("The modified columns are:")
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
The column object is:
Index(['Name', 'Roll', 'Language'], dtype='object')
The columns are:
['Name' 'Roll' 'Language']
The modified column object is:
Index(['First Name', 'Roll', 'Language'], dtype='object')
The modified columns are:
['First Name' 'Roll' 'Language'] 
```

在这种方法中，我们不能一次更改多个列名。要更改多个列名，需要逐个重命名每个列名，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The column object is:")
print(df.columns)
print("The columns are:")
print(df.columns.values)
df.columns.values[0]="First Name"
df.columns.values[1]="Roll Number"
print("The modified column object is:")
print(df.columns)
print("The modified columns are:")
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
The column object is:
Index(['Name', 'Roll', 'Language'], dtype='object')
The columns are:
['Name' 'Roll' 'Language']
The modified column object is:
Index(['First Name', 'Roll Number', 'Language'], dtype='object')
The modified columns are:
['First Name' 'Roll Number' 'Language']
```

我们还可以使用 rename()方法通过索引一次更改多个列名。让我们来讨论这种方法。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 在数据帧中使用 rename()方法更改列名

我们可以使用`rename()` 方法通过索引号来重命名多个列。当在 dataframe 上调用时,`rename()` 方法将一个字典作为它的输入参数。字典应该包含需要重命名为键的列名。新的列名应该是与原始键相关联的值。执行之后，`rename()`方法返回一个新的数据帧，其中包含修改后的列名。

为了使用索引号和`rename()`方法修改列名，我们将首先使用 dataframe 的`columns.values` 属性获得列名数组。之后，我们将创建一个字典，其中列名作为键，新列名作为键的关联值。然后，我们将把字典传递给`rename()` 方法。执行后，`rename()`方法将返回修改了列名的 dataframe，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
print("The column object is:")
print(df.columns)
print("The columns are:")
print(df.columns.values)
nameDict={"Name":"First Name","Roll":"Roll No."}
df=df.rename(columns=nameDict)
print("The modified column object is:")
print(df.columns)
print("The modified columns are:")
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
The column object is:
Index(['Name', 'Roll', 'Language'], dtype='object')
The columns are:
['Name' 'Roll' 'Language']
The modified column object is:
Index(['First Name', 'Roll No.', 'Language'], dtype='object')
The modified columns are:
['First Name' 'Roll No.' 'Language']
```

## 结论

在本文中，我们讨论了如何在 python 中通过索引重命名数据帧中的列。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中[字典理解的文章。你可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。
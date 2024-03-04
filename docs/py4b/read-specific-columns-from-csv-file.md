# 从 CSV 文件中读取特定列

> 原文：<https://www.pythonforbeginners.com/basics/read-specific-columns-from-csv-file>

CSV 文件是在文件系统中存储表格数据的最流行的方式。有时 csv 文件可以包含我们不需要分析的多个列。在本文中，我们将讨论如何用 python 从 csv 文件中读取特定的列。

## 使用 Pandas Dataframe 从 CSV 文件中读取特定列

为了用 python 读取 csv 文件，我们使用 pandas 模块中提供的`read_csv()`方法。`read_csv()`方法将 csv 文件的名称作为其输入参数。执行后，`read_csv()`方法返回包含 csv 文件数据的 dataframe。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
```

如您所见， `read_csv()`方法返回包含 csv 文件所有列的数据帧。为了从数据帧中读取特定的列，我们可以使用列名作为索引，就像我们从列表中获取元素一样。为此，我们可以简单地将列名传递给 dataframe 后面的方括号，如示例所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
specific_column=df["Name"]
print("The column is:")
print(specific_column)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The column is:
0    Aditya
1       Sam
2     Chris
3      Joel
Name: Name, dtype: object
```

要从 dataframe 中读取多个列，我们可以传递方括号中的列名列表，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv")
print("The dataframe is:")
print(df)
specific_columns=df[["Name","Roll"]]
print("The column are:")
print(specific_columns)
```

输出:

```py
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
The column are:
     Name  Roll
0  Aditya     1
1     Sam     2
2   Chris     3
3    Joel     4
```

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用“usecols”参数从 CSV 文件中读取特定列

读取整个数据帧并从数据帧中提取列不符合我们的目的。在上面的方法中，我们还将不需要的列读入数据帧。之后，我们正在阅读具体的栏目。我们可以通过在`read_csv()`方法中使用‘`usecols`’参数来避免整个过程。为了从 csv 文件中读取特定的列，我们将把要读取的列的列表作为输入参数传递给'`usecols`'参数。执行后，`read_csv()`方法返回包含特定列的 dataframe，如下例所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("demo_file.csv",usecols=["Name"])
print("The dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
     Name
0  Aditya
1     Sam
2   Chris
3    Joel
```

正如您在上面看到的，我们已经使用'`usecols`'参数从 csv 文件中读取了特定的列。

## 结论

在本文中，我们讨论了如何从 csv 文件中读取特定的列。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中[字典理解的文章。你可能也会喜欢这篇关于 python 的](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。
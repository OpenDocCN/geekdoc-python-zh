# 在 pandas 中向空数据框添加数据

> 原文：<https://www.blog.pythonlibrary.org/2022/08/09/adding-data-to-empty-dataframes-in-pandas/>

最流行的第三方 Python 包之一叫做 [pandas](https://pandas.pydata.org/) 。熊猫包“是一个快速、强大、灵活且易于使用的开源数据分析和操作工具，建立在 [Python](https://www.python.org) 编程语言之上。”它被世界各地的数据科学家和软件工程师所使用。

在本教程中，您将学习一些关于使用 pandas 创建不同类型的空或部分空数据帧的知识。然后，您将学习向该数据帧添加数据的几种不同方法。

具体来说，您将了解以下内容:

*   创建空数据帧并添加数据
*   创建包含列的空数据框架并添加数据
*   创建包含列和索引的空数据帧并添加数据

## 在 pandas 中创建空数据帧

有时您只需要创建一个空的数据框架，就像有时您需要创建一个空的 Python 字典或列表一样。

下面是一个用熊猫创建一个完全空的数据帧的例子:

```py
>>> import pandas as pd
>>> df = pd.DataFrame()
>>> df
Empty DataFrame
Columns: []
Index: []

```

当然，空的数据帧不是特别有用。因此，让我们在数据框架中添加一些数据！

```py
>>> import pandas as pd
>>> df = pd.DataFrame()
>>> df
Empty DataFrame
Columns: []
Index: []

>>> df["Name"] = ["Mike", "Steve", "Rodrigo"]
>>> df["Jobs"] = ["Engineer", "Core Dev", "Content Creator"]
>>> df
      Name             Jobs
0     Mike         Engineer
1    Steve         Core Dev
2  Rodrigo  Content Creator
```

这个例子演示了如何在 pandas 中指定列并向这些列添加数据。

现在让我们学习如何创建一个包含列但不包含数据的空 DataFrame！

## 创建包含列的空数据框架

下一个例子将向您展示如何创建一个包含列但不包含索引或列数据的 pandas DataFrame。

让我们来看看:

```py
>>> import pandas as pd

>>> df = pd.DataFrame(columns=["Name", "Job"])
>>> df
Empty DataFrame
Columns: [Name, Job]
Index: []

# Add some data using append()
>>> df = df.append({"Name": "Mike", "Job": "Blogger"}, ignore_index=True)
>>> df
   Name      Job
0  Mike  Blogger
>>> df = df.append({"Name": "Luciano", "Job": "Author"}, ignore_index=True)
>>> df
      Name      Job
0     Mike  Blogger
1  Luciano   Author

```

嗯，这比一个完全空的数据框要好！在本例中，您还将学习如何使用 DataFrame 的 **append()** 方法向每一列添加数据。

当您使用 **append()** 时，它接受一个列名和值的字典。您还将 **ignore_index** 设置为 **True** ，这将让熊猫自动为您更新索引。

现在让我们看看如何用熊猫创建另一种类型的空数据帧！

## 创建包含列和索引的空数据框架

对于这个例子，您将学习如何创建一个有两列和三个命名行或索引的 pandas 数据帧。

这是如何做到的:

```py
>>> import pandas as pd
>>> df = pd.DataFrame(columns = ["Name", "Job"], index = ["a", "b", "c"])
>>> df
  Name  Job
a  NaN  NaN
b  NaN  NaN
c  NaN  NaN
```

当您打印出 DataFrame 时，可以看到所有列都包含 NaN，它代表“不是一个数字”。NaN 有点像 Python 中的 None。

在 pandas 中向该数据帧添加数据的一种方法是使用 **loc** 属性:

```py
>>> df.loc["a"] = ["Mike", "Engineer"]
>>> df
   Name       Job
a  Mike  Engineer
b   NaN       NaN
c   NaN       NaN
>>> df.loc["b"] = ["Steve", "Core Dev"]
>>> df
    Name       Job
a   Mike  Engineer
b  Steve  Core Dev
c    NaN       NaN
>>> df.loc["c"] = ["Rodrigo", "Content Creator"]
>>> df
      Name              Job
a     Mike         Engineer
b    Steve         Core Dev
c  Rodrigo  Content Creator
```

当您使用 **loc** 属性时，您使用类似字典的语法来设置一个值列表的特定索引。上面的例子显示了如何添加三行数据。

## 包扎

这篇教程甚至还没有开始触及你能对熊猫做些什么的皮毛。但是不应该这样。这就是书的作用。在这里，您学习了如何创建三种不同类型的空数据帧。

具体来说，您学到了以下内容:

*   创建空数据帧并添加数据
*   创建包含列的空数据框架并添加数据
*   创建包含列和索引的空数据帧并添加数据

希望你会发现这些例子对你的熊猫之旅有用。编码快乐！
# 在 Python 中使用数据帧行和列

> 原文：<https://www.askpython.com/python-modules/pandas/dataframe-rows-and-columns>

在本文中，让我们看看如何使用 Python 创建类似表格的结构，以及如何处理它们的行和列。当我们创建需要处理大量数据的数据科学应用程序时，这将非常有用。让我们看看如何使用 Python 执行基本功能，比如创建、更新和删除行/列。

## 什么是数据框？

Python 作为一种广泛用于数据分析和处理的语言，有必要以结构化的形式存储数据，比如以行和列的形式存储在我们的传统表格中。我们使用 python 的 [Pandas 库](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)中的 DataFrame 对象来实现这个。在内部，数据以二维数组的形式存储。让我们在本文中了解更多关于 DataFrame 行和列的知识。

## 创建简单的数据框架

让我们通过一个例子来学习[创建一个简单的数据框架](https://www.askpython.com/python-modules/pandas/create-an-empty-dataframe)。

```py
import pandas as pd

data = {
  "TotalScore": [420, 380, 390],
  "MathScore": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df) 

```

## 结果

```py
       TotalScore  MathScore

  0       420        50
  1       380        40
  2       390        45
```

## 选择性地打印一个数据帧列

让我们看看如何在 python 中选择所需的列。假设我们有一个数据帧，如上例所示。我们可以通过它们的列来选择所需的列。

```py
print(df[['MathScore']])

```

上面的代码将只打印“MathScore”列的值。

## 在 Python 中向数据帧添加列

现在，有时，我们可能希望添加更多的列作为数据收集的一部分。我们可以通过声明一个新的列表并将其转换为 data frame 中的一列，向数据框中添加更多的列。

```py
# creating  a new list called name.
name = ['Rhema', 'Mehreen', 'Nitin']

# Using 'Name' as the column name
# and equating it to the list
df['Name'] = name

# Observe the result
print(df)

```

## 输出

```py
   TotalScore  MathScore     Name

0         420         50    Rhema
1         380         40  Mehreen
2         390         45    Nitin
```

## 删除列

我们可以在 pandas 数据帧中使用 drop()方法来删除特定的列。

```py
# dropping passed columns
df.drop(["Name"], axis = 1, inplace = True)

```

现在，列“名称”将从我们的数据框架中删除。

## 使用数据帧行

现在，让我们试着理解在行上执行这些操作的方法。

## 选择一行

要从数据帧中选择行，我们可以使用 loc[]方法或 iloc[]方法。在 loc[]方法中，我们可以使用行的索引值来检索行。我们还可以使用 iloc[]函数来检索使用整数 location to iloc[]函数的行。

```py
# importing pandas package
import pandas as pd

# making data frame from csv file
data = pd.read_csv("employees.csv", index_col ="Name")

# retrieving row by loc method
first = data.loc["Shubham"]
second = data.loc["Mariann"]

print(first, "\n\n\n", second)

```

在上面的代码中，我们将 CSV 文件作为 dataframe 加载，并将列“Name”指定为其索引值。稍后，我们使用行的索引来检索它们。

## 在 Python 中创建数据帧行

要在数据帧中插入新行，我们可以在数据帧中使用 append()函数、concat()函数或 loc[]函数。

```py
#adding a new row using the next index value.
df.loc[len(df.index)] = ['450', '80', 'Disha'] 

display(df)

#using append function

new_data = {'Name': 'Ripun', 'MathScore': 89, 'TotalScore': 465}
df = df.append(new_data, ignore_index = True)

#using concat function

concat_data = {'Name':['Sara', 'Daniel'],
        'MathScore':[89, 90],
        'TotalScore':[410, 445]
       }

df2 = pd.DataFrame(concat_data)

df3 = pd.concat([df, df2], ignore_index = True)
df3.reset_index()

print(df3)

```

## 输出

```py
Using loc[] method

 TotalScore MathScore     Name

0        420        50    Rhema
1        380        40  Mehreen
2        390        45    Nitin
3        450        80    Disha

Using append() function

 TotalScore MathScore     Name

0        420        50    Rhema
1        380        40  Mehreen
2        390        45    Nitin
3        450        80    Disha
4        465        89    Ripun

 Using Concat() function
 TotalScore MathScore     Name

0        420        50    Rhema
1        380        40  Mehreen
2        390        45    Nitin
3        450        80    Disha
4        465        89    Ripun
5        410        89     Sara
6        445        90   Daniel
```

## 删除一行

我们可以使用 drop()方法删除行。我们必须将行的索引值作为参数传递给方法。

```py
# importing pandas module
import pandas as pd

# making data frame from csv file
data = pd.read_csv("employees.csv", index_col ="Name" )

# dropping passed values
data.drop(["Shubham", "Mariann"], inplace = True)

```

## 结论

因此，在本文中，我们讨论了在 python 中处理行和列的各种方法。一般来说，数据框是 Python 中的二维结构，我们可以用它来存储数据和执行各种其他功能。

## 参考

在此找到 data frames-https://pandas.pydata.org/docs/reference/api/pandas.的官方文档 DataFrame.html
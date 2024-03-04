# 重命名数据框架中的特定列

> 原文：<https://www.pythonforbeginners.com/basics/rename-specific-columns-in-dataframe>

[Pandas Dataframes](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python) 用于处理 python 中的表格数据。在本文中，我们将讨论如何在 python 中重命名数据帧中的特定列。

## 通过索引重命名数据帧中的特定列

我们可以使用'`columns`'属性访问 pandas 数据帧中的列名。在 dataframe 对象上调用'`columns`'属性时，会返回一个索引对象。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("The column object is:")
print(columns)
```

输出:

```py
The dataframe is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The column object is:
Index(['Name', 'Roll Number', ' Subject'], dtype='object')
```

Index 对象包含'`values`'属性，其中所有的列名都存储在一个数组中，如下所示。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("The column object is:")
print(columns)
print("The column value is")
print(columns.values)
```

输出:

```py
The dataframe is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The column object is:
Index(['Name', 'Roll Number', ' Subject'], dtype='object')
The column value is
['Name' 'Roll Number' ' Subject']
```

要重命名数据帧中的特定列，我们可以更改值数组的元素。例如，我们可以将值数组中的值`“Roll Number”`更改为`“Registration Number”`，如下所示。

```py
df1.columns.values[1] = "Registration Number"
```

上述变化反映在熊猫数据框架的列名中。因此，数据帧中的`“Roll Number”`列名将更改为`“Registration Number”`列名。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe before modification is:")
print(df1)
df1.columns.values[1] = "Registration Number"
print("The dataframe after modification is:")
print(df1)
```

输出:

```py
The dataframe before modification is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The dataframe after modification is:
     Name  Registration Number      Subject
0  Aditya                   12       Python
1     Sam                   23         Java
2   Chris                   11          C++
3    Joel                   10   JavaScript
4  Mayank                    5   Typescript 
```

若要一次更改多个列名，还可以更改值数组中的多个值。这种变化也将反映在数据帧中。

建议阅读:[机器学习中的回归](https://codinginfinite.com/regression-in-machine-learning-with-examples/)

## 使用 Rename()方法重命名数据帧中的特定列

我们可以使用`rename()`方法来重命名 dataframe 中的特定列，而不是使用“`values`”数组。在 dataframe 上调用`rename()`方法时，该方法将一个字典映射作为其输入参数。映射应该包含需要重命名为 key 的列名，新的列名应该是与字典中的键相关联的值。执行后，`rename()`方法将返回一个新的 dataframe，其中输入字典中给定的特定列被重命名。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe before modification is:")
print(df1)
new_df = df1.rename(columns={'Roll Number': "Registration Number"})
print("The dataframe after modification is:")
print(new_df)
```

输出:

```py
The dataframe before modification is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The dataframe after modification is:
     Name  Registration Number      Subject
0  Aditya                   12       Python
1     Sam                   23         Java
2   Chris                   11          C++
3    Joel                   10   JavaScript
4  Mayank                    5   Typescript 
```

要重命名多个列，可以在作为输入参数提供给`rename()`方法的 [python 字典](https://www.pythonforbeginners.com/dictionary/python-dictionary-quick-guide)中将多个列名及其相应的更改名称作为键-值对传递，如下所示。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe before modification is:")
print(df1)
new_df = df1.rename(columns={' Subject': "Language", 'Roll Number': "Registration Number"})
print("The dataframe after modification is:")
print(new_df)
```

输出:

```py
The dataframe before modification is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The dataframe after modification is:
     Name  Registration Number     Language
0  Aditya                   12       Python
1     Sam                   23         Java
2   Chris                   11          C++
3    Joel                   10   JavaScript
4  Mayank                    5   Typescript 
```

您也可以使用`rename()`方法更改现有数据框架的列名，而不是创建一个列名已更改的新数据框架。为此，我们将使用`rename()`方法的参数`inplace`。'`inplace`'参数有默认值`False`，这意味着原始数据帧没有被修改，在重命名列后返回一个新的数据帧。要修改原始 dataframe 的列名，可以将值`True`作为输入变量传递给'`inplace`'参数，如下所示。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe before modification is:")
print(df1)
df1.rename(columns={' Subject': "Language", 'Roll Number': "Registration Number"},inplace=True)
print("The dataframe after modification is:")
print(df1) 
```

输出:

```py
The dataframe before modification is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The dataframe after modification is:
     Name  Registration Number     Language
0  Aditya                   12       Python
1     Sam                   23         Java
2   Chris                   11          C++
3    Joel                   10   JavaScript
4  Mayank                    5   Typescript 
```

在上面的示例中，您可以观察到在使用'`inplace`'参数后，原始数据帧已经被修改。

## 结论

在本文中，我们讨论了重命名数据帧中特定列的各种方法。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于用 python 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)

请继续关注更多内容丰富的文章。快乐学习！
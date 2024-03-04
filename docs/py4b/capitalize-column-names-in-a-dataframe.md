# 数据框架中的列名大写

> 原文：<https://www.pythonforbeginners.com/basics/capitalize-column-names-in-a-dataframe>

Pandas 数据帧用于在 python 中处理表格数据。在本文中，我们将讨论在 python 中大写数据帧中的列名的不同方法。

## 使用 str.upper()方法将列名大写

数据帧的列名存储在'`columns`'属性中。我们可以使用 columns 属性检索所有的列名，如下所示。

```py
import pandas as pd
df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("The column names are:")
for name in columns:
    print(name)
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
The column names are:
Name
Roll Number
 Subject 
```

由于列名存储在一个列表中，我们可以分配一个包含原始列名的大写值的新列表来大写数据帧的列名。为此，我们将首先创建一个空列表，比如`myList`。之后，我们将使用`upper()`方法将每个列名转换成大写。

在字符串上调用`upper()`方法时，会返回一个包含大写字符的新字符串。我们将为每个列名获取一个大写字符串，并将其存储在`myList`中。之后，我们将把`myList`分配给`columns`属性。这样，大写字符串将作为列名分配给数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("Originally, the column names are:")
for name in columns:
    print(name)
myList = []
for name in columns:
    myList.append(name.upper())
df1.columns = myList
columns = df1.columns
print("After capitalization, the column names are:")
for name in columns:
    print(name)
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
Originally, the column names are:
Name
Roll Number
 Subject
After capitalization, the column names are:
NAME
ROLL NUMBER
 SUBJECT
```

在上面的例子中，不使用 for 循环，您可以使用 [list comprehension](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 来大写列名，如下所示。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("Originally, the column names are:")
for name in columns:
    print(name)
df1.columns = [x.upper() for x in columns]
columns = df1.columns
print("After capitalization, the column names are:")
for name in columns:
    print(name)
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
Originally, the column names are:
Name
Roll Number
 Subject
After capitalization, the column names are:
NAME
ROLL NUMBER
 SUBJECT
```

## 使用 series.str.upper()方法将列名大写

不使用为字符串定义的`upper()`方法，我们可以使用`Series.str.upper()` 方法来大写数据帧中的列名。为了大写列名，我们可以简单地调用存储列名的索引对象上的`upper()`方法。当在`dataframe.columns`对象上调用`Series.str.upper()`时，返回另一个对象，其中所有的列名都是大写的。我们可以将由 `upper()`方法返回的对象分配给 dataframe 的`columns`属性，以大写列名，如下例所示。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("Originally, the column names are:")
for name in columns:
    print(name)
df1.columns = df1.columns.str.upper()
print("After capitalization, the column names are:")
columns = df1.columns
for name in columns:
    print(name)
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
Originally, the column names are:
Name
Roll Number
 Subject
After capitalization, the column names are:
NAME
ROLL NUMBER
 SUBJECT
```

## 使用 rename()方法将列名大写

我们还可以使用`rename()` 方法来大写数据帧中的列名。为此，我们可以将`upper()`方法作为输入参数传递给`rename()` 方法中的`columns`参数。执行后，`rename()`方法返回一个带有大写列名的 dataframe。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("Originally, the column names are:")
for name in columns:
    print(name)
newDf = df1.rename(columns=str.upper)
print("After capitalization, the column names are:")
columns = newDf.columns
for name in columns:
    print(name)
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
Originally, the column names are:
Name
Roll Number
 Subject
After capitalization, the column names are:
NAME
ROLL NUMBER
 SUBJECT
```

在这种方法中，不修改原始数据帧。取而代之的是，用大写的列名创建一个新的 dataframe。要修改原始数据帧中的列名，可以使用`inplace`参数，并在`rename()`方法中给它赋值`True`。这样，原始数据帧将被修改。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
columns = df1.columns
print("Originally, the column names are:")
for name in columns:
    print(name)
df1.rename(columns=str.upper, inplace=True)
print("After capitalization, the column names are:")
columns = df1.columns
for name in columns:
    print(name)
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
Originally, the column names are:
Name
Roll Number
 Subject
After capitalization, the column names are:
NAME
ROLL NUMBER
 SUBJECT
```

## 结论

在本文中，我们讨论了在 python 中大写数据帧中的列名的不同方法。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中[字典理解的文章。您可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)[字符串连接](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)的文章。
# 用 Python 在数据帧中插入新列

> 原文：<https://www.pythonforbeginners.com/basics/insert-new-column-into-a-dataframe-in-python>

在 python 中，数据帧通常用于处理表格数据。在本文中，我们将讨论如何在 python 中向数据帧中插入新列。

## 在 Python 中通过索引将新列插入数据帧

要在 dataframe 中添加新列，我们可以像在 python 字典中添加键值对一样使用索引。在这种方法中，我们首先将需要插入的列的所有元素放入一个列表中。之后，我们将使用下面的语法把这个列表作为一个新列添加到 dataframe 中。

`datframe_name[column_name]= element_lis` t

这里，

*   `datframe_name`是要插入列的数据帧的名称。
*   `column_name`表示包含新列名称的字符串。
*   `element_list`表示包含将被插入数据帧的元素的列表。

以下是通过索引将新列插入数据帧的 python 源代码。

```py
import pandas as pd

df = pd.read_csv('Demo.csv')
print("The dataframe before inserting the column:")
print(df)
column_data = [180, 164, 170]
df['Height'] = column_data
print("The dataframe after inserting the column:")
print(df) 
```

输出:

```py
The dataframe before inserting the column:
   Roll    Name Language
0     1  Aditya   Python
1     2     Sam     Java
2     3   Chris      C++
The dataframe after inserting the column:
   Roll    Name Language  Height
0     1  Aditya   Python     180
1     2     Sam     Java     164
2     3   Chris      C++     170
```

## 使用 assign()方法将新列插入数据帧

不使用索引，我们可以使用`assign()`方法向数据帧中添加一个新列。在 dataframe 上调用`assign()`方法时，使用以下语法将列名和新列的元素列表作为关键字参数。

`datframe_name.assign(column_name= element_list)`

这里，

*   `datframe_name`是要插入列的数据帧的名称。
*   `column_name`表示新列的名称。
*   `element_list`是包含将被插入数据帧的元素的列表。

执行后，`assign()`方法将列插入到数据帧中，并返回更新后的数据帧，如下所示。

```py
import pandas as pd

df = pd.read_csv('Demo.csv')
print("The dataframe before inserting the column:")
print(df)
column_data = [180, 164, 170]
df = df.assign(Height=column_data)
print("The dataframe after inserting the column:")
print(df)
```

输出:

```py
The dataframe before inserting the column:
   Roll    Name Language
0     1  Aditya   Python
1     2     Sam     Java
2     3   Chris      C++
The dataframe after inserting the column:
   Roll    Name Language  Height
0     1  Aditya   Python     180
1     2     Sam     Java     164
2     3   Chris      C++     170 
```

上述方法用于在末尾插入新列。我们还可以在数据帧的任何位置插入新列。为此，我们可以使用 insert()方法。

## 使用 Insert()方法将新列插入数据帧

使用 `insert()`方法，我们可以在数据帧中的任何位置插入一个新列。在 dataframe 上调用`insert()`方法时，该方法将新列插入的位置作为其第一个输入参数，新列的名称作为第二个输入参数，包含新列元素的列表作为第三个输入参数。执行后，它在 dataframe 中的指定位置插入列。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df = pd.read_csv('Demo.csv')
print("The dataframe before inserting the column:")
print(df)
column_data = [180, 164, 170]
df.insert(1, 'Height', column_data)
print("The dataframe after inserting the column:")
print(df)
```

输出:

```py
The dataframe before inserting the column:
   Roll    Name Language
0     1  Aditya   Python
1     2     Sam     Java
2     3   Chris      C++
The dataframe after inserting the column:
   Roll  Height    Name Language
0     1     180  Aditya   Python
1     2     164     Sam     Java
2     3     170   Chris      C++
```

## 结论

在本文中，我们讨论了在 python 中向数据帧中插入新列的三种方法。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中[列表理解的文章。你可能也会喜欢这篇关于 python 中的](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。
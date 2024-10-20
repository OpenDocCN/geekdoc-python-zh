# 检查数据帧是否为空的 4 种方法

> 原文：<https://www.askpython.com/python-modules/pandas/check-if-dataframe-is-empty>

读者你好！在本教程中，我们将讨论四种不同的方法来检查熊猫数据帧是否为空。那么，我们开始吧。

* * *

## 方法 1:使用 DataFrame.empty 属性

这是检查一个[熊猫数据框](https://www.askpython.com/python-modules/pandas/dataframe-indexing)对象是否为空的最常用的方法。在这个方法中，我们将使用 Pandas DataFrame 类的`DataFrame.empty`属性。

当`DataFrame.empty`属性应用于熊猫数据帧对象时，它返回一个布尔值，即`True`或`False`。首先，它检查 DataFrame 对象是否为空，返回**真**，如果 DataFrame 对象不为空，返回**假**。让我们通过 Python 代码来实现这一点。

```py
# Import pandas module
import pandas as pd 

# Create an empty DataFrame
# Using pd.DataFrame() function
df1 = pd.DataFrame()
print('\nThis is DataFrame-1:')
print(df1)

# Create a non-empty DataFrame
# Using pd.DataFrame() function
df2 = pd.DataFrame({'Char': ['A', 'B', 'C', 'D', 'E'],
                    'ASCII': [65, 66, 67, 68, 69]})
print('\nThis is DataFrame-2: ')
print(df2)

# Check if the above created DataFrames are empty 
# Or not using DataFrame.empty attribute
print(f'\nDataFrame-1 is empty: {df1.empty}')
print(f'\nDataFrame-2 is empty: {df2.empty}')

```

**输出:**

```py
This is DataFrame-1:
Empty DataFrame
Columns: []
Index: []

This is DataFrame-2:
  Char  ASCII
0    A     65
1    B     66
2    C     67
3    D     68
4    E     69

DataFrame-1 is empty: True

DataFrame-2 is empty: False

```

## 方法 2:使用 DataFrame.shape 属性

这是第二种最常用的方法来检查给定的 Pandas 数据帧是否为空。在这个方法中，我们将使用 Pandas DataFrame 类的`DataFrame.shape`属性。

`shape`属性返回代表 DataFrame 对象的维度(即行数和列数)的 **[元组](https://www.askpython.com/python/tuple/python-tuple)** 。为了检查 DataFrame 对象是否为空，我们必须在 DataFrame 对象上应用`shape`属性。

然后它检查 DataFrame 对象是否为空。它为返回的元组对象的第零个索引返回**零**值，表示数据帧中的行数为零。

如果 DataFrame 对象不为空，则返回 DataFrame 对象中的行数。让我们编写 Python 代码来实现这一点。

```py
# Import pandas module
import pandas as pd 

# Create an empty DataFrame with 5 columns
# Using pd.DataFrame() function
df1 = pd.DataFrame(columns = ['A', 'B', 'C', 'D', 'E'])
print('\nThis is DataFrame-1:')
print(df1)

# Create a non-empty DataFrame with 5 rows & 2 columns
# Using pd.DataFrame() function
df2 = pd.DataFrame({'Char': ['A', 'B', 'C', 'D', 'E'],
                    'ASCII': [65, 66, 67, 68, 69]})
print('\nThis is DataFrame-2:')
print(df2)

# Check if the above created DataFrames are empty 
# Or not using DataFrame.shape attribute
print(f'\nNumber of rows in DataFrame-1: {df1.shape[0]}')
print(f'\nNumber of rows in DataFrame-2: {df2.shape[0]}')

```

**输出:**

```py
This is DataFrame-1:
Empty DataFrame
Columns: [A, B, C, D, E]
Index: []

This is DataFrame-2:
  Char  ASCII
0    A     65
1    B     66
2    C     67
3    D     68
4    E     69

Number of rows in DataFrame-1: 0

Number of rows in DataFrame-2: 5

```

## 方法 3:将 DataFrame 对象传递给 len()函数

这是一种不常用的方法，用来检查给定的 pandas DataFrame 对象是否为空。在这个方法中，我们将使用`len()`函数。为了检查 DataFrame 是否为空，我们可以直接将 pandas DataFrame 对象传递给`len()`函数。

如果传递的数据帧对象是空数据帧，则 [`len()`函数](https://www.askpython.com/python/string/find-string-length-in-python)返回一个**零**值，表示数据帧对象中的行数为零。但是如果传递的 DataFrame 对象不为空，那么`len()`函数返回一个非零值**表示 DataFrame 对象中的行数。让我们通过 Python 代码来实现这一点。**

```py
# Import pandas module
import pandas as pd 

# Create an empty DataFrame with 3 columns
# Using pd.DataFrame() function
df1 = pd.DataFrame(columns = ['C1', 'C2', 'C3'])
print('\nThis is DataFrame-1:')
print(df1)

# Create a non-empty DataFrame with 4 rows & 2 columns
# Using pd.DataFrame() function
df2 = pd.DataFrame({'Char': ['a', 'b', 'c', 'd'], 'ASCII': [97, 98, 99, 100]})
print('\nThis is DataFrame-2:')
print(df2)

# Check if the above created DataFrames are empty 
# Or not passing the DataFrame object to the len() function
print(f'\nLength of DataFrame-1: {len(df1)}')
print(f'\nLength of DataFrame-2: {len(df2)}')

```

**输出:**

```py
This is DataFrame-1:
Empty DataFrame
Columns: [C1, C2, C3]
Index: []

This is DataFrame-2:
  Char  ASCII
0    a     97
1    b     98
2    c     99
3    d    100

Length of DataFrame-1: 0

Length of DataFrame-2: 4

```

在上面的输出中，DataFrame 的长度表示其中的行数。这就是为什么空数据帧的长度为零，因为其中没有行，而非空数据帧的长度非零，即它等于其中的行数。

## 方法 4:检查数据帧索引的长度

这是检查给定 Pandas DataFrame 对象是否为空的一种不太常用的方法。这里我们也将使用`len()`函数来检查数据帧是否为空。但是我们可以将数据帧索引列表传递给`len()`函数，而不是将整个熊猫数据帧对象传递给`len()`函数。

我们可以使用 pandas DataFrame 类的`DataFrame.index.values`属性获得 DataFrame 索引列表，该属性返回包含 DataFrame 对象索引作为其元素的 Python **列表**。

如果传递的数据帧索引列表为空，则`len()`函数返回一个**零**值。这意味着数据帧的行数为零。但是如果传递的数据帧索引列表不为空，那么`len()`函数返回一个**非零值**，这意味着数据帧索引列表有一些值。让我们看看实现这一点的 Python 代码。

```py
# Import pandas module
import pandas as pd 

# Create an empty DataFrame with 3 columns
# Using pd.DataFrame() function
df1 = pd.DataFrame(columns = ['Col-1', 'Col-2', 'Col-3'])
print('\nThis is DataFrame-1:')
print(df1)

# Create a non-empty DataFrame with 3 rows & 2 columns
# Using pd.DataFrame() function
df2 = pd.DataFrame({'Col-1': ['Python', 'Matlab', 'Csharp'],
                    'Col-2': ['.py', '.mat', '.cs']}, index = ['i', 'ii', 'iii'])
print('\nThis is DataFrame-2:')
print(df2)

# Obtain the DataFrame index list for
# DataFrame-1 & DataFrame-2
# Using the DataFrame.index.values attribute
print(f'\nIndex list of DataFrame-1: {df1.index.values}')
print(f'\nIndex list of DataFrame-2: {df2.index.values}')

# Check if the above created DataFrames are empty 
# Or not passing the DataFrame index list to the len() function
print(f'\nLength of DataFrame-1 index list: {len(df1.index.values)}')
print(f'\nLength of DataFrame-2 index list: {len(df2.index.values)}')

```

**输出:**

```py
This is DataFrame-1:
Empty DataFrame
Columns: [Col-1, Col-2, Col-3]
Index: []

This is DataFrame-2:
      Col-1 Col-2
i    Python   .py
ii   Matlab  .mat
iii  Csharp   .cs

Index list of DataFrame-1: []

Index list of DataFrame-2: ['i' 'ii' 'iii']

Length of DataFrame-1 index list: 0

Length of DataFrame-2 index list: 3

```

## 总结

在本教程中，我们学习了 Python 中四种不同的方法来检查 pandas DataFrame 对象是否为空。希望你已经理解了上面讨论的事情。想了解更多关于熊猫的信息，请继续关注我们。
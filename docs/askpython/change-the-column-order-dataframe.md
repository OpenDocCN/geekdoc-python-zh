# 在 Python 中改变熊猫数据帧的列顺序的 4 种方法

> 原文：<https://www.askpython.com/python-modules/pandas/change-the-column-order-dataframe>

在本教程中，我们将讨论如何改变一个给定的 pandas DataFrame 对象的列顺序。在数据预处理阶段，我们可能会遇到这样一种情况，即相关的 pandas 数据帧的列没有按照期望的顺序排列，那么我们必须改变数据帧的列顺序。

***也读作:[熊猫数据帧索引:设置一个熊猫数据帧的索引](https://www.askpython.com/python-modules/pandas/dataframe-indexing)***

* * *

## 如何改变熊猫数据框的列顺序？

让我们来看看在 Pandas 中改变数据帧的列顺序的不同方法。

### 方法 1:使用所需的顺序列列表

这是改变 pandas DataFrame 对象的列顺序的最简单的方法之一。在这个方法中，我们简单地将 DataFrame 的列的 Python 列表以期望的顺序传递给 DataFrame 对象。让我们看看如何用 Python 编写这个方法。

```py
# Method-1

# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame
df = pd.DataFrame({'Roll': [111, 112, 113, 114, 115],
                    'Name': ['Sanjay', 'Aman', 'Ankit', 'Ravi', 'Komal'],
                    'Branch': ['ECE', 'ICE', 'IT', 'CSE', 'CHE'],
                    'CGPA': [8.15, 9.03, 7.85, 8.55, 9.45]})
print('Given pandas DataFrame:\n')
print(df)

# Change the order of the DataFrame
# Using the desired order columns list
df_1 = df[['Name', 'CGPA', 'Roll', 'Branch']]
print('\nPandas DataFrame with changed column order:\n')
print(df_1)

```

**输出:**

```py
Given pandas DataFrame:

   Roll    Name Branch  CGPA
0   111  Sanjay    ECE  8.15
1   112    Aman    ICE  9.03
2   113   Ankit     IT  7.85
3   114    Ravi    CSE  8.55
4   115   Komal    CHE  9.45

Pandas DataFrame with changed column order:

     Name  CGPA  Roll Branch
0  Sanjay  8.15   111    ECE
1    Aman  9.03   112    ICE
2   Ankit  7.85   113     IT
3    Ravi  8.55   114    CSE
4   Komal  9.45   115    CHE

```

### 方法 2:使用 loc 方法

在这个方法中，我们将利用 pandas DataFrame 类的`loc`方法。使用`loc`方法，我们可以通过提供列名的 [Python 列表](https://www.askpython.com/python/difference-between-python-list-vs-array)来重新排序 pandas DataFrame 对象的列。让我们编写 Python 代码来实现这个方法。

```py
# Method-2

# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame
df = pd.DataFrame({'Name': ['Sanjay', 'Aman', 'Ankit', 'Ravi', 'Komal'],
                    'Roll': [111, 112, 113, 114, 115],
                    'Branch': ['ECE', 'ICE', 'IT', 'CSE', 'CHE'],
                    'CGPA': [8.15, 9.03, 7.85, 8.55, 9.45]})
print('Given pandas DataFrame:\n')
print(df)

# Change the order of the DataFrame
# Using the loc method of pandas DataFrame class
df_2 = df.loc[2:4, ['Roll', 'Name', 'CGPA', 'Branch']]
print('\nPandas DataFrame with changed column order:\n')
print(df_2)

```

**输出:**

```py
Given pandas DataFrame:

     Name  Roll Branch  CGPA
0  Sanjay   111    ECE  8.15
1    Aman   112    ICE  9.03
2   Ankit   113     IT  7.85
3    Ravi   114    CSE  8.55
4   Komal   115    CHE  9.45

Pandas DataFrame with changed column order:

   Roll   Name  CGPA Branch
2   113  Ankit  7.85     IT
3   114   Ravi  8.55    CSE
4   115  Komal  9.45    CHE

```

### 方法 3:使用 iloc 方法

在这个方法中，我们将使用 pandas DataFrame 类的`iloc`方法。使用`iloc`方法，我们可以通过提供列索引的 Python 列表(即 0，1，2，3，…)而不是列名来重新排序 pandas DataFrame 对象的列。让我们看看如何通过 Python 代码实现这个方法。

```py
# Method-3

# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame
df = pd.DataFrame({'CGPA': [8.15, 9.03, 7.85, 8.55, 9.45],
                    'Name': ['Sanjay', 'Aman', 'Ankit', 'Ravi', 'Komal'],
                    'Roll': [111, 112, 113, 114, 115],
                    'Branch': ['ECE', 'ICE', 'IT', 'CSE', 'CHE']})
print('Given pandas DataFrame:\n')
print(df)

# Change the order of the DataFrame
# Using the iloc method of pandas DataFrame class
df_3 = df.iloc[1:4, [1, 2, 0, 3]]
print('\nPandas DataFrame with changed column order:\n')
print(df_3)

```

**输出:**

```py
Given pandas DataFrame:

   CGPA    Name  Roll Branch
0  8.15  Sanjay   111    ECE
1  9.03    Aman   112    ICE
2  7.85   Ankit   113     IT
3  8.55    Ravi   114    CSE
4  9.45   Komal   115    CHE

Pandas DataFrame with changed column order:

    Name  Roll  CGPA Branch
1   Aman   112  9.03    ICE
2  Ankit   113  7.85     IT
3   Ravi   114  8.55    CSE

```

**注意:**在上面的两个方法`loc`和`iloc`中，我们有一个额外的优势，即在给定的 pandas DataFrame 对象中只选择一系列行。

### 方法 4:使用 reindex()函数

在这个方法中，我们将使用 pandas DataFrame 对象的`reindex()`函数。使用`reindex()`函数，我们可以通过传递列名的 Python 列表来重新排列 pandas DataFrame 对象的列顺序。让我们通过 Python 代码来实现这个方法。

```py
# Method-4

# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame
df = pd.DataFrame({'Branch': ['ECE', 'ICE', 'IT', 'CSE', 'CHE'],
                    'CGPA': [8.15, 9.03, 7.85, 8.55, 9.45],
                    'Name': ['Sanjay', 'Aman', 'Ankit', 'Ravi', 'Komal'],
                    'Roll': [111, 112, 113, 114, 115]})
print('Given pandas DataFrame:\n')
print(df)

# Change the order of the DataFrame
# Using the reindex() function of pandas DataFrame class
df_4 = df.reindex(columns = ['Roll', 'CGPA', 'Name', 'Branch'])
print('\nPandas DataFrame with changed column order:\n')
print(df_4)

```

**输出:**

```py
Given pandas DataFrame:

  Branch  CGPA    Name  Roll
0    ECE  8.15  Sanjay   111
1    ICE  9.03    Aman   112
2     IT  7.85   Ankit   113
3    CSE  8.55    Ravi   114
4    CHE  9.45   Komal   115

Pandas DataFrame with changed column order:

   Roll  CGPA    Name Branch
0   111  8.15  Sanjay    ECE
1   112  9.03    Aman    ICE
2   113  7.85   Ankit     IT
3   114  8.55    Ravi    CSE
4   115  9.45   Komal    CHE

```

## 总结

在本教程中，我们学习了如何用四种不同的方法来改变 pandas DataFrame 对象的列顺序。希望你已经理解了上面讨论的所有方法，并且很乐意自己使用它们。感谢您的阅读，请继续关注我们，了解更多关于 Python 编程的精彩内容。
# 如何在 Python 熊猫中获取一个数据帧的索引？

> 原文：<https://www.askpython.com/python-modules/pandas/get-index-of-dataframe>

大家好！在本教程中，我们将讨论获取熊猫数据帧对象的索引或行的不同方法。那么，我们开始吧。

* * *

## Python 中获取数据帧索引的方法

让我们进入寻找数据帧索引的步骤。此外，查看如何[重置数据帧](https://www.askpython.com/python-modules/pandas/reset-index-of-a-dataframe)的索引，以确保每次追加或排序数据帧时，索引号都是对齐的。

### 方法 1:使用 for 循环

在 Python 中，我们可以使用循环的[很容易地获得熊猫 DataFrame 对象的索引或行。在这个方法中，我们将使用 Python 中的](https://www.askpython.com/course/python-course-for-loop) [pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)的`pd.DataFrame()`函数从 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-comprehension)中创建一个 pandas DataFrame 对象。然后，我们将在 pandas DataFrame 索引对象上运行一个 for 循环来打印索引。让我们通过 Python 代码来实现这一点。

```py
# Method-1

# Import pandas
import pandas as pd

# Create a Python dictionary
data = {"Name": ['Sanjay', 'Shreya', 'Raju', 'Gopal', 'Ravi'],
        "Roll": [101, 102, 103, 104, 105]}

# Create a DataFrame object from above dictionary
df = pd.DataFrame(data, index = [1, 2, 3, 4, 5])
print("This is DataFrame:\n")
print(df)

# Get the index/rows of the above DataFrame 
# Using for loop iteration
print("\nThis is index of DataFrame:\n")
for idx in df.index:
    print(idx, end = ' ')

```

**输出:**

```py
This is DataFrame:

     Name  Roll
1  Sanjay   101
2  Shreya   102
3    Raju   103
4   Gopal   104
5    Ravi   105

This is index of DataFrame:

1 2 3 4 5

```

### 方法 2:使用索引属性

这是获取 DataFrame 对象索引的最广泛使用的方法。在这个方法中，我们将照常使用的`pd.DataFrame()`函数创建一个 pandas DataFrame 对象。然后我们将使用 pandas DataFrame 类的`index`属性来获取 pandas DataFrame 对象的索引。当我们在 pandas DataFrame 对象上应用`index`属性时，它返回一个包含 DataFrame 的**索引列表**的[元组](https://www.askpython.com/python/tuple/python-tuple)。让我们看看如何在 Python 编程中实现这一点。

```py
# Method-2

# Import pandas
import pandas as pd

# Create a Python dictionary
data = {"Name": ['Sanjay', 'Shreya', 'Raju', 'Gopal', 'Ravi'],
        "Roll": [101, 102, 103, 104, 105],
        "CGPA": [8.15, 8.18, 9.32, 8.85, 7.87]}

# Create a DataFrame object from above dictionary
df = pd.DataFrame(data, index = ['s1', 's2', 's3', 's4', 's5'])
print("This is DataFrame:\n")
print(df)

# Get the index/rows of the above DataFrame 
# Using index attribute
print("\nThis is index of DataFrame:\n")
index_list = df.index
print(index_list)

```

**输出:**

```py
This is DataFrame:

      Name  Roll  CGPA
s1  Sanjay   101  8.15
s2  Shreya   102  8.18
s3    Raju   103  9.32
s4   Gopal   104  8.85
s5    Ravi   105  7.87

This is index of DataFrame:

Index(['s1', 's2', 's3', 's4', 's5'], dtype='object')

```

### 方法 3:使用 index.values 属性

首先，我们将使用 pd 创建一个熊猫 DataFrame 对象。熊猫 Python 模块的 DataFrame()函数。然后，我们将使用 pandas DataFrame 对象的 index.values 属性来访问它的索引列表。当我们在 pandas DataFrame 对象上应用 index.values 属性时，它返回一个数组，表示 pandas DataFrame 对象的索引列表中的数据。让我们进入 Python 代码来实现这种获取数据帧索引列表的方法。

```py
# Method-3

# Import pandas
import pandas as pd

# Create a Python dictionary
data = {"Name": ['Sanjay', 'Shreya', 'Raju', 'Gopal', 'Ravi'],
        "Roll": [101, 102, 103, 104, 105],
        "Branch": ['ECE', 'CSE', 'EEE', 'ICE', 'IPE'],
        "CGPA": [8.15, 8.18, 9.32, 8.85, 7.87]}

# Create a DataFrame object from above dictionary
df = pd.DataFrame(data)
print("This is DataFrame:\n")
print(df)

# Get the index/rows of the above DataFrame 
# Using index.values property
print("\nThis is index of DataFrame:\n")
index_list = df.index.values
print(index_list)

```

**输出:**

```py
This is DataFrame:

     Name  Roll Branch  CGPA
0  Sanjay   101    ECE  8.15
1  Shreya   102    CSE  8.18
2    Raju   103    EEE  9.32
3   Gopal   104    ICE  8.85
4    Ravi   105    IPE  7.87

This is index of DataFrame:

[0 1 2 3 4]

```

### 方法 4:使用 tolist()函数

这是 pandas 模块的一个便利工具，它将 pandas DataFrame 对象的索引转换成一个 [Python 列表](https://www.askpython.com/python/difference-between-python-list-vs-array)。在这个方法中，我们使用 pd 创建一个 pandas DataFrame 对象。DataFrame()函数，就像我们在前面的方法中所做的那样。然后我们将使用 pandas DataFrame 类的`index`属性访问 pandas DataFrame 索引对象。最后，我们将应用`tolist()`函数，它实际上以 Python 列表的形式返回数据帧的索引。让我们编写 Python 程序来实现这个方便的方法，以获得 Python 列表中熊猫数据帧的索引。

```py
# Method-4

# Import pandas
import pandas as pd

# Create a Python dictionary
data = {"Name": ['Sanjay', 'Shreya', 'Raju', 'Gopal', 'Ravi'],
        "Roll": [101, 102, 103, 104, 105],
        "Branch": ['ECE', 'CSE', 'EEE', 'ICE', 'IPE'],
        "CGPA": [8.15, 8.18, 9.32, 8.85, 7.87]}

# Create a DataFrame object from above dictionary
df = pd.DataFrame(data, index = ['R1', 'R2', 'R3', 'R4', 'R5'])
print("This is DataFrame:\n")
print(df)

# Get the index/rows of the above DataFrame 
# Using tolist() function
print("\nThis is index of DataFrame:\n")
index_list = df.index.tolist()
print(index_list)

```

**输出:**

```py
This is DataFrame:

      Name  Roll Branch  CGPA
R1  Sanjay   101    ECE  8.15
R2  Shreya   102    CSE  8.18
R3    Raju   103    EEE  9.32
R4   Gopal   104    ICE  8.85
R5    Ravi   105    IPE  7.87

This is index of DataFrame:

['R1', 'R2', 'R3', 'R4', 'R5']

```

### 方法 5:使用 query()和 tolist()函数

使用这种方法，我们可以只获得 pandas DataFrame 对象的满足特定标准的特定索引。在这个方法中，我们将使用`pd.DataFrame()`函数创建一个 pandas DataFrame 对象，并使用 pandas DataFrame 类的`query()`函数。当我们在数据帧上应用`query()`函数并传递一个条件时，它返回一个数据帧，该数据帧只包含满足传递给它的条件的行。

之后，我们将应用 DataFrame 类的`index`属性，并使用`tolist()`函数返回 DataFrame 索引值的 Python 列表。

让我们看看实现这个有用方法的 Python 代码，以获得满足给定条件的 pandas DataFrame 对象的选定行或索引。

```py
# Method-5

# Import pandas
import pandas as pd

# Create a Python dictionary
data = {"Name": ['Sanjay', 'Shreya', 'Raju', 'Gopal', 'Ravi'],
        "Roll": [101, 102, 103, 104, 105],
        "Branch": ['ECE', 'CSE', 'EEE', 'ICE', 'IPE'],
        "CGPA": [8.15, 9.32, 8.78, 7.87, 8.85]}

# Create a DataFrame object from above dictionary
df = pd.DataFrame(data, index = ['I', 'II', 'III', 'IV', 'V'])
print("This is DataFrame:\n")
print(df)

# Get the index/rows of the above DataFrame
# Using query() and tolist() functions
print("\nThis is index of DataFrame:\n")
index_list = df.query("CGPA > 8.5").index.tolist()
print(index_list)

```

**输出:**

```py
This is DataFrame:

       Name  Roll Branch  CGPA
I    Sanjay   101    ECE  8.15
II   Shreya   102    CSE  9.32
III    Raju   103    EEE  8.78
IV    Gopal   104    ICE  7.87
V      Ravi   105    IPE  8.85

This is index of DataFrame:

['II', 'III', 'V']

```

## 总结

在本教程中，我们学习了四种不同的方法来获取 DataFrame 对象的索引。希望你已经理解了以上内容，并对自己尝试这些方法感到兴奋。谢谢你，请继续关注我们的更多这类 Python 教程。
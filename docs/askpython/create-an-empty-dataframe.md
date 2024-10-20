# 如何在 Python 中创建一个空的 DataFrame？

> 原文：<https://www.askpython.com/python-modules/pandas/create-an-empty-dataframe>

读者你好！在本教程中，我们将讨论在 Python 中创建空数据帧的不同方法。我们还将讨论空数据帧和具有 n an 值的数据帧之间的区别。那么，我们开始吧。

* * *

## Python 中的空数据帧是什么？

在 Python 中，数据帧是由 Python [**pandas 模块**](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 提供的二维数据结构，该模块以表格格式即行和列的形式存储数据。一个**空数据帧**是一个完全空的熊猫数据帧对象(其中没有数据)，所有轴的长度为零。它必须有零行数或零列数。

我们可以使用 pandas DataFrame 对象的`DataFrame.empty`属性来检查 pandas DataFrame 对象是否为空。当我们在 pandas DataFrame 对象上应用这个属性时，它返回一个布尔值，即*真*或*假*，这取决于相关 DataFrame 对象是否为空的条件。

## 创建空数据帧的方法

在 Python 中，我们可以通过以下方式创建一个空的熊猫数据帧。让我们一个一个地了解这些。

### 1.创建一个没有任何行或列的完整的空数据帧

这是使用`pd.DataFrame()`函数创建一个空的 pandas DataFrame 对象的最简单和最容易的方法。在这个方法中，我们简单地调用 pandas DataFrame 类构造函数，不带任何参数，这又返回一个空的 pandas DataFrame 对象。让我们看看实现这个方法的 Python 代码。

```py
# Method-1

# Import pandas module
import pandas as pd 

# Create an empty DataFrame without 
# Any any row or column
# Using pd.DataFrame() function
df1 = pd.DataFrame()
print('This is our DataFrame with no row or column:\n')
print(df1)

# Check if the above created DataFrame
# Is empty or not using the empty property
print('\nIs this an empty DataFrame?\n')
print(df1.empty)

```

**输出:**

```py
This is our DataFrame with no row or column:

Empty DataFrame
Columns: []
Index: []

Is this an empty DataFrame?

True

```

### 2.创建一个只有行的空数据帧

这是使用`pd.DataFrame()`函数创建一个空的 [pandas DataFrame](https://www.askpython.com/python-modules/pandas/dataframe-indexing) 对象的另一个简单方法，该对象只包含行。在这个方法中，我们将调用带有一个参数的 pandas DataFrame 类构造函数- **index** ，它反过来返回一个空的 Pandas DataFrame 对象以及传递的行或索引列表。让我们编写 Python 代码来实现这个方法。

```py
# Method-2

# Import pandas module
import pandas as pd 

# Create an empty DataFrame with
# Five rows but no columns
# Using pd.DataFrame() function with rows parameter
df2 = pd.DataFrame(index = ['R1', 'R2', 'R3', 'R4', 'R5'])
print('This is our DataFrame with rows only no columns:\n')
print(df2)

# Check if the above created DataFrame
# Is empty or not using the empty property
print('\nIs this an empty DataFrame?\n')
print(df2.empty)

```

**输出:**

```py
This is our DataFrame with rows only no columns:

Empty DataFrame
Columns: []
Index: [R1, R2, R3, R4, R5]

Is this an empty DataFrame?

True

```

### 3.创建一个只有列的空数据框架

为了使用`pd.DataFrame()`函数创建一个只包含列的空 Pandas DataFrame 对象，我们调用带有一个参数的 Pandas DataFrame 类构造函数—**columns**,它反过来返回一个空 Pandas DataFrame 对象和传递的列列表。让我们通过 Python 代码来实现这个方法。

```py
# Method-3

# Import pandas module
import pandas as pd 

# Create an empty DataFrame with
# Five columns but no rows
# Using pd.DataFrame() function with columns parameter
df3 = pd.DataFrame(columns = ['C1', 'C2', 'C3', 'C4', 'C5'])
print('This is our DataFrame with columns only no rows:\n')
print(df3)

# Check if the above created DataFrame
# Is empty or not using the empty property
print('\nIs this an empty DataFrame?\n')
print(df3.empty)

```

**输出:**

```py
This is our DataFrame with columns only no rows:

Empty DataFrame
Columns: [C1, C2, C3, C4, C5]
Index: []

Is this an empty DataFrame?

True

```

### 4.创建一个包含行和列的空数据框架

在这个方法中，我们创建一个空的 Pandas DataFrame 对象，它包含行和列。当我们用两个参数调用 pandas DataFrame 类构造函数时- **列**和**索引**，它返回一个空的 pandas DataFrame 对象，带有传递的索引和列列表。让我们看看如何通过 Python 代码实现这个方法。

```py
# Method-4

# Import pandas module
import pandas as pd 

# Create an empty DataFrame with
# Five rows and five columns
# Using pd.DataFrame() function 
# With columns & index parameters
df4 = pd.DataFrame(columns = ['C1', 'C2', 'C3', 'C4', 'C5'],
                   index = ['R1', 'R2', 'R3', 'R4', 'R5'])
print('This is our DataFrame with both rows and columns:\n')
print(df4)

# Check if the above created DataFrame
# Is empty or not using the empty property
print('\nIs this an empty DataFrame?\n')
print(df4.empty)

```

**输出:**

```py
This is our DataFrame with both rows and columns:

     C1   C2   C3   C4   C5
R1  NaN  NaN  NaN  NaN  NaN
R2  NaN  NaN  NaN  NaN  NaN
R3  NaN  NaN  NaN  NaN  NaN
R4  NaN  NaN  NaN  NaN  NaN
R5  NaN  NaN  NaN  NaN  NaN

Is this an empty DataFrame?

False

```

**注意:**这个方法有一个问题，我们可以看到它的输出,`empty`属性返回了 False。这意味着我们在这个方法中创建的数据帧不会被 pandas 模块视为空数据帧。

## 空数据帧与具有 NaN 值的数据帧

我们已经看到了上述 Python 代码输出的问题。Pandas 模块对空数据帧和具有所有 NaN 值的数据帧进行不同的处理。

发生这种情况是因为当我们试图使用这种方法创建一个空的 pandas DataFrame 时，我们没有在 DataFrame 对象中提供或输入任何数据，但默认情况下，它会填充有 [**NaN** 值](https://www.askpython.com/python/examples/nan-in-numpy-and-pandas)。

这就是为什么当我们将`empty`属性应用于这种熊猫数据帧时，它返回 False。

因此，克服这个问题的一个简单的解决方案是删除所有默认放置在数据帧中的 NaN 值。我们可以使用 pandas DataFrame 类的 [`dropna()`函数](https://www.askpython.com/python/examples/detection-removal-outliers-in-python)删除 DataFrame 中的所有 NaN 值。然后我们在 DataFrame 对象上应用`empty`属性来检查结果，它将返回 True。让我们通过 Python 代码来实现这一点。

```py
# Compare an empty DataFrame
# With a DataFrame with all NaN values

# Import pandas module
import pandas as pd 

# Create an empty DataFrame with
# Three rows and four columns
# Using pd.DataFrame() function 
# With columns & index parameters
df = pd.DataFrame(columns = ['Col-1', 'Col-2', 'Col-3', 'Col-4'],
                   index = ['Row-1', 'Row-2', 'Row-3'])
print('This is our DataFrame with NaN values:\n')
print(df)

# Check if the above created DataFrame
# Is empty or not using the empty property
print('\nIs this an empty DataFrame?\n')
print(df.empty)

# Remove all the NaN values using dropna() function
# Then apply the empty attribute/property on the DataFrame
print('\nAfter removing all the NaN values:\n')
print('Is this an empty DataFrame?\n')
print(df.dropna().empty)

```

**输出:**

```py
This is our DataFrame with NaN values:

      Col-1 Col-2 Col-3 Col-4
Row-1   NaN   NaN   NaN   NaN
Row-2   NaN   NaN   NaN   NaN
Row-3   NaN   NaN   NaN   NaN

Is this an empty DataFrame?

False

After removing all the NaN values:

Is this an empty DataFrame?

True

```

## 结论

在本教程中，我们已经学习了四种创建空 Pandas DataFrame 对象的方法，以及空 DataFrame 和具有 n an 值的 DataFrame 之间的区别。希望你已经理解了上面讨论的所有内容，并对自己尝试这些方法感到兴奋。谢谢你，请继续关注我们，以获得更多精彩的 Python 教程。
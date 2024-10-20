# 将索引转到熊猫数据框架中的列

> 原文：<https://www.askpython.com/python-modules/pandas/index-to-column-dataframe>

你好。在本 Python 教程中，我们将讨论如何将数据帧索引转换为列。我们还将看到如何将多索引数据帧的多级索引转换成它的多列。所以让我们开始吧。

* * *

## 熊猫数据框架中的索引是什么？

Pandas 是一个健壮的 Python 库，广泛用于数据分析。它为我们提供了一个名为 **DataFrame** 的数据结构，以行和列的形式存储数据，其中每一行都有一个唯一的索引值。一个 pandas DataFrame 对象可以有一个以上的索引级别，在这种情况下，它被称为多索引 DataFrame。

每当我们创建一个 panda DataFrame 对象时，默认情况下，从**零**到**行数–1**的索引值按顺序分配给 DataFrame 的每一行。尽管我们也可以使用 pandas 中的`DataFrame.set_index()`函数为 pandas DataFrame 对象的每一行手动设置索引值。

我们可以使用以下两种方法将 pandas DataFrame 对象的一级或多级索引转换成它的列。为了演示将 DataFrame 索引转换为列的过程，让我们首先创建一个 pandas DataFrame 对象。

***也读作:[熊猫数据帧索引:设置一个熊猫数据帧的索引](https://www.askpython.com/python-modules/pandas/dataframe-indexing)***

## 熊猫数据框中索引到列的转换方法

```py
# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame object
df = pd.DataFrame({'Dept': ['ECE', 'ICE', 'IT', 'CSE', 'CHE'],
                    'GPA': [8.15, 9.03, 7.85, 8.55, 9.45],
                    'Name': ['Kirti', 'Sarthak', 'Anubhav', 'Ranjan', 'Kartik'],
                    'RegNo': [111, 112, 113, 114, 115]})

# Set 'RegNo' as index of the pandas DataFrame
df.set_index('RegNo', inplace=True)                    

# Print the created pandas DataFrame object
print('Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Sample pandas DataFrame:

      Dept   GPA     Name
RegNo                    
111    ECE  8.15    Kirti
112    ICE  9.03  Sarthak
113     IT  7.85  Anubhav
114    CSE  8.55   Ranjan
115    CHE  9.45   Kartik

```

### 方法 1:创建一个新的 DataFrame 列并传递索引

这是将 DataFrame 索引转换为列的最简单方法。在这个方法中，我们简单地在 DataFrame 中创建一个新列，并使用 pandas DataFrame 类的`DataFrame.index`方法将索引传递给它。让我们看看实现这个方法的 Python 代码。

```py
# Method 1

# Convert the index of the sample DataFrame into column
# Using the new column method
df['Roll'] = df.index                    

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

      Dept   GPA     Name  Roll
RegNo                          
111    ECE  8.15    Kirti   111
112    ICE  9.03  Sarthak   112
113     IT  7.85  Anubhav   113
114    CSE  8.55   Ranjan   114
115    CHE  9.45   Kartik   115

```

### 方法 2:在 pandas 中使用 DataFrame.reset_index()函数

这是将一个或多个级别的 DataFrame 索引转换为一个或多个列的广泛使用的方法。在这个方法中，我们将使用 pandas DataFrame 类的`DataFrame.reset_index()`函数。让我们编写 Python 代码来实现这个方法。

```py
# Method 2

# Convert the index of the sample DataFrame into column
# Using the DataFrame.reset_index() function
df.reset_index(inplace=True)                    

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

   RegNo Dept   GPA     Name
0    111  ECE  8.15    Kirti
1    112  ICE  9.03  Sarthak
2    113   IT  7.85  Anubhav
3    114  CSE  8.55   Ranjan
4    115  CHE  9.45   Kartik

```

## 将多索引数据框架的一个或多个级别转换为列

让我们首先通过使用`DataFrame.set_index()`函数将`RegNo`和`Name`设置为样本数据帧的多级索引，将上述样本数据帧转换为多索引数据帧。

```py
# Convert the sample DataFrame into MultiIndex DataFrame
# By setting the 'RegNo' and 'Name' as Multi-level index
df.set_index(['RegNo', 'Name'], inplace=True)                    

# Print the modified pandas DataFrame
print('Modified Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified Sample pandas DataFrame:

              Dept   GPA
RegNo Name              
111   Kirti    ECE  8.15
112   Sarthak  ICE  9.03
113   Anubhav   IT  7.85
114   Ranjan   CSE  8.55
115   Kartik   CHE  9.45

```

现在让我们编写 Python 代码，使用`DataFrame.reset_index()`函数将示例多索引数据帧中的一个索引级别转换成一个列。

```py
# Convert one level of the MultiIndex DataFrame into column
# Using the DataFrame.reset_index() function
df.reset_index(level='Name', inplace=True)                  

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

          Name Dept   GPA
RegNo                    
111      Kirti  ECE  8.15
112    Sarthak  ICE  9.03
113    Anubhav   IT  7.85
114     Ranjan  CSE  8.55
115     Kartik  CHE  9.45

```

## 总结

在本教程中，我们学习了如何将熊猫数据帧的索引转换成它的列。我们还学习了将多索引数据帧的一个或多个级别的索引转换成它的列。希望你已经理解了上面讨论的东西，并准备好用你自己的熊猫数据框架进行实验。感谢阅读！请继续关注我们，了解更多与 Python 编程相关的精彩学习内容。
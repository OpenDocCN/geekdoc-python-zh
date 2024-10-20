# 获得熊猫数据帧或系列的头部和尾部

> 原文：<https://www.askpython.com/python-modules/pandas/head-and-tail-of-dataframe-series>

在本 Python 教程中，我们将讨论获得熊猫数据帧或系列对象的头部和尾部的不同方法。所以让我们开始吧。

* * *

## 为什么要得到熊猫的头部和尾部？

我们都知道 **[Pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)** 是一个必不可少的 Python 库，广泛用于数据分析。众所周知，数据分析处理非常大的数据集。因此，为了快速浏览大样本数据集(以 pandas DataFrame 或 Series 对象的形式加载)，我们需要 pandas DataFrame 或 Series 的头部和尾部。

我们主要使用 pandas DataFrame 类的`DataFrame.head()`和`DataFrame.tail()`函数来分别获取 pandas DataFrame 或系列的第一行和最后一行`N`(默认情况下是这个 **N** = 5 的值)。

## 熊猫数据帧的头部和尾部

因此，在继续讨论熊猫数据帧对象的头部和尾部之前，让我们创建一个样本熊猫数据帧对象。

### 创建一个示例 pandas DataFrame 对象

```py
# Import pandas Python module
import pandas as pd

# Create a large pandas DataFrame object
df = pd.DataFrame({'RegNo': [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
                   'Dept': ['ECE', 'ICE', 'IT', 'CSE', 'CHE', 'EE', 'ME', 'CSE', 'ICE', 'TT', 'ECE', 'IT', 'ME', 'BT', 'EE']})

# Print the created pandas DataFrame
print('Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Sample pandas DataFrame:

    RegNo Dept
0     111  ECE
1     112  ICE
2     113   IT
3     114  CSE
4     115  CHE
5     116   EE
6     117   ME
7     118  CSE
8     119  ICE
9     120   TT
10    121  ECE
11    122   IT
12    123   ME
13    124   BT
14    125   EE

```

### 得到熊猫的头像数据帧:熊猫。DataFrame.head()

```py
# Get the head of the sample pandas Series
print('First 10 rows of the sample pandas DataFrame:\n')
temp_df = df.head(10)
print(temp_df)

```

**输出:**

```py
First 10 rows of the sample pandas DataFrame:

   RegNo Dept
0    111  ECE
1    112  ICE
2    113   IT
3    114  CSE
4    115  CHE
5    116   EE
6    117   ME
7    118  CSE
8    119  ICE
9    120   TT

```

### 得到熊猫的尾巴数据帧:熊猫。DataFrame.tail()

```py
# Get the tail of the sample pandas Series
print('Last 10 rows of the sample pandas DataFrame:\n')
temp_df = df.tail(10)
print(temp_df)

```

**输出:**

```py
Last 10 rows of the sample pandas DataFrame:

    RegNo Dept
5     116   EE
6     117   ME
7     118  CSE
8     119  ICE
9     120   TT
10    121  ECE
11    122   IT
12    123   ME
13    124   BT
14    125   EE

```

### 将熊猫数据帧的头部和尾部放在一起:pandas.option_context()

```py
# Get the head and tail of the sample pandas DataFrame
# Using the pd.option_context() function in Pandas
print('First and Last 5 rows of the sample pandas DataFrame:\n')
with pd.option_context('display.max_rows',10):
    print(df)

```

**输出:**

```py
First and Last 5 rows of the sample pandas DataFrame:

    RegNo Dept
0     111  ECE
1     112  ICE
2     113   IT
3     114  CSE
4     115  CHE
..    ...  ...
10    121  ECE
11    122   IT
12    123   ME
13    124   BT
14    125   EE

[15 rows x 2 columns]

```

## 熊猫系列的头尾

因此，在继续讨论熊猫系列对象的头部和尾部之前，让我们创建一个熊猫系列对象的样本。

### 创建一个示例熊猫系列对象

```py
# Import pandas Python module
import pandas as pd
# Import NumPy Python module
import numpy as np

# Create a pandas Series
sr = pd.Series(np.random.randn(1000))

# Print the created pandas Series
print('Sample pandas Series:\n')
print(sr)

```

**输出:**

```py
Sample pandas Series:

0     -0.157775
1     -0.108095
2     -0.876501
3     -0.591994
4     -0.435755
         ...   
995    1.186127
996   -0.898278
997   -0.267392
998    1.295608
999   -2.024564
Length: 1000, dtype: float64

```

### 得到一个熊猫系列的头:熊猫。Series.head()

```py
# Get the head of the sample pandas Series
print('First 10 values of the sample pandas Series:\n')
temp_sr = sr.head(10)
print(temp_sr)

```

**输出:**

```py
First 10 values of the sample pandas Series:

0   -0.157775
1   -0.108095
2   -0.876501
3   -0.591994
4   -0.435755
5   -1.204434
6   -0.035977
7    0.015345
8   -0.453117
9   -0.695302
dtype: float64

```

### 得到一个熊猫系列的尾巴:熊猫。Series.tail()

```py
# Get the tail of the sample pandas Series
print('Last 10 values of the sample pandas Series:\n')
temp_sr = sr.tail(10)
print(temp_sr)

```

**输出:**

```py
Last 10 values of the sample pandas Series:

990   -0.239336
991   -1.475996
992   -0.162860
993    0.405505
994    0.458872
995    1.186127
996   -0.898278
997   -0.267392
998    1.295608
999   -2.024564
dtype: float64

```

## 总结

在这个 Python 教程中，我们已经学习了如何使用`head()`和`tail()`函数获得熊猫数据帧或系列的头部和尾部。我们还看到了如何在 pandas 中使用`pandas.option_context()`功能同时获得熊猫数据帧的头部和尾部。希望您已经理解了上面讨论的内容，并对使用熊猫功能快速浏览您的大熊猫数据框架感到兴奋。感谢阅读！请继续关注我们，获取更多关于 Python 编程的学习资源。
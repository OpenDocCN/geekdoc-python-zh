# 删除熊猫数据框多列的 8 种方法

> 原文：<https://www.askpython.com/python-modules/pandas/drop-multiple-columns-dataframe>

在这个 Python 教程中，我们将探索**不同的方法来删除一个** [熊猫数据帧](https://www.askpython.com/python-modules/pandas/create-an-empty-dataframe)的多个列。所以，让我们开始吧！

* * *

## 方法来删除数据帧的多列

在开始之前，我们需要一个示例数据框架。下面是我将在本教程中使用的数据帧的一小段代码。请随意复制粘贴这段代码，并按照本教程进行操作。

```py
# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame object
df = pd.DataFrame({'Dept': ['ECE', 'ICE', 'IT', 'CSE', 'CHE', 'EE', 'TE', 'ME', 'CSE', 'IPE', 'ECE'],
                    'GPA': [8.15, 9.03, 7.85, 8.55, 9.45, 7.45, 8.85, 9.35, 6.53,8.85, 7.83],
                    'Name': ['Mohan', 'Gautam', 'Tanya', 'Rashmi', 'Kirti', 'Ravi', 'Sanjay', 'Naveen', 'Gaurav', 'Ram', 'Tom'],
                    'RegNo': [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]})

# Print the created sample pandas DataFrame
print('Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Sample pandas DataFrame:

   Dept   GPA    Name  RegNo
0   ECE  8.15   Mohan    111
1   ICE  9.03  Gautam    112
2    IT  7.85   Tanya    113
3   CSE  8.55  Rashmi    114
4   CHE  9.45   Kirti    115
5    EE  7.45    Ravi    116
6    TE  8.85  Sanjay    117
7    ME  9.35  Naveen    118
8   CSE  6.53  Gaurav    119
9   IPE  8.85     Ram    120
10  ECE  7.83     Tom    121

```

### 方法 1:使用 del 关键字

```py
# Drop 'GPA' column using del keyword
del df['GPA']

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

   Dept    Name  RegNo
0   ECE   Mohan    111
1   ICE  Gautam    112
2    IT   Tanya    113
3   CSE  Rashmi    114
4   CHE   Kirti    115
5    EE    Ravi    116
6    TE  Sanjay    117
7    ME  Naveen    118
8   CSE  Gaurav    119
9   IPE     Ram    120
10  ECE     Tom    121

```

### 方法 2:使用 DataFrame.pop()函数

```py
# Drop 'RegNo' column using DataFrame.pop() function
df.pop('RegNo')

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

   Dept   GPA    Name
0   ECE  8.15   Mohan
1   ICE  9.03  Gautam
2    IT  7.85   Tanya
3   CSE  8.55  Rashmi
4   CHE  9.45   Kirti
5    EE  7.45    Ravi
6    TE  8.85  Sanjay
7    ME  9.35  Naveen
8   CSE  6.53  Gaurav
9   IPE  8.85     Ram
10  ECE  7.83     Tom

```

### 方法 3:使用带有列参数的 DataFrame.drop()函数

```py
# Drop 'GPA' and 'Name' column using DataFrame.drop() function with columns parameter
df.drop(columns=['GPA','Name'], inplace=True)

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

   Dept  RegNo
0   ECE    111
1   ICE    112
2    IT    113
3   CSE    114
4   CHE    115
5    EE    116
6    TE    117
7    ME    118
8   CSE    119
9   IPE    120
10  ECE    121

```

### 方法 4:使用带有轴参数的 DataFrame.drop()函数

```py
# Drop 'Dept' and 'GPA' columns using DataFrame.drop() function with axis parameter
df.drop(['Dept','GPA'], axis=1, inplace=True)

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

      Name  RegNo
0    Mohan    111
1   Gautam    112
2    Tanya    113
3   Rashmi    114
4    Kirti    115
5     Ravi    116
6   Sanjay    117
7   Naveen    118
8   Gaurav    119
9      Ram    120
10     Tom    121

```

### 方法 5:使用 DataFrame.drop()函数和 DataFrame.iloc[]

```py
# Drop 'Name' and 'GPA' column using DataFrame.drop() function and DataFrame.iloc[]
df.drop(df.iloc[:,1:3], axis=1, inplace=True)

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

输出:

```py
Modified pandas DataFrame:

   Dept  RegNo
0   ECE    111
1   ICE    112
2    IT    113
3   CSE    114
4   CHE    115
5    EE    116
6    TE    117
7    ME    118
8   CSE    119
9   IPE    120
10  ECE    121

```

### 方法 6:使用 DataFrame.drop()函数和 DataFrame.columns[]

```py
# Drop 'Name' and 'Dept' columns using DataFrame.drop() function and DataFrame.columns[]
df.drop(df.columns[[0,2]], axis=1, inplace=True)

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified pandas DataFrame:

     GPA  RegNo
0   8.15    111
1   9.03    112
2   7.85    113
3   8.55    114
4   9.45    115
5   7.45    116
6   8.85    117
7   9.35    118
8   6.53    119
9   8.85    120
10  7.83    121

```

### 方法 7:仅选择所需的列

```py
# Drop 'RegNo' and 'Dept' columns by selecting only the required columns
df2 = df[['Name','GPA']]

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Modified pandas DataFrame:

      Name   GPA
0    Mohan  8.15
1   Gautam  9.03
2    Tanya  7.85
3   Rashmi  8.55
4    Kirti  9.45
5     Ravi  7.45
6   Sanjay  8.85
7   Naveen  9.35
8   Gaurav  6.53
9      Ram  8.85
10     Tom  7.83

```

### 方法 8:使用 DataFrame.dropna()函数

首先，创建一个具有 NaN 值的 pandas 数据帧。下面是相同的代码片段。

```py
# Import pandas Python module
import pandas as pd
# Import NumPy module
import numpy as np

# Create a pandas DataFrame object with NaN values
df = pd.DataFrame({'Dept': ['ECE', 'ICE', 'IT', 'CSE', 'CHE', 'EE', 'TE', 'ME', 'CSE', 'IPE', 'ECE'],
                    'GPA': [8.15, 9.03, 7.85, np.nan, 9.45, 7.45, np.nan, 9.35, 6.53,8.85, 7.83],
                    'Name': ['Mohan', 'Gautam', 'Tanya', 'Rashmi', 'Kirti', 'Ravi', 'Sanjay', 'Naveen', 'Gaurav', 'Ram', 'Tom'],
                    'RegNo': [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
                    'City': ['Biharsharif','Ranchi',np.nan,'Patiala', 'Rajgir', 'Patna', np.nan,'Mysore',np.nan,'Mumbai',np.nan]})

# Print the created pandas DataFrame
print('Sample pandas DataFrame with NaN values:\n')
print(df)

```

**输出:**

```py
Sample pandas DataFrame with NaN values:

   Dept   GPA    Name  RegNo         City
0   ECE  8.15   Mohan    111  Biharsharif
1   ICE  9.03  Gautam    112       Ranchi
2    IT  7.85   Tanya    113          NaN
3   CSE   NaN  Rashmi    114      Patiala
4   CHE  9.45   Kirti    115       Rajgir
5    EE  7.45    Ravi    116        Patna
6    TE   NaN  Sanjay    117          NaN
7    ME  9.35  Naveen    118       Mysore
8   CSE  6.53  Gaurav    119          NaN
9   IPE  8.85     Ram    120       Mumbai
10  ECE  7.83     Tom    121          NaN

```

现在，我们将删除具有 NaN 值的列。

```py
# Drop columns with NaN values using the DataFrame.dropna() function
df2 = df.dropna(axis='columns')

# Print the modified pandas DataFrame
print('Modified pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Modified pandas DataFrame:

   Dept    Name  RegNo
0   ECE   Mohan    111
1   ICE  Gautam    112
2    IT   Tanya    113
3   CSE  Rashmi    114
4   CHE   Kirti    115
5    EE    Ravi    116
6    TE  Sanjay    117
7    ME  Naveen    118
8   CSE  Gaurav    119
9   IPE     Ram    120
10  ECE     Tom    121

```

## 结论

在本教程中，我们学习了不同的方法来删除熊猫数据帧的多个列。希望你已经理解了上面讨论的方法，并乐于在你的数据分析项目中使用它们。感谢阅读！请继续关注我们，了解更多关于 Python 编程的精彩学习内容。
# DataFrame.query()函数:如何查询熊猫 DataFrame？

> 原文：<https://www.askpython.com/python-modules/pandas/query-pandas-dataframe>

在本 Python 教程中，我们将讨论如何使用 DataFrame.query()函数来查询熊猫数据帧。那么，我们开始讨论吧。

* * *

## pandas 中 DataFrame.query()函数的语法

`pandas.DataFrame.query(expr, inplace=False, **kwargs)`

**expr** =包含逻辑表达式的字符串，根据该表达式选择熊猫数据帧的行(当 expr=True 时)。
**原地** =这是一个布尔值(“`True`”或“`False`”)，它将决定是原地修改数据帧还是返回修改后的数据帧的新副本。
****kwargs** =引用其他关键字参数(如果有)。

## 何时使用 DataFrame.query()函数？

**Pandas** 为我们提供了许多从 pandas DataFrame 对象中选择或过滤行的方式/方法。pandas 中的`DataFrame.query()`函数是过滤 pandas DataFrame 对象的行的健壮方法之一。

并且最好使用`DataFrame.query()`函数来选择或过滤熊猫数据帧对象的行，而不是传统的和常用的索引方法。这个`DataFrame.query()`函数也可以和其他 pandas 方法一起使用，使数据操作变得平滑和简单。

## DataFrame.query()函数的示例

让我们创建一个示例 pandas DataFrame 对象，并借助几个例子来理解`DataFrame.query()`函数的功能。

### 创建一个示例 pandas DataFrame 对象

```py
# Import pandas Python module
import pandas as pd

# Create a pandas DataFrame object
df = pd.DataFrame({'Dept': ['ECE', 'ICE', 'IT', 'CSE', 'CHE', 'EE', 'TE', 'ME', 'CSE', 'IPE', 'ECE'],
                    'GPA': [8.85, 9.03, 7.85, 8.85, 9.45, 7.45, 6.85, 9.35, 6.53,8.85, 7.83],
                    'Name': ['Mohan', 'Gautam', 'Tanya', 'Rashmi', 'Kirti', 'Ravi', 'Sanjay', 'Naveen', 'Gaurav', 'Ram', 'Tom'],
                    'RegNo': [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
                    'City': ['Biharsharif','Ranchi','Patna','Patiala','Rajgir','Patna','Patna','Mysore','Patna','Mumbai','Patna']})

# Print the created pandas DataFrame
print('Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Sample pandas DataFrame:

   Dept   GPA    Name  RegNo         City
0   ECE  8.85   Mohan    111  Biharsharif
1   ICE  9.03  Gautam    112       Ranchi
2    IT  7.85   Tanya    113        Patna
3   CSE  8.85  Rashmi    114      Patiala
4   CHE  9.45   Kirti    115       Rajgir
5    EE  7.45    Ravi    116        Patna
6    TE  6.85  Sanjay    117        Patna
7    ME  9.35  Naveen    118       Mysore
8   CSE  6.53  Gaurav    119        Patna
9   IPE  8.85     Ram    120       Mumbai
10  ECE  7.83     Tom    121        Patna

```

### 示例#1

选择示例数据框中的行，其中(City = "Patna ")。

```py
# Filter the rows of the sample DataFrame which has City = 'Patna'
# Using the DataFrame.query() function
df2 = df.query('City=="Patna"')

# Print the filtered sample pandas DataFrame
print('Filtered sample pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Filtered sample pandas DataFrame:

   Dept   GPA    Name  RegNo   City
2    IT  7.85   Tanya    113  Patna
5    EE  7.45    Ravi    116  Patna
6    TE  6.85  Sanjay    117  Patna
8   CSE  6.53  Gaurav    119  Patna
10  ECE  7.83     Tom    121  Patna

```

### 实施例 2

选择样本数据框的行，其中(GPA < 8)。

```py
# Filter the rows of the sample DataFrame which has GPA < 8
# Using the DataFrame.query() function
df2 = df.query('GPA < 8' & City == "Patna")

# Print the filtered sample pandas DataFrame
print('Filtered sample pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Filtered sample pandas DataFrame:

   Dept   GPA    Name  RegNo   City
2    IT  7.85   Tanya    113  Patna
5    EE  7.45    Ravi    116  Patna
6    TE  6.85  Sanjay    117  Patna
8   CSE  6.53  Gaurav    119  Patna
10  ECE  7.83     Tom    121  Patna

```

### 实施例 3

选择示例数据框中的行，其中(GPA < 7 且 City = 'Patna ')。

```py
# Filter the rows of the sample DataFrame which has GPA < 7 & City = 'Patna'
# Using the DataFrame.query() function
df2 = df.query('GPA < 7 & City == "Patna"')

# Print the filtered sample pandas DataFrame
print('Filtered sample pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Filtered sample pandas DataFrame:

  Dept   GPA    Name  RegNo   City
6   TE  6.85  Sanjay    117  Patna
8  CSE  6.53  Gaurav    119  Patna

```

### 实施例 4

选择在[ECE，CSE，IT]中有 Dept 的样本数据帧的行。

```py
# Filter the rows of the sample DataFrame which has Dept in (ECE, CSE, IT)
# Using the DataFrame.query() function
df2 = df.query("Dept in ['CSE','ECE','IT']")

# Print the filtered sample pandas DataFrame
print('Filtered sample pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Filtered sample pandas DataFrame:

   Dept   GPA    Name  RegNo         City
0   ECE  8.85   Mohan    111  Biharsharif
2    IT  7.85   Tanya    113        Patna
3   CSE  8.85  Rashmi    114      Patiala
8   CSE  6.53  Gaurav    119        Patna
10  ECE  7.83     Tom    121        Patna

```

### 实施例 5

选择样本数据帧的行，其中(RegNo < 115 and GPA > 7)。

```py
# Filter the rows of the sample DataFrame which has (RegNo < 115 & GPA > 7)
# Using the DataFrame.query() function
df2 = df.query("RegNo < 115 & GPA > 7")

# Print the filtered sample pandas DataFrame
print('Filtered sample pandas DataFrame:\n')
print(df2)

```

**输出:**

```py
Filtered sample pandas DataFrame:

  Dept   GPA    Name  RegNo         City
0  ECE  8.85   Mohan    111  Biharsharif
1  ICE  9.03  Gautam    112       Ranchi
2   IT  7.85   Tanya    113        Patna
3  CSE  8.85  Rashmi    114      Patiala

```

## 总结

在这个 Python 教程中，我们学习了如何使用 Pandas 中的`DataFrame.query()`函数来查询 pandas DataFrame 对象。希望您已经理解了上面讨论的概念和示例，并准备好使用它们来查询您自己的 pandas 数据框架。感谢阅读！请继续关注我们，了解更多关于 Python 编程的精彩学习内容。
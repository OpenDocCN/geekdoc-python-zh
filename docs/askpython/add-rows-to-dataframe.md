# 向熊猫数据框架添加行的 5 种简单方法

> 原文：<https://www.askpython.com/python-modules/pandas/add-rows-to-dataframe>

在本 Python 教程中，我们将讨论向 pandas DataFrame 对象添加或插入一行或多行的前五种方法。那么，我们开始讨论吧。

* * *

## 向 Pandas 数据帧添加行的方法

让我们首先创建一个示例 pandas DataFrame 对象，然后我们将使用以下方法向它添加一行或多行。

```py
# Import pandas Python module
import pandas as pd

# Create a sample pandas DataFrame object
df = pd.DataFrame({'RegNo': [111, 112, 113, 114, 115],
                   'Name': ['Gautam', 'Tanya', 'Rashmi', 'Kirti', 'Ravi'],
                   'CGPA': [8.85, 9.03, 7.85, 8.85, 9.45],
                   'Dept': ['ECE', 'ICE', 'IT', 'CSE', 'CHE'],
                   'City': ['Jalandhar','Ranchi','Patna','Patiala','Rajgir']})

# Print the created pandas DataFrame
print('Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Sample pandas DataFrame:

   RegNo    Name  CGPA Dept       City
0    111  Gautam  8.85  ECE  Jalandhar
1    112   Tanya  9.03  ICE     Ranchi
2    113  Rashmi  7.85   IT      Patna
3    114   Kirti  8.85  CSE    Patiala
4    115    Ravi  9.45  CHE     Rajgir

```

### 方法 1

将 pandas 系列对象作为一行添加到现有的 pandas DataFrame 对象中。

```py
# Create a pandas Series object with all the column values passed as a Python list
s_row = pd.Series([116,'Sanjay',8.15,'ECE','Biharsharif'], index=df.columns)

# Append the above pandas Series object as a row to the existing pandas DataFrame
# Using the DataFrame.append() function
df = df.append(s_row,ignore_index=True)

# Print the modified pandas DataFrame object after addition of a row
print('Modified Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified Sample pandas DataFrame:

   RegNo    Name  CGPA Dept         City
0    111  Gautam  8.85  ECE    Jalandhar
1    112   Tanya  9.03  ICE       Ranchi
2    113  Rashmi  7.85   IT        Patna
3    114   Kirti  8.85  CSE      Patiala
4    115    Ravi  9.45  CHE       Rajgir
5    116  Sanjay  8.15  ECE  Biharsharif

```

### 方法 2

将一个 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-comprehension)作为一行添加到现有的 pandas DataFrame 对象中。

```py
# Create a Python dictionary object with all the column values
d_row = {'RegNo':117,'Name':"Sarthak",'CGPA':8.88,'Dept':"ECE",'City':"Allahabad"}

# Append the above Python dictionary object as a row to the existing pandas DataFrame
# Using the DataFrame.append() function
df = df.append(d_row,ignore_index=True)

# Print the modified pandas DataFrame object after addition of a row
print('Modified Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified Sample pandas DataFrame:

   RegNo     Name  CGPA Dept         City
0    111   Gautam  8.85  ECE    Jalandhar
1    112    Tanya  9.03  ICE       Ranchi
2    113   Rashmi  7.85   IT        Patna
3    114    Kirti  8.85  CSE      Patiala
4    115     Ravi  9.45  CHE       Rajgir
5    116   Sanjay  8.15  ECE  Biharsharif
6    117  Sarthak  8.88  ECE    Allahabad

```

**注意:**在传递 Python 字典或熊猫系列时，请将`DataFrame.append()`函数的`ignore_index`参数设置为`True`，否则会抛出错误。

### 方法 3

使用`DataFrame.loc[]`方法将 Python 列表对象作为一行添加到现有的 pandas DataFrame 对象中。

```py
# Create a Python list object with all the column values
l_row = [118,"Kanika",7.88,"EE","Varanasi"]

# Append the above Python list object as a row to the existing pandas DataFrame
# Using the DataFrame.loc[]
df.loc[7] = l_row

# Print the modified pandas DataFrame object after addition of a row
print('Modified Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified Sample pandas DataFrame:

   RegNo     Name  CGPA Dept         City
0    111   Gautam  8.85  ECE    Jalandhar
1    112    Tanya  9.03  ICE       Ranchi
2    113   Rashmi  7.85   IT        Patna
3    114    Kirti  8.85  CSE      Patiala
4    115     Ravi  9.45  CHE       Rajgir
5    116   Sanjay  8.15  ECE  Biharsharif
6    117  Sarthak  8.88  ECE    Allahabad
7    118   Kanika  7.88   EE     Varanasi

```

### 方法 4

使用`DataFrame.append()`函数将一个 pandas DataFrame 对象的行添加到另一个 pandas DataFrame 对象中。

```py
# Create a new pandas DataFrame object
df2 = pd.DataFrame({'RegNo': [119, 120, 121],
                   'Name': ['Gaurav', 'Thaman', 'Radha'],
                   'CGPA': [8.85, 9.03, 7.85],
                   'Dept': ['ECE', 'ICE', 'IT'],
                   'City': ['Jalandhar','Ranchi','Patna']})

# Print the newly created pandas DataFrame object
print('New pandas DataFrame:\n')
print(df2)

# Append the rows of the above pandas DataFrame to the existing pandas DataFrame
# Using the DataFrame.append()
df = df.append(df2,ignore_index=True)

# Print the modified pandas DataFrame object after addition of rows
print('\nModified Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
New pandas DataFrame:

   RegNo    Name  CGPA Dept       City
0    119  Gaurav  8.85  ECE  Jalandhar
1    120  Thaman  9.03  ICE     Ranchi
2    121   Radha  7.85   IT      Patna

Modified Sample pandas DataFrame:

    RegNo    Name  CGPA Dept         City
0     111  Gautam  8.85  ECE    Jalandhar
1     112   Tanya  9.03  ICE       Ranchi
2     113  Rashmi  7.85   IT        Patna
3     114   Kirti  8.85  CSE      Patiala
4     115    Ravi  9.45  CHE       Rajgir
5     116  Sanjay  8.15  ECE  Biharsharif
6     116  Sanjay  8.15  ECE  Biharsharif
7     118  Kanika  7.88   EE     Varanasi
8     119  Gaurav  8.85  ECE    Jalandhar
9     120  Thaman  9.03  ICE       Ranchi
10    121   Radha  7.85   IT        Patna

```

### 方法 5

使用`DataFrame.iloc[]`方法在现有 pandas DataFrame 对象的特定索引位置添加一行。

```py
# Create a Python list object with all the column values
i_row = [122,"Zahir",6.88,"ME","Kolkata"]

# Append the above Python list object as a row to the existing pandas DataFrame
# At index 2 using the DataFrame.iloc[]
df.iloc[2] = i_row

# Print the modified pandas DataFrame object after addition of a row
print('Modified Sample pandas DataFrame:\n')
print(df)

```

**输出:**

```py
Modified Sample pandas DataFrame:

    RegNo    Name  CGPA Dept         City
0     111  Gautam  8.85  ECE    Jalandhar
1     112   Tanya  9.03  ICE       Ranchi
2     122   Zahir  6.88   ME      Kolkata
3     114   Kirti  8.85  CSE      Patiala
4     115    Ravi  9.45  CHE       Rajgir
5     116  Sanjay  8.15  ECE  Biharsharif
6     116  Sanjay  8.15  ECE  Biharsharif
7     118  Kanika  7.88   EE     Varanasi
8     119  Gaurav  8.85  ECE    Jalandhar
9     120  Thaman  9.03  ICE       Ranchi
10    121   Radha  7.85   IT        Patna

```

**注意:**使用`DataFrame.iloc[]`方法时请小心，因为它会用新行替换索引位置的现有行。

## 结论

在本教程中，我们已经学习了在现有的 pandas DataFrame 对象中添加或插入一行或多行的前五种方法。希望你已经很好地理解了上面讨论的东西，并准备在自己的数据分析项目中使用这些方法。感谢阅读！请继续关注我们，获取更多关于 Python 编程的精彩学习资源。
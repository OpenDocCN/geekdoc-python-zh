# 用 Python 从数据帧中选择行

> 原文：<https://www.pythonforbeginners.com/basics/select-row-from-a-dataframe-in-python>

Pandas 数据帧用于在 Python 中处理表格数据。在本文中，我们将讨论如何用 Python 从数据帧中选择一行。我们还将讨论如何使用布尔运算符从 pandas 数据帧中选择数据。

## 使用 iloc 属性从数据帧中选择行

属性包含一个作为数据帧中行的有序集合的对象。`iloc`属性的功能类似于[列表索引](https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet-2)。您可以使用`iloc`属性从数据框中选择一行。为此，您可以简单地使用带有`iloc`属性的方括号内的行的位置来选择熊猫数据帧的一行，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
position=1
row=myDf.iloc[position]
print("The row at position {} is :{}".format(position,row))
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The row at position 1 is :Class        1
Roll        12
Name     Chris
Name: 1, dtype: object
```

在这里，您可以观察到`iloc`属性将指定位置的行作为输出。

## 使用 Python 中的 loc 属性从数据帧中选择行

dataframe 的`loc`属性的工作方式类似于 python 字典的键。`loc`属性包含一个`_LocIndexer`对象，您可以用它从一个[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)中选择行。您可以使用带有`loc`属性的方括号内的索引标签来访问[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)的元素，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
index=2
row=myDf.loc[index]
print("The row at index {} is :{}".format(index,row))
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The row at index 2 is :Class      1
Roll      13
Name     Sam
Name: 2, dtype: object
```

如果您已经为数据帧定义了一个自定义索引，您可以使用行的索引值从 pandas 数据帧中选择该行，如下所示。

```py
myDf=pd.read_csv("samplefile.csv",index_col=0)
print("The dataframe is:")
print(myDf)
index=1
row=myDf.loc[index]
print("The row at index {} is :{}".format(index,row))
```

输出:

```py
The dataframe is:
       Roll      Name
Class                
1        11    Aditya
1        12     Chris
1        13       Sam
2         1      Joel
2        22       Tom
2        44  Samantha
3        33      Tina
3        34       Amy
The row at index 1 is :       Roll    Name
Class              
1        11  Aditya
1        12   Chris
1        13     Sam
```

如果有多级索引，可以使用这些索引从数据帧中选择行，如下所示。

```py
myDf=pd.read_csv("samplefile.csv",index_col=[0,1])
print("The dataframe is:")
print(myDf)
index=(1,12)
row=myDf.loc[index]
print("The row at index {} is :{}".format(index,row))
```

输出:

```py
The dataframe is:
                Name
Class Roll          
1     11      Aditya
      12       Chris
      13         Sam
2     1         Joel
      22         Tom
      44    Samantha
3     33        Tina
      34         Amy
The row at index (1, 12) is :Name    Chris
Name: (1, 12), dtype: object
```

## 在 Pandas 数据框架中使用列名选择列

要从数据帧中选择一列，可以使用带方括号的列名，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
column_name="Class"
column=myDf[column_name]
print("The {} column is :{}".format(column_name,column))
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The Class column is :0    1
1    1
2    1
3    2
4    2
5    2
6    3
7    3
Name: Class, dtype: int64
```

如果要从数据帧中选择多个列，可以将列名列表传递给方括号，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
column_names=["Class","Name"]
column=myDf[column_names]
print("The {} column is :{}".format(column_names,column))
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The ['Class', 'Name'] column is :   Class      Name
0      1    Aditya
1      1     Chris
2      1       Sam
3      2      Joel
4      2       Tom
5      2  Samantha
6      3      Tina
7      3       Amy
```

## 熊猫数据帧中的布尔屏蔽

布尔屏蔽用于检查数据帧中的条件。当我们在 dataframe 列上应用一个[布尔操作符](https://www.pythonforbeginners.com/basics/boolean)时，它根据如下所示的条件返回一个包含`True`和`False`值的 pandas 系列对象。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
boolean_mask=myDf["Class"]>1
print("The boolean mask is:")
print(boolean_mask)
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The boolean mask is:
0    False
1    False
2    False
3     True
4     True
5     True
6     True
7     True
Name: Class, dtype: bool
```

您可以使用布尔掩码从数据帧中选择行。为此，您需要将包含布尔掩码的序列传递给方括号运算符，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
boolean_mask=myDf["Class"]>1
print("The boolean mask is:")
print(boolean_mask)
print("The rows in which class>1 is:")
rows=myDf[boolean_mask]
print(rows)
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The boolean mask is:
0    False
1    False
2    False
3     True
4     True
5     True
6     True
7     True
Name: Class, dtype: bool
The rows in which class>1 is:
   Class  Roll      Name
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
```

除了使用方括号，您还可以使用`where()`方法通过布尔屏蔽从数据帧中选择行。在 dataframe 上调用`where()`方法时，该方法将布尔掩码作为其输入参数，并返回包含所需行的新 dataframe，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
boolean_mask=myDf["Class"]>1
print("The boolean mask is:")
print(boolean_mask)
print("The rows in which class>1 is:")
rows=myDf.where(boolean_mask)
print(rows)
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The boolean mask is:
0    False
1    False
2    False
3     True
4     True
5     True
6     True
7     True
Name: Class, dtype: bool
The rows in which class>1 is:
   Class  Roll      Name
0    NaN   NaN       NaN
1    NaN   NaN       NaN
2    NaN   NaN       NaN
3    2.0   1.0      Joel
4    2.0  22.0       Tom
5    2.0  44.0  Samantha
6    3.0  33.0      Tina
7    3.0  34.0       Amy
```

在上面使用`where()`方法的例子中，布尔掩码的值为`False`、`NaN`的行存储在 dataframe 中。您可以删除包含`NaN`值的行，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
boolean_mask=myDf["Class"]>1
print("The boolean mask is:")
print(boolean_mask)
print("The rows in which class>1 is:")
rows=myDf.where(boolean_mask).dropna()
print(rows)
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The boolean mask is:
0    False
1    False
2    False
3     True
4     True
5     True
6     True
7     True
Name: Class, dtype: bool
The rows in which class>1 is:
   Class  Roll      Name
3    2.0   1.0      Joel
4    2.0  22.0       Tom
5    2.0  44.0  Samantha
6    3.0  33.0      Tina
7    3.0  34.0       Amy
```

您还可以使用逻辑运算符从两个或多个条件创建布尔掩码，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
print(myDf)
boolean_mask=(myDf["Class"]>1) & (myDf["Class"]<3)
print("The boolean mask is:")
print(boolean_mask)
print("The rows in which class>1 and <3 is:")
rows=myDf.where(boolean_mask).dropna()
print(rows)
```

输出:

```py
The dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The boolean mask is:
0    False
1    False
2    False
3     True
4     True
5     True
6    False
7    False
Name: Class, dtype: bool
The rows in which class>1 and <3 is:
   Class  Roll      Name
3    2.0   1.0      Joel
4    2.0  22.0       Tom
5    2.0  44.0  Samantha
```

创建布尔掩码后，您可以使用它来选择布尔掩码包含 True 的行，如下所示。

## 结论

在本文中，我们讨论了如何从数据帧中选择一行。我们还讨论了如何从数据帧中选择一列，以及如何使用布尔掩码从数据帧中选择多行。

要了解更多关于 python 编程的知识，你可以阅读这篇关于[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。如果你想进入机器学习领域，你可以阅读这篇关于机器学习中的[回归的文章。](https://codinginfinite.com/regression-in-machine-learning-with-examples/)
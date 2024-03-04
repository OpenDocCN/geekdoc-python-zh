# 用 Python 创建熊猫数据框架

> 原文：<https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python>

Pandas dataframe 是 Python 中处理表格数据的主要数据结构。在本文中，我们将讨论使用 pandas 模块在 Python 中创建数据帧的不同方法。

## 在 Python 中创建一个空数据帧

要创建一个空的数据帧，可以使用`DataFrame()` 功能。在没有任何输入参数的情况下执行时，`DataFrame()`函数将返回一个没有任何列或行的空数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
myDf=pd.DataFrame()
print(myDf)
```

输出:

```py
Empty DataFrame
Columns: []
Index: []
```

要创建具有指定列名的空数据帧，可以使用`DataFrame()`函数中的 columns 参数。`columns`参数将一个列表作为其输入参数，并将列表元素分配给 dataframe 的列名，如下所示。

```py
import pandas as pd
myDf=pd.DataFrame(columns=["A", "B", "C"])
print(myDf)
```

输出:

```py
Empty DataFrame
Columns: [A, B, C]
Index: []
```

这里，我们创建了一个数据帧，其中包含 A、B 和 C 列，行中没有任何数据。

## 从字典创建熊猫数据框架

你可以使用`DataFrame()`函数从 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中创建一个熊猫数据帧。为此，您首先需要创建一个字典列表。之后，您可以将字典列表传递给`DataFrame()`函数。执行后，`DataFrame()`函数将返回一个新的数据帧，如下例所示。

```py
import pandas as pd
dict1={"A":1,"B":12,"C":14}
dict2={"A":13,"B":17,"C":12}
dict3={"A":2,"B":11,"C":14}
dictList=[dict1,dict2,dict3]
myDf=pd.DataFrame(dictList)
print(myDf)
```

输出:

```py
 A   B   C
0   1  12  14
1  13  17  12
2   2  11  14
```

从字典的[列表创建数据帧时，字典的关键字被用作数据帧的列名。如果所有的字典都不包含相同的键，那么对应于一个字典的行将包含列中的`NaN`值，这些值在字典中不作为键出现。您可以在下面的示例中观察到这一点。](https://www.pythonforbeginners.com/basics/read-csv-into-list-of-dictionaries-in-python)

```py
import pandas as pd
dict1={"A":1,"B":12,"C":14}
dict2={"A":13,"B":17,"C":12}
dict3={"A":2,"B":11,"C":14,"D":1117}
dictList=[dict1,dict2,dict3]
myDf=pd.DataFrame(dictList)
print(myDf)
```

输出:

```py
 A   B   C       D
0   1  12  14     NaN
1  13  17  12     NaN
2   2  11  14  1117.0
```

在这个例子中，第一行和第二行对应于没有 D 作为关键字的字典。因此，这些行包含列 d 中的`NaN`值。

## 从 Python 中的系列创建熊猫数据帧

dataframe 由 pandas 系列对象作为其列组成。您也可以将一系列对象的列表传递给`DataFrame()`函数来创建一个数据帧，如下所示。

```py
series1 = pd.Series([1,2,3])
series2 = pd.Series([4,12,34])
series3 = pd.Series([22,33,44])
seriesList=[series1,series2,series3]
myDf=pd.DataFrame(seriesList)
print(myDf)
```

输出:

```py
 0   1   2
0   1   2   3
1   4  12  34
2  22  33  44
```

正如您所看到的，series 对象的键标签被转换成数据帧的列。因此，如果作为输入给出的系列对象具有不同的索引标签，则结果数据帧的列名将是所有系列对象的索引标签的并集。此外，数据帧中对应于一个系列的行将包含列中的`NaN`值，这些值不作为索引标签出现在该系列中。您可以在下面的示例中观察到这一点。

```py
series1 = pd.Series({"A":1,"B":12,"C":14})
series2 = pd.Series({"A":13,"B":17,"C":12})
series3 = pd.Series({"A":2,"B":11,"C":14,"D":1117})
seriesList=[series1,series2,series3]
myDf=pd.DataFrame(seriesList)
print(myDf)
```

输出:

```py
 A     B     C       D
0   1.0  12.0  14.0     NaN
1  13.0  17.0  12.0     NaN
2   2.0  11.0  14.0  1117.0
```

这里，第一行和第二行对应于没有 D 作为键的序列。因此，这些行包含列 d 中的`NaN`值。

## 要数据帧的列表列表

你也可以用 Python 中的[列表创建一个数据帧。为此，您可以将列表列表作为输入参数传递给`DataFrame()`函数，如下所示。](https://www.pythonforbeginners.com/basics/list-of-lists-in-python)

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList)
print(myDf)
```

输出:

```py
 0   1   2
0   1   2   3
1   3  55  34
2  12  32  45
```

在上面的例子中，您可以看到列名和索引都是自动分配的。您还可以观察到，数据帧中行的长度被视为所有列表的长度。如果列表中的元素数量不相等，那么元素数量较少的行将被填充以`NaN`值，如下所示。

```py
import pandas as pd
list1=[1,2,3,4,55]
list2=[3,55,34]
list3=[12,32,45,32]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList)
print(myDf)
```

输出:

```py
 0   1   2     3     4
0   1   2   3   4.0  55.0
1   3  55  34   NaN   NaN
2  12  32  45  32.0   NaN
```

这里，数据帧中的列数等于输入列表的最大长度。对应于较短列表的行在最右边的列中包含`NaN`值。

如果您有一个长度相等的列表，您也可以使用`DataFrame()`函数的`columns`参数来给 dataframe 指定列名。为此，可以将列名列表传递给 columns 参数，如下例所示。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"])
print(myDf)
```

输出:

```py
 A   B   C
0   1   2   3
1   3  55  34
2  12  32  45
```

在上面的例子中，确保“`columns`”参数中给定的列数应该大于最大输入列表的长度。否则，程序将会出错。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C", "D"])
print(myDf)
```

输出:

```py
ValueError: 4 columns passed, passed data had 3 columns
```

在上面的例子中，输入列表的最大长度是 3。但是，我们已经向 columns 参数传递了四个值。因此，程序运行到[值错误异常](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10)。

## 用 Python 从 CSV 文件创建数据帧

要从 csv 文件创建熊猫数据帧，您可以使用`read_csv()`功能。`read_csv()`函数将 csv 文件的文件名作为其输入参数。执行后，它返回一个 pandas 数据帧，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print(myDf)
```

输出:

```py
 Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
```

## 创建一个带索引的熊猫数据框架

默认情况下，pandas 数据帧的行使用从 0 开始的整数进行索引。但是，我们可以为数据帧中的行创建自定义索引。为此，我们需要将一个索引名列表传递给函数`DataFrame()`的 index 参数，如下所示。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"],index=["a","b","c"])
print(myDf)
```

输出:

```py
 A   B   C
a   1   2   3
b   3  55  34
c  12  32  45
```

在这个例子中，我们将列表`[a, b, c]`传递给了函数`DataFrame()`的索引参数。执行后，值 a、b 和 c 被分配给行，作为它们在数据帧中的索引。

还可以在创建数据帧后为数据帧中的行创建索引。为此，您可以将索引列表分配给 dataframe 的 index 属性，如下例所示。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"])
myDf.index=["a","b","c"]
print(myDf)
```

输出:

```py
 A   B   C
a   1   2   3
b   3  55  34
c  12  32  45
```

在本例中，我们没有将索引列表传递给 index 参数，而是在创建 dataframe 后将其分配给 dataframe 的 index 属性。

如果从 csv 文件创建数据帧，可以使用如下所示的`index_col`参数将其中一列指定为索引。

```py
myDf=pd.read_csv("samplefile.csv",index_col="Class")
print(myDf)
```

输出:

```py
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
```

您还可以创建多级索引。为此，您需要将列名列表传递给`index_col`参数。

```py
myDf=pd.read_csv("samplefile.csv",index_col=["Class","Roll"])
print(myDf)
```

输出:

```py
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
```

## 用 Python 转置熊猫数据帧

我们也可以通过转置另一个数据帧来创建一个熊猫数据帧。为此，您可以使用 T 运算符。在数据帧上调用 T 操作符时，将返回原始 pandas 数据帧的转置，如下所示。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"])
myDf.index=["a","b","c"]
newDf=myDf.T
print(newDf)
```

输出:

```py
 a   b   c
A  1   3  12
B  2  55  32
C  3  34  45
```

在输出中，您可以看到原始数据帧的行变成了新数据帧的列。随后，原始数据帧的列成为新数据帧的索引，反之亦然。

## 用 Python 复制数据帧

您也可以通过复制现有的数据帧来创建 pandas 数据帧。为此，您可以使用`copy()`方法。在 dataframe 上调用`copy()`方法时，会返回一个与原始数据具有相同数据的新 dataframe，如下所示。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"])
myDf.index=["a","b","c"]
newDf=myDf.copy()
print(newDf)
```

输出:

```py
 A   B   C
a   1   2   3
b   3  55  34
c  12  32  45
```

## 结论

在本文中，我们讨论了用 Python 创建熊猫数据帧的不同方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何用 Python 创建聊天应用的文章。您可能也会喜欢这篇关于使用 Python 中的 sklearn 模块进行[线性回归的文章。](https://codinginfinite.com/linear-regression-using-sklearn-in-python/)

请继续关注更多内容丰富的文章。

快乐学习！
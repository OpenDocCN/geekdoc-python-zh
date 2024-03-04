# Python 中的熊猫数据帧索引

> 原文：<https://www.pythonforbeginners.com/basics/pandas-dataframe-index-in-python>

Pandas dataframes 是 Python 中数据分析和[机器学习](https://codinginfinite.com/machine-learning-an-introduction/)任务最常用的数据结构之一。在本文中，我们将讨论如何从 pandas 数据帧中创建和删除索引。我们还将讨论 pandas 数据帧中的多级索引，以及如何使用数据帧索引来访问数据帧中的元素。

## 什么是熊猫数据框架指数？

就像 dataframe 有列名一样，您可以将索引视为行标签。当我们创建数据帧时，数据帧的行被赋予从 0 开始的索引，直到行数减 1，如下所示。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"])
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
```

输出:

```py
The dataframe is:
    A   B   C
0   1   2   3
1   3  55  34
2  12  32  45
The index is:
[0, 1, 2]
```

## 创建熊猫数据框架时创建索引

您还可以在创建数据框架时创建自定义索引。为此，您可以使用`DataFrame()`函数的`index`参数。`index`参数接受一个值列表，并将这些值指定为数据帧中行的索引。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"],index=[101,102,103])
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
```

输出:

```py
The dataframe is:
      A   B   C
101   1   2   3
102   3  55  34
103  12  32  45
The index is:
[101, 102, 103]
```

在上面的例子中，我们已经使用列表`[101, 102, 103]`和`DataFrame()`函数的`index`参数创建了数据帧的索引。

这里，您需要确保传递给参数`index`的列表中的元素数量应该等于 dataframe 中的行数。否则，程序会遇到如下所示的[值错误异常](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10)。

```py
import pandas as pd
list1=[1,2,3]
list2=[3,55,34]
list3=[12,32,45]
myList=[list1,list2,list3]
myDf=pd.DataFrame(myList,columns=["A", "B", "C"],index=[101,102,103,104])
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
```

输出:

```py
ValueError: Length of values (3) does not match length of index (4)
```

在上面的例子中，您可以观察到我们将列表中的 4 个元素传递给了 index 参数。然而，数据帧只有三行。因此，程序会遇到 Python ValueError 异常。

## 加载 CSV 文件时创建数据帧索引

如果您正在创建一个 csv 文件的数据帧，并希望将 csv 文件的一列作为数据帧索引，您可以使用`read_csv()`函数中的`index_col`参数。

`index_col`参数将列名作为其输入参数。执行`read_csv()`函数后，指定的列被指定为数据帧的索引。您可以在下面的示例中观察到这一点。

```py
myDf=pd.read_csv("samplefile.csv",index_col="Class")
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[1, 1, 1, 2, 2, 2, 3, 3]
```

您还可以将列名在列列表中的位置而不是它的名称作为输入参数传递给`index_col`参数。例如，如果想将熊猫数据帧的第一列作为索引，可以将 0 传递给`DataFrame()`函数中的`index_col`参数，如下所示。

```py
myDf=pd.read_csv("samplefile.csv",index_col=0)
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[1, 1, 1, 2, 2, 2, 3, 3]
```

这里，`Class`列是 csv 文件中的第一列。因此，它被转换成数据帧的索引。

`index_col`参数也接受多个值作为它们的输入。我们已经在数据帧中的多级索引一节中讨论了这一点。

## 创建熊猫数据框架后创建索引

当创建数据帧时，数据帧的行被分配从 0 开始直到行数减 1 的索引。但是，我们可以使用 index 属性为 dataframe 创建自定义索引。

为了在 pandas 数据帧中创建自定义索引，我们将为数据帧的 index 属性分配一个索引标签列表。执行赋值语句后，将为数据帧创建一个新索引，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
myDf.index=[101,102,103,104,105,106,107,108]
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
```

输出:

```py
The dataframe is:
     Class  Roll      Name
101      1    11    Aditya
102      1    12     Chris
103      1    13       Sam
104      2     1      Joel
105      2    22       Tom
106      2    44  Samantha
107      3    33      Tina
108      3    34       Amy
The index is:
[101, 102, 103, 104, 105, 106, 107, 108] 
```

在这里，您可以看到我们将一个包含从 101 到 108 的数字的列表分配给了 dataframe 的 index 属性。因此，列表的元素被转换成数据帧中行的索引。

请记住，列表中索引标签的总数应该等于数据帧中的行数。否则，程序将遇到 ValueError 异常。

## 将数据帧的列转换为索引

我们也可以使用列作为数据帧的索引。为此，我们可以使用`set_index()` 方法。在 dataframe 上调用`set_index()`方法时，该方法将列名作为其输入参数。执行后，它返回一个新的 dataframe，并将指定的列作为其索引，如下例所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
myDf=myDf.set_index("Class")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[1, 1, 1, 2, 2, 2, 3, 3]
```

在上面的例子中，我们使用了`set_index()`方法从数据帧的现有列而不是新序列中创建索引。

## 熊猫数据帧的变化指数

您可以使用`set_index()`方法更改数据帧的索引列。为此，您只需要将新索引列的列名作为输入传递给`set_index()` 方法，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
myDf=myDf.set_index("Class")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
print("The modified dataframe is:")
newDf=myDf.set_index("Roll")
print(newDf)
print("The index is:")
index=list(newDf.index)
print(index)
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
The index is:
[1, 1, 1, 2, 2, 2, 3, 3]
The modified dataframe is:
          Name
Roll          
11      Aditya
12       Chris
13         Sam
1         Joel
22         Tom
44    Samantha
33        Tina
34         Amy
The index is:
[11, 12, 13, 1, 22, 44, 33, 34]
```

如果要将一个序列作为新的索引分配给数据帧，可以将该序列分配给 pandas 数据帧的 index 属性，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
myDf=myDf.set_index("Class")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
print("The modified dataframe is:")
myDf.index=[101, 102, 103, 104, 105, 106, 107, 108]
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[1, 1, 1, 2, 2, 2, 3, 3]
The modified dataframe is:
     Roll      Name
101    11    Aditya
102    12     Chris
103    13       Sam
104     1      Joel
105    22       Tom
106    44  Samantha
107    33      Tina
108    34       Amy
The index is:
[101, 102, 103, 104, 105, 106, 107, 108]
```

当我们更改数据帧的索引列时，现有的索引列将从数据帧中删除。因此，在更改索引列之前，应该首先将索引列存储到数据帧的新列中。否则，您将丢失数据帧中存储在索引列中的数据。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
myDf=myDf.set_index("Class")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
print("The modified dataframe is:")
myDf["Class"]=myDf.index
myDf.index=[101, 102, 103, 104, 105, 106, 107, 108]
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[1, 1, 1, 2, 2, 2, 3, 3]
The modified dataframe is:
     Roll      Name  Class
101    11    Aditya      1
102    12     Chris      1
103    13       Sam      1
104     1      Joel      2
105    22       Tom      2
106    44  Samantha      2
107    33      Tina      3
108    34       Amy      3
The index is:
[101, 102, 103, 104, 105, 106, 107, 108]
```

在这里，您可以看到，在更改数据帧的索引之前，我们首先将索引存储到了`Class`列中。在前一个例子中，我们没有这样做。因此，`Class`列中的数据丢失。

## 在熊猫数据框架中创建多级索引

您还可以在数据帧中创建多级索引。多级索引有助于您访问分层数据，如具有不同抽象级别的人口普查数据。我们可以在创建数据帧的同时以及在创建数据帧之后创建多级索引。这一点讨论如下。

### 创建数据框架时创建多级索引

要使用 dataframe 的不同列创建多级索引，可以使用`read_csv()`函数中的`index_col`参数。`index_col`参数接受必须用作索引的列的列表。列表中赋予`index_col`参数的列名从左到右的顺序是从最高到最低的索引级别。在执行了`read_csv()`函数后，你将得到一个多级索引的数据帧，如下例所示。

```py
myDf=pd.read_csv("samplefile.csv",index_col=["Class","Roll"])
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[(1, 11), (1, 12), (1, 13), (2, 1), (2, 22), (2, 44), (3, 33), (3, 34)]
```

在上面的示例中，`Class`列包含第一级索引，`Roll`列包含第二级索引。要从 dataframe 中访问元素，您需要知道任意行的两个级别的索引。

除了使用列名，您还可以将列名在列列表中的位置而不是其名称作为输入参数传递给`index_col`参数。例如，您可以将 dataframe 的第一列和第三列指定为其索引，如下所示。

```py
myDf=pd.read_csv("samplefile.csv",index_col=[0,1])
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[(1, 11), (1, 12), (1, 13), (2, 1), (2, 22), (2, 44), (3, 33), (3, 34)]
```

### 创建数据帧后创建多级索引

您还可以在使用`set_index()`方法创建数据帧之后创建多级索引。为此，您只需要将列名列表传递给`set_index()`方法。同样，列表中赋予`index_col`参数的列名从左到右的顺序是从最高到最低的索引级别，如下所示。

```py
myDf=pd.read_csv("samplefile.csv")
print("The dataframe is:")
myDf=myDf.set_index(["Class","Roll"])
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[(1, 11), (1, 12), (1, 13), (2, 1), (2, 22), (2, 44), (3, 33), (3, 34)] 
```

您需要记住,`set_index()`方法会删除现有的索引列。如果要保存存储在索引列中的数据，应该在创建新索引之前将数据复制到另一列中。

## 从熊猫数据帧中删除索引

要从 pandas 数据帧中删除索引，可以使用`reset_index()`方法。在 dataframe 上调用`reset_index()`方法时，会返回一个没有任何索引列的新 dataframe。如果现有索引是特定的列，则该列将再次转换为普通列，如下所示。

```py
myDf=pd.read_csv("samplefile.csv",index_col=[0,1])
print("The dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
myDf=myDf.reset_index()
print("The modified dataframe is:")
print(myDf)
print("The index is:")
index=list(myDf.index)
print(index)
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
The index is:
[(1, 11), (1, 12), (1, 13), (2, 1), (2, 22), (2, 44), (3, 33), (3, 34)]
The modified dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The index is:
[0, 1, 2, 3, 4, 5, 6, 7]
```

## 结论

在这篇文章中，我们讨论了如何创建熊猫数据帧索引。此外，我们还创建了多级索引，并学习了如何从熊猫数据帧中删除索引。要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中的[列表理解的文章。如果你对机器学习感兴趣，你可以阅读这篇关于机器学习](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)中的[正则表达式的文章。](https://codinginfinite.com/regression-in-machine-learning-with-examples/)

请继续关注更多内容丰富的文章。

快乐学习！
# Python 中的熊猫系列数据结构

> 原文：<https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python>

Python 中使用系列数据结构处理一维数据。在本文中，我们将讨论如何使用 pandas 模块创建一个系列，它的属性，以及使用示例的操作。

## 熊猫系列是什么？

你可以把熊猫系列看作是一个列表和一个字典的组合。在一个序列中，所有元素都按顺序存储，您可以使用索引来访问它们。

就像我们使用键名从 python 字典中访问值一样，您可以为 pandas 系列中的元素分配标签，并使用标签来访问它们。

## 用 Python 创建熊猫系列

为了创建一个系列，我们使用`pandas.Series()`函数。它将一个列表或一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)作为它的输入参数，并返回一个序列。我们已经在下面的章节中讨论了`Series()`函数的使用。

### 将 Python 列表转换为熊猫系列

您可以使用列表元素创建熊猫系列。`Series()`方法将列表作为其输入参数，并返回一个 Series 对象，如下所示。

```py
import pandas as pd
names = ['Aditya', 'Chris', 'Joel']
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
```

输出:

```py
The input list is:
['Aditya', 'Chris', 'Joel']
The series is:
0    Aditya
1     Chris
2      Joel
dtype: object 
```

在输出中，您可以看到列表元素位于第二列。数列的第一列由数列的索引组成。索引用于访问序列中的元素。

默认情况下，序列中的索引从 0 开始。但是，您可以使用`Series()`函数中的 index 参数将索引明确分配给序列。

index 参数接受一个索引值列表，并将索引分配给序列中的元素，如下所示。

```py
import pandas as pd
names = ['Aditya', 'Chris', 'Joel']
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names, index=["A","B", "C"])
print(mySeries)
```

输出:

```py
The input list is:
['Aditya', 'Chris', 'Joel']
The series is:
A    Aditya
B     Chris
C      Joel
dtype: object
```

这里，我们将列表`["A", "B", "C"]`传递给了函数`Series()`的索引参数。因此，“A”、“B”和“C”被指定为系列行的索引。

这里，您需要记住，索引的数量应该等于序列中元素的数量。否则，程序将出现如下所示的错误。

```py
import pandas as pd
names = ['Aditya', 'Chris', 'Joel']
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names, index=["A","B", "C", "D"])
print(mySeries)
```

输出:

```py
The input list is:
['Aditya', 'Chris', 'Joel']
The series is:

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_6004/861141107.py in <module>
      4 print(names)
      5 print("The series is:")
----> 6 mySeries=pd.Series(names, index=["A","B", "C", "D"])
      7 print(mySeries)

ValueError: Length of values (3) does not match length of index (4)
```

在上面的例子中，我们向 index 参数传递了一个包含四个元素的列表。但是，这个系列只有三个元素。因此，程序会遇到 ValueError 异常。

除了将标签列表传递给`Series()`函数，您还可以将标签列表分配给序列的 index 属性。这将为系列创建索引标签，如下所示。

```py
import pandas as pd
names = ['Aditya', 'Chris', 'Joel']
print("The input list is:")
print(names)
print("Series before index creation:")
mySeries=pd.Series(names)
print(mySeries)
mySeries.index=["A","B", "C"]
print("Series after index creation:")
print(mySeries)
```

输出:

```py
The input list is:
['Aditya', 'Chris', 'Joel']
Series before index creation:
0    Aditya
1     Chris
2      Joel
dtype: object
Series after index creation:
A    Aditya
B     Chris
C      Joel
dtype: object
```

在本例中，我们没有使用 index 参数，而是使用 Series 对象的 index 属性为系列中的行创建索引。

### 将 Python 字典转换为 Python 中的系列

做一个带标签的熊猫系列，也可以用 python 字典。当我们将字典传递给`Series()`函数时，字典的键就变成了索引标签。对应于某个键的值成为序列中的数据值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
names = {"A":'Aditya', "B":'Chris', "C":'Joel'}
print("The input dictionary is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
```

输出:

```py
The input dictionary is:
{'A': 'Aditya', 'B': 'Chris', 'C': 'Joel'}
The series is:
A    Aditya
B     Chris
C      Joel
dtype: object
```

在上面的例子中，您可以观察到字典的键已经变成了索引标签。字典的相应值被分配给与索引相关联的行。

除了列表或字典，您还可以将一个元组或其他有序的可迭代对象传递给`Series()` 函数来创建 pandas 系列。但是，您不能将一个无序的可迭代对象(比如集合)作为输入传递给`Series()`函数来创建一个序列。这样做将使你的程序运行到如下所示的错误中。

```py
import pandas as pd
names = {'Aditya', 'Chris', 'Joel'}
print("The input set is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
```

输出:

```py
The input set is:
{'Joel', 'Aditya', 'Chris'}
The series is:

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_6004/4101083988.py in <module>
      4 print(names)
      5 print("The series is:")
----> 6 mySeries=pd.Series(names)
      7 print(mySeries)

TypeError: 'set' type is unordered 
```

这里，我们将一个集合传递给了`Series()`函数。因此，程序运行到 [Python TypeError](https://www.pythonforbeginners.com/basics/typeerror-in-python) 异常，并显示 set type 是无序的消息。

## 熊猫系列中元素的数据类型

当从列表或字典的元素创建序列时，序列中元素的数据类型是根据输入元素的数据类型决定的。

例如，如果您将一个整数列表传递给`Series()`函数，那么序列的结果数据类型将是如下所示的`int64`。

```py
import pandas as pd
names = [1,2,3,4]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
[1, 2, 3, 4]
The series is:
0    1
1    2
2    3
3    4
dtype: int64
The datatype of series elements is:
int64
```

上述条件也适用于浮点数。然而，当我们将一个 floats 和 int 的列表传递给`Series()`函数时，序列中的结果数据集是`float64`，因为所有的元素都被转换为最高级别的兼容数据类型。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
names = [1,2,3.1,4.2]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
[1, 2, 3.1, 4.2]
The series is:
0    1.0
1    2.0
2    3.1
3    4.2
dtype: float64
The datatype of series elements is:
float64
```

在上面的例子中，数据类型被写成`float64`和`int64`，因为程序是在 64 位机器上执行的。如果你在 32 位机器上运行程序，你将得到 int32 和 float32 的数据类型。所以，如果你得到这种类型的输出，不用担心。

当您将字符串列表传递给`Series()`函数时，序列元素的结果数据类型是"`object`"而不是字符串，如下例所示。

```py
import pandas as pd
names = ['Aditya', 'Chris', 'Joel']
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
['Aditya', 'Chris', 'Joel']
The series is:
0    Aditya
1     Chris
2      Joel
dtype: object
The datatype of series elements is:
object
```

当我们将包含整型、浮点型和字符串的列表传递给`Series()`函数时，series 元素的结果数据类型是“`object`”。

```py
import pandas as pd
names = [1, 2.2, 'Joel']
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
[1, 2.2, 'Joel']
The series is:
0       1
1     2.2
2    Joel
dtype: object
The datatype of series elements is:
object
```

元素被赋予 object 数据类型，因为我们可以在 object 数据类型中包含任何值。将值存储为对象数据类型有助于解释器以简单的方式处理元素的数据类型。

## 熊猫系列中的无类型值

在创建 series 对象时，当我们将一个包含值 None 的列表传递给`Series()`函数时，会出现一种特殊情况。

当我们将包含值`None`的字符串列表传递给`Series()`函数时，该序列的数据类型是“`object`”。这里，值`None`被存储为对象类型。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
The datatype of series elements is:
object
```

然而，当我们传递一个包含值`None`的整数列表时，`None`被转换为`NaN`，这是一个不存在的值的浮点表示。因此，序列的数据类型变成了`float64`。类似地，当我们传递浮点数列表中的值`None`时，`None`被转换为`NaN`。

```py
import pandas as pd
names = [1, 2.2, 3.2, None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
[1, 2.2, 3.2, None]
The series is:
0    1.0
1    2.2
2    3.2
3    NaN
dtype: float64
The datatype of series elements is:
float64
```

在前面的示例中，`None`被存储为一个`NoneType`对象，因为该系列包含一个字符串。在本例中，`None`被存储为浮点值`NaN`，因为该系列只包含数字。因此，可以说 python 解释器根据现有元素的兼容性为系列选择了最佳数据类型。

当我们将包含整型、浮点型和字符串的列表传递给`String()`函数时，`None`被存储为对象类型。你可以在上面的例子中观察到这一点。

```py
import pandas as pd
names = [1, 2.2, 3.2, None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
print("The datatype of series elements is:")
print(mySeries.dtype)
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
The datatype of series elements is:
object
```

## 使用索引运算符从系列中访问数据

您可以使用索引运算符访问序列中的数据，就像访问列表元素一样。为此，您可以在索引操作符中传递 series 元素的位置，如下所示。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
index=2
myVal=mySeries[index]
print("Element at index {} is {}".format(index,myVal))
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
Element at index 2 is Aditya
```

如果已经为索引分配了标签，则可以使用索引运算符中的标签来访问序列元素。这类似于我们如何使用键和索引操作符来访问字典的值。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names, index=["A","B", "C", "D"])
print(mySeries)
index="B"
myVal=mySeries[index]
print("Element at index {} is {}".format(index,myVal))
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
Element at index B is 2.2
```

使用索引运算符时，当整数用作索引标签时，不能使用元素的位置来访问元素。例如，考虑以下示例中的系列。这里，索引标签是整数。因此，我们不能使用索引 0 来访问序列中的第一个元素，也不能使用索引 1 来访问序列中的第二个元素，依此类推。这样做会导致如下所示的 [KeyError 异常](https://www.pythonforbeginners.com/basics/python-keyerror)。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names, index=[4,5,6,7])
print(mySeries)
index=0
myVal=mySeries[index]
print("Element at index {} is {}".format(index,myVal))
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
4         1
5       2.2
6    Aditya
7      None
dtype: object

---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_6004/208185265.py in <module>
      7 print(mySeries)
      8 index=0
----> 9 myVal=mySeries[index]
     10 print("Element at index {} is {}".format(index,myVal))

KeyError: 0
```

因此，在这些情况下，您只能在使用索引操作符访问元素时使用索引标签。但是，您可以使用 pandas 系列对象的`iloc`属性，通过它们在系列中的位置来访问元素。

## 使用 Python 中的 iloc 访问系列数据

`iloc`属性的功能类似于[列表索引](https://www.pythonforbeginners.com/basics/index-of-minimum-element-in-a-list-in-python)。`iloc`属性包含一个`_iLocIndexer`对象，您可以用它来访问序列中的数据。您可以简单地使用方括号内的位置和`iloc`属性来访问熊猫系列的元素，如下所示。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
position=0
myVal=mySeries.iloc[position]
print("Element at position {} is {}".format(position,myVal))
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
Element at position 0 is 1
```

如果您使用整数作为系列的索引标签，这对`iloc`属性的工作没有任何影响。`iloc`属性用于访问某个位置的列表。因此，我们使用什么索引并不重要，`iloc`属性以同样的方式工作。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names,index=[1,2,3,4])
print(mySeries)
position=0
myVal=mySeries.iloc[position]
print("Element at position {} is {}".format(position,myVal))
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
1         1
2       2.2
3    Aditya
4      None
dtype: object
Element at position 0 is 1
```

## 使用 Python 中的 loc 属性从系列中访问数据

系列的`loc`属性的工作方式类似于 python 字典的键。`loc`属性包含一个`_LocIndexer`对象，您可以用它来访问序列中的数据。您可以使用方括号内的索引标签和`loc`属性来访问 pandas 系列的元素，如下所示。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
index="A"
myVal=mySeries.loc[index]
print("Element at index {} is {}".format(index,myVal))
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
Element at index A is 1
```

## 将数据插入熊猫系列

要将单个元素插入到序列中，可以使用`loc`属性或`append()`方法。

要将数据插入到带有索引标签的序列中，可以使用`loc`属性。在这里，我们将以在 python 字典中添加新的键-值对的相同方式为系列分配标签和值。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
index="D"
mySeries.loc[index]=1117
print("The modified series is:")
print(mySeries)
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
A         1
B       2.2
C    Aditya
D      1117
dtype: object 
```

`append()`方法用于将一个系列追加到另一个系列。当在一个系列上调用时，它将另一个系列作为其输入参数，将其附加到原始系列，并返回一个包含两个系列中元素的新系列。

为了将一个元素插入到一个序列中，我们将首先用给定的元素创建一个新的序列。之后，我们将使用下面的例子所示的`append()`方法将新的序列添加到现有的序列中。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
newSeries=pd.Series([1117])
mySeries=mySeries.append(newSeries)
print("The modified series is:")
print(mySeries)
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
A         1
B       2.2
C    Aditya
D      None
0      1117
dtype: object
```

您可以观察到输出序列的索引没有按顺序排列。这是因为新系列和现有系列的索引已经与元素合并。为了保持索引的顺序，您可以在如下所示的`append()`函数中使用`ignore_index=True`参数。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
newSeries=pd.Series([1117])
mySeries=mySeries.append(newSeries, ignore_index=True )
print("The modified series is:")
print(mySeries)
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
0         1
1       2.2
2    Aditya
3      None
4      1117
dtype: object
```

如果现有序列具有索引标签，并且要插入的数据也包含索引的特定标签，您也可以使用`append()`方法向序列添加新元素，如下所示。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The input list is:")
print(names)
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
newSeries=pd.Series([1117],index=["P"])
mySeries=mySeries.append(newSeries)
print("The modified series is:")
print(mySeries)
```

输出:

```py
The input list is:
[1, 2.2, 'Aditya', None]
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
A         1
B       2.2
C    Aditya
D      None
P      1117
dtype: object
```

`append()`方法已经被弃用，它将从熊猫的未来版本中移除(我目前使用的是熊猫 1.4.3)。如果您正在使用`append()`方法并得到错误，那么您可能正在使用一个新版本的 pandas。因此，寻找一种替代方法来将元素添加到系列中。

建议阅读: [K-Means 使用 Python 中的 sklearn 模块进行聚类](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 在 Python 中删除熊猫系列的数据

要在 Python 中删除序列中的数据，可以使用`drop()`方法。在 series 对象上调用时，`drop()`方法将一个索引标签或一组索引标签作为其输入参数。执行后，它会在删除指定索引处的数据后返回一个新序列。

要从具有索引标签的序列中删除单个元素，可以将索引标签传递给如下所示的`drop()` 函数。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
mySeries=mySeries.drop("A")
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
B       2.2
C    Aditya
D      None
dtype: object
```

要删除多个索引标签处的元素，可以将索引标签列表传递给如下所示的`drop()` 方法。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
mySeries=mySeries.drop(["A","D"])
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
B       2.2
C    Aditya
dtype: object 
```

在上面的例子中，原始系列没有被修改。要删除原始序列中的元素，可以使用如下所示的`drop()` 方法中的`inplace=True`参数。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
mySeries.drop(["A","D"], inplace=True)
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
B       2.2
C    Aditya
dtype: object
```

要从没有索引标签的序列中删除元素，可以使用元素在索引处的位置，并将其传递给如下所示的`drop()`方法。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
mySeries.drop(0, inplace=True)
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
The modified series is:
1       2.2
2    Aditya
3      None
dtype: object
```

要删除多个位置的元素，可以将索引列表传递给如下所示的`drop()`方法。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
mySeries.drop([0,1], inplace=True)
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
The modified series is:
2    Aditya
3      None
dtype: object
```

## 更新熊猫系列中的数据

要更新给定索引处的元素，可以使用索引操作符和赋值操作符，如下所示。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names)
print(mySeries)
mySeries[0]=12345
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
0         1
1       2.2
2    Aditya
3      None
dtype: object
The modified series is:
0     12345
1       2.2
2    Aditya
3      None
dtype: object
```

对于具有索引标签的系列，可以将索引标签与赋值运算符一起使用，如下所示。

```py
import pandas as pd
names = [1, 2.2, "Aditya", None]
print("The series is:")
mySeries=pd.Series(names,index=["A","B","C","D"])
print(mySeries)
mySeries["D"]="Chris"
print("The modified series is:")
print(mySeries)
```

输出:

```py
The series is:
A         1
B       2.2
C    Aditya
D      None
dtype: object
The modified series is:
A         1
B       2.2
C    Aditya
D     Chris
dtype: object
```

## 结论

在本文中，我们讨论了如何使用 pandas 模块在 Python 中创建一个系列数据结构。我们还讨论了序列中的索引，如何从序列中删除元素，如何更新序列中的元素，以及如何在序列中插入元素。

想了解更多关于 Python 编程的知识，可以阅读这篇关于 Python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢关于如何用 Python 创建聊天应用程序的文章[。](https://codinginfinite.com/python-chat-application-tutorial-source-code/)
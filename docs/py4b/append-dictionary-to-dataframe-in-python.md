# 在 Python 中将字典追加到数据帧

> 原文：<https://www.pythonforbeginners.com/basics/append-dictionary-to-dataframe-in-python>

我们使用 python 字典来存储键值对。类似地，数据帧用于以表格格式存储包含与关键字相关联的值的记录。在本文中，我们将讨论如何在 Python 中将字典附加到数据帧中。

## 如何在 Python 中给数据帧追加字典？

为了给一个[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)添加一个字典，我们将使用`append()`方法。在 dataframe 上调用`append()` 方法时，该方法将一个字典作为其输入参数，并返回一个新的 dataframe。`append()`方法也将值`True`作为其`ignore_index`参数的输入变量。

输出数据帧包含作为原始数据帧中的一行追加的字典值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
names=pd.read_csv("name.csv")
print("The input dataframe is:")
print(names)
myDict={"Class":3, "Roll":22, "Name":"Sakshi"}
print("The dictionary is:")
print(myDict)
names=names.append(myDict,ignore_index=True)
print("The output dataframe is:")
print(names)
```

输出

```py
The input dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The dictionary is:
{'Class': 3, 'Roll': 22, 'Name': 'Sakshi'}
The output dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
8      3    22    Sakshi
```

如果字典的键数少于数据帧中指定的列数，则在字典所附加的行中，剩余的列被赋予值`NaN`。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
names=pd.read_csv("name.csv")
print("The input dataframe is:")
print(names)
myDict={"Class":3, "Name":"Sakshi"}
print("The dictionary is:")
print(myDict)
names=names.append(myDict,ignore_index=True)
print("The output dataframe is:")
print(names)
```

输出:

```py
The input dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The dictionary is:
{'Class': 3, 'Name': 'Sakshi'}
The output dataframe is:
   Class  Roll      Name
0      1  11.0    Aditya
1      1  12.0     Chris
2      1  13.0       Sam
3      2   1.0      Joel
4      2  22.0       Tom
5      2  44.0  Samantha
6      3  33.0      Tina
7      3  34.0       Amy
8      3   NaN    Sakshi
```

如果字典不具有与数据帧中的列相同的关键字，则为数据帧中不存在的每个关键字添加一个新列作为列名。向字典追加与列名不同的键值时，现有行中新列的值被填充为`NaN`。类似地，不存在于字典中但存在于数据帧中的列被赋予值`NaN`,即字典被附加到的行。您可以在下面的例子中观察到这一点。

```py
import pandas as pd
names=pd.read_csv("name.csv")
print("The input dataframe is:")
print(names)
myDict={"Class":3, "Roll":22, "Name":"Sakshi","Height":160}
print("The dictionary is:")
print(myDict)
names=names.append(myDict,ignore_index=True)
print("The output dataframe is:")
print(names)
```

输出:

```py
The input dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The dictionary is:
{'Class': 3, 'Roll': 22, 'Name': 'Sakshi', 'Height': 160}
The output dataframe is:
   Class  Roll      Name  Height
0      1    11    Aditya     NaN
1      1    12     Chris     NaN
2      1    13       Sam     NaN
3      2     1      Joel     NaN
4      2    22       Tom     NaN
5      2    44  Samantha     NaN
6      3    33      Tina     NaN
7      3    34       Amy     NaN
8      3    22    Sakshi   160.0
```

要将两个或更多字典或一个字典列表追加到数据帧中，可以使用 for 循环和`append()`方法。在 for 循环中，可以将每个字典追加到数据帧中。

建议阅读:如果你对机器学习感兴趣，你可以阅读这篇关于 [k-prototypes 聚类的文章，并给出数值示例](https://codinginfinite.com/k-prototypes-clustering-with-numerical-example/)。你可能也会喜欢这篇关于[数据油墨比率](https://www.codeconquest.com/blog/data-ink-ratio-explained-with-example/)的文章。

在不久的将来，`append()`方法将被弃用。因此，您可以使用 pandas `concat()` 方法向 dataframe 添加一个字典，如下所示。

```py
import pandas as pd
names=pd.read_csv("name.csv")
print("The input dataframe is:")
print(names)
myDict={"Class":3, "Roll":22, "Name":"Sakshi","Height":160}
print("The dictionary is:")
print(myDict)
tempDf=pd.DataFrame([myDict])
names=pd.concat([names,tempDf],ignore_index=True)
print("The output dataframe is:")
print(names)
```

输出:

```py
The input dataframe is:
   Class  Roll      Name
0      1    11    Aditya
1      1    12     Chris
2      1    13       Sam
3      2     1      Joel
4      2    22       Tom
5      2    44  Samantha
6      3    33      Tina
7      3    34       Amy
The dictionary is:
{'Class': 3, 'Roll': 22, 'Name': 'Sakshi', 'Height': 160}
The output dataframe is:
   Class  Roll      Name  Height
0      1    11    Aditya     NaN
1      1    12     Chris     NaN
2      1    13       Sam     NaN
3      2     1      Joel     NaN
4      2    22       Tom     NaN
5      2    44  Samantha     NaN
6      3    33      Tina     NaN
7      3    34       Amy     NaN
8      3    22    Sakshi   160.0
```

## 结论

在本文中，我们讨论了如何在 python 中将字典附加到数据帧中。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。
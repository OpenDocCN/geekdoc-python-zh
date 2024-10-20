# 熊猫 fillna()方法——完全指南

> 原文：<https://www.askpython.com/python/pandas-fillna-method>

数据分析已经成为我们日常生活的重要组成部分。每天我们都要处理来自不同领域的不同种类的数据。数据分析的主要挑战之一是数据中缺失值的存在。在本文中，我们将学习如何在 fillna()方法的帮助下处理数据集中的缺失值。我们开始吧！

## 熊猫 fillna()方法是什么，为什么有用？

Pandas Fillna()是一种用于填充数据集中缺失值或 na 值的方法。您可以填充像零这样的缺失值，也可以输入一个值。当您处理 CSV 或 Excel 文件时，这种方法通常会很方便。

不要与我们删除丢失值的 [dropna()方法](https://www.askpython.com/python-modules/pandas/drop-multiple-columns-dataframe)混淆。在这种情况下，我们将用零或用户输入的值替换丢失的值。

让我们来看看 fillna()函数的语法。

```py
DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)

```

让我们看看下面的例子，看看如何在不同的场景中使用 fillna()方法。

### **Pandas DataFrame fillna()方法**

在下面的例子中，我们将用零填充 NAN 值的位置。

```py
import pandas as pd
import numpy as np

df = pd.DataFrame([[np.nan, 300, np.nan, 330],
                     [589, 700, np.nan, 103],
                     [np.nan, np.nan, np.nan, 675],
                     [np.nan, 3]],
                    columns=list('abcd'))
print(df)

#Filling the NaN values with zeros.
print("\n")
print(df.fillna(0))

```

输出

```py
   a      b   c      d
0    NaN  300.0 NaN  330.0
1  589.0  700.0 NaN  103.0
2    NaN    NaN NaN  675.0
3    NaN    3.0 NaN    NaN

       a      b    c      d
0    0.0  300.0  0.0  330.0
1  589.0  700.0  0.0  103.0
2    0.0    0.0  0.0  675.0
3    0.0    3.0  0.0    0.0

```

### **只对一列应用 fillna()方法**

```py
df = pd.DataFrame([[np.nan, 300, np.nan, 330],
                     [589, 700, np.nan, 103],
                     [np.nan, np.nan, np.nan, 675],
                     [np.nan, 3]],
                    columns=list('abcd'))

print(df)

#Filling the NaN value 
print("\n")
newDF = df['b'].fillna(0)
print(newDF)

```

输出

```py
 a      b   c      d
0    NaN  300.0 NaN  330.0
1  589.0  700.0 NaN  103.0
2    NaN    NaN NaN  675.0
3    NaN    3.0 NaN    NaN

0    300.0
1    700.0
2      0.0
3      3.0
Name: b, dtype: float64

```

还可以使用 limit 方法指定要填充 NAN 值的行。

```py
import pandas as pd
import numpy as np
df = pd.DataFrame([[np.nan, 300, np.nan, 330],
                     [589, 700, np.nan, 103],
                     [np.nan, np.nan, np.nan, 675],
                     [np.nan, 3]],
                    columns=list('abcd'))

print(df)

# Filing the NaN value 
print("\n")
print(df.fillna(0, limit=2))

```

输出

```py
a      b   c      d
0    NaN  300.0 NaN  330.0
1  589.0  700.0 NaN  103.0
2    NaN    NaN NaN  675.0
3    NaN    3.0 NaN    NaN

       a      b    c      d
0    0.0  300.0  0.0  330.0
1  589.0  700.0  0.0  103.0
2    0.0    0.0  NaN  675.0
3    NaN    3.0  NaN    0.0

```

在上面的方法中，我们应用了 limit=2，这意味着我们只替换了前两行中的 NAN 值。

## 结论

总之，我们学习了在数据帧中填充 NAN 值的不同方法。所有这些方法在你的任何数据分析项目中都会派上用场。
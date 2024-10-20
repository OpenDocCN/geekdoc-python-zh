# 将 Pandas 系列转换为 Numpy 数组的 4 种方法

> 原文：<https://www.askpython.com/python-modules/numpy/convert-pandas-series-to-numpy-array>

你们可能都已经熟悉熊猫有两个主要的数据结构，即[系列](https://www.askpython.com/python-modules/pandas/head-and-tail-of-dataframe-series)和[数据框架](https://www.askpython.com/python-modules/pandas/dataframe-rows-and-columns)。在以前的文章中，我们已经了解了如何将数据帧转换成 Numpy 数组。所以今天，在这篇文章中，我们将学习如何在 python 中将一个序列转换成一个 Numpy 数组。

## Python 中的熊猫系列是什么？

Pandas series 是一维数据结构 Pandas，可以接受多种数据类型，如整数、对象和浮点数据类型。Pandas 系列相对于数据框的优势在于它可以存储多种数据类型。可以用多种方法创建序列，例如从列表、元组或字典创建序列，或者通过传递标量值来创建序列。

在这篇文章中，我们将用 python 制作一系列字典。在本文的其余部分，我们也将使用这个系列。

```py
import pandas as pd

list = ['a', 'b', 'c', 'd', 'e']

my_series = pd.Series(list)
print(my_series)

```

**输出:**

```py
0  a
1  b
2  c
3  d
4  e

```

## Python 中的 Numpy 数组是什么？

一个 [NumPy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)是一个只接受相似类型数据的数据结构。Numpy 数组几乎就像列表，但不要混淆。数组比列表更有效，也更紧凑。

让我们看看如何创建一个 NumPy 数组。

```py
import numpy as np

my_arr = np.array([1, 2, 3, 4, 5])

print(my_arr)

```

**输出:**

```py
[1 2 3 4 5]

```

## 将 Pandas 系列转换为 Numpy 数组的方法

现在我们将学习如何使用一些函数和属性将 Pandas 系列转换成 NumPy 数组。

### 1.使用 Pandas.index.to_numpy()函数

这是一个相当简单的方法，因为它直接将序列中的元素转换成 NumPy 数组。我们将首先与 pd 一起创建一个系列。DataFrame()函数，然后将其转换为 Numpy 数组。

**例如，**

```py
import pandas as pd

df = pd.DataFrame({'A1': [1, 2, 3], 'A2': [4, 5, 6]}, index=['a', 'b', 'c']); 

array = df.index.to_numpy()
print(array)

```

**输出:**

```py
['a' , 'b' , 'c']

```

### 2.使用 pandas.index.values 属性

在这种方法中，我们将把系列转换成两个步骤。首先，我们将使用熊猫。index.values 属性该属性将以数组的形式返回索引处的值。在 NumPy.array 函数的帮助下，这个数组将被转换为 NumPy 数组。

```py
import pandas as pd
import numpy as np

df = pd.DataFrame({'A1': [1, 2, 3], 'A2': [4, 5, 6]}, index=['a', 'b', 'c']); 

array = np.array(df.index.values)
print(array)

```

**输出:**

```py
['a' , 'b', 'c']

```

### 3.使用 pandas.index.array 属性

该属性也分两步工作。首先，它将熊猫系列转换成熊猫数组。然后，在 numpy.array()函数的帮助下，Pandas 数组被转换为 Numpy 数组。

```py
import pandas as pd
import numpy as np

df = pd.DataFrame({'A1': [1, 2, 3], 'A2': [4, 5, 6]}, index=['a', 'b', 'c']); 

array = np.array(df.index.array)
print(array)

```

**输出:**

```py
['a' , 'b' , 'c']

```

### 4.使用 Pandas series.to_numpy()函数

通过这个函数，我们将使用一个数据集，我们将首先从数据集中的一列创建一个序列，然后将其转换为一个 Numpy 数组。在这里，我们首先从电影信息栏创建了一个系列。然后我们使用 series.to_numpy()函数创建一个 numpy 数组。

```py
import pandas as pd 

data = pd.read_csv("/content/Highest Holywood Grossing Movies.csv") 

data.dropna(inplace = True)

my_ser = pd.Series(data['Movie Info'].head())

# using to_numpy() function
print((my_ser.to_numpy()))

```

**输出:**

```py
['As a new threat to the galaxy rises, Rey, a desert scavenger, and Finn, an ex-stormtrooper, must join Han Solo and Chewbacca to search for the one hope of restoring peace.'
 "After the devastating events of Avengers: Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe."
 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.'
 'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.'
 'A new theme park, built on the original site of Jurassic Park, creates a genetically modified hybrid dinosaur, the Indominus Rex, which escapes containment and goes on a killing spree.']

```

## 结论

在本文中，我们学习了很多将序列转换成 Numpy 数组的不同方法。一些方法分两步完成，而另一些方法一步完成。
# 熊猫数据框:创建和操作数据的教程

> 原文：<https://www.pythoncentral.io/pandas-data-frame-a-tutorial-for-creating-and-manipulating-data/>

Pandas DataFrames 是一种数据结构，以二维形式保存数据，类似于 SQL 或 Excel 电子表格中的表格，但速度更快，功能更强大。它们有行和列，也有对应于这些行和列的标签。

数据帧是一种无价的工具，它构成了机器学习、数据科学、科学计算以及几乎所有其他数据密集型领域的基石。

在本 Python 指南中，我们将看到如何在 Pandas 数据帧中创建和操作数据。

## **熊猫基础数据框**

要使用数据帧，你必须首先导入熊猫，就像这样:

```py
>>> import pandas as pd
```

让我们借助一个例子来理解数据帧的基础。假设你想用熊猫来分析求职者。除了跟踪他们的详细信息，包括他们的姓名、位置和年龄，您还想跟踪他们在公司指定的编程测试中的分数。

|   | **名称** | **城市** | **年龄** | **测试分数** |
| **101** | 约翰 | Oslo | 25 | 88.0 |
| **102** | 丽莎 | 伯尔尼 | 32 | 79.0 |
| **103** | 基因 | 布拉格 | 35 | 81.0 |
| **104** | 凌 | 东京 | 29 | 80.0 |
| **105** | 布鲁斯 | 苏黎世 | 37 | 68.0 |
| **106** | 艾伦 | 哥本哈根 | 34 | 61.0 |
| **107** | 常 | 东京 | 32 | 84.0 |

如您所见，该表的第一行有列标签，可帮助您跟踪详细信息。另外，第一列有行标签，它们是数字。表格中的其他单元格用其他数据值填充。

既然背景已经确定，我们也知道需要创建什么样的数据框架，我们就可以研究如何去做了。

创建熊猫数据框架有很多方法。最突出的方法是用 DataFrame 构造函数提供标签、数据和其他细节。

在传递数据方面，你也可以有几种选择。除了将数据作为字典或 Pandas Series 实例传递之外，还可以将其作为二维元组、NumPy 数组或列表传递。您并不局限于这些方法——您可以使用 Python 提供的许多数据类型中的一种来传递数据。

假设你想用字典来传递数据。下面是代码的样子:

```py
>>> data = {
...     'name': ['John', 'Lisa', 'Gene', 'Ling', 'Bruce', 'Alan', 'Chang'],
...     'city': ['Oslo', 'Bern', 'Prague', 'Tokyo',
...              'Manchester', 'Copenhagen', 'Hong Kong'],
...     'age': [41, 28, 33, 34, 38, 31, 37],
...     'test-score': [88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0]

... }

>>> row_labels = [101, 102, 103, 104, 105, 106, 107]
```

“数据”变量是一个内置的 Python 变量，指的是保存数据的字典。“row_labels”变量做了您期望它做的事情——它保存行的标签。

下面是创建数据帧的方法:

```py
>>> df = pd.DataFrame(data=data, index=row_labels)
>>> df
name city age test-score
101 John Oslo 41 88.0
102 Lisa Bern 28 79.0
103 Gene Prague 33 81.0
104 Ling Tokyo 34 80.0
105 Bruce Manchester 38 68.0
106 Alan Copenhagen 31 61.0
107 Chang Hong Kong 37 84.0
```

至此，数据框已创建完毕。“df”变量保存对数据帧的引用。

## **使用 Python 数据框架**

虽然我们创建的不是，但数据帧可能非常大。熊猫有两个命令。头()和。tail()，它允许您查看数据帧中的第一项和最后几项。

下面是你如何使用它们:

```py
>>> df.head(n=2) # Outputs the first 2 rows
>>> df.tail(n=2) # Outputs the final 2 rows
```

“n”值是一个参数，它告诉命令要显示的角色数量。

也可以访问数据帧中的一列。你可以用同样的方法从字典中得到一个值:

```py
>>> cities = df['city']
>>> cities # Returns the column
```

这是目前为止从熊猫数据框架中获取列的最方便的方法。但是在某些情况下，列名与 Python 标识符相同。

在这些情况下，你可以使用点符号来访问它，就像你访问一个类实例的属性一样:

```py
>>> df.city # This will have the same result as the previous code
```

在上面的代码中，我们提取了与“city”标签对应的列，该列保存了所有求职者的位置。

有趣的是，Pandas DataFrame 中的每一列都是 pandas.Series 的一个实例。这是一个保存一维数据及其标签的数据结构。

你也可以像使用字典一样从一个系列对象中获取单个项目。这个想法是使用标签作为一个键，就像这样:

```py
>>> cities[102]
```

要访问一整行，可以使用。loc[]访问器如下:

```py
>>> df.loc[103]
name Gene
city Prague
age 33
test-score 81
Name: 103, dtype: object
```

## **创建熊猫数据帧的不同方式**

本节将介绍四种最常见的创建数据框的方法。但是你可以在 [官方文档](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe) 中找到在熊猫中创建数据帧的所有各种方法的细节。

在您尝试运行下面任何一个例子的代码之前，您需要导入 Pandas 和 NumPy:

```py
>>> import numpy as np
>>> import pandas as pd
```

### **#1 带字典**

使用字典创建数据帧时，其关键字成为列标签，值成为列。你可以这样定义一个:

```py
>>> d = {'x': [1, 2, 3], 'y': np.array([2, 4, 8]), 'z': 100}

>>> pd.DataFrame(d)
   x  y    z
0  1  2  100
1  2  4  100
2  3  8  100
```

你可以使用 columns 参数来控制列的顺序。index 参数做同样的事情，只是针对行。

```py
>>> pd.DataFrame(d, index=[100, 200, 300], columns=['z', 'y', 'x'])
       z  y  x
100  100  2  1
200  100  4  2
300  100  8  3
```

### **#2 带列表**

Pandas 还提供了使用字典列表创建数据框架的灵活性:

```py
>>> l = [{'x': 1, 'y': 2, 'z': 100},
...      {'x': 2, 'y': 4, 'z': 100},
...      {'x': 3, 'y': 8, 'z': 100}]

>>> pd.DataFrame(l)
   x  y    z
0  1  2  100
1  2  4  100
2  3  8  100
```

此外，您还可以灵活地使用嵌套列表来创建数据帧。但是在这样做的时候，最好是显式地指定行和/或列的标签，就像这样:

```py
>>> l = [[1, 2, 100],
...      [2, 4, 100],
...      [3, 8, 100]]

>>> pd.DataFrame(l, columns=['x', 'y', 'z'])
   x  y    z
0  1  2  100
1  2  4  100
2  3  8  100
```

值得注意的是，你也可以使用元组列表来代替上面的字典。

### **#3 带 NumPy 数组**

以同样的方式将二维列表传递给 DataFrame 构造函数，也可以传递 NumPy 数组:

```py
>>> arr = np.array([[1, 2, 100],
...                 [2, 4, 100],
...                 [3, 8, 100]])

>>> df_ = pd.DataFrame(arr, columns=['x', 'y', 'z'])
>>> df_
   x  y    z
0  1  2  100
1  2  4  100
2  3  8  100
```

看起来这个例子和我们讨论过的嵌套列表例子是一样的。这种方法有一个优点。您可以选择使用“复制”参数。

默认情况下，NumPy 数组的数据不会被复制，导致数组中的原始数据被分配给 DataFrame。所以，如果你修改数组，数据帧也会改变:

```py
>>> arr[0, 0] = 1000 # Changing the first item of the array

>>> df_ # Checking if the DataFrame is modified 
      x  y    z
0  1000  2  100
1     2  4  100
2     3  8  100
```

如果你想用 arr 中的值的副本而不是 arr 来创建你的 DataFrame，在 DataFrame 构造函数中提到“copy=True”。这样，即使 arr 被修改，数据帧也将保持不变。

### **#4 来自文件**

Pandas 允许您在各种文件类型之间保存和加载数据和标签，包括但不限于 CSV、JSON 和 SQL。要将数据帧保存为 CSV 文件，可以运行以下命令:

```py
>>> df.to_csv('data.csv')
```

一个 CSV 文件“data.csv”将出现在您的工作目录中。你也可以读取一个 CSV 文件，数据如下:

```py
>>> pd.read_csv('data.csv', index_col=0)
```

这段代码将会给你与我们在这篇文章中创建的第一个相同的数据框架。请记住，代码的“index_col=0”部分表示行标签位于文件的第一列。

## **访问和修改数据帧中的数据**

### **用存取器获取数据**

The。loc[]访问器允许你通过标签获取行和列，比如:

```py
>>> df.loc[10]
name John
city Oslo
age 41
test-score 88
Name: 10, dtype: object
```

你也可以使用。iloc[]访问器，用于按整数索引检索行或列。

```py
>>> df.iloc[0] # Returns first row
name John
city Oslo
age 41
test-score 88
Name: 10, dtype: object
```

在大多数情况下，使用这些访问器都是有帮助的。Python 还有另外两个访问器:。在[]和。iat[]。但是自从。loc[]和。iloc[]支持切片和 NumPy 风格的索引，你可以用它们来访问列，比如:

```py
>>> df.loc[:, 'city']

>>> df.iloc[:, 1]
```

也可以为列表或数组编写切片，而不是索引来检索行和列:

```py
>>> df.loc[11:15, ['name', 'city']] # Slicing to get rows

>>> df.iloc[1:6, [0, 1]] # Lists to get columns
```

如您所知，您可以对列表、元组和 NumPy 数组进行切片，并根据需要跳过值。好的一面是。iloc[]支持这个列表。

```py
>>> df.iloc[1:6:2, 0]
11 Lisa
13 Ling
15 Alan
Name: name, dtype: object
```

我们上面所做的切片从第二行开始，在索引为 6 的行之前停止，并且每隔一行跳过一行。

你不一定要使用切片构造来切片——你也可以使用 slice() Python 类来做。当然了，警察局。IndexSlice[]或 numpy.s []也可用于这些目的:

```py
>>> df.iloc[slice(1, 6, 2), 0]

>>> df.iloc[np.s_[1:6:2], 0]

>>> df.iloc[pd.IndexSlice[1:6:2], 0]
```

虽然所有这些方法完成的是同样的事情，但你可能会发现根据你的情况使用其中一种更容易。

如果你想获取一个单一的值，你可以使用。在[]和。iat[]，像这样:

```py
>>> df.at[12, 'name']

>>> df.iat[2, 0]

# Output of both is 'Gene'
```

### **用访问器修改数据**

您可以通过传递 Python 序列或 NumPy 数组来修改数据帧的一部分，就像这样:

```py
>>> df.loc[:13, 'test-score'] = [40, 50, 60, 70] # Modifying the first four items in the test-score column
>>> df.loc[14:, 'test-score'] = 0 # Setting 0 to the remaining columns 

>>> df['test-score']
10    40.0
11    50.0
12    60.0
13    70.0
14     0.0
15     0.0
16     0.0

Name: test-score, dtype: float64
```

你也可以使用负指数。iloc[]修改数据:

```py
>>> df.iloc[:, -1] = np.array([88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0])
```

## **在数据帧中插入和删除数据**

### **修改行**

假设你想在数据框中添加一个新的人。下面是如何创建一个新的 Series 对象:

```py
>>> rich = pd.Series(data=['Rich', 'Boston', 34, 79],
...                  index=df.columns, name=17)

>>> rich
name Rich
city Boston
age 31
test-score 79
Name: 17, dtype: object

>>> rich.name
17
```

您将需要使用“index=df.columns ”,因为对象的标签对应于 df 数据帧中的标签。

要将这个新的候选对象添加到 df 的末尾，可以使用。append()像这样:

```py
>>> df = df.append(rich)
```

如果您以后需要删除这个新行，您可以使用。drop()方法:

```py
>>> df = df.drop(labels=[17])
```

The。drop()方法返回删除了指定行的 DataFrame，但是您可以使用“inplace=True”来获取 None 作为返回值。

### **修改列**

假设有第二次考试，你需要将每个考生的分数加到表格中。下面是如何在一个新列中添加分数:

```py
>>> df['second-test-score'] = np.array([71.0, 95.0, 88.0, 79.0, 91.0, 91.0, 80.0])
```

如果您使用过 Python 中的字典，您可能会发现这种插入列的方法很熟悉。这段代码将在“测试分数”列的右边添加一个新的“第二次测试分数”列。

您也可以在整个列中分配相同的值。你所要做的就是运行下面的代码:

```py
>>> df['total-score'] = 0.0
```

虽然这种插入方法很简单，但它不允许在特定位置插入列。如果需要在表中的特定位置插入列，可以使用。insert()方法，像这样:

```py
>>> df.insert(loc=4, column='soft-skills-score',
...           value=np.array([86.0, 81.0, 78.0, 88.0, 74.0, 70.0, 81.0])) # Makes this new column the fourth column in the table
```

要从数据帧中删除一列，可以像使用 Python 字典一样使用 del 语句。

```py
>>> del df['soft-skills-score']
```

数据帧的另一个类似于字典的特征是数据帧与。pop()。此方法移除指定的列并将其返回。

换句话说，你也可以用“df.pop('soft-skills-score ')”来代替 del。

但是您可能需要从表格中删除不止一列。在这种情况下，您可以使用。drop()函数。您需要做的就是指定要删除的列的标签。

另外，当您打算删除列时，您需要提供参数“axis=1”。但是请记住，该方法将返回没有指定列的 DataFrame。如果你想看到指定的列，你可以传递“inplace=True”
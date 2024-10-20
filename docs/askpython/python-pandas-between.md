# Python 熊猫 between()方法——快速指南！

> 原文：<https://www.askpython.com/python-modules/pandas/python-pandas-between>

读者朋友们，你们好！在我们的 Pandas 模块系列中，我们将详细讨论一个尚未解决但很重要的函数——Python**Pandas between()函数**。

所以，让我们开始吧！

* * *

## 使用熊猫 between()方法

[Python Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)主要用于处理行和列中的数据值，即以一种表格/矩阵的形式。在其中，我们经常遇到保存数值类型值的数据变量。

在将数据处理成任何类型的动作(例如建模等)之前，数据的分析和转换是必要的。

简单地说，Python Pandas between()函数帮助我们在比较和最后时刻检查方面进行简单的分析。

**between()函数检查传递给该函数的起始值和结束值之间的值。**

**也就是说，在一系列值中，它将检查哪些数据元素落在传递的起始值和结束值之间。**

现在让我们试着理解同样的结构！

* * *

### 语法–Python Pandas between()方法

看看下面的语法！

```py
Series.between(start, end, inclusive=True)

```

*   **start** :开始检查的起始值。
*   **结束**:检查停止在该值。
*   **包含**:如果**为真**，则包含所传递的‘开始’和‘结束’值，并进行校验。如果设置为“**假**，则在执行检查时不包括“开始”和“结束”值。

补充一下，Python Pandas between()函数仅适用于数值和一维数据帧。

现在让我们通过一些例子来分析这个函数。

* * *

### 1.包含设置为“真”的 Python between()函数

在本例中，我们使用`pandas.DataFrame()` 函数创建了一个一维数据帧。

**举例:**

```py
import pandas as pd 
data = {"Roll-num": [10,20,30,40,50,60,70], "Age":[12,21,13,20,14,13,15], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}

block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)

```

**输出:**

看看下面的数据图！

```py
Original Data frame:

   Roll-num  Age    NAME
0        10   12    John
1        20   21  Camili
2        30   13  Rheana
3        40   20  Joseph
4        50   14  Amanti
5        60   13   Alexa
6        70   15    Siri

```

现在，我们已经在数据框的“年龄”变量上应用了 between()方法。

通过将 inclusive 设置为 true，它现在将包含并检查所有介于 12 和 15(包括 12 和 15)之间的值，然后为年龄介于设置范围内的索引返回 True。

```py
block["Age"].between(12, 15, inclusive = True)  

```

**输出:**

因此，对于索引 1 和 3，它返回 False，因为这些值超出了范围 12 到 15。

```py
0     True
1    False
2     True
3    False
4     True
5     True
6     True
Name: Age, dtype: bool

```

* * *

### 2.带分类变量的 Python between()函数

现在，让我们看看字符串或分类数据的结果。

如果我们将一个[字符串](https://www.askpython.com/python/string/strings-in-python)或非数字变量传递给 Pandas between()函数，它会将起始值和结束值与传递的数据进行比较，如果数据值与起始值或结束值匹配，则返回 True。

**举例:**

```py
block["NAME"].between("John", "Joseph", inclusive = True)   

```

**输出:**

因此，只有两个值返回为真。

```py
0     True
1    False
2    False
3     True
4    False
5    False
6    False
Name: NAME, dtype: bool

```

* * *

### 3.打印从 between()函数获得的值

在本例中，我们将尝试使用 Pandas between()函数打印介于 12 和 15 之间的数据。

**举例:**

```py
btwn = block["Age"].between(12, 15, inclusive = False)  
block[btwn] 

```

**输出:**

由于我们已经将 inclusive 设置为 False，它将检查位于 12 和 15 之间的值，不包括 12 和 15 本身。因此，它将 13、14 和 15 作为输出。

```py
     Roll-num	Age	NAME
2	30	13	Rheana
4	50	14	Amanti
5	60	13	Alexa

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂
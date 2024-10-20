# python astype()–数据列的类型转换

> 原文：<https://www.askpython.com/python/built-in-methods/python-astype>

在本文中，我们将详细讨论一个重要的概念——使用 Python astype()方法对数据帧中的列进行数据类型转换。

* * *

## 了解 Python astype()函数

在深入研究使用 Python astype()方法进行数据类型转换的概念之前，让我们首先考虑下面的场景。

在数据科学和机器学习领域，我们经常会遇到需要预处理和转换数据的阶段。事实上，准确地说，数据值的转换是走向建模的敏锐的一步。

这是数据列转换开始的时候。

**Python astype()方法使我们能够设置或转换数据集或数据框中现有数据列的数据类型。**

这样，我们可以使用 astype()函数将单个或多个列的数据值的类型改变或转换为另一种形式。

现在让我们在下一节中详细关注 astype()函数的语法。

* * *

## 语法–astype()函数

看看下面的语法！

```py
DataFrame.astype(dtype, copy=True, errors=’raise’)

```

*   **dtype** :我们要应用于整个数据框的数据类型。
*   **复制**:通过将它设置为**真**，它创建数据集的另一个副本，并向其灌输更改。
*   **错误**:通过将其设置为“**引发**，我们允许该函数引发异常。如果没有，我们可以将其设置为“**忽略**”。

理解了函数的语法之后，现在让我们把注意力放在函数的实现上！

* * *

### 1.带有数据帧的 Python astype()

在这个例子中，我们使用`pandas.DataFrame()` 方法从[字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)中创建了一个[数据帧](https://www.askpython.com/python-modules/pandas/dataframes-in-python)，如下所示。

**举例:**

```py
import pandas as pd 
data = {"Gender":['M','F','F','M','F','F','F'], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}

block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)
block.dtypes

```

**输出:**

让我们看一下键的原始数据类型。

```py
Original Data frame:

  Gender    NAME
0      M    John
1      F  Camili
2      F  Rheana
3      M  Joseph
4      F  Amanti
5      F   Alexa
6      F    Siri

Gender    object
NAME      object
dtype: object

```

现在，我们已经对“性别”列应用了 astype()方法，并将数据类型更改为“类别”。

```py
block['Gender'] = block['Gender'].astype('category')
block.dtypes

```

**输出:**

```py
Gender    category
NAME        object
dtype: object

```

* * *

### **2。使用数据集**实现 Python astype()

这里，我们使用 [pandas.read_csv()](https://www.askpython.com/python-modules/python-csv-module) 函数导入了数据集。你可以在这里找到数据集。

**举例:**

```py
import pandas 
BIKE = pandas.read_csv("Bike.csv")
BIKE.dtypes

```

**列的原始数据类型—**

```py
temp            float64
hum             float64
windspeed       float64
cnt               int64
season_1          int64
season_2          int64
season_3          int64
season_4          int64
yr_0              int64
yr_1              int64
mnth_1            int64
mnth_2            int64
mnth_3            int64
mnth_4            int64
mnth_5            int64
mnth_6            int64
mnth_7            int64
mnth_8            int64
mnth_9            int64
mnth_10           int64
mnth_11           int64
mnth_12           int64
weathersit_1      int64
weathersit_2      int64
weathersit_3      int64
holiday_0         int64
holiday_1         int64
dtype: object

```

现在，我们尝试更改变量“season_1”和“temp”的数据类型。因此，我们说使用 astype()函数，我们可以一次改变多个列的数据类型！

```py
BIKE = BIKE.astype({"season_1":'category', "temp":'int64'}) 
BIKE.dtypes

```

**输出:**

```py
temp               int64
hum              float64
windspeed        float64
cnt                int64
season_1        category
season_2           int64
season_3           int64
season_4           int64
yr_0               int64
yr_1               int64
mnth_1             int64
mnth_2             int64
mnth_3             int64
mnth_4             int64
mnth_5             int64
mnth_6             int64
mnth_7             int64
mnth_8             int64
mnth_9             int64
mnth_10            int64
mnth_11            int64
mnth_12            int64
weathersit_1       int64
weathersit_2       int64
weathersit_3       int64
holiday_0          int64
holiday_1          int64
dtype: object

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂
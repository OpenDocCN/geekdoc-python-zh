# 如何在 Python 中重置数据帧的索引？

> 原文：<https://www.askpython.com/python-modules/pandas/reset-index-of-a-dataframe>

读者你好！在本教程中，我们将讨论如何使用 reset_index()和 [concat()函数](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)重置 DataFrame 对象的索引。我们还将讨论需要重置熊猫数据帧索引的不同场景。

***也读作:[用 Python 索引一个数据帧](https://www.askpython.com/python-modules/pandas/dataframe-indexing)***

* * *

## pandas 中 reset_index()函数的语法

在 Python 中，我们可以使用 pandas DataFrame 类的`reset_index()`函数来重置 pandas DataFrame 对象的索引。默认情况下，`reset_index()`函数将熊猫数据帧的索引重置为熊猫默认索引，并返回带有新索引或`None`值的熊猫数据帧对象。如果 Pandas 数据帧有不止一级的索引，那么这个函数可以删除一级或多级。让我们快速了解一下`**reset_index()**`函数的语法。

```py
# Syntax of the reset_index() function in pandas
DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')

```

大多数情况下，我们将只使用两个参数 *drop* 和 *inplace* ，其余参数使用频率较低。

*   **drop** :尝试不将索引插入 DataFrame 列。它将相关熊猫数据帧的索引重置为默认的整数索引。它采用布尔值，即 True 或 False，默认情况下为 False。
*   **就地**:它不创建新的 pandas DataFrame 对象，而是进行修改，即就地重置 DataFrame 的索引。它还接收一个布尔值，该值默认为 False。

## 使用 reset_index()函数重置数据帧的索引

在 Python 中，我们需要在以下场景中重置 pandas DataFrame 对象的索引:

### 1.当在数据帧中插入行时

如果我们在原始 DataFrame 对象中添加几行，新的行索引从 0 开始。这里，我们可以应用`reset_index()`函数来重置数据帧的索引。看看下面的演示

```py
# Case-1
# When some rows are inserted in the DataFrame

# Import pandas
import pandas as pd

# Create a DataFrame object 
# Using DataFrame() function
df1 = pd.DataFrame({"Date": ['11/05/21', '15/06/21', '17/07/21'],
                   "Item": ['Television', 'Speaker', 'Monitor'],
                   "Sales": [200, 300, 115]})
print("Original DataFrame:\n")
print(df1)

# Create an another DataFrame
df2 = pd.DataFrame({"Date": ['04/05/20', '29/07/20', '30/08/20'],
                    "Item": ['Mixer', 'Bulb', 'Cooler'],
                    "Sales": [803, 178, 157]})

# Add the rows of DataFrame (df2) to the DataFrame (df1)
# Using the concat() function
df = pd.concat([df1, df2])
print("\nDataFrame after inserting some rows:\n") 
print(df)

# Reset the index of the final DataFrame 
# Using reset_index() function
df.reset_index(drop = True, inplace = True)
print("\nDataFrame after the resetting the index:\n")
print(df)

```

**输出:**

```py
Original DataFrame:

       Date        Item  Sales
0  11/05/21  Television    200
1  15/06/21     Speaker    300
2  17/07/21     Monitor    115

DataFrame after inserting some rows:

       Date        Item  Sales
0  11/05/21  Television    200
1  15/06/21     Speaker    300
2  17/07/21     Monitor    115
0  04/05/20       Mixer    803
1  29/07/20        Bulb    178
2  30/08/20      Cooler    157

DataFrame after the resetting the index:

       Date        Item  Sales
0  11/05/21  Television    200
1  15/06/21     Speaker    300
2  17/07/21     Monitor    115
3  04/05/20       Mixer    803
4  29/07/20        Bulb    178
5  30/08/20      Cooler    157

```

### 2.当数据帧中的行被删除时

在这种情况下，我们首先从原始 DataFrame 对象中删除一些被选中的行，在那里索引变得混乱。然后我们在最终的数据帧上应用`reset_index()`函数来重新计算这些值。让我们看看实现这种情况的 Python 代码。

```py
# Case-2
# When few rows from DataFrame are deleted

# Import pandas Python module
import pandas as pd

# Create a DataFrame object 
# Using DataFrame() function
df = pd.DataFrame({"Date": ['11/05/21', '15/06/21', '17/07/21', '19/11/20', '21/12/20'],
                   "Item": ['Television', 'Speaker', 'Desktop', 'Dish-Washer', 'Mobile'],
                   "Sales": [200, 300, 115, 303, 130]})
print("Original DataFrame:\n")
print(df)

# Delete few rows of the DataFrame (df)
# Using drop() function 
df = df.drop(labels = [0, 3], axis = 0)
print("\nDataFrame after deleting few rows:\n") 
print(df)

# Reset the index of the final DataFrame 
# Using reset_index() function
df.reset_index(drop = True, inplace = True)
print("\nDataFrame after the resetting the index:\n")
print(df)

```

**输出:**

```py
Original DataFrame:

       Date         Item  Sales
0  11/05/21   Television    200
1  15/06/21      Speaker    300
2  17/07/21      Desktop    115
3  19/11/20  Dish-Washer    303
4  21/12/20       Mobile    130

DataFrame after deleting few rows:

       Date     Item  Sales
1  15/06/21  Speaker    300
2  17/07/21  Desktop    115
4  21/12/20   Mobile    130

DataFrame after the resetting the index:

       Date     Item  Sales
0  15/06/21  Speaker    300
1  17/07/21  Desktop    115
2  21/12/20   Mobile    130

```

### 3.当在数据帧中对行进行排序时

在这种情况下，我们首先根据一列或多列对原始 DataFrame 对象的行进行排序，然后对最终的 DataFrame 对象应用`reset_index()`函数。让我们看看如何通过 Python 代码实现这个案例。

```py
# Case-3
# When rows of the DataFrame are sorted

# Import pandas Python module
import pandas as pd

# Create a DataFrame object 
# Using DataFrame() function
df = pd.DataFrame({"Date": ['11/05/21', '15/06/21', '17/07/21', '19/11/20', '21/12/20'],
                   "Item": ['Television', 'Speaker', 'Desktop', 'Dish-Washer', 'Mobile'],
                   "Sales": [200, 300, 115, 303, 130]})
print("Original DataFrame:\n")
print(df)

# Sort the rows of the DataFrame (df)
# Using sort_values() function
df.sort_values(by = "Sales", inplace = True)
print("\nDataFrame after sorting the rows by Sales:\n") 
print(df)

# Reset the index of the final DataFrame 
# Using reset_index() function
df.reset_index(drop = True, inplace = True)
print("\nDataFrame after the resetting the index:\n")
print(df)

```

**输出:**

```py
Original DataFrame:

       Date         Item  Sales
0  11/05/21   Television    200
1  15/06/21      Speaker    300
2  17/07/21      Desktop    115
3  19/11/20  Dish-Washer    303
4  21/12/20       Mobile    130

DataFrame after sorting the rows by Sales:

       Date         Item  Sales
2  17/07/21      Desktop    115
4  21/12/20       Mobile    130
0  11/05/21   Television    200
1  15/06/21      Speaker    300
3  19/11/20  Dish-Washer    303

DataFrame after the resetting the index:

       Date         Item  Sales
0  17/07/21      Desktop    115
1  21/12/20       Mobile    130
2  11/05/21   Television    200
3  15/06/21      Speaker    300
4  19/11/20  Dish-Washer    303

```

### 4.当附加两个数据帧时

同样，这是一种常用的情况，我们必须重置 pandas DataFrame 对象的索引。在这种情况下，我们首先将另一个 DataFrame 对象附加到我们的原始 DataFrame 对象，然后在最终组合的 DataFrame 对象上应用`reset_index()`函数。让我们编写 Python 代码来实现这种情况。

```py
# Case-4
# When two DataFrames are appended

# Import pandas Python module
import pandas as pd

# Create a DataFrame object 
# Using DataFrame() function
df1 = pd.DataFrame({"Date": ['11/05/21', '15/06/21', '17/07/21'],
                   "Item": ['Television', 'Speaker', 'Desktop'],
                   "Sales": [200, 300, 115]})
print("Original DataFrame:\n")
print(df1)

# Create a new DataFrame
df2 = pd.DataFrame({"Date": ['19/11/20', '21/12/20'],
                    "Item": ['Dish-Washer', 'Mobile'],
                    "Sales": [403, 130]})

# Append the new DataFrame (df1) to the previous one (df2)
df = df1.append(df2)
print("\nDataFrame after appending the new DataFrame:\n") 
print(df)

# Reset the index of the final DataFrame 
# Using reset_index() function
df.reset_index(drop = True, inplace = True)
print("\nDataFrame after the resetting the index:\n")
print(df)

```

**输出:**

```py
Original DataFrame:

       Date        Item  Sales
0  11/05/21  Television    200
1  15/06/21     Speaker    300
2  17/07/21     Desktop    115

DataFrame after appending the new DataFrame:

       Date         Item  Sales
0  11/05/21   Television    200
1  15/06/21      Speaker    300
2  17/07/21      Desktop    115
0  19/11/20  Dish-Washer    403
1  21/12/20       Mobile    130

DataFrame after the resetting the index:

       Date         Item  Sales
0  11/05/21   Television    200
1  15/06/21      Speaker    300
2  17/07/21      Desktop    115
3  19/11/20  Dish-Washer    403
4  21/12/20       Mobile    130

```

## 使用 concat()函数重置数据帧的索引

在 Python 中，我们还可以使用 pandas `concat()`函数和`ignor_index`参数来重置 pandas DataFrame 对象的索引。默认情况下，`ignore_index`参数的值为**假**。为了重置数据帧的索引，我们必须将其值设置为**真**。让我们通过 Python 代码来实现这一点。

```py
# Reset the index of DataFrame using concat() function

# Import pandas Python module
import pandas as pd

# Create a DataFrame object 
# Using DataFrame() function
df1 = pd.DataFrame({"Date": ['11/05/21', '15/06/21', '17/07/21'],
                   "Item": ['Television', 'Speaker', 'Desktop'],
                   "Sales": [200, 300, 115]})
print("Original DataFrame:\n")
print(df1)

# Create a new DataFrame
df2 = pd.DataFrame({"Date": ['14/10/20', '19/11/20', '21/12/20'],
                    "Item": ['Oven', 'Toaster', 'Fan'],
                    "Sales": [803, 178, 157]})

# Concat the new DataFrame (df2) with the prevous one (df1)
# And reset the index of the DataFrame
# Using the concat() function with ignor_index parameter
df = pd.concat([df1, df2], ignore_index = True)
print("\nDataFrame after concatenation and index reset:\n") 
print(df)

```

**输出:**

```py
Original DataFrame:

       Date        Item  Sales
0  11/05/21  Television    200
1  15/06/21     Speaker    300
2  17/07/21     Desktop    115

DataFrame after concatenation and index reset:

       Date        Item  Sales
0  11/05/21  Television    200
1  15/06/21     Speaker    300
2  17/07/21     Desktop    115
3  14/10/20        Oven    803
4  19/11/20     Toaster    178
5  21/12/20         Fan    157

```

## 结论

在本教程中，我们学习了如何以及何时使用 pandas `reset_index()`函数来重置修改后的 pandas DataFrame 对象的索引。希望您已经理解了上面讨论的内容，并对自己执行这些数据帧操作感到兴奋。感谢阅读，请继续关注我们，获取更多与 Python 相关的足智多谋的文章。
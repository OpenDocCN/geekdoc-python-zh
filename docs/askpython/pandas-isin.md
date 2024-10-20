# 熊猫的 isin()函数——一个完整的指南

> 原文：<https://www.askpython.com/python-modules/pandas/pandas-isin>

大家好！在本教程中，我们将学习熊猫模块中的`isin()`方法，并且我们将研究当不同类型的值被传递时这个函数的行为。所以让我们开始吧。

## DataFrame.isin()方法

Pandas `isin()`方法用于过滤数据帧中的数据。该方法检查数据帧中的每个元素是否包含在指定的值中。这个方法返回布尔值的数据帧。如果元素出现在指定的值中，返回的数据帧包含`True`，否则显示`False`。因此，这种方法在过滤数据帧时很有用，我们将通过下面的例子看到这一点。

`isin()`方法的语法如下所示。它只需要一个参数:

```py
DataFrame.isin(values)

```

这里参数`values`可以是其中的任何一个:

*   列表或可迭代
*   词典
*   熊猫系列
*   熊猫数据框

让我们看看当不同的值被传递给方法时`isin()`方法的结果。

## isin()方法的示例

让我们考虑一些通过传递不同类型的值的`isin()`方法的例子。对于以下示例，我们将使用以下数据:

```py
import pandas as pd

data = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'Age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

print(data)

```

```py
    Name  Age      Department
0   John   25           Sales
1    Sam   45     Engineering
2   Luna   23     Engineering
3  Harry   32  Human Resource

```

### 值为列表时的 isin()方法

当一个列表作为参数值传递给`isin()`方法时，它检查 DataFrame 中的每个元素是否都在列表中，如果找到，就显示`True`。例如，如果我们传递一个包含一些部门的值的列表，`Department`列中的那些值将被标记为`True`。

```py
import pandas as pd
# Creating DataFrame
data = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'Age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

#List of Departments to filter
departments_to_filter = ['Engineering', 'Sales', 'Finance']

result = data.isin(departments_to_filter)

print(result)

```

```py
    Name    Age  Department
0  False  False        True
1  False  False        True
2  False  False        True
3  False  False       False

```

因此，使用这种方法，我们也可以根据情况过滤数据帧。例如，我们想要查找年龄在 20 到 30 岁之间的员工，我们可以在`Age`列上使用`isin()`方法。

```py
import pandas as pd
# Creating DataFrame
data = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'Age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

start_age=20
end_age=30
# Using isin() method to filter employees on age
age_filter = data['Age'].isin(range(start_age, end_age+1))
# Using the filter to retrieve the data
result = data[ age_filter ]

print(result)

```

```py
   Name  Age   Department
0  John   25        Sales
2  Luna   23  Engineering

```

### 值为字典时的 isin()方法

当一个字典作为参数值传递给`isin()`方法时，对于 DataFrame 的不同列，要搜索的数据范围是不同的。因此，我们可以分别搜索每一列。例如，在字典中，我们可以为`Name`和`Department`传递一个带有各自值的列表来进行搜索，如下所示。

```py
import pandas as pd
# Creating DataFrame
data = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'Age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

#Dictionary data to filter DataFrame
dict_data_to_filter = {'Name': ['Sam', 'Harry'], 'Department': ['Engineering']}

result = data.isin(dict_data_to_filter)

print(result)

```

```py
    Name    Age  Department
0  False  False       False
1   True  False        True
2  False  False        True
3   True  False       False

```

### 值为序列时的 isin()方法

当熊猫序列作为参数值传递给`isin()`方法时，序列中写入值的顺序变得很重要。数据帧的每一列都将按照写入的顺序，用序列中的值逐一检查。考虑下面的例子。

```py
import pandas as pd
# Creating DataFrame
data = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'Age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

#Series data, changing index of Sam and Luna
series_data = pd.Series(['John', 'Luna', 'Sam', 'Harry'])

result = data.isin(series_data)

print(result)

```

```py
    Name    Age  Department
0   True  False       False
1  False  False       False
2  False  False       False
3   True  False       False

```

尽管序列中出现的值包含数据 DataFrame 中出现的所有`Names`，但索引 1 和 2 处的结果包含`False`，因为我们互换了“Sam”和“Luna”的索引。因此，当序列作为值传递时，索引很重要。

### 当值为 DataFrame 时，使用 isin()方法

当 Pandas 数据帧作为参数值传递给`isin()`方法时，传递的数据帧的索引和列必须匹配。如果两个数据帧相同，但列名不匹配，结果将显示这些列的`False`。如果两个数据帧中的数据相同，但顺序不同，那么对于那些不同的行，结果将是`False`。因此，如果传递数据帧，索引和列都很重要。考虑这个例子。

```py
import pandas as pd
# Creating DataFrame
data = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'Age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

# DataFrame to filter, here column name Age to lowercased to age
df = pd.DataFrame({
  'Name': ['John', 'Sam', 'Luna', 'Harry'],
  'age': [25, 45, 23, 32],
  'Department': ['Sales', 'Engineering', 'Engineering', 'Human Resource']
})

result = data.isin(df)
print(result)

print("-----------------")

# DataFrame to filter, here last 2 rows are swapped
df = pd.DataFrame({
  'Name': ['John', 'Sam', 'Harry', 'Luna'],
  'Age': [25, 45, 32, 23],
  'Department': ['Sales', 'Engineering', 'Human Resource', 'Engineering']
})

result = data.isin(df)
print(result)

```

```py
   Name    Age  Department
0  True  False        True
1  True  False        True
2  True  False        True
3  True  False        True
-----------------
    Name    Age  Department
0   True   True        True
1   True   True        True
2  False  False       False
3  False  False       False

```

## 结论

在本教程中，我们学习了 Pandas `isin()`方法，它的不同用例，以及该方法如何帮助从数据帧中过滤出数据。现在你知道了如何使用`isin()`方法，并且可以在数据框架中轻松过滤数据，所以恭喜你。

感谢阅读！！
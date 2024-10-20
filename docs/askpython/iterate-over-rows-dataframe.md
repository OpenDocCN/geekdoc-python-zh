# 如何迭代熊猫 Dataframe 中的行？

> 原文：<https://www.askpython.com/python-modules/pandas/iterate-over-rows-dataframe>

“迭代”一词意味着一个接一个地获取数据结构中包含的每个元素的过程。在 python 中，我们使用循环多次遍历条目。我们也可以将迭代称为“项目的重复执行”。Pandas 是 Python 中一个非常有用的库，因为它提供了许多数据分析工具。在本文中，我们将学习如何在 Pandas 数据帧中迭代行。所以让我们开始吧！

## 熊猫的数据框架是什么？

Pandas DataFrame 是由行和列组成的二维表格数据结构。DataFrame 是 Python 中的可变数据结构。

例如:

```py
import pandas as pd

#Creating the data
data = {'Name':['Tommy','Linda','Justin','Brendon'], 'Marks':[100,200,300,600]}
df= pd.DataFrame(data)
print(df)

```

输出:

```py
      Name        Marks
0    Tommy    100
1    Linda       200
2   Justin       300
3  Brendon    600

```

现在让我们来看看迭代行的方法。

## 在 Pandas 数据帧中迭代行的方法

有许多方法可以用来迭代 Pandas 数据帧中的行，但是每种方法都有自己的优点和缺点。

### 1.使用 iterrows()方法

这是 Python 中迭代行的简单明了的方法之一。虽然这是最简单的方法，但迭代进行得很慢，效率也不高。该方法将返回整行以及行索引。

例如:

```py
import pandas as pd

data = {'Name': ['Tommy', 'Linda', 'Justin', 'Brendon'],
                'Age': [21, 19, 20, 18],
                'Subject': ['Math', 'Commerce', 'Arts', 'Biology'],
                'Scores': [88, 92, 95, 70]}

df = pd.DataFrame(data, columns = ['Name', 'Age', 'Subject', 'Scores'])

print("The DataFrame is :\n", df)

print("\nPerforming Interation using iterrows() method :\n")

# iterate through each row and select 'Name' and 'Scores' column respectively.
for index, row in df.iterrows():
    print (row["Name"], row["Scores"])

```

输出:

```py
The DataFrame is :
       Name  Age   Subject  Scores
0    Tommy   21      Math      88
1    Linda   19  Commerce      92
2   Justin   20      Arts      95
3  Brendon   18   Biology      70

Performing Interation using iterrows() method :

Tommy 88
Linda 92
Justin 95
Brendon 70

```

### 2.使用 itertuples()方法

该方法与 iterrows()方法非常相似，只是它返回命名元组。在元组的帮助下，您可以访问作为属性的特定值，或者换句话说，我们可以访问列中某一行的特定值。这是一种更健壮的方法，并且迭代速度比 iterrows()方法更快。

例如:

```py
import pandas as pd

# Creating a dictionary containing students data
data = {'Name': ['Tommy', 'Linda', 'Justin', 'Brendon'],
                'Age': [21, 19, 20, 18],
                'Subject': ['Math', 'Commerce', 'Arts', 'Biology'],
                'Scores': [88, 92, 95, 70]}

# Converting the dictionary into DataFrame
df = pd.DataFrame(data, columns = ['Name', 'Age', 'Subject', 'Scores'])

print("Given Dataframe :\n", df)

print("\n Performing iteration over rows using itertuples() method :\n")

# iterate through each row and select 'Name' and 'Scores' column respectively.
for row in df.itertuples(index = True, name ='Pandas'):
    print (getattr(row, "Name"), getattr(row, "Scores"))

```

输出:

```py
Given Dataframe :
       Name  Age   Subject  Scores
0    Tommy   21      Math      88
1    Linda   19  Commerce      92
2   Justin   20      Arts      95
3  Brendon   18   Biology      70

Performing iteration over rows using itertuples() method :

Tommy 88
Linda 92
Justin 95
Brendon 70

```

### 3.使用 apply()方法

这种方法是最有效的方法，比上面两种方法运行时间更快。

例如:

```py
import pandas as pd
import pandas as pd

# Creating a dictionary containing students data
data = {'Name': ['Tommy', 'Linda', 'Justin', 'Brendon'],
                'Age': [21, 19, 20, 18],
                'Subject': ['Math', 'Commerce', 'Arts', 'Biology'],
                'Scores': [88, 92, 95, 70]}

# Converting the dictionary into DataFrame
df = pd.DataFrame(data, columns = ['Name', 'Age', 'Stream', 'Scores'])

print("Given Dataframe :\n", df)

print("\nPerforming Iteration over rows using apply function :\n")

# iterate through each row and concatenate 'Name' and 'Scores' column 
print(df.apply(lambda row: row["Name"] + " " + str(row["Scores"]), axis = 1)) 

```

输出:

```py
Given Dataframe :
       Name  Age Stream  Scores
0    Tommy   21    NaN      88
1    Linda   19    NaN      92
2   Justin   20    NaN      95
3  Brendon   18    NaN      70

Performing Iteration over rows using apply function :

0      Tommy 88
1      Linda 92
2     Justin 95
3    Brendon 70
dtype: object

```

### 4.使用 iloc []功能

这是我们可以用来遍历行的另一个简单函数。我们将在迭代后使用 iloc[]函数选择列的索引。

例如:

```py
import pandas as pd

# Creating a dictionary containing students data
data = {'Name': ['Tommy', 'Linda', 'Justin', 'Brendon'],
                'Age': [21, 19, 20, 18],
                'Subject': ['Math', 'Commerce', 'Arts', 'Biology'],
                'Scores': [88, 92, 95, 70]}

# Converting the dictionary into DataFrame
df = pd.DataFrame(data, columns = ['Name', 'Age', 'Subject', 'Scores'])

print("Given Dataframe :\n", df)

print("\nIterating over rows using iloc function :\n")

# iterate through each row and select 0th and 3rd index column 
for i in range(len(df)) :
  print(df.iloc[i, 0], df.iloc[i, 3])

```

输出:

```py
Given Dataframe :
       Name  Age   Subject  Scores
0    Tommy   21      Math      88
1    Linda   19  Commerce      92
2   Justin   20      Arts      95
3  Brendon   18   Biology      70

Performing Iteration over rows using iloc function :

Tommy 88
Linda 92
Justin 95
Brendon 70

```

## 结论

在本文中，我们学习了在 python 中迭代行的不同方法。iterrows()和 itertuples()方法并不是迭代数据帧行的最有效方法，尽管它们相当简单。为了获得更好的结果和更快的运行时间，您应该寻找 apply()方法。
# 熊猫里的泡菜档案怎么看？

> 原文：<https://www.askpython.com/python-modules/pandas/read-pickle-files-in-pandas>

我们通常以 CSV、excel 或文本文件的形式使用或存储数据。但是我们也可以将数据保存为 Pickle 文件。Pickles 是在磁盘上表示 Python 对象的一种方式。它们以序列化格式存储对象，这种格式可用于以后重建对象。Pickles 对于存储需要快速方便地访问的数据很有用。在本文中，我们将学习如何从 pickle 文件中存储和读取 Pandas 中的数据。让我们开始吧！

## 使用熊猫读取泡菜文件

Pandas 提供了读写 pickle 文件的方法。读取 pickle 文件最基本的方法是使用 read_pickle()函数。该函数将 pickle 文件的名称作为参数，并返回一个 pandas 数据帧。

可以使用 read_pickle()函数在 Python 中读取 pickle 文件。

**函数的语法:**

```py
pd.read_pickle(path, compression='infer')

```

类似于 [read_csv()函数](https://www.askpython.com/python-modules/pandas/read-csv-with-delimiters)，这个函数也将返回一个 Pandas DataFrame 作为输出。

**例如:**

```py
df = pd.read_pickle('data.pkl')

```

现在让我们看看如何在 python 中将数据保存到 pickle 文件中。我们将从创建数据帧开始。

```py
import pandas as pd
data = {
    'Name': ['Microsoft Corporation', 'Google, LLC', 'Tesla, Inc.',\
             'Apple Inc.', 'Netflix, Inc.'],
    'Icon': ['MSFT', 'GOOG', 'TSLA', 'AAPL', 'NFLX'],
    'Field': ['Tech', 'Tech', 'Automotive', 'Tech', 'Entertainment'],
    'Market Shares': [100, 50, 160, 300, 80]
           }
df = pd.DataFrame(data)
# print dataframe
print(df)

```

**输出**

```py
  Name  Icon          Field  Market Shares
0  Microsoft Corporation  MSFT           Tech            100
1            Google, LLC  GOOG           Tech             50
2            Tesla, Inc.  TSLA     Automotive            160
3             Apple Inc.  AAPL           Tech            300
4          Netflix, Inc.  NFLX  Entertainment             80

```

现在让我们将数据帧保存到 pickle 文件中。

```py
df.to_pickle('company info.pkl')

```

现在让我们来读泡菜文件。

```py
df2 = pd.read_pickle('company info.pkl')
# print the dataframe
print(df2)

```

输出

```py
   Name  Icon          Field  Market Shares
0  Microsoft Corporation  MSFT           Tech            100
1            Google, LLC  GOOG           Tech             50
2            Tesla, Inc.  TSLA     Automotive            150
3             Apple Inc.  AAPL           Tech            200
4          Netflix, Inc.  NFLX  Entertainment             80

```

## 结论

总之，我们学习了如何在 Pandas 中使用 read_pickle()函数读取 pickle 文件。还可以使用 read_pickle()函数读取序列化为 pickle 对象的数据帧。Pickle 文件非常适合存储数据，但是如果您使用 pickle 文件中的数据，请确保它来自可靠的来源。
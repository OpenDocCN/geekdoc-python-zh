# 用 Pandas 和 NumPy 清理 Pythonic 数据

> 原文：<https://realpython.com/python-data-cleaning-numpy-pandas/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用熊猫和 NumPy**](/courses/data-cleaning-with-pandas-and-numpy/) 进行数据清洗

数据科学家花费大量时间清理数据集，并将它们转换成他们可以使用的形式。事实上，许多数据科学家认为，获取和清理数据的初始步骤构成了 80%的工作。

因此，如果您刚刚进入这个领域或者[计划进入这个领域](https://realpython.com/tutorials/data-science/)，能够处理杂乱的数据是很重要的，无论这意味着丢失的值、不一致的格式、畸形的记录还是无意义的异常值。

在本教程中，我们将利用 Python 的 [Pandas](https://realpython.com/pandas-python-explore-dataset/) 和 NumPy 库来清理数据。

我们将讨论以下内容:

*   删除`DataFrame`中不必要的列
*   改变一个`DataFrame`的索引
*   使用`.str()`方法清洁色谱柱
*   使用`DataFrame.applymap()`函数逐个元素地清理整个数据集
*   将列重命名为更容易识别的标签集
*   跳过 CSV 文件中不必要的行

**免费奖励:** [点击此处获取免费的 NumPy 资源指南](#)，它会为您指出提高 NumPy 技能的最佳教程、视频和书籍。

以下是我们将使用的数据集:

*   BL-Flickr-Images-book . csv–包含大英图书馆书籍信息的 CSV 文件
*   [university _ towns . txt](https://github.com/realpython/python-data-cleaning/blob/master/Datasets/university_towns.txt)–包含美国各州大学城名称的文本文件
*   [Olympics . csv](https://github.com/realpython/python-data-cleaning/blob/master/Datasets/olympics.csv)–总结所有国家参加夏季和冬季奥运会的 CSV 文件

您可以从 Real Python 的 [GitHub 仓库](https://github.com/realpython/python-data-cleaning)下载数据集，以便了解这里的示例。

**注意**:我推荐使用 Jupyter 笔记本来跟进。

本教程假设对 Pandas 和 NumPy 库有基本的了解，包括 Panda 的主力 [`Series`和`DataFrame`对象](https://pandas.pydata.org/pandas-docs/stable/dsintro.html)，可以应用于这些对象的常用方法，以及熟悉 NumPy 的 [`NaN`](https://docs.scipy.org/doc/numpy-1.13.0/user/misc.html) 值。

让我们导入所需的模块并开始吧！

>>>

```py
>>> import pandas as pd
>>> import numpy as np
```

## 在`DataFrame`中拖放列

通常，您会发现并非数据集中的所有数据类别都对您有用。例如，您可能有一个包含学生信息(姓名、年级、标准、父母姓名和地址)的数据集，但您希望专注于分析学生的成绩。

在这种情况下，地址或父母的名字对你来说并不重要。保留这些不需要的类别会占用不必要的空间，还可能会影响运行时间。

Pandas 提供了一种简便的方法，通过 [`drop()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html) 功能从`DataFrame`中删除不需要的列或行。让我们看一个简单的例子，我们从一个`DataFrame`中删除了一些列。

首先，让我们从 CSV 文件“BL-Flickr-Images-Book.csv”中创建一个 [`DataFrame`](https://realpython.com/pandas-dataframe/) 。在下面的例子中，我们传递了一个到`pd.read_csv`的相对路径，这意味着所有的数据集都在我们当前工作目录中一个名为`Datasets`的文件夹中:

>>>

```py
>>> df = pd.read_csv('Datasets/BL-Flickr-Images-Book.csv')
>>> df.head()

 Identifier             Edition Statement      Place of Publication  \
0         206                           NaN                    London
1         216                           NaN  London; Virtue & Yorston
2         218                           NaN                    London
3         472                           NaN                    London
4         480  A new edition, revised, etc.                    London

 Date of Publication              Publisher  \
0         1879 [1878]       S. Tinsley & Co.
1                1868           Virtue & Co.
2                1869  Bradbury, Evans & Co.
3                1851          James Darling
4                1857   Wertheim & Macintosh

 Title     Author  \
0                  Walter Forbes. [A novel.] By A. A      A. A.
1  All for Greed. [A novel. The dedication signed...  A., A. A.
2  Love the Avenger. By the author of “All for Gr...  A., A. A.
3  Welsh Sketches, chiefly ecclesiastical, to the...  A., E. S.
4  [The World in which I live, and my place in it...  A., E. S.

 Contributors  Corporate Author  \
0                               FORBES, Walter.               NaN
1  BLAZE DE BURY, Marie Pauline Rose - Baroness               NaN
2  BLAZE DE BURY, Marie Pauline Rose - Baroness               NaN
3                   Appleyard, Ernest Silvanus.               NaN
4                           BROOME, John Henry.               NaN

 Corporate Contributors Former owner  Engraver Issuance type  \
0                     NaN          NaN       NaN   monographic
1                     NaN          NaN       NaN   monographic
2                     NaN          NaN       NaN   monographic
3                     NaN          NaN       NaN   monographic
4                     NaN          NaN       NaN   monographic

 Flickr URL  \
0  http://www.flickr.com/photos/britishlibrary/ta...
1  http://www.flickr.com/photos/britishlibrary/ta...
2  http://www.flickr.com/photos/britishlibrary/ta...
3  http://www.flickr.com/photos/britishlibrary/ta...
4  http://www.flickr.com/photos/britishlibrary/ta...

 Shelfmarks
0    British Library HMNTS 12641.b.30.
1    British Library HMNTS 12626.cc.2.
2    British Library HMNTS 12625.dd.1.
3    British Library HMNTS 10369.bbb.15.
4    British Library HMNTS 9007.d.28.
```

当我们使用`head()`方法查看前五个条目时，我们可以看到一些列提供了对图书馆有帮助的辅助信息，但并没有很好地描述书籍本身:`Edition Statement`、`Corporate Author`、`Corporate Contributors`、`Former owner`、`Engraver`、`Issuance type`和`Shelfmarks`。

我们可以通过以下方式删除这些列:

>>>

```py
>>> to_drop = ['Edition Statement',
...            'Corporate Author',
...            'Corporate Contributors',
...            'Former owner',
...            'Engraver',
...            'Contributors',
...            'Issuance type',
...            'Shelfmarks']

>>> df.drop(to_drop, inplace=True, axis=1)
```

上面，我们定义了一个[列表](https://realpython.com/python-lists-tuples/)，其中包含了我们想要删除的所有列的名称。接下来，我们调用对象上的`drop()`函数，将`inplace`参数作为`True`传入，将`axis`参数作为`1`传入。这告诉 Pandas 我们希望直接在我们的对象中进行更改，并且它应该在对象的列中寻找要删除的值。

当我们再次检查`DataFrame`时，我们会看到不需要的列已经被删除:

>>>

```py
>>> df.head()
 Identifier      Place of Publication Date of Publication  \
0         206                    London         1879 [1878]
1         216  London; Virtue & Yorston                1868
2         218                    London                1869
3         472                    London                1851
4         480                    London                1857

 Publisher                                              Title  \
0       S. Tinsley & Co.                  Walter Forbes. [A novel.] By A. A
1           Virtue & Co.  All for Greed. [A novel. The dedication signed...
2  Bradbury, Evans & Co.  Love the Avenger. By the author of “All for Gr...
3          James Darling  Welsh Sketches, chiefly ecclesiastical, to the...
4   Wertheim & Macintosh  [The World in which I live, and my place in it...

 Author                                         Flickr URL
0      A. A.  http://www.flickr.com/photos/britishlibrary/ta...
1  A., A. A.  http://www.flickr.com/photos/britishlibrary/ta...
2  A., A. A.  http://www.flickr.com/photos/britishlibrary/ta...
3  A., E. S.  http://www.flickr.com/photos/britishlibrary/ta...
4  A., E. S.  http://www.flickr.com/photos/britishlibrary/ta...
```

或者，我们也可以通过将列直接传递给`columns`参数来删除列，而不是单独指定要删除的标签和熊猫应该在哪个轴上寻找标签:

>>>

```py
>>> df.drop(columns=to_drop, inplace=True)
```

这种语法更直观，可读性更强。我们要做的事情很明显。

如果您事先知道想要保留哪些列，另一个选项是将它们传递给`pd.read_csv`的`usecols`参数。

[*Remove ads*](/account/join/)

## 改变一个`DataFrame`的索引

Pandas `Index`扩展了 NumPy 数组的功能，允许更多的切片和标记。在许多情况下，使用数据的唯一值标识字段作为索引是很有帮助的。

例如，在上一节使用的数据集中，可以预计当图书管理员搜索记录时，他们可能会输入一本书的唯一标识符(`Identifier`列中的值):

>>>

```py
>>> df['Identifier'].is_unique
True
```

让我们使用`set_index`用这个列替换现有的索引:

>>>

```py
>>> df = df.set_index('Identifier')
>>> df.head()
 Place of Publication Date of Publication  \
206                           London         1879 [1878]
216         London; Virtue & Yorston                1868
218                           London                1869
472                           London                1851
480                           London                1857

 Publisher  \
206              S. Tinsley & Co.
216                  Virtue & Co.
218         Bradbury, Evans & Co.
472                 James Darling
480          Wertheim & Macintosh

 Title     Author  \
206                         Walter Forbes. [A novel.] By A. A      A. A.
216         All for Greed. [A novel. The dedication signed...  A., A. A.
218         Love the Avenger. By the author of “All for Gr...  A., A. A.
472         Welsh Sketches, chiefly ecclesiastical, to the...  A., E. S.
480         [The World in which I live, and my place in it...  A., E. S.

 Flickr URL
206         http://www.flickr.com/photos/britishlibrary/ta...
216         http://www.flickr.com/photos/britishlibrary/ta...
218         http://www.flickr.com/photos/britishlibrary/ta...
472         http://www.flickr.com/photos/britishlibrary/ta...
480         http://www.flickr.com/photos/britishlibrary/ta...
```

**技术细节**:与 [SQL](https://realpython.com/python-sql-libraries/) 中的主键不同，Pandas `Index`不保证唯一性，尽管许多索引和合并操作会注意到运行时的加速。

我们可以用`loc[]`直接访问每条记录。虽然`loc[]`可能没有名字那么直观，但它允许我们做*基于标签的索引*，这是对行或记录的标签，而不考虑其位置:

>>>

```py
>>> df.loc[206]
Place of Publication                                               London
Date of Publication                                           1879 [1878]
Publisher                                                S. Tinsley & Co.
Title                                   Walter Forbes. [A novel.] By A. A
Author                                                              A. A.
Flickr URL              http://www.flickr.com/photos/britishlibrary/ta...
Name: 206, dtype: object
```

换句话说，206 是索引的第一个标签。要通过*位置*访问它，我们可以使用`df.iloc[0]`，它执行基于位置的索引。

**技术细节** : `.loc[]`在技术上是一个[类实例](https://github.com/pandas-dev/pandas/blob/7273ea0709590e6264607f227bb8def0ef656c50/pandas/core/indexing.py#L1415)，并且有一些特殊的[语法](https://pandas.pydata.org/pandas-docs/stable/indexing.html#selection-by-label)，这些语法并不完全符合大多数普通的 Python 实例方法。

以前，我们的索引是一个 RangeIndex:从`0`开始的整数，类似于 Python 的内置`range`。通过将一个列名传递给`set_index`，我们将索引更改为`Identifier`中的值。

您可能已经注意到，我们将[变量](https://realpython.com/python-variables/)重新分配给了由带有`df = df.set_index(...)`的方法返回的对象。这是因为，默认情况下，该方法返回我们的对象的修改副本，并不直接对对象进行更改。我们可以通过设置`inplace`参数来避免这种情况:

```py
df.set_index('Identifier', inplace=True)
```

## 整理数据中的字段

到目前为止，我们已经删除了不必要的列，并将`DataFrame`的索引改为更合理的。在这一节中，我们将清理特定的列，并将它们转换为统一的格式，以便更好地理解数据集并增强一致性。特别是，我们将清洁`Date of Publication`和`Place of Publication`。

经检查，目前所有的数据类型都是`object` [dtype](http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes) ，这大致类似于原生 Python 中的`str`。

它封装了任何不能作为数字或分类数据的字段。这是有意义的，因为我们处理的数据最初是一堆杂乱的[字符串](https://realpython.com/python-strings/):

>>>

```py
>>> df.get_dtype_counts()
object    6
```

强制使用数字值有意义的一个字段是出版日期，这样我们可以在以后进行计算:

>>>

```py
>>> df.loc[1905:, 'Date of Publication'].head(10)
Identifier
1905           1888
1929    1839, 38-54
2836        [1897?]
2854           1865
2956        1860-63
2957           1873
3017           1866
3131           1899
4598           1814
4884           1820
Name: Date of Publication, dtype: object
```

一本书只能有一个出版日期。因此，我们需要做到以下几点:

*   删除方括号中的多余日期:1879 [1878]
*   将日期范围转换为它们的“开始日期”，如果有的话:1860-63；1839, 38-54
*   完全去掉我们不确定的日期，用 NumPy 的`NaN`:【1897？]
*   将字符串`nan`转换为 NumPy 的`NaN`值

综合这些模式，我们实际上可以利用一个正则表达式来提取出版年份:

>>>

```py
regex = r'^(\d{4})'
```

上面的正则表达式旨在查找字符串开头的任意四位数字，这就满足了我们的情况。上面是一个*原始字符串*(意思是反斜杠不再是转义字符)，这是正则表达式的标准做法。

`\d`代表任意数字，`{4}`重复这个规则四次。`^`字符匹配一个字符串的开头，圆括号表示一个捕获组，这向 Pandas 发出信号，表明我们想要提取正则表达式的这一部分。(我们希望`^`避免`[`开始串的情况。)

让我们看看在数据集上运行这个正则表达式会发生什么:

>>>

```py
>>> extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
>>> extr.head()
Identifier
206    1879
216    1868
218    1869
472    1851
480    1857
Name: Date of Publication, dtype: object
```

**延伸阅读:**不熟悉 regex？你可以在 regex101.com[查看上面](https://regex101.com/r/3AJ1Pv/1)的表达式，用[正则表达式:Python 中的正则表达式](https://realpython.com/regex-python/)学习所有关于正则表达式的知识。

从技术上讲，这个列仍然有`object` dtype，但是我们可以很容易地用`pd.to_numeric`得到它的数字版本:

>>>

```py
>>> df['Date of Publication'] = pd.to_numeric(extr)
>>> df['Date of Publication'].dtype
dtype('float64')
```

这导致大约十分之一的值丢失，对于现在能够对剩余的有效值进行计算来说，这是一个很小的代价:

>>>

```py
>>> df['Date of Publication'].isnull().sum() / len(df)
0.11717147339205986
```

太好了！就这么定了！

[*Remove ads*](/account/join/)

## 将`str`方法与 NumPy 结合起来清洗色谱柱

以上，你可能注意到了`df['Date of Publication'].str`的用法。这个属性是在 Pandas 中访问快速的[字符串操作](https://pandas.pydata.org/pandas-docs/stable/text.html)的一种方式，这些操作很大程度上模仿了原生 Python 字符串或编译后的正则表达式的操作，如`.split()`、[、](https://realpython.com/replace-string-python/)和`.capitalize()`。

为了清理`Place of Publication`字段，我们可以将 Pandas `str`方法与 NumPy 的`np.where`函数结合起来，该函数基本上是 Excel 的`IF()`宏的矢量化形式。它具有以下语法:

>>>

```py
>>> np.where(condition, then, else)
```

这里，`condition`或者是一个类数组对象，或者是一个[布尔](https://realpython.com/python-boolean/)掩码。`then`是在`condition`评估为`True`时使用的值，而`else`是在其他情况下使用的值。

本质上，`.where()`获取用于`condition`的对象中的每个元素，检查该特定元素在条件的上下文中是否评估为`True`，并返回包含`then`或`else`的`ndarray`，这取决于哪一个适用。

它可以嵌套在一个复合 if-then 语句中，允许我们基于多个条件计算值:

>>>

```py
>>> np.where(condition1, x1, 
 np.where(condition2, x2, 
 np.where(condition3, x3, ...)))
```

我们将利用这两个函数来清理`Place of Publication`,因为这个列有 string 对象。以下是该专栏的内容:

>>>

```py
>>> df['Place of Publication'].head(10)
Identifier
206                                  London
216                London; Virtue & Yorston
218                                  London
472                                  London
480                                  London
481                                  London
519                                  London
667     pp. 40\. G. Bryan & Co: Oxford, 1898
874                                 London]
1143                                 London
Name: Place of Publication, dtype: object
```

我们看到，对于某些行，发布位置被其他不必要的信息所包围。如果我们要查看更多的值，我们会发现只有一些发布地点为“London”或“Oxford”的行是这种情况。

让我们来看看两个具体条目:

>>>

```py
>>> df.loc[4157862]
Place of Publication                                  Newcastle-upon-Tyne
Date of Publication                                                  1867
Publisher                                                      T. Fordyce
Title                   Local Records; or, Historical Register of rema...
Author                                                        T.  Fordyce
Flickr URL              http://www.flickr.com/photos/britishlibrary/ta...
Name: 4157862, dtype: object

>>> df.loc[4159587]
Place of Publication                                  Newcastle upon Tyne
Date of Publication                                                  1834
Publisher                                                Mackenzie & Dent
Title                   An historical, topographical and descriptive v...
Author                                               E. (Eneas) Mackenzie
Flickr URL              http://www.flickr.com/photos/britishlibrary/ta...
Name: 4159587, dtype: object
```

这两本书是在同一个地方出版的，但是一本在地名上有连字符，而另一本没有。

为了在一次扫描中清理这个列，我们可以使用`str.contains()`来获得一个布尔掩码。

我们按照以下步骤清洗色谱柱:

>>>

```py
>>> pub = df['Place of Publication']
>>> london = pub.str.contains('London')
>>> london[:5]
Identifier
206    True
216    True
218    True
472    True
480    True
Name: Place of Publication, dtype: bool

>>> oxford = pub.str.contains('Oxford')
```

我们将它们与`np.where`结合起来:

>>>

```py
df['Place of Publication'] = np.where(london, 'London',
 np.where(oxford, 'Oxford',
 pub.str.replace('-', ' ')))

>>> df['Place of Publication'].head()
Identifier
206    London
216    London
218    London
472    London
480    London
Name: Place of Publication, dtype: object
```

这里，`np.where`函数在一个嵌套结构中被调用，其中`condition`是用`str.contains()`获得的布尔值的`Series`。`contains()`方法的工作方式类似于内置的 [`in`关键字](https://realpython.com/python-keywords/#the-in-keyword)，用于查找 iterable 中实体(或字符串中的子字符串)的出现。

要使用的替换是一个表示我们想要的发布地点的字符串。我们还将连字符替换为带有`str.replace()`的空格，并重新分配给我们的`DataFrame`中的列。

尽管这个数据集中有更多的脏数据，我们现在只讨论这两列。

让我们来看看前五个条目，它们看起来比我们开始时清晰得多:

>>>

```py
>>> df.head()
 Place of Publication Date of Publication              Publisher  \
206                      London                1879        S. Tinsley & Co.
216                      London                1868           Virtue & Co.
218                      London                1869  Bradbury, Evans & Co.
472                      London                1851          James Darling
480                      London                1857   Wertheim & Macintosh

 Title    Author  \
206                         Walter Forbes. [A novel.] By A. A        AA
216         All for Greed. [A novel. The dedication signed...   A. A A.
218         Love the Avenger. By the author of “All for Gr...   A. A A.
472         Welsh Sketches, chiefly ecclesiastical, to the...   E. S A.
480         [The World in which I live, and my place in it...   E. S A.

 Flickr URL
206         http://www.flickr.com/photos/britishlibrary/ta...
216         http://www.flickr.com/photos/britishlibrary/ta...
218         http://www.flickr.com/photos/britishlibrary/ta...
472         http://www.flickr.com/photos/britishlibrary/ta...
480         http://www.flickr.com/photos/britishlibrary/ta...
```

**注意**:在这一点上，`Place of Publication`将是转换为 [`Categorical` dtype](https://pandas.pydata.org/pandas-docs/stable/categorical.html) 的一个很好的候选，因为我们可以用整数对相当小的唯一的一组城市进行编码。(*一个分类的内存使用量与分类的数量加上数据的长度成正比；对象数据类型是一个常数乘以数据的长度。*)

[*Remove ads*](/account/join/)

## 使用`applymap`函数清理整个数据集

在某些情况下，您会看到“污垢”并不局限于某一列，而是更加分散。

在某些情况下，将定制函数应用于数据帧的每个单元格或元素会很有帮助。Pandas `.applymap()`方法类似于内置的 [`map()`函数](https://realpython.com/python-map-function/)，只是将一个函数应用于`DataFrame`中的所有元素。

让我们看一个例子。我们将从“university_towns.txt”文件中创建一个`DataFrame`:

```py
$ head Datasets/univerisity_towns.txt
Alabama[edit]
Auburn (Auburn University)[1]
Florence (University of North Alabama)
Jacksonville (Jacksonville State University)[2]
Livingston (University of West Alabama)[2]
Montevallo (University of Montevallo)[2]
Troy (Troy University)[2]
Tuscaloosa (University of Alabama, Stillman College, Shelton State)[3][4]
Tuskegee (Tuskegee University)[5]
Alaska[edit]
```

我们看到，我们有周期性的州名，后面跟着该州的大学城:`StateA TownA1 TownA2 StateB TownB1 TownB2...`。如果我们观察状态名在文件中的书写方式，我们会发现所有的状态名中都有“[edit]”子字符串。

我们可以通过创建一个由`(state, city)`元组组成的*列表并将该列表包装在一个`DataFrame`中来利用这种模式:*

>>>

```py
>>> university_towns = []
>>> with open('Datasets/university_towns.txt') as file:
...     for line in file:
...         if '[edit]' in line:
...             # Remember this `state` until the next is found
...             state = line
...         else:
...             # Otherwise, we have a city; keep `state` as last-seen
...             university_towns.append((state, line))

>>> university_towns[:5]
[('Alabama[edit]\n', 'Auburn (Auburn University)[1]\n'),
 ('Alabama[edit]\n', 'Florence (University of North Alabama)\n'),
 ('Alabama[edit]\n', 'Jacksonville (Jacksonville State University)[2]\n'),
 ('Alabama[edit]\n', 'Livingston (University of West Alabama)[2]\n'),
 ('Alabama[edit]\n', 'Montevallo (University of Montevallo)[2]\n')]
```

我们可以将这个列表包装在一个 DataFrame 中，并将列设置为“State”和“RegionName”。Pandas 将获取列表中的每个元素，并将`State`设置为左边的值，将`RegionName`设置为右边的值。

生成的数据帧如下所示:

>>>

```py
>>> towns_df = pd.DataFrame(university_towns,
...                         columns=['State', 'RegionName'])

>>> towns_df.head()
 State                                         RegionName
0  Alabama[edit]\n                    Auburn (Auburn University)[1]\n
1  Alabama[edit]\n           Florence (University of North Alabama)\n
2  Alabama[edit]\n  Jacksonville (Jacksonville State University)[2]\n
3  Alabama[edit]\n       Livingston (University of West Alabama)[2]\n
4  Alabama[edit]\n         Montevallo (University of Montevallo)[2]\n
```

虽然我们可以在上面的 for 循环中清理这些字符串，但 Pandas 让它变得很容易。我们只需要州名和镇名，其他的都可以去掉。虽然我们可以在这里再次使用 Pandas 的`.str()`方法，但是我们也可以使用`applymap()`将一个 Python callable 映射到 DataFrame 的每个元素。

我们一直在使用术语*元素*，但是它到底是什么意思呢？考虑以下“玩具”数据帧:

>>>

```py
 0           1
0    Mock     Dataset
1  Python     Pandas
2    Real     Python
3   NumPy     Clean
```

在这个例子中，每个单元格(' Mock '，' Dataset '，' Python '，' Pandas '等)。)是一个元素。因此，`applymap()`将独立地对其中的每一个应用一个函数。让我们来定义这个函数:

>>>

```py
>>> def get_citystate(item):
...     if ' (' in item:
...         return item[:item.find(' (')]
...     elif '[' in item:
...         return item[:item.find('[')]
...     else:
...         return item
```

Pandas 的`.applymap()`只有一个参数，它是应该应用于每个元素的函数(可调用的):

>>>

```py
>>> towns_df =  towns_df.applymap(get_citystate)
```

首先，我们定义一个 Python 函数，它将来自`DataFrame`的一个元素作为它的参数。在函数内部，执行检查以确定元素中是否有`(`或`[`。

根据检查结果，函数会相应地返回值。最后，在我们的对象上调用`applymap()`函数。现在数据框架更加整洁了:

>>>

```py
>>> towns_df.head()
 State    RegionName
0  Alabama        Auburn
1  Alabama      Florence
2  Alabama  Jacksonville
3  Alabama    Livingston
4  Alabama    Montevallo
```

`applymap()`方法从 DataFrame 中取出每个元素，将其传递给函数，原始值被返回值替换。就这么简单！

**技术细节**:虽然它是一个方便且通用的方法，但是`.applymap`对于较大的数据集来说有很长的运行时间，因为它将一个可调用的 Python 映射到每个单独的元素。在某些情况下，利用 Cython 或 NumPY(反过来，用 C 进行调用)进行*矢量化*操作会更有效。

[*Remove ads*](/account/join/)

## 重命名列和跳过行

通常，您将使用的数据集要么具有不容易理解的列名，要么在前几行和/或最后几行中具有不重要的信息，如数据集中术语的定义或脚注。

在这种情况下，我们希望重命名列并跳过某些行，这样我们就可以使用正确和合理的标签深入到必要的信息。

为了演示我们如何去做，让我们先看一下“olympics.csv”数据集的前五行:

```py
$ head -n 5 Datasets/olympics.csv
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
,? Summer,01 !,02 !,03 !,Total,? Winter,01 !,02 !,03 !,Total,? Games,01 !,02 !,03 !,Combined total
Afghanistan (AFG),13,0,0,2,2,0,0,0,0,0,13,0,0,2,2
Algeria (ALG),12,5,2,8,15,3,0,0,0,0,15,5,2,8,15
Argentina (ARG),23,18,24,28,70,18,0,0,0,0,41,18,24,28,70
```

现在，我们将[把它](https://realpython.com/pandas-read-write-files/#read-a-csv-file)读入熊猫数据帧:

>>>

```py
>>> olympics_df = pd.read_csv('Datasets/olympics.csv')
>>> olympics_df.head()
 0         1     2     3     4      5         6     7     8  \
0                NaN  ? Summer  01 !  02 !  03 !  Total  ? Winter  01 !  02 !
1  Afghanistan (AFG)        13     0     0     2      2         0     0     0
2      Algeria (ALG)        12     5     2     8     15         3     0     0
3    Argentina (ARG)        23    18    24    28     70        18     0     0
4      Armenia (ARM)         5     1     2     9     12         6     0     0

 9     10       11    12    13    14              15
0  03 !  Total  ? Games  01 !  02 !  03 !  Combined total
1     0      0       13     0     0     2               2
2     0      0       15     5     2     8              15
3     0      0       41    18    24    28              70
4     0      0       11     1     2     9              12
```

这真的很乱！这些列是索引为 0 的整数的字符串形式。本应是我们标题的行(即用于设置列名的行)位于`olympics_df.iloc[0]`。这是因为我们的 CSV 文件以 0，1，2，…，15 开头。

此外，如果我们转到数据集的[源，我们会看到上面的`NaN`应该是类似于“国家”的东西，`? Summer`应该代表“夏季运动会”，`01 !`应该是“黄金”，等等。](https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table)

因此，我们需要做两件事:

*   跳过一行，将标题设置为第一行(索引为 0)
*   重命名列

我们可以通过向`read_csv()`函数传递一些参数，在读取 CSV 文件时跳过行并设置标题。

这个函数需要 [*很多*](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) 的可选参数，但是在这种情况下我们只需要一个(`header`)来删除第 0 行:

>>>

```py
>>> olympics_df = pd.read_csv('Datasets/olympics.csv', header=1)
>>> olympics_df.head()
 Unnamed: 0  ? Summer  01 !  02 !  03 !  Total  ? Winter  \
0        Afghanistan (AFG)        13     0     0     2      2         0
1            Algeria (ALG)        12     5     2     8     15         3
2          Argentina (ARG)        23    18    24    28     70        18
3            Armenia (ARM)         5     1     2     9     12         6
4  Australasia (ANZ) [ANZ]         2     3     4     5     12         0

 01 !.1  02 !.1  03 !.1  Total.1  ? Games  01 !.2  02 !.2  03 !.2  \
0       0       0       0        0       13       0       0       2
1       0       0       0        0       15       5       2       8
2       0       0       0        0       41      18      24      28
3       0       0       0        0       11       1       2       9
4       0       0       0        0        2       3       4       5

 Combined total
0               2
1              15
2              70
3              12
4              12
```

现在，我们已经将正确的行设置为标题，并删除了所有不必要的行。请注意熊猫如何将包含国家名称的列的名称从`NaN`更改为`Unnamed: 0`。

为了重命名列，我们将利用 DataFrame 的`rename()`方法，该方法允许您基于*映射*(在本例中为`dict`)重新标记轴。

让我们首先定义一个字典，将当前的列名(作为键)映射到更有用的列名(字典的值):

>>>

```py
>>> new_names =  {'Unnamed: 0': 'Country',
...               '? Summer': 'Summer Olympics',
...               '01 !': 'Gold',
...               '02 !': 'Silver',
...               '03 !': 'Bronze',
...               '? Winter': 'Winter Olympics',
...               '01 !.1': 'Gold.1',
...               '02 !.1': 'Silver.1',
...               '03 !.1': 'Bronze.1',
...               '? Games': '# Games',
...               '01 !.2': 'Gold.2',
...               '02 !.2': 'Silver.2',
...               '03 !.2': 'Bronze.2'}
```

我们在对象上调用`rename()`函数:

>>>

```py
>>> olympics_df.rename(columns=new_names, inplace=True)
```

将*就地*设置为`True`指定我们的更改直接作用于对象。让我们看看这是否属实:

>>>

```py
>>> olympics_df.head()
 Country  Summer Olympics  Gold  Silver  Bronze  Total  \
0        Afghanistan (AFG)               13     0       0       2      2
1            Algeria (ALG)               12     5       2       8     15
2          Argentina (ARG)               23    18      24      28     70
3            Armenia (ARM)                5     1       2       9     12
4  Australasia (ANZ) [ANZ]                2     3       4       5     12

 Winter Olympics  Gold.1  Silver.1  Bronze.1  Total.1  # Games  Gold.2  \
0                0       0         0         0        0       13       0
1                3       0         0         0        0       15       5
2               18       0         0         0        0       41      18
3                6       0         0         0        0       11       1
4                0       0         0         0        0        2       3

 Silver.2  Bronze.2  Combined total
0         0         2               2
1         2         8              15
2        24        28              70
3         2         9              12
4         4         5              12
```

[*Remove ads*](/account/join/)

## Python 数据清理:概述和资源

在本教程中，您学习了如何使用`drop()`函数从数据集中删除不必要的信息，以及如何为数据集设置索引，以便可以轻松引用其中的项目。

此外，您还学习了如何使用`.str()`访问器清理`object`字段，以及如何使用`applymap()`方法清理整个数据集。最后，我们探索了如何跳过 CSV 文件中的行并使用`rename()`方法重命名列。

了解数据清理非常重要，因为它是数据科学的一大部分。现在，您已经对如何利用 Pandas 和 NumPy 清理数据集有了基本的了解！

请查看下面的链接，找到对您的 Python 数据科学之旅有所帮助的其他资源:

*   熊猫[文档](https://pandas.pydata.org/pandas-docs/stable/index.html)
*   NumPy [文档](https://docs.scipy.org/doc/numpy/reference/)
*   熊猫的创造者韦斯·麦金尼的数据分析 Python
*   数据科学培训师兼顾问泰德·彼得鲁的熊猫食谱

**免费奖励:** [点击此处获取免费的 NumPy 资源指南](#)，它会为您指出提高 NumPy 技能的最佳教程、视频和书籍。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。配合文字教程一起看，加深理解: [**用熊猫和 NumPy**](/courses/data-cleaning-with-pandas-and-numpy/) 进行数据清洗*******
# Python CSV 模块–读取和写入 CSV 文件

> 原文：<https://www.askpython.com/python-modules/python-csv-module>

在本教程中，我们将了解 Python CSV 模块，它对于 CSV 文件处理非常有用。

使用 Python 附带的这个模块，我们可以轻松地读写 CSV 文件。

我们开始吧！

* * *

## 使用 Python csv 模块

我们必须导入 csv 模块才能使用相关方法。

```py
import csv

```

现在，根据您想要做什么，我们可以使用适当的对象读取或写入 csv 文件。

我们先来看看读取 csv 文件。

## 使用 csv.reader()读取 csv 文件

要读取 csv 文件，我们必须构造一个 reader 对象，然后它将解析文件并填充我们的 Python 对象。

Python 的`csv`模块有一个名为`csv.reader()`的方法，它会自动构造 csv reader 对象！

我们必须在已经打开的文件对象上调用`csv.reader()`方法，使用`open()`。

```py
import csv
reader = csv.reader(file_object)

```

通常，推荐的方法是使用`with`上下文管理器来封装所有内容。

您可以做类似的事情:

```py
import csv

# Open the csv file object
with open('sample.csv', 'r') as f:
    # Construct the csv reader object from the file object
    reader = csv.reader(f)

```

reader 对象将是一个包含 csv 文件中所有行的 iterable 对象。默认情况下，每个`row`都会是一个 Python 列表，所以对我们来说会非常方便！

因此，您可以使用循环的[直接打印行，如下所示:](https://www.askpython.com/python/python-for-loop)

```py
for row in reader:
    print(row)

```

好吧。现在我们有了一个基本的模板代码，让我们使用`csv.reader()`打印下面文件的内容。

让我们考虑`sample.csv`有以下内容。

```py
Club,Country,Rating
Man Utd,England,7.05
Man City,England,8.75
Barcelona,Spain,8.72
Bayern Munich,Germany,8.75
Liverpool,England,8.81

```

现在，让我们运行代码:

```py
import csv
with open('sample.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

```

**输出**

```py
['Club', 'Country', 'Rating']
['Man Utd', 'England', '7.05']
['Man City', 'England', '8.75']
['Barcelona', 'Spain', '8.72']
['Bayern Munich', 'Germany', '8.75']
['Liverpool', 'England', '8.81']

```

好的，我们得到了所有的行。在这里，你可以看到，`csv`在逗号后面给了我们空格。

如果您想要解析单个单词，通过使用空白字符进行分隔，您可以简单地将它作为分隔符传递给`csv.reader(delimiter=' ')`。

现在让我们尝试修改后的代码:

```py
import csv

with open('sample.csv', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        print(row)

```

输出

```py
['Club,', 'Country,', 'Rating']
['Man', 'Utd,', 'England,', '7.05']
['Man', 'City,', 'England,', '8.75']
['Barcelona,', 'Spain,', '8.72']
['Bayern', 'Munich,', 'Germany,', '8.75']
['Liverpool,', 'England,', '8.81']

```

事实上，我们现在已经把单词分开了，所以`Man Utd`变成了`Man`和`Utd`。

类似地，如果您想要解析带分隔符的内容，只需将该字符作为分隔符传递给`csv.reader()`。

现在让我们看看如何写入 csv 文件。

* * *

## 使用 csv.writer()写入 csv 文件

类似于用于读取的`csv.reader()`方法，我们有用于写入文件的`csv.writer()`方法。

这将返回一个`writer`对象，我们可以用它将行写入我们的目标文件。

让我们看看如何使用它。首先，创建`writer`对象:

```py
import csv

with open('output.csv', 'w') as f:
    writer = csv.writer(f)

```

我们现在可以使用`writer.writerow(row)`方法来写一行。在这里，类似于 reader 对象，`row`是一个列表。

所以，我们可以这样调用它:

```py
writer.writerow(['Club', 'Country', 'Rating'])

```

现在让我们看看运行整个程序:

```py
import csv

with open('output.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Club', 'Country', 'Rating'])
    clubs = [['Real Madrid', 'Spain', 9.1], ['Napoli', 'Italy', 7.5]]
    for club in clubs:
        writer.writerow(club)

```

我们现在来看看`output.csv`。

```py
Club,Country,Rating
Real Madrid,Spain,9.1
Napoli,Italy,7.5

```

事实上，我们在输出文件中有自己的行！

**注**:类似于`csv.reader(delimiter)`，我们也可以通过一个分隔符来写使用`csv.writer(delimiter)`

如果您仔细观察，我们已经手动遍历了我们的行列表(列表列表),并逐个写入了每一行。

原来还有一个叫`writer.writerows(rows)`的方法可以直接写我们所有的行！

让我们来测试一下。删除`output.csv`并运行以下代码。

```py
import csv

with open('output.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Club', 'Country', 'Rating'])
    clubs = [['Real Madrid', 'Spain', 9.1], ['Napoli', 'Italy', 7.5]]
    writer.writerows(clubs)

```

**输出**

```py
Club,Country,Rating
Real Madrid,Spain,9.1
Napoli,Italy,7.5

```

我们确实得到了和以前一样的输出！

* * *

## 使用 csv。DictReader()和 csv。DictWriter()将 csv 作为字典来读写

还记得当使用`reader`对象读取时，我们以列表的形式按行获取对象吗？

如果您想要精确的`column_name: row_name`映射，我们可以使用`csv.DictReader`类并获得一个字典来代替！

让我们看看如何将 csv 文件读入字典。

```py
import csv

with open("sample.csv", 'r') as file:
    csv_file = csv.DictReader(file)

    for row in csv_file:
        print(dict(row))

```

这里，`csv.DictReader()`返回一个`OrderedDict()`对象的 iterable。我们需要使用`dict(row)`将每个`OrderedDict`行转换成一个`dict`。

让我们看看输出:

```py
{'Club': 'Man Utd', ' Country': ' England', ' Rating': ' 7.05'}
{'Club': 'Man City', ' Country': ' England', ' Rating': ' 8.75'}
{'Club': 'Barcelona', ' Country': ' Spain', ' Rating': ' 8.72'}
{'Club': 'Bayern Munich', ' Country': ' Germany', ' Rating': ' 8.75'}
{'Club': 'Liverpool', ' Country': ' England', ' Rating': ' 8.81'}

```

事实上，我们有列名和行值！

现在，为了从字典写入 csv 文件，您有了`csv.DictWriter()`类。

这几乎和`csv.write()`一样，除了你是从字典而不是从列表开始写。

尽管语法有点不同。我们必须预先指定列名，作为我们的`fieldnames`的一部分。

然后我们需要使用`writer.writeheader()`写入第一行(header)。

```py
    fieldnames = ['Club', 'Country', 'Rating']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

```

现在，我们可以遍历我们的列表`dicts`，它包含相关的信息。

让我们用`csv.DictWriter()`重写我们以前的`writer`例子。

```py
import csv

with open('output.csv', 'w') as f:
    fieldnames = ['Club', 'Country', 'Rating']
    # Set the fieldnames
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    clubs = [{'Club': 'Real Madrid', 'Country': 'Spain', 'Rating': 9.1}, {'Club': 'Napoli', 'Country': 'Italy', 'Rating': 7.5}]

    for club in clubs:
        writer.writerow(club)

```

我们现在将获得与之前相同的输出，表明我们已经使用我们的`csv.DictWriter()`对象成功地写入了 csv 文件！

* * *

## 结论

希望您已经理解了如何使用`csv`模块轻松处理 csv 文件。我们使用合适的对象使 csv 文件的读写变得容易。

## 参考

*   关于用 Python 读写 csv 文件的 JournalDev 文章

* * *
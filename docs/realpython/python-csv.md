# 用 Python 读写 CSV 文件

> 原文：<https://realpython.com/python-csv/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**读写 CSV 文件**](/courses/reading-and-writing-csv-files/)

让我们面对现实吧:你需要通过不仅仅是键盘和控制台来将信息输入和输出你的程序。通过文本文件交换信息是程序间共享信息的常见方式。交换数据最流行的格式之一是 CSV 格式。但是怎么用呢？

让我们弄清楚一件事:您不必(也不会)从头构建自己的 CSV 解析器。有几个完全可以接受的库可供您使用。Python [`csv`库](https://docs.python.org/3/library/csv.html)将适用于大多数情况。如果您的工作需要大量数据或数值分析，那么 [`pandas`库](http://pandas.pydata.org/)也有 CSV 解析功能，它应该可以处理剩下的事情。

在本文中，您将学习如何使用 Python 从文本文件中读取、处理和解析 CSV。您将看到 CSV 文件是如何工作的，学习 Python 内置的非常重要的`csv`库，并使用 [`pandas`库](https://realpython.com/pandas-python-explore-dataset/)查看 CSV 解析是如何工作的。

所以让我们开始吧！

**免费下载:** [从《Python 基础:Python 3 实用入门》中获取一个示例章节](https://realpython.com/bonus/python-basics-sample-download/)，看看如何通过 Python 3.8 的最新完整课程从初级到中级学习 Python。

***参加测验:****通过我们的交互式“用 Python 读写 CSV 文件”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-csv/)

## 什么是 CSV 文件？

CSV 文件(逗号分隔值文件)是一种纯文本文件，它使用特定的结构来排列表格数据。因为它是一个纯文本文件，所以它只能包含实际的文本数据——换句话说，可打印的 [ASCII](https://en.wikipedia.org/wiki/ASCII) 或 [Unicode](https://en.wikipedia.org/wiki/Unicode) 字符。

CSV 文件的结构由它的名称给出。通常，CSV 文件使用逗号分隔每个特定的数据值。这个结构看起来是这样的:

```py
column 1 name,column 2 name, column 3 name
first row data 1,first row data 2,first row data 3
second row data 1,second row data 2,second row data 3
...
```

注意每段数据是如何被逗号分隔的。通常，第一行标识每条数据，换句话说，就是数据列的名称。其后的每一行都是实际数据，仅受文件大小的限制。

一般来说，分隔符称为分隔符，逗号不是唯一使用的分隔符。其他流行的分隔符包括制表符(`\t`)、冒号(`:`)和分号(`;`)字符。正确解析 CSV 文件需要我们知道使用了哪个分隔符。

[*Remove ads*](/account/join/)

### CSV 文件从哪里来？

CSV 文件通常由处理大量数据的程序创建。它们是从电子表格和数据库导出数据以及在其他程序中导入或使用数据的便捷方式。例如，您可以将数据挖掘程序的结果导出到 CSV 文件，然后将其导入到电子表格中以分析数据、为演示文稿生成图表或准备发布报告。

CSV 文件很容易以编程方式处理。任何支持文本文件输入和字符串操作的语言(比如 Python)都可以直接处理 CSV 文件。

## 用 Python 内置的 CSV 库解析 CSV 文件

[`csv`库](https://docs.python.org/3/library/csv.html)提供了读取和写入 CSV 文件的功能。它设计用于 Excel 生成的 CSV 文件，可以很容易地适应各种 CSV 格式。`csv`库包含从 CSV 文件读取、写入和处理数据的对象和其他代码。

### 用`csv` 读取 CSV 文件

使用`reader`对象读取 CSV 文件。CSV 文件通过 Python 内置的`open()`函数作为文本文件打开，该函数返回一个 file 对象。这然后被传递给`reader`，它做繁重的工作。

下面是`employee_birthday.txt`文件:

```py
name,department,birthday month
John Smith,Accounting,November
Erica Meyers,IT,March
```

下面是阅读它的代码:

```py
import csv

with open('employee_birthday.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
    print(f'Processed {line_count} lines.')
```

这会产生以下输出:

```py
Column names are name, department, birthday month
 John Smith works in the Accounting department, and was born in November.
 Erica Meyers works in the IT department, and was born in March.
Processed 3 lines.
```

由`reader`返回的每一行都是一个由`String`元素组成的列表，其中包含了通过删除分隔符找到的数据。返回的第一行包含列名，这是以特殊方式处理的。

### 用`csv` 将 CSV 文件读入字典

不需要处理单个`String`元素的列表，您也可以将 CSV 数据直接读入字典(技术上来说，是一个[有序字典](https://realpython.com/python-ordereddict/))。

再次，我们的输入文件，`employee_birthday.txt`如下:

```py
name,department,birthday month
John Smith,Accounting,November
Erica Meyers,IT,March
```

下面是这次作为[字典](https://realpython.com/python-dicts/)读入的代码:

```py
import csv

with open('employee_birthday.txt', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
        line_count += 1
    print(f'Processed {line_count} lines.')
```

这将产生与之前相同的输出:

```py
Column names are name, department, birthday month
 John Smith works in the Accounting department, and was born in November.
 Erica Meyers works in the IT department, and was born in March.
Processed 3 lines.
```

字典键是从哪里来的？假设 CSV 文件的第一行包含用于构建字典的键。如果您的 CSV 文件中没有这些，您应该通过将可选参数`fieldnames`设置为包含它们的列表来指定您自己的键。

[*Remove ads*](/account/join/)

### 可选 Python CSV `reader`参数

`reader`对象可以通过指定[附加参数](https://docs.python.org/3/library/csv.html?highlight=csv#csv-fmt-params)来处理不同样式的 CSV 文件，其中一些如下所示:

*   `delimiter`指定用于分隔每个字段的字符。默认为逗号(`','`)。

*   `quotechar`指定用于包围包含分隔符的字段的字符。默认为双引号(`' " '`)。

*   `escapechar`指定在不使用引号的情况下，用于转义分隔符的字符。默认情况下没有转义字符。

这些参数值得更多的解释。假设您正在使用下面的`employee_addresses.txt`文件:

```py
name,address,date joined
john smith,1132 Anywhere Lane Hoboken NJ, 07030,Jan 4
erica meyers,1234 Smith Lane Hoboken NJ, 07030,March 2
```

这个 CSV 文件包含三个字段:`name`、`address`和`date joined`，用逗号分隔。问题是`address`字段的数据还包含一个逗号来表示邮政编码。

有三种不同的方法来处理这种情况:

*   **使用不同的分隔符**
    这样，逗号可以安全地用在数据本身中。使用可选参数`delimiter`来指定新的分隔符。

*   **用引号将数据括起来**
    在带引号的字符串中，您选择的分隔符的特殊性质会被忽略。因此，您可以用可选参数`quotechar`指定用于引用的字符。只要这个字符没有出现在数据中，就没有问题。

*   **转义数据中的分隔符**
    转义字符的作用就像在格式字符串中一样，使被转义字符的解释无效(在本例中是分隔符)。如果使用转义字符，必须使用可选参数`escapechar`指定。

### 用`csv` 编写 CSV 文件

您还可以使用`writer`对象和`.write_row()`方法写入 CSV 文件:

```py
import csv

with open('employee_file.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(['John Smith', 'Accounting', 'November'])
    employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
```

可选参数`quotechar`告诉`writer`在写入时使用哪个字符来引用字段。然而，是否使用引用由`quoting`可选参数决定:

*   如果`quoting`被设置为`csv.QUOTE_MINIMAL`，那么`.writerow()`将引用包含`delimiter`或`quotechar`的字段。这是默认情况。
*   如果`quoting`被设置为`csv.QUOTE_ALL`，那么`.writerow()`将引用所有字段。
*   如果`quoting`设置为`csv.QUOTE_NONNUMERIC`，那么`.writerow()`将引用所有包含文本数据的字段，并将所有数值字段转换为`float`数据类型。
*   如果`quoting`被设置为`csv.QUOTE_NONE`，那么`.writerow()`将转义分隔符而不是引用它们。在这种情况下，您还必须为可选参数`escapechar`提供一个值。

以纯文本形式读回该文件显示，该文件是按如下方式创建的:

```py
John Smith,Accounting,November
Erica Meyers,IT,March
```

### 用`csv` 从字典写入 CSV 文件

既然您可以将我们的数据读入字典，那么您也应该能够将它从字典中写出来，这才是公平的:

```py
import csv

with open('employee_file2.csv', mode='w') as csv_file:
    fieldnames = ['emp_name', 'dept', 'birth_month']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
    writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})
```

与`DictReader`不同的是，编写字典时需要`fieldnames`参数。仔细想想，这是有道理的:没有一个`fieldnames`列表，`DictWriter`就不知道使用哪个键从字典中检索值。它还使用`fieldnames`中的键写出第一行作为列名。

上面的代码生成以下输出文件:

```py
emp_name,dept,birth_month
John Smith,Accounting,November
Erica Meyers,IT,March
```

[*Remove ads*](/account/join/)

## 使用`pandas`库解析 CSV 文件

当然，Python CSV 库并不是唯一的游戏。[在](https://realpython.com/pandas-read-write-files/#read-a-csv-file) [`pandas`](http://pandas.pydata.org/index.html) 中也可以读取 CSV 文件。如果你有很多数据要分析，强烈推荐。

`pandas`是一个开源的 Python 库，提供高性能的数据分析工具和易于使用的数据结构。`pandas`可用于所有 Python 安装，但它是 [Anaconda](https://www.anaconda.com/) 发行版的关键部分，在 [Jupyter 笔记本](https://jupyter.org/)中非常好地工作，以共享数据、代码、分析结果、可视化和叙述性文本。

在`Anaconda`中安装`pandas`及其依赖项很容易:

```py
$ conda install pandas
```

对于其他 Python 安装，使用 [`pip` / `pipenv`](https://realpython.com/pipenv-guide/) :

```py
$ pip install pandas
```

我们不会深究`pandas`如何工作或如何使用它的细节。关于使用`pandas`读取和分析大型数据集的深入讨论，请查看 [Shantnu Tiwari 的](https://realpython.com/team/stiwari/)关于[在 pandas](https://realpython.com/working-with-large-excel-files-in-pandas/) 中使用大型 Excel 文件的精彩文章。

### 用`pandas` 读取 CSV 文件

为了展示`pandas` CSV 功能的一些威力，我创建了一个稍微复杂一点的文件来阅读，名为`hrdata.csv`。它包含公司员工的数据:

```py
Name,Hire Date,Salary,Sick Days remaining
Graham Chapman,03/15/14,50000.00,10
John Cleese,06/01/15,65000.00,8
Eric Idle,05/12/14,45000.00,10
Terry Jones,11/01/13,70000.00,3
Terry Gilliam,08/12/14,48000.00,7
Michael Palin,05/23/13,66000.00,8
```

将 CSV 读取为`pandas` [`DataFrame`](https://realpython.com/pandas-dataframe/) 既快速又简单:

```py
import pandas
df = pandas.read_csv('hrdata.csv')
print(df)
```

就是这样:三行代码，其中只有一行在做实际的工作。`pandas.read_csv()`打开、分析并读取提供的 CSV 文件，并将数据存储在[数据帧](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)中。打印`DataFrame`会产生以下输出:

```py
 Name Hire Date   Salary  Sick Days remaining
0  Graham Chapman  03/15/14  50000.0                   10
1     John Cleese  06/01/15  65000.0                    8
2       Eric Idle  05/12/14  45000.0                   10
3     Terry Jones  11/01/13  70000.0                    3
4   Terry Gilliam  08/12/14  48000.0                    7
5   Michael Palin  05/23/13  66000.0                    8
```

这里有几点不值一提:

*   首先，`pandas`识别出 CSV 的第一行包含列名，并自动使用它们。我称之为善。
*   然而，`pandas`在`DataFrame`中也使用从零开始的整数索引。那是因为我们没有告诉它我们的索引应该是什么。
*   此外，如果您查看我们的列的数据类型，您会看到`pandas`已经正确地将`Salary`和`Sick Days remaining`列转换为数字，但是`Hire Date`列仍然是一个`String`。这在交互模式下很容易确认:

    >>>

    ```py
    >>> print(type(df['Hire Date'][0]))
    <class 'str'>` 
    ```

让我们一次解决一个问题。要使用不同的列作为`DataFrame`索引，添加`index_col`可选参数:

```py
import pandas
df = pandas.read_csv('hrdata.csv', index_col='Name')
print(df)
```

现在`Name`字段是我们的`DataFrame`索引:

```py
 Hire Date   Salary  Sick Days remaining
Name 
Graham Chapman  03/15/14  50000.0                   10
John Cleese     06/01/15  65000.0                    8
Eric Idle       05/12/14  45000.0                   10
Terry Jones     11/01/13  70000.0                    3
Terry Gilliam   08/12/14  48000.0                    7
Michael Palin   05/23/13  66000.0                    8
```

接下来，让我们修复`Hire Date`字段的数据类型。您可以使用可选参数`parse_dates`强制`pandas`将数据作为日期读取，该参数被定义为列名列表，以作为日期处理:

```py
import pandas
df = pandas.read_csv('hrdata.csv', index_col='Name', parse_dates=['Hire Date'])
print(df)
```

请注意输出中的差异:

```py
 Hire Date   Salary  Sick Days remaining
Name 
Graham Chapman 2014-03-15  50000.0                   10
John Cleese    2015-06-01  65000.0                    8
Eric Idle      2014-05-12  45000.0                   10
Terry Jones    2013-11-01  70000.0                    3
Terry Gilliam  2014-08-12  48000.0                    7
Michael Palin  2013-05-23  66000.0                    8
```

日期现在已正确格式化，这在交互模式下很容易确认:

>>>

```py
>>> print(type(df['Hire Date'][0]))
<class 'pandas._libs.tslibs.timestamps.Timestamp'>
```

如果您的 CSV 文件在第一行没有列名，您可以使用`names`可选参数来提供列名列表。如果您想覆盖第一行中提供的列名，也可以使用这个方法。在这种情况下，您还必须使用可选参数`header=0`告诉`pandas.read_csv()`忽略现有的列名:

```py
import pandas
df = pandas.read_csv('hrdata.csv', 
            index_col='Employee', 
            parse_dates=['Hired'], 
            header=0, 
            names=['Employee', 'Hired','Salary', 'Sick Days'])
print(df)
```

注意，由于列名发生了变化，在可选参数`index_col`和`parse_dates`中指定的列也必须发生变化。这会产生以下输出:

```py
 Hired   Salary  Sick Days
Employee 
Graham Chapman 2014-03-15  50000.0         10
John Cleese    2015-06-01  65000.0          8
Eric Idle      2014-05-12  45000.0         10
Terry Jones    2013-11-01  70000.0          3
Terry Gilliam  2014-08-12  48000.0          7
Michael Palin  2013-05-23  66000.0          8
```

[*Remove ads*](/account/join/)

### 用`pandas` 编写 CSV 文件

当然，如果你不能再次把你的数据从`pandas`中取出来，对你没有太大的好处。将`DataFrame`写入 CSV 文件就像读入一样简单。让我们将带有新列名的数据写入一个新的 CSV 文件:

```py
import pandas
df = pandas.read_csv('hrdata.csv', 
            index_col='Employee', 
            parse_dates=['Hired'],
            header=0, 
            names=['Employee', 'Hired', 'Salary', 'Sick Days'])
df.to_csv('hrdata_modified.csv')
```

这段代码与上面的读取代码的唯一区别是，`print(df)`调用被替换为`df.to_csv()`，提供了文件名。新的 CSV 文件如下所示:

```py
Employee,Hired,Salary,Sick Days
Graham Chapman,2014-03-15,50000.0,10
John Cleese,2015-06-01,65000.0,8
Eric Idle,2014-05-12,45000.0,10
Terry Jones,2013-11-01,70000.0,3
Terry Gilliam,2014-08-12,48000.0,7
Michael Palin,2013-05-23,66000.0,8
```

## 结论

如果你理解了阅读 CSV 文件的基础，那么当你需要处理导入数据时，你就不会手足无措。基本的`csv` Python 库可以轻松处理大多数 CSV 读取、处理和编写任务。如果您有大量数据要读取和处理，那么`pandas`库也提供了快速简单的 CSV 处理功能。

***参加测验:****通过我们的交互式“用 Python 读写 CSV 文件”测验来测试您的知识。完成后，您将收到一个分数，以便您可以跟踪一段时间内的学习进度:*

*[参加测验](/quizzes/python-csv/)

还有其他解析文本文件的方法吗？当然啦！像 [ANTLR](http://www.antlr.org/) 、 [PLY](http://www.dabeaz.com/ply/) 、 [PlyPlus](https://pypi.org/project/PlyPlus/) 这样的库都可以处理重型解析，如果简单的`String`操纵不行，总有[正则表达式](https://realpython.com/regex-python/)。

但是那些是其他文章的主题…

**免费下载:** [从《Python 基础:Python 3 实用入门》中获取一个示例章节](https://realpython.com/bonus/python-basics-sample-download/)，看看如何通过 Python 3.8 的最新完整课程从初级到中级学习 Python。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**读写 CSV 文件**](/courses/reading-and-writing-csv-files/)********
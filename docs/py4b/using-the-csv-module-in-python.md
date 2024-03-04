# 在 Python 中使用 CSV 模块

> 原文：<https://www.pythonforbeginners.com/csv/using-the-csv-module-in-python>

如果要导入或导出电子表格和数据库以便在 Python 解释器中使用，则必须依赖 CSV 模块或逗号分隔值格式。

### 什么是 CSV 文件？

CSV 文件用于存储大量变量或数据。它们是极其简化的电子表格——想想 Excel——只是内容以明文形式存储。

CSV 模块是一个内置函数，允许 Python 解析这些类型的文件。

值得注意的是，当您处理 CSV 文件时，您正在涉足 JSON 开发。

JSON——代表 JavaScript 对象符号——是一种格式，用于在明文文件中以 JavaScript 代码的形式存储信息。您不需要了解 JavaScript 来处理这些文件，实践也不局限于该语言。显然，因为我们在这里使用的是 Python。

CSV 文件中的文本按行排列，每一行都有列，用逗号分隔。文件中的每一行都是电子表格中的一行，而逗号用于定义和分隔单元格。

### 使用 CSV 模块

要从 CSV 文件中提取信息，可以使用循环和分割方法从各个列中获取数据。

CSV 模块的存在就是为了处理这个任务，使得处理 CSV 格式的文件变得更加容易。当您处理从实际的电子表格和数据库导出到文本文件的数据时，这变得尤其重要。这些信息本身很难读懂。

不幸的是，没有标准，所以 CSV 模块使用“方言”来支持使用不同参数的解析。除了通用读取器和写入器，该模块还包括一种用于处理 Microsoft Excel 和相关文件的方言。

### CSV 功能

CSV 模块包括所有内置的必要功能。它们是:

*   csv .阅读器
*   csv.writer
*   csv.register_dialect
*   csv.unregister_dialect
*   csv.get_dialect
*   csv.list _ 方言
*   csv.field_size_limit

在本指南中，我们将只关注 reader 和 writer 函数，它们允许您编辑、修改和操作 CSV 文件中存储的数据。

### 读取 CSV 文件

要从 CSV 文件中提取数据，必须使用 reader 函数来生成 reader 对象。

reader 函数被设计成获取文件的每一行并列出所有列。然后，您只需选择需要可变数据的列。

听起来比实际复杂多了。为了证明这一点，我们来看一个例子。

```py
*i**mport CSV*
*With* *open(**‘some.csv’, ‘**rb**’) as f:*
*r**eader =* *csv.reader**(f)*
*f**or row in reader:*
*p**rint row*
```

注意第一个命令是如何用于导入 CSV 模块的？

让我们看另一个例子。

```py
*i**mport csv* 
*i**mport sys*

*f* *= open(**sys.argv**[1], ‘**rb**’)*
*r**eader =* *csv.reader**(f)*
*f**or row in reader*
*p**rint row*

*f.close**()*
```

在前两行中，我们导入 CSV 和 sys 模块。然后，我们打开想要从中提取信息的 CSV 文件。

接下来，我们创建 reader 对象，迭代文件的行，然后打印它们。最后，我们结束操作。

### CSV 样本文件

我们将看看一个示例 CSV 文件。注意信息是如何存储和呈现的。

```py
*Title,Release* *Date,Director*
*And Now For Something Completely Different,**1971,Ian* *MacNaughton*
*Monty Python And The Holy Grail,**1975,Terry* *Gilliam and Terry Jones*
*Monty Python's Life Of Brian,**1979,Terry* *Jones*
*Monty Python Live At The Hollywood Bowl,**1982,Terry* *Hughes*
*Monty Python's The Meaning Of Life,**1983,Terry* *Jones*
```

### 读取 CSV 文件示例

我们将从一个基本的 CSV 文件开始，它有 3 列，包含变量“A”、“B”、“C”和“D”。

```py
*$ cat test.csv*
*A,B**,”C D”*
*1,2,”3 4”*
*5,6,7*
```

然后，我们将使用下面的 Python 程序来读取和显示上述 CSV 文件的内容。

```py
*import csv*

*ifile* *=* *open(**‘test.csv’, “**rb**”)*
*reader =* *csv.reader**(**ifile**)*

*rownum* *= 0*
*for row in reader:*
*# Save header row.*
*i**f* *rownum* *==0:*
*h**eader = row*
*e**lse:*
*c**olnum* *= 0*
*f**or col in row:*
*p**rint ‘%-8s: %s’ % (header[**colnum**], col)*
*c**olnum* *+ = 1*

*r**ownum* *+ = 1*

*i**file.close**()*
```

当我们用 Python 执行这个程序时，输出将如下所示:

```py
*$ python csv1.py*
*A    **  :* *1* 
*B    **  :* *2*
*C D    **  :* *3 4*
*A    **  :** 5* 
*B    **  :* *6*
*C D    **  :* *7*
```

### 写入 CSV 文件

当您有一组想要存储在 CSV 文件中的数据时，是时候反过来使用 write 函数了。信不信由你，这和阅读它们一样容易做到。

***writer()*** 函数将创建一个适合写的对象。要迭代行上的数据，需要使用***writerow**(**)***函数。

这里有一个例子。

以下 Python 程序将名为“test.csv”的文件转换为 csv 文件，该文件使用制表符作为值分隔符，所有值都用引号括起来。分隔符和引号，以及如何/何时引用，是在创建编写器时指定的。创建 reader 对象时，这些选项也是可用的。

```py
*import csv*

*ifile**  =* *open('test.csv', "**rb**")*
*reader =* *csv.reader**(**ifile**)*
*ofile**  =* *open('ttest.csv', "**wb**")*
*writer =* *csv.writer**(**ofile**, delimiter='**',* *quotechar**='"', quoting=**csv.QUOTE_ALL**)*

*for row in reader:*
*writer.writerow**(row)*

*ifile.close**()*
*ofile.close**()*
```

当您执行这个程序时，输出将是:

```py
*$ python csv2.py*
*$ cat ttest.csv*
*"A"     "B"     "C D"*
*"1"     "2"     "3 4"*
*"5"     "6"     "7"*
```

### 引用 CSV 文件

使用 CSV 模块，您还可以执行各种报价功能。

它们是:

*   ***csv。引用**_ 所有***–引用所有内容，不考虑类型。
*   ***csv。引用**_ 最小***–引用包含特殊字符的字段
*   ***csv。引用**_ 非数字***–引用所有非整数或浮点数的字段
*   ***csv。引用**_ 无***–不引用输出上的任何内容

## 更多 Python 阅读和资源

[http://docs.python.org/2/library/csv.html](https://docs.python.org/2/library/csv.html "csv-module")[http://www.doughellmann.com/PyMOTW/csv/](http://www.doughellmann.com/PyMOTW/csv/ "pymotwcsv")[http://effbot.org/librarybook/csv.htm](http://effbot.org/librarybook/csv.htm "effbot.org")[http://www . Linux journal . com/content/handling-CSV-files-python](https://www.linuxjournal.com/content/handling-csv-files-python "linuxjournal.com")[http://programming-crash-course . code point . net/there _ are _ columns](http://programming-crash-course.codepoint.net "programming_crash")
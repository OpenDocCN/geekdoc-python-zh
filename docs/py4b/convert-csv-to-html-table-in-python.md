# 在 Python 中将 CSV 转换为 HTML 表格

> 原文：<https://www.pythonforbeginners.com/basics/convert-csv-to-html-table-in-python>

CSV 文件包含逗号分隔的值，这些值通常包含表格。有时，我们可能需要将 csv 文件呈现为 HTML 页面。在本文中，我们将讨论如何用 python 将 csv 文件转换成 HTML 表格。

## 使用 pandas 模块将 CSV 转换为 HTML 表格

熊猫模块为我们提供了不同的工具来处理 csv 文件。要将 csv 文件转换成 HTML 表格，我们将首先使用`read_csv()`方法打开文件。`read_csv()`方法将 csv 文件的文件名作为输入参数，并返回包含来自 csv 文件的数据的 dataframe。

从 csv 文件获取数据到数据帧后，我们可以使用`to_html()`方法将数据帧转换成 HTML 字符串。在 dataframe 上调用`to_html()`方法时，会将 dataframe 转换为 HTML 表，并以字符串的形式返回 HTML 文本。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
html_string = df1.to_html()
print("The html string is:")
print(html_string)
```

输出:

```py
The dataframe is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
The html string is:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Roll Number</th>
      <th>Subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aditya</td>
      <td>12</td>
      <td>Python</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sam</td>
      <td>23</td>
      <td>Java</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chris</td>
      <td>11</td>
      <td>C++</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joel</td>
      <td>10</td>
      <td>JavaScript</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mayank</td>
      <td>5</td>
      <td>Typescript</td>
    </tr>
  </tbody>
</table>
```

您也可以将数据直接保存到 HTML 文件中。为此，您必须将文件名作为输入参数传递给`to_html()`方法。在 dataframe 上调用`to_html()`方法时，该方法将 HTML 文件的文件名作为输入参数，并将其保存在当前工作目录中。在执行之后，`to_html()`方法在这种情况下返回 None。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
df1.to_html("html_output.html")
print("CSV file saved into html file.") 
```

输出:

```py
The dataframe is:
     Name  Roll Number      Subject
0  Aditya           12       Python
1     Sam           23         Java
2   Chris           11          C++
3    Joel           10   JavaScript
4  Mayank            5   Typescript
CSV file saved into html file.
```

下面是使用上述程序创建的 html 表格的快照。

![](img/d82b572d8de9c4b4d9045cefac107f86.png)



## 使用 PrettyTable 模块将 CSV 转换为 HTML 表格

我们还可以使用`PrettyTable()`方法将 csv 文件转换成 HTML 文件。为此，我们将首先在读取模式下使用`open()`方法打开 csv 文件。`open()`方法将文件名作为第一个输入参数，将文字“r”作为第二个输入参数。执行后，它返回一个包含文件内容的 file 对象。

打开文件后，我们将使用`readlines()` 方法读取文件内容。在 file 对象上调用`readlines()`方法时，它将文件的内容作为字符串列表返回，其中列表的每个元素都包含输入文件中的一行。

现在，csv 文件的头将出现在由`readlines()` 方法返回的列表中的索引 0 处。我们将在由`readlines()`方法返回的列表的第一个元素上使用字符串分割操作提取 csv 文件的列名。

获得列表中的列名后，我们将使用`PrettyTable`()方法创建一个漂亮的表。`PrettyTable()`方法将包含列名的列表作为输入参数，并返回一个漂亮的表。创建表格后，我们将使用`add_row()`方法将数据添加到表格中。`add_row()`方法获取包含一行中的值的列表，并将其添加到 pretty 表中。

创建表格后，我们将使用`get_html_string()`方法获得表格的 HTML 字符串。当在 pretty table 对象上调用`get_html_string()`方法时，它以字符串的形式返回表格的 HTML 文本。

您可以在下面的示例中观察整个过程。

```py
import prettytable

csv_file = open('student_details.csv', 'r')
data = csv_file.readlines()
column_names = data[0].split(',')
table = prettytable.PrettyTable()
table.add_row(column_names)
for i in range(1, len(data)):
    row = data[i].split(",")
    table.add_row(row)
html_string = table.get_html_string()
print("The html string obtained from the csv file is:")
print(html_string)
```

输出:

```py
The html string obtained from the csv file is:
<table>
    <thead>
        <tr>
            <th>Field 1</th>
            <th>Field 2</th>
            <th>Field 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Name</td>
            <td>Roll Number</td>
            <td> Subject<br></td>
        </tr>
        <tr>
            <td>Aditya</td>
            <td> 12</td>
            <td> Python<br></td>
        </tr>
        <tr>
            <td>Sam</td>
            <td> 23</td>
            <td> Java<br></td>
        </tr>
        <tr>
            <td>Chris</td>
            <td> 11</td>
            <td> C++<br></td>
        </tr>
        <tr>
            <td>Joel</td>
            <td> 10</td>
            <td> JavaScript<br></td>
        </tr>
        <tr>
            <td>Mayank</td>
            <td> 5</td>
            <td> Typescript</td>
        </tr>
    </tbody>
</table> 
```

## 结论

在本文中，我们讨论了如何用 python 将 csv 文件转换成 HTML 文件。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中[列表理解的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)
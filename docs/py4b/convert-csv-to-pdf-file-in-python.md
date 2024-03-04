# 用 python 将 CSV 转换成 PDF 文件

> 原文：<https://www.pythonforbeginners.com/basics/convert-csv-to-pdf-file-in-python>

CSV 文件包含逗号分隔的值，这些值通常包含表格。有时，我们可能需要将 csv 文件转换成 PDF 文件。在本文中，我们将讨论如何用 python 将 csv 文件转换成 PDF。

## 如何用 Python 把 CSV 转换成 PDF 文件？

要将 csv 文件转换为 PDF，我们将首先使用 pandas 模块创建 csv 文件内容的 HTML 字符串。熊猫模块为我们提供了不同的工具来处理 csv 文件。

要将 csv 文件转换成 HTML 字符串，我们将首先使用`read_csv()`方法打开文件。`read_csv()`方法将 csv 文件的文件名作为输入参数，并返回包含来自 csv 文件的数据的 dataframe。

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

获得 HTML 字符串形式的 csv 文件后，我们将把 HTML 字符串转换成 pdf 文件。为此，我们将使用 pdfkit 模块，它构建在 wkhtmltopdf 库之上。pdfkit 模块为我们提供了`from_string()`方法，我们可以用它将 HTML 字符串转换成 pdf 文件。为此，我们将使用`from_string()`方法。`from_string()`方法将 HTML 字符串作为第一个输入参数，将 pdf 文件的文件名作为第二个输入参数。执行后，HMTL 字符串保存在 pdf 文件中。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import pdfkit

df1 = pd.read_csv('student_details.csv')
print("The dataframe is:")
print(df1)
html_string = df1.to_html()
pdfkit.from_string(html_string, "output_file.pdf")
print("PDF file saved.")
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
PDF file saved.
```

附件是从 csv 文件创建的 PDF 文件。

## 结论

在本文中，我们讨论了如何用 python 将 csv 文件转换成 pdf 文件。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中[列表理解的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)
# Python 制表模块:如何在 Python 中轻松创建表格？

> 原文：<https://www.askpython.com/python-modules/tabulate-tables-in-python>

各位程序员好！在今天的教程中，您将学习如何在 Python 中借助`tabulate`函数创建表格，并了解与所创建的表格相关的各种属性。

## 使用制表模块在 Python 中创建表格的步骤

事不宜迟，让我们直接进入使用制表模块在 Python 中创建表格的步骤。

### 1.导入`tabulate`

第一步是从制表库中导入制表函数。如果这导致一个错误，通过在命令提示符下执行`pip install`命令，确保您已经安装了制表库。

```py
from tabulate import tabulate

```

现在让我们在制表函数的帮助下创建我们的第一个表格。

### 2.创建简单表格

表格的数据以嵌套的[列表](https://www.askpython.com/python/list/python-list)的形式存储，如下面提到的代码所示。

```py
all_data = [["Roll Number","Student name","Marks"],
            [1,"Sasha",34],
            [2,"Richard",36],
            [3,"Judy",20],
            [4,"Lori",39],
            [5,"Maggie",40]]

```

为了将数据制成表格，我们只需将数据传递给`tabulate`函数。我们还可以使用一个名为`headers`的属性将第一个嵌套列表作为表的头部。

```py
table1 = tabulate(all_data)
table2 = tabulate(all_data,headers='firstrow')

```

两个表的结果如下所示。

```py
-----------  ------------  -----
Roll Number  Student name  Marks
1            Sasha         34
2            Richard       36
3            Judy          20
4            Lori          39
5            Maggie        40
-----------  ------------  -----

```

```py
Roll Number  Student name      Marks
-------------  --------------  -------
            1  Sasha                34
            2  Richard              36
            3  Judy                 20
            4  Lori                 39
            5  Maggie               40

```

### 3.格式化 Python 表以使其看起来更好

为了让 Python 中的表格看起来更好，我们可以为表格添加边框，使其看起来更像表格而不是文本数据。可以在`tablefmt`属性的帮助下添加边框，并将其值设置为`grid`。

```py
print(tabulate(all_data,headers='firstrow',tablefmt='grid'))

```

```py
+---------------+----------------+---------+
|   Roll Number | Student name   |   Marks |
+===============+================+=========+
|             1 | Sasha          |      34 |
+---------------+----------------+---------+
|             2 | Richard        |      36 |
+---------------+----------------+---------+
|             3 | Judy           |      20 |
+---------------+----------------+---------+
|             4 | Lori           |      39 |
+---------------+----------------+---------+
|             5 | Maggie         |      40 |
+---------------+----------------+---------+

```

为了让它看起来更好，我们可以使用`fancy_grid`来代替简单的网格。

```py
print(tabulate(all_data,headers='firstrow',tablefmt='fancy_grid'))

```

```py
╒═══════════════╤════════════════╤═════════╕
│   Roll Number │ Student name   │   Marks │
╞═══════════════╪════════════════╪═════════╡
│             1 │ Sasha          │      34 │
├───────────────┼────────────────┼─────────┤
│             2 │ Richard        │      36 │
├───────────────┼────────────────┼─────────┤
│             3 │ Judy           │      20 │
├───────────────┼────────────────┼─────────┤
│             4 │ Lori           │      39 │
├───────────────┼────────────────┼─────────┤
│             5 │ Maggie         │      40 │
╘═══════════════╧════════════════╧═════════╛

```

### 4.从制表中提取表格的 HTML 代码

为了提取表格的 HTML 代码，我们需要将`tablefmt`属性设置为`html`。同样显示在下面。

```py
print(tabulate(all_data,headers='firstrow',tablefmt='html'))

```

```py
<table>
<thead>
<tr><th style="text-align: right;">  Roll Number</th><th>Student name  </th><th style="text-align: right;">  Marks</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">            1</td><td>Sasha         </td><td style="text-align: right;">     34</td></tr>
<tr><td style="text-align: right;">            2</td><td>Richard       </td><td style="text-align: right;">     36</td></tr>
<tr><td style="text-align: right;">            3</td><td>Judy          </td><td style="text-align: right;">     20</td></tr>
<tr><td style="text-align: right;">            4</td><td>Lori          </td><td style="text-align: right;">     39</td></tr>
<tr><td style="text-align: right;">            5</td><td>Maggie        </td><td style="text-align: right;">     40</td></tr>
</tbody>
</table>

```

## 结论

在本教程中，我们使用`tabulate`函数创建了自己的表格数据，并了解了表格的一些属性。希望你喜欢它！

感谢您的阅读！
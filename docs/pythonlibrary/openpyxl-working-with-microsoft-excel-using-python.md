# OpenPyXL -使用 Python 处理 Microsoft Excel

> 原文：<https://www.blog.pythonlibrary.org/2020/11/03/openpyxl-working-with-microsoft-excel-using-python/>

商业世界使用**微软 Office** 。他们的电子表格软件解决方案**微软 Excel** 尤其受欢迎。Excel 用于存储表格数据、创建报告、绘制趋势图等等。在开始使用 Excel 和 Python 之前，让我们澄清一些特殊术语:

*   电子表格或工作簿-文件本身(。xls 或者。xlsx)。
*   工作表-工作簿中的一张内容表。电子表格可以包含多个工作表。
*   列-以字母标记的垂直数据行，以“A”开头。
*   Row -用数字标记的水平数据行，从 1 开始。
*   单元格-列和行的组合，如“A1”。

在本文中，您将使用 Python 处理 Excel 电子表格。您将了解以下内容:

*   Python Excel 包
*   从工作簿中获取工作表
*   读取单元格数据
*   遍历行和列
*   编写 Excel 电子表格
*   添加和移除工作表
*   添加和删除行和列

Excel 被大多数公司和大学使用。它可以以多种不同的方式使用，并使用 Visual Basic for Applications (VBA)进行增强。然而，VBA 有点笨拙——这就是为什么学习如何在 Python 中使用 Excel 是件好事。

现在让我们来看看如何使用 Python 编程语言处理 Microsoft Excel 电子表格！

### Python Excel 包

您可以使用 Python 创建、读取和编写 Excel 电子表格。但是，Python 的标准库不支持使用 Excel 为此，您需要安装第三方软件包。最受欢迎的是 **OpenPyXL** 。您可以在此处阅读其文档:

*   [https://openpyxl.readthedocs.io/en/stable/](https://openpyxl.readthedocs.io/en/stable/)

OpenPyXL 不是你唯一的选择。还有其他几个支持 Microsoft Excel 的软件包:

*   xlrd -用于读取旧的 Excel(。xls)文档
*   xlwt -用于编写较老的 Excel(。xls)文档
*   xlwings -可处理新的 Excel 格式，并具有宏功能

几年前，前两个库曾经是最流行的 Excel 文档库。然而，这些软件包的作者已经停止支持它们。xlwings 包很有前途，但是不能在所有平台上工作，并且需要安装 Microsoft Excel。

您将在本文中使用 OpenPyXL，因为它正在被积极开发和支持。OpenPyXL 不需要安装 Microsoft Excel，在所有平台上都可以使用。

您可以使用`pip`安装 OpenPyXL:

```py
$ python -m pip install openpyxl
```

安装完成后，让我们看看如何使用 OpenPyXL 来读取 Excel 电子表格！

### 从工作簿中获取工作表

第一步是找到一个与 OpenPyXL 一起使用的 Excel 文件。这本书的 Github 资源库中有一个为您提供的`books.xlsx`文件。您可以通过以下网址下载:

*   [https://github . com/dris collis/python 101 code/tree/master/chapter 38 _ excel](https://github.com/driscollis/python101code/tree/master/chapter38_excel)

请随意使用您自己的文件，尽管您自己的文件的输出不会与本书中的示例输出相匹配。

下一步是编写一些代码来打开电子表格。为此，创建一个名为`open_workbook.py`的新文件，并向其中添加以下代码:

```py
# open_workbook.py

from openpyxl import load_workbook

def open_workbook(path):
    workbook = load_workbook(filename=path)
    print(f'Worksheet names: {workbook.sheetnames}')
    sheet = workbook.active
    print(sheet)
    print(f'The title of the Worksheet is: {sheet.title}')

if __name__ == '__main__':
    open_workbook('books.xlsx')
```

在这个例子中，您从`openpyxl`导入`load_workbook()`，然后创建`open_workbook()`，它接受 Excel 电子表格的路径。接下来，使用`load_workbook()`创建一个`openpyxl.workbook.workbook.Workbook`对象。该对象允许您访问电子表格中的工作表和单元格。是的，它的名字中确实有两个`workbook`。那不是错别字！

函数的其余部分演示了如何打印电子表格中所有当前定义的工作表，获取当前活动的工作表并打印出该工作表的标题。

当您运行此代码时，您将看到以下输出:

```py
Worksheet names: ['Sheet 1 - Books']
<Worksheet "Sheet 1 - Books">
The title of the Worksheet is: Sheet 1 - Books
```

现在您已经知道了如何访问电子表格中的工作表，您已经准备好继续访问单元格数据了！

### 读取单元格数据

使用 Microsoft Excel 时，数据存储在单元格中。您需要一种从 Python 访问这些单元格的方法，以便能够提取这些数据。OpenPyXL 让这个过程变得简单明了。

创建一个名为`workbook_cells.py`的新文件，并将以下代码添加到其中:

```py
# workbook_cells.py

from openpyxl import load_workbook

def get_cell_info(path):
    workbook = load_workbook(filename=path)
    sheet = workbook.active
    print(sheet)
    print(f'The title of the Worksheet is: {sheet.title}')
    print(f'The value of {sheet["A2"].value=}')
    print(f'The value of {sheet["A3"].value=}')
    cell = sheet['B3']
    print(f'{cell.value=}')

if __name__ == '__main__':
    get_cell_info('books.xlsx')
```

这段代码将在 OpenPyXL 工作簿中加载 Excel 文件。您将获取活动工作表，然后打印出它的`title`和几个不同的单元格值。您可以通过使用 sheet 对象后跟方括号(其中包含列名和行号)来访问单元格。例如，`sheet["A2"]`将获取“A”列第 2 行的单元格。要获得该单元格的值，可以使用`value`属性。

**注意:**这段代码使用了 Python 3.8 中添加到 f 字符串的新特性。如果您使用早期版本运行此程序，将会收到一个错误。

当您运行这段代码时，您将得到以下输出:

```py
<Worksheet "Sheet 1 - Books">
The title of the Worksheet is: Sheet 1 - Books
The value of sheet["A2"].value='Title'
The value of sheet["A3"].value='Python 101'
cell.value='Mike Driscoll'
```

您可以使用单元格的一些其他属性来获取有关单元格的附加信息。将以下函数添加到文件中，并更新末尾的条件语句以运行它:

```py
def get_info_by_coord(path):
    workbook = load_workbook(filename=path)
    sheet = workbook.active
    cell = sheet['A2']
    print(f'Row {cell.row}, Col {cell.column} = {cell.value}')
    print(f'{cell.value=} is at {cell.coordinate=}')

if __name__ == '__main__':
    get_info_by_coord('books.xlsx')
```

在这个例子中，您使用`cell`对象的`row`和`column`属性来获取行和列信息。注意，列“A”映射到“1”，“B”映射到“2”，等等。如果要迭代 Excel 文档，可以使用`coordinate`属性获取单元格名称。

当您运行此代码时，输出将如下所示:

```py
Row 2, Col 1 = Title
cell.value='Title' is at cell.coordinate='A2'
```

说到迭代，让我们看看下一步怎么做！

### 遍历行和列

有时，您需要迭代整个 Excel 电子表格或部分电子表格。OpenPyXL 允许您以几种不同的方式做到这一点。创建一个名为`iterating_over_cells.py`的新文件，并向其中添加以下代码:

```py
# iterating_over_cells.py

from openpyxl import load_workbook

def iterating_range(path):
    workbook = load_workbook(filename=path)
    sheet = workbook.active
    for cell in sheet['A']:
        print(cell)

if __name__ == '__main__':
    iterating_range('books.xlsx')
```

在这里，您加载电子表格，然后循环遍历“A”列中的所有单元格。对于每个单元格，打印出`cell`对象。如果您想更精细地格式化输出，可以使用在上一节中学习的一些单元格属性。

运行这段代码的结果如下:

```py
<Cell 'Sheet 1 - Books'.A1>
<Cell 'Sheet 1 - Books'.A2>
<Cell 'Sheet 1 - Books'.A3>
<Cell 'Sheet 1 - Books'.A4>
<Cell 'Sheet 1 - Books'.A5>
<Cell 'Sheet 1 - Books'.A6>
<Cell 'Sheet 1 - Books'.A7>
<Cell 'Sheet 1 - Books'.A8>
<Cell 'Sheet 1 - Books'.A9>
<Cell 'Sheet 1 - Books'.A10>
# output truncated for brevity
```

默认情况下，输出会被截断，因为它会打印出相当多的单元格。OpenPyXL 通过使用`iter_rows()`和`iter_cols()`函数提供了其他方法来迭代行和列。这些方法接受几个参数:

*   `min_row`
*   `max_row`
*   `min_col`
*   `max_col`

您还可以添加一个`values_only`参数，告诉 OpenPyXL 返回单元格的值，而不是单元格对象的值。继续创建一个名为`iterating_over_cell_values.py`的新文件，并将以下代码添加到其中:

```py
# iterating_over_cell_values.py

from openpyxl import load_workbook

def iterating_over_values(path):
    workbook = load_workbook(filename=path)
    sheet = workbook.active
    for value in sheet.iter_rows(
            min_row=1, max_row=3,
            min_col=1, max_col=3,
            values_only=True,
        ):
        print(value)

if __name__ == '__main__':
    iterating_over_values('books.xlsx')
```

这段代码演示了如何使用`iter_rows()`遍历 Excel 电子表格中的行，并打印出这些行的值。当您运行此代码时，您将获得以下输出:

```py
('Books', None, None)
('Title', 'Author', 'Publisher')
('Python 101', 'Mike Driscoll', 'Mouse vs Python')
```

输出是一个 Python 元组，其中包含每一列中的数据。至此，您已经学会了如何打开电子表格和读取数据——既可以从特定的单元格读取，也可以通过迭代读取。现在您已经准备好学习如何使用 OpenPyXL 来**创建** Excel 电子表格了！

### 编写 Excel 电子表格

使用 OpenPyXL 创建 Excel 电子表格并不需要很多代码。您可以使用`Workbook()`类创建一个电子表格。继续创建一个名为`writing_hello.py`的新文件，并将以下代码添加到其中:

```py
# writing_hello.py

from openpyxl import Workbook

def create_workbook(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet['A1'] = 'Hello'
    sheet['A2'] = 'from'
    sheet['A3'] = 'OpenPyXL'
    workbook.save(path)

if __name__ == '__main__':
    create_workbook('hello.xlsx')
```

在这里实例化`Workbook()`并获得活动表。然后将“A”列的前三行设置为不同的字符串。最后，您调用`save()`并把保存新文档的`path`传递给它。恭喜你！您刚刚用 Python 创建了一个 Excel 电子表格。

接下来让我们看看如何在工作簿中添加和删除工作表！

### 添加和移除工作表

许多人喜欢在工作簿的多个工作表中组织他们的数据。OpenPyXL 支持通过其`create_sheet()`方法向`Workbook()`对象添加新工作表的能力。

创建一个名为`creating_sheets.py`的新文件，并将以下代码添加到其中:

```py
# creating_sheets.py

import openpyxl

def create_worksheets(path):
    workbook = openpyxl.Workbook()
    print(workbook.sheetnames)
    # Add a new worksheet
    workbook.create_sheet()
    print(workbook.sheetnames)
    # Insert a worksheet
    workbook.create_sheet(index=1,
                          title='Second sheet')
    print(workbook.sheetnames)
    workbook.save(path)

if __name__ == '__main__':
    create_worksheets('sheets.xlsx')
```

这里您使用了两次`create_sheet()`来向工作簿添加两个新的工作表。第二个示例显示了如何设置工作表的标题以及在哪个索引处插入工作表。参数`index=1`意味着工作表将被添加到第一个现有工作表之后，因为它们的索引从`0`开始。

当您运行此代码时，您将看到以下输出:

```py
['Sheet']
['Sheet', 'Sheet1']
['Sheet', 'Second sheet', 'Sheet1']
```

您可以看到新工作表已逐步添加到工作簿中。保存文件后，您可以通过打开 Excel 或其他与 Excel 兼容的应用程序来验证是否有多个工作表。

在这个自动创建工作表的过程之后，您突然得到了太多的工作表，所以让我们去掉一些。有两种方法可以移除板材。继续创建`delete_sheets.py`，看看如何使用 Python 的`del`关键字删除工作表:

```py
# delete_sheets.py

import openpyxl

def create_worksheets(path):
    workbook = openpyxl.Workbook()
    workbook.create_sheet()
    # Insert a worksheet
    workbook.create_sheet(index=1,
                          title='Second sheet')
    print(workbook.sheetnames)
    del workbook['Second sheet']
    print(workbook.sheetnames)
    workbook.save(path)

if __name__ == '__main__':
    create_worksheets('del_sheets.xlsx')
```

这段代码将创建一个新工作簿，然后向其中添加两个新工作表。然后它用 Python 的`del`关键字删除`workbook['Second sheet']`。您可以通过查看`del`命令前后的表单列表的打印结果来验证它是否按预期工作:

```py
['Sheet', 'Second sheet', 'Sheet1']
['Sheet', 'Sheet1']
```

从工作簿中删除工作表的另一种方法是使用`remove()`方法。创建一个名为`remove_sheets.py`的新文件，并输入以下代码以了解其工作原理:

```py
# remove_sheets.py

import openpyxl

def remove_worksheets(path):
    workbook = openpyxl.Workbook()
    sheet1 = workbook.create_sheet()
    # Insert a worksheet
    workbook.create_sheet(index=1,
                          title='Second sheet')
    print(workbook.sheetnames)
    workbook.remove(sheet1)
    print(workbook.sheetnames)
    workbook.save(path)

if __name__ == '__main__':
    remove_worksheets('remove_sheets.xlsx')
```

这一次，通过将结果赋给`sheet1`，您保留了对您创建的第一个工作表的引用。然后在代码中删除它。或者，您也可以使用与前面相同的语法删除该工作表，如下所示:

```py
workbook.remove(workbook['Sheet1'])
```

无论您选择哪种方法删除工作表，输出都是一样的:

```py
['Sheet', 'Second sheet', 'Sheet1']
['Sheet', 'Second sheet']
```

现在让我们继续学习如何添加和删除行和列。

### 添加和删除行和列

OpenPyXL 有几个有用的方法，可以用来在电子表格中添加和删除行和列。以下是您将在本节中了解的四种方法的列表:

*   `.insert_rows()`
*   `.delete_rows()`
*   `.insert_cols()`
*   `.delete_cols()`

这些方法中的每一个都可以接受两个参数:

*   `idx` -插入行或列的索引
*   `amount` -要添加的行数或列数

要了解这是如何工作的，创建一个名为`insert_demo.py`的文件，并向其中添加以下代码:

```py
# insert_demo.py

from openpyxl import Workbook

def inserting_cols_rows(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet['A1'] = 'Hello'
    sheet['A2'] = 'from'
    sheet['A3'] = 'OpenPyXL'
    # insert a column before A
    sheet.insert_cols(idx=1)
    # insert 2 rows starting on the second row
    sheet.insert_rows(idx=2, amount=2)
    workbook.save(path)

if __name__ == '__main__':
    inserting_cols_rows('inserting.xlsx')
```

这里，您创建一个工作表，并在列“A”之前插入一个新列。列的索引从 1 开始，而相比之下，工作表从 0 开始。这实际上将 A 列中的所有单元格移动到 b 列，然后从第 2 行开始插入两个新行。

既然您已经知道了如何插入列和行，那么是时候了解如何删除它们了。

要了解如何删除列或行，创建一个名为`delete_demo.py`的新文件，并添加以下代码:

```py
# delete_demo.py

from openpyxl import Workbook

def deleting_cols_rows(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet['A1'] = 'Hello'
    sheet['B1'] = 'from'
    sheet['C1'] = 'OpenPyXL'
    sheet['A2'] = 'row 2'
    sheet['A3'] = 'row 3'
    sheet['A4'] = 'row 4'
    # Delete column A
    sheet.delete_cols(idx=1)
    # delete 2 rows starting on the second row
    sheet.delete_rows(idx=2, amount=2)
    workbook.save(path)

if __name__ == '__main__':
    deleting_cols_rows('deleting.xlsx')
```

这段代码在几个单元格中创建文本，然后使用`delete_cols()`删除 A 列。它还通过`delete_rows()`从第二行开始删除两行。在组织数据时，能够添加和删除列和行非常有用。

### 包扎

由于 Excel 在许多行业的广泛使用，能够使用 Python 与 Excel 文件进行交互是一项极其有用的技能。在本文中，您了解了以下内容:

*   Python Excel 包
*   从工作簿中获取工作表
*   读取单元格数据
*   遍历行和列
*   编写 Excel 电子表格
*   添加和移除工作表
*   添加和删除行和列

OpenPyXL 可以做的甚至比这里介绍的更多。例如，您可以使用 OpenPyXL 向单元格添加公式、更改字体以及对单元格应用其他类型的样式。阅读文档并尝试在您自己的一些电子表格上使用 OpenPyXL，这样您就可以发现它的全部功能。
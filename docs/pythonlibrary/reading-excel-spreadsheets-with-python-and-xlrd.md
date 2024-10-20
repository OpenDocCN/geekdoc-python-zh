# 用 Python 和 xlrd 读取 Excel 电子表格

> 原文：<https://www.blog.pythonlibrary.org/2014/04/30/reading-excel-spreadsheets-with-python-and-xlrd/>

上个月，我们学习了如何创建 Microsoft Excel(即*。xls)文件使用 **xlwt** 包。今天我们将学习如何阅读*。xls/*。xlsx 文件使用一个名为 **xlrd** 的包。xlrd 包可以在 Linux 和 Mac 上运行，也可以在 Windows 上运行。当您需要在 Linux 服务器上处理 Excel 文件时，这非常有用。

我们将从阅读我们在[上一篇文章](https://www.blog.pythonlibrary.org/2014/03/24/creating-microsoft-excel-spreadsheets-with-python-and-xlwt/)中创建的第一个 Excel 文件开始。

我们开始吧！

* * *

## 阅读 Excel 电子表格

在这一节中，我们将看到一个函数，它演示了读取 Excel 文件的不同方法。下面是代码示例:

```py

import xlrd

#----------------------------------------------------------------------
def open_file(path):
    """
    Open and read an Excel file
    """
    book = xlrd.open_workbook(path)

    # print number of sheets
    print book.nsheets

    # print sheet names
    print book.sheet_names()

    # get the first worksheet
    first_sheet = book.sheet_by_index(0)

    # read a row
    print first_sheet.row_values(0)

    # read a cell
    cell = first_sheet.cell(0,0)
    print cell
    print cell.value

    # read a row slice
    print first_sheet.row_slice(rowx=0,
                                start_colx=0,
                                end_colx=2)

#----------------------------------------------------------------------
if __name__ == "__main__":
    path = "test.xls"
    open_file(path)

```

让我们把它分解一下。首先我们导入 **xlrd** ，然后在我们的函数中，我们打开传入的 Excel 工作簿。接下来的几行显示了如何反思这本书。我们找出工作簿中有多少工作表，并打印出它们的名称。接下来，我们通过 **sheet_by_index** 方法提取第一个工作表。我们可以使用 **row_values** 方法从工作表中读取一整行。如果我们想获得一个特定单元格的值，我们可以调用**单元格**方法，并向其传递行和列索引。最后，我们使用 xlrd 的 **row_slice** 方法来读取行的一部分。如您所见，最后一个方法接受一个行索引以及开始和结束列索引来确定要返回的内容。row_slice 方法返回单元格实例的列表。

这使得迭代一组单元格变得非常容易。这里有一个小片段来演示:

```py

cells = first_sheet.row_slice(rowx=0,
                              start_colx=0,
                              end_colx=2)
for cell in cells:
    print cell.value

```

xlrd 包支持以下类型的单元格:文本、数字(即浮点)、日期(任何“看起来”像日期的数字格式)、布尔、错误和空/空白。该包还支持从命名单元格中提取数据，尽管该项目并不支持所有类型的命名单元格。[参考文本](http://www.simplistix.co.uk/presentations/python-excel.pdf)对它到底不支持什么有点含糊。

如果你需要复制单元格格式，你需要下载 [xlutils 包](https://pypi.python.org/pypi/xlutils/1.7.0)。

* * *

### 包扎

至此，你应该足够了解如何阅读大多数使用微软 **XLS** 格式构建的 Excel 文件。还有另一个包也支持读取 xls/xlsx 文件，名为 [openpyxl](http://pythonhosted.org/openpyxl/) 项目。你也许会想试试看。

* * *

### 相关阅读

*   xlrd / xlwt / xlutils [主页](http://www.python-excel.org/)
*   Excel [阅读指南](https://classic.scraperwiki.com/docs/python/python_excel_guide/)
*   [使用 Python 和 xlwt 创建 Microsoft Excel 电子表格](https://www.blog.pythonlibrary.org/2014/03/24/creating-microsoft-excel-spreadsheets-with-python-and-xlwt/)
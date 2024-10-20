# Python 和 Microsoft Office -使用 PyWin32

> 原文：<https://www.blog.pythonlibrary.org/2010/07/16/python-and-microsoft-office-using-pywin32/>

大多数典型用户都用过微软 Office。虽然 Office 可能是技术支持的克星，但我们仍然必须应对它。Python 可以用来编写 Office 脚本(也称为自动化),让我们或我们的用户更容易使用。这可能不像录制宏那样容易，但也很接近了。在本文中，您将学习如何使用 PyWin32 模块访问一些 Office 程序，并使用 Python 操作它们。有些论坛说，你需要在 Microsoft Word(和 Excel)上运行 PythonWin 的 makepy 实用程序，然后才能访问 Office 应用程序。我不认为我需要这样做来使它工作(至少，在 2007 版本中没有)。然而，PythonWin 附带了 PyWin32，所以如果您遇到麻烦，可以尝试一下。

## Python 和 Microsoft Excel

如果你寻找过使用 Python 和 Office 的例子，你通常会发现最常被黑的组件是 Excel。事实上，有几个专门为读写 Excel 文件而创建的非 PyWin32 模块。它们分别被称为 xlrd 和 xlwt。但这是另一篇文章的主题。在这里，我们将看到如何使用 PyWin32 接口来处理 Excel。请注意，以下脚本仅适用于 Windows。xlrd 和 xlwt 的一个优点是可以在任何平台上使用。

我们来看一个简单的例子，好吗？

```py

import time
import win32com.client as win32

#----------------------------------------------------------------------
def excel():
    """"""
    xl = win32.gencache.EnsureDispatch('Excel.Application')
    ss = xl.Workbooks.Add()
    sh = ss.ActiveSheet

    xl.Visible = True
    time.sleep(1)

    sh.Cells(1,1).Value = 'Hacking Excel with Python Demo'

    time.sleep(1)
    for i in range(2,8):
        sh.Cells(i,1).Value = 'Line %i' % i
        time.sleep(1)

    ss.Close(False)
    xl.Application.Quit()

if __name__ == "__main__":
    excel()

```

上面的例子和你通常在网上找到的很相似。它实际上是基于我在 Wesley Chun 的优秀著作*核心 Python 编程*中看到的一个例子。让我们花些时间来解开代码。为了访问 Excel，我们导入 *win32com.client* ，然后调用它的 *gencache。EnsureDispatch* ，传入我们想要打开的应用程序名称。在这种情况下，要传递的字符串是“Excel。应用”。所做的就是在后台打开 Excel。此时，用户甚至不会知道 Excel 已经打开，除非他们运行了任务管理器。下一行是通过调用 Excel 实例的“Workbooks ”,向 Excel 添加一个新工作簿。Add()"方法。这将返回一个 sheets 对象(我想)。为了获得 ActiveSheet，我们调用 *ss。活动表*。最后，我们通过将该属性设置为 True 来使 Excel 程序本身可见。

要设置特定单元格的值，可以这样调用: *sh。单元格(行，列)。Value = "某值"*。请注意，我们的实例不是从零开始的，实际上会将值放在正确的行/列组合中。如果我们想提取一个值，我们只需去掉等号。如果我们想要分子式呢？为了解决这个问题，我在 Excel 中记录了一个宏，并执行了一个仅粘贴公式的选择性粘贴命令。使用生成的代码，我发现要获得 Python 中的公式，只需这样做:

```py

formula = sh.Cells(row, col).Formula

```

如果您需要更改您所在的工作表，该怎么办？录制宏也向我展示了如何完成这一壮举。这是来自 Excel 的 VBA 代码:

 `Sub Macro1()
'
' Macro1 Macro
'
Sheets("Sheet2").Select
End Sub` 

从这段代码中，我得出结论，我需要调用我的 sheets 对象的“Sheets”方法，在做了一点小改动之后，我通过执行以下操作让它工作了:

```py

sheet2 = ss.Sheets("Sheet2")

```

现在我们有了工作簿中第二个工作表的句柄。如果您想要编辑或检索值，只需在前面使用的相同方法前面加上您调用的 sheet2 实例(即 sheet2。单元格(1，1)。值)。原始程序的最后两行将关闭工作表，然后退出整个 Excel 实例。

您可能会想，到目前为止，我所展示的只是如何创建一个新文档。如果你想打开一个现有的文件呢？在代码的开头做这样的事情:

```py

xl = win32.gencache.EnsureDispatch('Excel.Application')
ss = xl.Workbooks.Open(filename)

```

现在你知道了！现在，您已经了解了使用 Excel 的 COM 对象模型用 Python 入侵 Excel 的基本知识。如果您需要了解更多，我建议尝试录制一个宏，然后将结果翻译成 Python。注意:我找不到一个可以保存电子表格的例子...有几个例子声称他们的工作，但他们没有为我。

## Python 和 Microsoft Word

使用 Python 访问 Microsoft Word 遵循我们用于 Excel 的相同语法。让我们快速了解一下如何访问 Word。

```py

from time import sleep
import win32com.client as win32

RANGE = range(3, 8)

def word():
    word = win32.gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Add()
    word.Visible = True
    sleep(1)

    rng = doc.Range(0,0)
    rng.InsertAfter('Hacking Word with Python\r\n\r\n')
    sleep(1)
    for i in RANGE:
        rng.InsertAfter('Line %d\r\n' % i)
        sleep(1)
    rng.InsertAfter("\r\nPython rules!\r\n")

    doc.Close(False)
    word.Application.Quit()

if __name__ == '__main__':
    word()

```

这个特殊的例子也是基于春的书。然而，网上有很多其他的例子看起来也几乎和这个一模一样。现在让我们解开这段代码。为了获得 Microsoft Word 应用程序的句柄，我们调用*win32 . gen cache . ensure dispatch(' Word。应用')*；然后我们通过调用 word 实例的*文档来添加一个新文档。添加()*。如果您想向用户展示您在做什么，可以将 Word 的 visibility 设置为 True。

如果您想在文档中添加文本，那么您需要告诉 Word 您想将文本添加到哪里。这就是 Range 方法的用武之地。虽然您看不到它，但有一种“网格”告诉 Word 如何在屏幕上布局文本。因此，如果我们想在文档的最顶端插入文本，我们告诉它从(0，0)开始。要在 Word 中添加新的一行，我们需要将“\r\n”追加到字符串的末尾。如果你不知道不同平台上的行尾的恼人之处，你应该花些时间在 Google 上学习一下，这样你就不会被奇怪的 bug 咬了！

代码的其余部分是不言自明的，将留给读者去解释。我们现在将继续打开和保存文档:

```py

# Based on examples from http://code.activestate.com/recipes/279003/
word.Documents.Open(doc)
word.ActiveDocument.SaveAs("c:\\a.txt", FileFormat=win32com.client.constants.wdFormatTextLineBreaks)

```

这里我们展示了如何打开一个现有的 Word 文档并将其保存为文本。我还没有完全测试这个，所以你的里程可能会有所不同。如果您想阅读文档中的文本，您可以执行以下操作:

```py

docText = word.Documents[0].Content

```

关于 Word 文档的 Python 黑客课程到此结束。由于我在微软 Word 和 Python 上找到的许多信息都是陈旧的，而且似乎有一半时间都不起作用，所以我不会添加到糟糕信息的混乱中。希望这能让你开始自己的文字处理之旅。

## 进一步阅读

*   [Python for Windows 示例](http://win32com.goermezer.de/content/view/94/192/)
*   通过 Python [线程](http://www.velocityreviews.com/forums/t330073-opening-ms-word-files-via-python.html)打开 MS Word 文件
*   [配方 279003](http://code.activestate.com/recipes/279003/) :将 Word 文档转换成文本
*   Dzone: [从 Python 中编写 excel 脚本](http://snippets.dzone.com/posts/show/2036)
*   Python-Excel [网站](http://www.python-excel.org/)，[邮件列表](http://groups.google.com/group/python-excel)
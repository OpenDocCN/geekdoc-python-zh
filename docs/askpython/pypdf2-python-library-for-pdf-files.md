# PyPDF2:用于 PDF 文件操作的 Python 库

> 原文：<https://www.askpython.com/python-modules/pypdf2-python-library-for-pdf-files>

PyPDF2 是一个处理 PDF 文件的纯 python 库。我们可以使用 PyPDF2 模块来处理现有的 PDF 文件。我们无法使用此模块创建新的 PDF 文件。

## PyPDF2 特性

PyPDF2 模块的一些令人兴奋的特性包括:

*   PDF 文件元数据，如页数、作者、创建者、创建时间和上次更新时间。
*   逐页提取 PDF 文件内容。
*   合并多个 PDF 文件。
*   将 PDF 文件页面旋转一个角度。
*   PDF 页面的缩放。
*   使用 Pillow library 从 PDF 页面提取图像并将其保存为图像。

## 安装 PyPDF2 模块

我们可以使用 PIP 来安装 PyPDF2 模块。

```py
$ pip install PyPDF2

```

## PyPDF2 示例

让我们看一些使用 PyPDF2 模块处理 PDF 文件的例子。

### 1.提取 PDF 元数据

我们可以得到 PDF 文件的页数。我们还可以获得关于 PDF 作者、创建者应用程序和创建日期的信息。

```py
import PyPDF2

with open('Python_Tutorial.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    print(f'Number of Pages in PDF File is {pdf_reader.getNumPages()}')
    print(f'PDF Metadata is {pdf_reader.documentInfo}')
    print(f'PDF File Author is {pdf_reader.documentInfo["/Author"]}')
    print(f'PDF File Creator is {pdf_reader.documentInfo["/Creator"]}')

```

样本输出:

```py
Number of Pages in PDF File is 2
PDF Metadata is {'/Author': 'Microsoft Office User', '/Creator': 'Microsoft Word', '/CreationDate': "D:20191009091859+00'00'", '/ModDate': "D:20191009091859+00'00'"}
PDF File Author is Microsoft Office User
PDF File Creator is Microsoft Word

```

**推荐阅读** : [Python 带语句](https://www.journaldev.com/33273/python-with-statement-with-open-file)和 [Python f-strings](https://www.journaldev.com/23592/python-f-strings-literal-string-interpolation)

*   PDF 文件应该以二进制模式打开。这就是为什么文件打开模式被作为' rb '传递。
*   PdfFileReader 类用于读取 PDF 文件。
*   documentInfo 是一个包含 PDF 文件元数据的[字典](https://www.journaldev.com/14401/python-dictionary)。
*   我们可以使用 getNumPages()函数获得 PDF 文件的页数。另一种方法是使用`numPages`属性。

### 2.提取 PDF 页面的文本

```py
import PyPDF2

with open('Python_Tutorial.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)

    # printing first page contents
    pdf_page = pdf_reader.getPage(0)
    print(pdf_page.extractText())

    # reading all the pages content one by one
    for page_num in range(pdf_reader.numPages):
        pdf_page = pdf_reader.getPage(page_num)
        print(pdf_page.extractText())

```

*   PdfFileReader getPage(int)方法返回`PyPDF2.pdf.PageObject`实例。
*   我们可以在 page 对象上调用 extractText()方法来获取页面的文本内容。
*   extractText()不会返回任何二进制数据，如图像。

### 3.旋转 PDF 文件页面

PyPDF2 允许多种类型的操作，可以逐页进行。我们可以顺时针或逆时针旋转页面一个角度。

```py
import PyPDF2

with open('Python_Tutorial.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    pdf_writer = PyPDF2.PdfFileWriter()

    for page_num in range(pdf_reader.numPages):
        pdf_page = pdf_reader.getPage(page_num)
        pdf_page.rotateClockwise(90)  # rotateCounterClockwise()

        pdf_writer.addPage(pdf_page)

    with open('Python_Tutorial_rotated.pdf', 'wb') as pdf_file_rotated:
        pdf_writer.write(pdf_file_rotated)

```

*   PdfFileWriter 用于从源 PDF 写入 PDF 文件。
*   我们使用 rotateClockwise(90)方法将页面顺时针旋转 90 度。
*   我们将旋转的页面添加到 PdfFileWriter 实例中。
*   最后，使用 PdfFileWriter 的 write()方法生成旋转后的 PDF 文件。

PDF writer 可以从一些源 PDF 文件编写 PDF 文件。我们不能用它从一些文本数据创建 PDF 文件。

### 4.合并 PDF 文件

```py
import PyPDF2

pdf_merger = PyPDF2.PdfFileMerger()
pdf_files_list = ['Python_Tutorial.pdf', 'Python_Tutorial_rotated.pdf']

for pdf_file_name in pdf_files_list:
    with open(pdf_file_name, 'rb') as pdf_file:
        pdf_merger.append(pdf_file)

with open('Python_Tutorial_merged.pdf', 'wb') as pdf_file_merged:
    pdf_merger.write(pdf_file_merged)

```

上面的代码看起来很适合合并 PDF 文件。但是，它生成了一个空的 PDF 文件。原因是在实际写入创建合并的 PDF 文件之前，源 PDF 文件已经关闭。

是 PyPDF2 最新版本的 bug。你可以在本期 GitHub 中读到。

有一种替代方法是使用`contextlib`模块保持源文件打开，直到写操作完成。

```py
import contextlib
import PyPDF2

pdf_files_list = ['Python_Tutorial.pdf', 'Python_Tutorial_rotated.pdf']

with contextlib.ExitStack() as stack:
    pdf_merger = PyPDF2.PdfFileMerger()
    files = [stack.enter_context(open(pdf, 'rb')) for pdf in pdf_files_list]
    for f in files:
        pdf_merger.append(f)
    with open('Python_Tutorial_merged_contextlib.pdf', 'wb') as f:
        pdf_merger.write(f)

```

你可以通过这个 [StackOverflow 问题](https://stackoverflow.com/questions/49927338/merge-2-pdf-files-giving-me-an-empty-pdf)了解更多信息。

### 5.将 PDF 文件分割成单页文件

```py
import PyPDF2

with open('Python_Tutorial.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    for i in range(pdf_reader.numPages):
        pdf_writer = PyPDF2.PdfFileWriter()
        pdf_writer.addPage(pdf_reader.getPage(i))
        output_file_name = f'Python_Tutorial_{i}.pdf'
        with open(output_file_name, 'wb') as output_file:
            pdf_writer.write(output_file)

```

Python_Tutorial.pdf 有 2 页。输出文件被命名为 Python_Tutorial_0.pdf 和 Python_Tutorial_1.pdf。

### 6.从 PDF 文件中提取图像

我们可以使用 PyPDF2 和 Pillow (Python 图像库)从 PDF 页面中提取图像并保存为图像文件。

首先，您必须使用以下命令安装 Pillow 模块。

```py
$ pip install Pillow

```

下面是一个从 PDF 文件的第一页提取图像的简单程序。我们可以很容易地扩展它，从 PDF 文件中提取所有的图像。

```py
import PyPDF2
from PIL import Image

with open('Python_Tutorial.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)

    # extracting images from the 1st page
    page0 = pdf_reader.getPage(0)

    if '/XObject' in page0['/Resources']:
        xObject = page0['/Resources']['/XObject'].getObject()

        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    mode = "P"

                if '/Filter' in xObject[obj]:
                    if xObject[obj]['/Filter'] == '/FlateDecode':
                        img = Image.frombytes(mode, size, data)
                        img.save(obj[1:] + ".png")
                    elif xObject[obj]['/Filter'] == '/DCTDecode':
                        img = open(obj[1:] + ".jpg", "wb")
                        img.write(data)
                        img.close()
                    elif xObject[obj]['/Filter'] == '/JPXDecode':
                        img = open(obj[1:] + ".jp2", "wb")
                        img.write(data)
                        img.close()
                    elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                        img = open(obj[1:] + ".tiff", "wb")
                        img.write(data)
                        img.close()
                else:
                    img = Image.frombytes(mode, size, data)
                    img.save(obj[1:] + ".png")
    else:
        print("No image found.")

```

我的示例 PDF 文件在第一页有一个 PNG 图像，程序用“image20.png”文件名保存它。

## 参考

*   [PyPI.org 页面](https://pypi.org/project/PyPDF2/)
*   [PyPDF2 GitHub Page](https://github.com/mstamy2/PyPDF2)
*   [PDF 图像提取器脚本](https://github.com/mstamy2/PyPDF2/blob/master/Scripts/pdf-image-extractor.py)
*   [枕头模块](https://pypi.org/project/Pillow/)
# Python 101:如何遍历目录

> 原文：<https://www.blog.pythonlibrary.org/2016/01/26/python-101-how-to-traverse-a-directory/>

您经常会发现自己需要编写遍历目录的代码。以我的经验来看，它们往往是一次性脚本或运行在 cron 中的清理脚本。总之，Python 提供了一种非常有用的遍历目录结构的方法，被恰当地称为 **os.walk** 。我通常使用该功能浏览一组文件夹和子文件夹，在这些文件夹和子文件夹中，我需要删除旧文件或将它们移动到归档目录中。让我们花一些时间来学习如何在 Python 中遍历目录！

* * *

### 使用 os.walk

使用 os.walk 需要一点练习才能正确。下面是一个例子，它只打印出您传递给它的路径中的所有文件名:

```py

import os

def pywalker(path):
    for root, dirs, files in os.walk(path):
        for file_ in files:
            print( os.path.join(root, file_) )

if __name__ == '__main__':
    pywalker('/path/to/some/folder')

```

通过连接**根**和**文件 _** 元素，您最终得到了文件的完整路径。如果你想检查文件的创建日期，那么你可以使用 **os.stat** 。例如，我曾经用它创建了一个[清理脚本](https://www.blog.pythonlibrary.org/2013/11/14/python-101-how-to-write-a-cleanup-script/)。

如果您想要做的只是检查指定路径中的文件夹和文件列表，那么您正在寻找 **os.listdir** 。大多数时候，我通常需要深入到最低的子文件夹，所以 listdir 不够好，我需要使用 os.walk 来代替。

* * *

### 直接使用 os.scandir()

Python 3.5 最近增加了 **os.scandir()** ，这是一个新的目录迭代函数。你可以在 [PEP 471](https://www.python.org/dev/peps/pep-0471/) 里读到。在 Python 3.5 中， **os.walk** 是使用 **os.scandir** 实现的，根据 [Python 3.5 公告](https://docs.python.org/3/whatsnew/3.5.html#pep-471-os-scandir-function-a-better-and-faster-directory-iterator)“这使得它在 POSIX 系统上快 3 到 5 倍，在 Windows 系统上快 7 到 20 倍”。

让我们直接使用 **os.scandir** 来尝试一个简单的例子。

```py

import os

folders = []
files = []

for entry in os.scandir('/'):
    if entry.is_dir():
        folders.append(entry.path)
    elif entry.is_file():
        files.append(entry.path)

print('Folders:')
print(folders)

```

Scandir 返回一个由 [DirEntry](https://docs.python.org/3/library/os.html#os.DirEntry) 对象组成的迭代器，这些对象是轻量级的，有方便的方法可以告诉你正在迭代的路径。在上面的例子中，我们检查条目是文件还是目录，并将条目添加到适当的列表中。你还可以通过 DirEntry 的 **stat** 方法创建一个 **stat** 对象，这非常简单！

* * *

### 包扎

现在您知道了如何在 Python 中遍历目录结构。如果你想在 than 3.5 之前的版本中获得 os.scandir 的速度提升，你可以在 T2 的 PyPI 上获得 scandir 包。

* * *

### 相关阅读

*   Python 3.5 - [新功能](https://docs.python.org/3/whatsnew/3.5.html)
*   [Python 增强提案 0471 (PEP471)](https://www.python.org/dev/peps/pep-0471/)
*   PyPI 版本的 [scandir](https://pypi.python.org/pypi/scandir)
*   Webucator 在 os.scandir 上的文章
*   Python 101 - [如何编写清理脚本](https://www.blog.pythonlibrary.org/2013/11/14/python-101-how-to-write-a-cleanup-script/)
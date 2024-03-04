# OS。Python 中的 Walk 和 Fnmatch

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/os-walk-and-fnmatch-in-python>

## 概观

在之前的一篇文章中，我描述了如何使用 OS.walk，并展示了一些如何在脚本中使用它的例子。

在本文中，我将展示如何使用 os.walk()模块函数遍历目录树，以及使用 fnmatch 模块匹配文件名。

## OS.walk 是什么？

它通过自顶向下或自底向上遍历目录树来生成目录树中的文件名。

对于以目录顶层为根的树中的每个目录(包括顶层本身)，它产生一个三元组(目录路径、目录名、文件名)。

**dirpath #** 是一个字符串，指向目录的路径。

**目录名#** 是目录路径中子目录的名称列表(不包括'.'和'..').

**filenames #** 是 dirpath 中非目录文件的名称列表。

请注意，列表中的名称不包含路径组件。

要在 dirpath 中获取文件或目录的完整路径(以 top 开头)，请执行 os.path.join(dirpath，name)。更多信息，请参见 [Python 文档](https://docs.python.org/dev/library/os.html#os.walk "python_docs_os.walk")。

## 什么是 Fnmatch

fnmatch [模块](https://www.pythonforbeginners.com/modules-in-python/python-modules)将文件名与诸如 Unix shells 所使用的 glob 样式的模式进行比较。

这些和更复杂的[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)规则不一样。纯粹是字符串匹配操作。

如果您发现使用不同的模式样式更方便，例如正则表达式，那么只需使用正则表达式操作来匹配您的文件名。[http://www.doughellmann.com/PyMOTW/fnmatch/](http://www.doughellmann.com/PyMOTW/fnmatch/ "pymotw_fnmatch")

## 它是做什么的？

fnmatch 模块用于通配符模式匹配。

**简单匹配**

fnmatch()将单个文件名与一个模式进行比较，并返回一个布尔值来指示它们是否匹配。当操作系统使用区分大小写的文件系统时，比较是区分大小写的。

**过滤**

要测试文件名序列，可以使用 filter()。它返回与模式参数匹配的名称列表。

## 查找所有 mp3 文件

这个脚本将从根路径(“/”)中搜索*.mp3 文件

```py
 import fnmatch
import os

rootPath = '/'
pattern = '*.mp3'

for root, dirs, files in os.walk(rootPath):
    for filename in fnmatch.filter(files, pattern):
        print( os.path.join(root, filename)) 
```

## 在计算机中搜索特定文件

此脚本使用带有过滤器的“os.walk”和“fnmatch”在硬盘中搜索所有图像文件

```py
import fnmatch
import os

images = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
matches = []

for root, dirnames, filenames in os.walk("C:\"):
    for extensions in images:
        for filename in fnmatch.filter(filenames, extensions):
            matches.append(os.path.join(root, filename)) 
```

有许多其他(更快)的方法可以做到这一点，但是现在您已经了解了它的基本原理。

#### 更多阅读

[http://Rosetta code . org/wiki/Walk _ a _ directory/Recursively # Python](https://rosettacode.org/wiki/Walk_a_directory/Recursively#Python "rosettacode")

[堆栈溢出匹配模式](https://stackoverflow.com/questions/7541976/fnmatch-how-exactly-do-you-implement-the-match-any-chars-in-seq-pattern "SO-fnmatch")

[使用 fnmatch 的 stack overflow oswalk](https://stackoverflow.com/questions/10660284/how-to-improve-searching-with-os-walk-and-fnmatch-in-python-2-7-2 "SO_post")

[Python 中的 osWalk](https://www.pythonforbeginners.com/code-snippets-source-code/python-os-walk)

[与 osWalk 同乐](https://www.pythonforbeginners.com/code-snippets-source-code/having-fun-with-os-walk-in-python)
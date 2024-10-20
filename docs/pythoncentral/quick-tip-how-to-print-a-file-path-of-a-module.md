# 快速提示:如何打印模块的文件路径

> 原文：<https://www.pythoncentral.io/quick-tip-how-to-print-a-file-path-of-a-module/>

Python 提供了一种非常容易和简单的方法来检索导入模块的文件路径。如果您试图快速找到文件路径，并且您正在处理一个具有多个子目录的项目，或者如果您正在使用主要通过命令行访问的脚本或程序，这将非常有用。如果您处于类似的情况，您可以使用下面的方法来查找模块的确切文件路径:

```py
import os
print(os)
```

就是这样。每当您想知道代码中模块文件的确切位置时，就使用这个技巧。然后应该会为您打印出模块的准确文件路径。它可能看起来像这样:

```py
<module 'os' from '/usr/lib/python2.7/os.pyc'>
```
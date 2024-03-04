# Python 中的 OS.walk

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-os-walk>

Python 在 OS 模块中有一个很酷的内置函数，叫做 os.walk()。

### OS.Walk()

OS.walk()通过自顶向下或自底向上遍历目录树来生成目录树中的文件名。

对于以目录顶层为根的树中的每个目录(包括顶层本身)，它产生一个三元组(目录路径、目录名、文件名)。

### 小路

```py
 root :	Prints out directories only from what you specified
dirs :	Prints out sub-directories from root. 
files:  Prints out all files from root and directories 
```

### 制作剧本

有了这些信息，我们就可以创建一个简单的脚本来做到这一点。这个脚本将打印出我指定的路径(/var/log)中的所有目录、子目录和文件

```py
import os
print "root prints out directories only from what you specified"
print "dirs prints out sub-directories from root"
print "files prints out all files from root and directories"
print "*" * 20
for root, dirs, files in os.walk("/var/log"):
    print root
    print dirs
    print files 
```

### 使用 getsize

第二个示例扩展了第一个示例，使用 getsize 函数显示了每个文件消耗了多少资源。

```py
print "This is using getsize to see how much every file consumes"
print "---------------"
from os.path import join, getsize
for root, dirs, files in os.walk('/tmp'):
    print root, "consumes",
    print sum([getsize(join(root, name)) for name in files]),
    print "bytes in", len(files), "non-directory files" 
```
## Python 标准库 04 文件管理 (部分 os 包，shutil 包)

[`www.cnblogs.com/vamei/archive/2012/09/14/2684775.html`](http://www.cnblogs.com/vamei/archive/2012/09/14/2684775.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

我们可以通过操作系统的命令来管理文件，如同我们在[linux 文件管理相关命令](http://www.cnblogs.com/vamei/archive/2012/09/13/2682519.html)中看到的那样。另一方面，python 标准库则允许我们在 python 内部管理文件。相同的目的，我们就有了两条不同的途径实现。在 python 内部实现的优势在于你可以同时利用 python 语言，并配合其他 python 工具，比如[正则表达式工具](http://www.cnblogs.com/vamei/archive/2012/08/31/2661870.html)。但操作系统同样可以通过 shell 编程，来整合 linux 文件管理命令，shell 也拥有自己的正则表达式工具。python or shell? 这是留给用户的选择。本文中会尽量将两者相似的功能相对应。

同样，本文也是基于[linux 文件管理背景知识](http://www.cnblogs.com/vamei/archive/2012/09/09/2676792.html)

（python 并不是调用操作系统的命令来实现这些功能的。python 独立地调用 c 标准库以及系统调用函数来实现。当然，在 python 中也可以调用操作系统的命令，我们会在以后介绍如何实现这一功能。）

1\. os 包： 

os 包包括各种各样的函数，以实现操作系统的许多功能。这个包非常庞杂。os 包的一些命令就是用于文件管理。我们这里列出最常用的:

mkdir(*path*)

创建新目录，path 为一个字符串，表示新目录的路径。相当于$mkdir 命令

rmdir(*path*)

删除空的目录，path 为一个字符串，表示想要删除的目录的路径。相当于$rmdir 命令

listdir(*path*)

返回目录中所有文件。相当于$ls 命令。

remove(*path*)

删除 path 指向的文件。

rename(*src, dst*)

重命名文件，src 和 dst 为两个路径，分别表示重命名之前和之后的路径。 

chmod(*path, mode*)

改变 path 指向的文件的权限。相当于$chmod 命令。

chown(*path, uid, gid*)

改变 path 所指向文件的拥有者和拥有组。相当于$chown 命令。

stat(*path*)

查看 path 所指向文件的附加信息，相当于$ls -l 命令。

symlink(*src, dst*)

为文件 dst 创建软链接，src 为软链接文件的路径。相当于$ln -s 命令。

getcwd() 

查询当前工作路径 (cwd, current working directory)，相当于$pwd 命令。 

比如说我们要新建目录 new：

```py
import os
os.mkdir('/home/vamei/new')

```

2\. shutil 包

copy(*src, dst*)

复制文件，从 src 到 dst。相当于$cp 命令。

move(*src, dst*)

移动文件，从 src 到 dst。相当于$mv 命令。

比如我们想复制文件 a.txt:

```py
import shutil
shutil.copy('a.txt', 'b.txt')

```

关于本文中的各个命令的细节，请参照官方文档。[os](http://docs.python.org/library/os.html), [shutil](http://docs.python.org/library/shutil.html)。实际上，结合本章以及之前的内容，我们已经可以把 Python 作为一个系统文件管理的利器使用了。

总结:

os 包: rmdir, mkdir, listdir, remove, rename, chmod, chown, stat, symlink

shutil 包: copy, move
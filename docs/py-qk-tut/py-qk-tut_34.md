## Python 标准库 03 路径与文件 (os.path 包, glob 包)

[`www.cnblogs.com/vamei/archive/2012/09/05/2671198.html`](http://www.cnblogs.com/vamei/archive/2012/09/05/2671198.html)

作者：Vamei 出处：http://www.cnblogs.com/vamei 欢迎转载，也请保留这段声明。谢谢！

1\. 路径与文件简介

请参看 Linux 文件系统

[`www.cnblogs.com/vamei/archive/2012/09/09/2676792.html`](http://www.cnblogs.com/vamei/archive/2012/09/09/2676792.html)

2\. os.path 包

os.path 包主要是处理路径字符串，比如说'/home/vamei/doc/file.txt'，提取出有用信息。

```py
import os.path
path = '/home/vamei/doc/file.txt'

print(os.path.basename(path))    # 查询路径中包含的文件名
print(os.path.dirname(path))     # 查询路径中包含的目录
 info = os.path.split(path)       # 将路径分割成文件名和目录两个部分，放在一个表中返回
path2 = os.path.join('\', 'home', 'vamei', 'doc', 'file1.txt')  # 使用目录名和文件名构成一个路径字符串
 p_list = [path, path2] print(os.path.commonprefix(p_list))    # 查询多个路径的共同部分

```

此外，还有下面的方法： 

os.path.normpath(*path*)   # 去除路径 path 中的冗余。比如'/home/vamei/../.'被转化为'/home'

os.path 还可以查询文件的相关信息(metadata)。文件的相关信息不存储在文件内部，而是由操作系统维护的，关于文件的一些信息(比如文件类型，大小，修改时间)。

```py
import os.path 
path = '/home/vamei/doc/file.txt'

print(os.path.exists(path))    # 查询文件是否存在

print(os.path.getsize(path))   # 查询文件大小
print(os.path.getatime(path))  # 查询文件上一次读取的时间
print(os.path.getmtime(path))  # 查询文件上一次修改的时间

print(os.path.isfile(path))    # 路径是否指向常规文件
print(os.path.isdir(path))     # 路径是否指向目录文件

```

 (实际上，这一部份类似于 Linux 中的 ls 命令的某些功能)

3\. glob 包

glob 包最常用的方法只有一个, glob.glob()。该方法的功能与 Linux 中的 ls 相似(参看[Linux 文件管理命令](http://www.cnblogs.com/vamei/archive/2012/09/13/2682519.html))，接受一个 Linux 式的文件名格式表达式(filename pattern expression)，列出所有符合该表达式的文件（与正则表达式类似），将所有文件名放在一个表中返回。所以 glob.glob()是一个查询目录下文件的好方法。

该文件名表达式的语法与[Python 自身的正则表达式](http://www.cnblogs.com/vamei/archive/2012/08/31/2661870.html)不同 (你可以同时看一下 fnmatch 包，它的功能是检测一个文件名是否符合 Linux 的文件名格式表达式)。 如下：

Filename Pattern Expression      Python Regular Expression 

*                                .*

?                                .

[0-9]                            same

[a-e]                            same

[^mnp]                           same

我们可以用该命令找出/home/vamei 下的所有文件: 

```py
import glob print(glob.glob('/home/vamei/*'))

```

文件系统

os.path

glob.glob
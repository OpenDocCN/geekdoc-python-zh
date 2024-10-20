# Python 101:编写清理脚本

> 原文：<https://www.blog.pythonlibrary.org/2014/01/24/python-101-writing-a-cleanup-script/>

*编者按:这是 Yasoob Khalid 的客座博文，他是[免费 Python 技巧博客](http://freepythontips.wordpress.com/)* 的作者

嗨，伙计们！我希望你一切都好。那么这个帖子里是什么呢？今天我们将编写一个清理脚本。这篇文章的想法来自于 Mike Driscoll，他最近写了一篇关于用 python 写清理脚本的非常有用的文章。那么我的帖子和他的帖子有什么不同呢？在我的帖子里我会用

```py
path.py
```

。当我使用

```py
path.py
```

我第一次爱上了它。

* * *

### 正在安装 path.py:

因此，有几种安装 path.py 的方法。Path.py 可以使用 setuptools 或 distribute 或 pip 来安装:

```py
easy_install path.py
```

最新版本总是更新到 [Python 包索引](http://pypi.python.org/pypi/path.py)。源代码托管在 [Github](https://github.com/jaraco/path.py) 上。

* * *

### 查找目录中文件的数量:

所以我们的第一个任务是找到目录中存在的文件数量。在本例中，我们不会遍历子目录，而是只计算顶层目录中的文件数量。这个很简单。以下是我的解决方案:

```py
from path import path
d = path(DIRECTORY) 
#Replace DIRECTORY with your required directory
num_files = len(d.files())

print num_files

```

在这个脚本中，我们首先导入了路径模块。然后我们设置

```py
num_file
```

变量设置为 0。这个变量将记录我们目录中文件的数量。然后我们用一个目录名调用 path 函数。此外，我们遍历目录根目录中的文件，并增加

```py
num_files
```

可变。最后，我们打印出

```py
num_files
```

可变。下面是这个脚本的一个小修改版本，它输出了我们目录的根目录下子目录的数量。

```py
from path import path
d = path(DIRECTORY) 
#Replace DIRECTORY with your required directory
num_dirs = len(d.dirs())

print num_dirs 

```

* * *

### 查找目录中递归文件的数量:

那很容易！不是吗？所以现在我们的工作是找到一个目录中递归文件的数量。为了完成这项任务，我们被赋予了

```py
walk()
```

方法依据

```py
path.py
```

。这与相同

```py
os.walk()
```

。因此，让我们编写一个简单的脚本，用 Python 递归地列出一个目录及其子目录中的所有文件。

```py
from path import path
file_count = 0
dir_count = 0
total = 0
d = path(DIRECTORY)
#Replace DIRECTORY with your required directory
for i in d.walk():
    if i.isfile():
        file_count += 1
    elif i.isdir():
        dir_count += 1
    else:
        pass
    total += 1

print "Total number of files == {0}".format(file_count)
print "Total number of directories == {0}".format(dir_count)

```

这也很简单。现在，如果我们想漂亮地打印目录名呢？我知道有一些终端一行程序，但是这里我们只讨论 Python。让我们看看如何实现这一目标。

```py
files_loc = {}
file_count = 0
dir_count = 0
total = 0
for i in d.walk():
    if i.isfile():
        if i.dirname().basename() in files_loc:
            files_loc[i.dirname().basename()].append(i.basename())
        else:
            files_loc[i.dirname().basename()] = []
            files_loc[i.dirname().basename()].append(i.basename())
        file_count += 1
    elif i.isdir():
        dir_count += 1
    else:
        pass
    total += 1

for i in files_loc:
    print "|---"+i
    for i in files_loc[i]:
        print "|   |"
        print "|   `---"+i
    print "|"

```

这里没有什么花哨的东西。在这个脚本中，我们只是打印了一个目录和其中包含的文件。现在让我们继续。

* * *

### 从目录中删除特定文件:

假设我们有一个名为

```py
this_file_sucks.py
```

。现在我们如何删除它。让我们假设我们不知道它被放在哪个目录中，让这个场景更真实一些。解决这个问题也很简单。只需转到顶层目录并执行以下脚本:

```py
from path import path
d = path(DIRECTORY)
#replace directory with your desired directory
for i in d.walk():
    if i.isfile():
        if i.name == 'php.py':
            i.remove()

```

在上面的脚本中，我没有实现任何日志记录和错误处理。这是留给读者的练习。

* * *

### 根据扩展名删除文件

假设你想删除所有的。pyc '文件。你将如何着手处理这个问题。这是我在 1999 年提出的一个解决方案

```py
path.py
```

。

```py
from path import path
d = path(DIRECTORY)
files = d.walkfiles("*.pyc")
for file in files:
    file.remove()
    print "Removed {} file".format(file)

```

* * *

### 根据文件大小删除文件:

另一个有趣的场景是。如果我们想删除那些超过 5Mb 的文件怎么办？
**注:**Mb 和 Mb 是有区别的。我将在这里报道 Mb。
有没有可能

```py
path.py
```

？是的，它是！这是一个完成这项工作的脚本:

```py
d = path('./')
del_size = 4522420
for i in d.walk():
    if i.isfile():
        if i.size > del_size:
        #4522420 is approximately equal to 4.1Mb
        #Change it to your desired size
            i.remove()

```

因此，我们看到了如何根据文件大小删除文件。

* * *

### 根据文件的上次访问时间删除文件

在这一部分中，我们将了解如何根据文件的最后访问时间来删除文件。我写了下面的代码来实现这个目标。把天数改成自己喜欢的就行了。该脚本将删除在

```py
DAYS
```

可变。

```py
from path import path
import time

#Change the DAYS to your liking
DAYS = 6
removed = 0
d = path(DIRECTORY)
#Replace DIRECTORY with your required directory
time_in_secs = time.time() - (DAYS * 24 * 60 * 60)

for i in d.walk():
    if i.isfile():
        if i.mtime <= time_in_secs:
            i.remove()
            removed += 1

print removed

```

我们还学习了如何根据文件的最后修改时间来删除文件。如果您想根据上次访问时间删除文件，只需更改

```py
i.mtime
```

到

```py
i.atime
```

你就可以走了。

* * *

### 再见

原来如此。我希望你喜欢这篇文章。最后，我想公开道歉，我的英语不好，所以你可能会发现一些语法错误。请你把它们发邮件给我，这样我可以提高我的英语水平。如果你喜欢这篇文章，那么别忘了在 twitter 和 facebook 上关注我。一条转发也没坏处！如果你想发给我一个下午，然后使用[这个](mailto:yasoob.khld@gmail.com)电子邮件。

### **注:**

这是来自 [Pytips](http://freepythontips.wordpress.com) 博客的官方交叉帖子。如果你想阅读原文，请点击[链接](http://freepythontips.wordpress.com/2014/01/23/python-101-writing-a-cleanup-script/)。
# Python 中的递归文件和目录操作(第 2 部分)

> 原文：<https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-2/>

在[第 1 部分](https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/ "Recursive File and Directory Manipulation in Python (Part 1)")中，我们看了如何使用 [os.path.walk 和 os.walk 方法在目录树](https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/ "Recursive File and Directory Manipulation in Python (Part 1)")下查找并列出特定扩展名的文件。前一个函数只出现在 Python 2.x 中，后一个函数在 Python 2.x 和 Python 3.x 中都可用。正如我们在上一篇文章中看到的，`os.path.walk`方法可能很难使用，所以从现在开始我们将坚持使用`os.walk`方法，这样脚本将更简单，并且与两个分支都兼容。

在[第 1 部分](https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-1/ "Recursive File and Directory Manipulation in Python (Part 1)")中，我们的脚本遍历了`topdir`变量下的所有文件夹，但只找到一个扩展名的文件。现在让我们展开它，在选择的文件夹中的`topdir`路径下查找多个扩展名的文件。我们将首先搜索三种不同文件扩展名的文件:。txt，。我们的`extens`变量将是一个字符串列表，而不是一个:

```py

extens = ['txt', 'pdf', 'doc']

```

字符`.`没有像之前一样包含在变量`ext`中，我们很快就会看到原因。为了保存结果(文件名)，我们将使用一个以扩展名为关键字的字典:

```py

# List comprehension form of instantiation

found = { x: [] for x in extens }

```

其他变量暂时保持不变；然而，脚本文件本身将被放在(并将从)我的系统的“Documents”文件夹中执行，所以`topdir`变量将成为那个路径。

之前我们用`str.endswith`方法测试了扩展。如果我们要再次使用它，我们必须遍历扩展名列表并用`endswith`测试每个文件名，但是我们将使用稍微不同的方法。对于行走过程中踩过的每个文件，我们将提取扩展，然后测试在*扩展*中的成员资格。下面是我们如何提取它:

```py

for name in files:

    # Split the name by '.' & get the last element

    ext = name.lower().rsplit(“.”, 1)[-1]

```

与前一部分一样，我们将这一行放在 for 循环中，该循环与由`os.walk`返回的*文件*列表交互。在这一行中，我们结合了三个操作:改变文件名的大小写，拆分文件名，提取一个元素。在文件名上调用`str.lower`会将其变为小写。与`extens`中的所有琴弦相同。在 *name* 上调用`str.rsplit`，然后将字符串拆分成一个列表(从右边开始)，第一个参数`.`对其进行分隔，并且只进行与第二个参数(1)一样多的拆分。第三部分(`[-1]`)检索列表的最后一个元素——我们使用它而不是索引 1，因为如果没有进行拆分(如果`name`中没有`.`)，就不会引发`IndexError`。

现在我们已经提取了`name`的扩展名(如果有的话)，我们可以测试它是否在我们的扩展名列表中:

```py

if ext in extens:

```

这就是为什么在`extens`中`.`不在任何扩展名之前，因为`ext`永远不会有扩展名。如果条件为真，我们将把找到的名字添加到我们的`found`字典中:

```py

if ext in extens:

    found[ext].append(os.path.join(dirpath, name))

```

上面的行将把结果路径(`dirpath`连接到`os.walk`返回的`name`)追加到`found`中`ext`键的列表中。既然我们已经更改了搜索扩展和结果列表，我们还必须调整如何将结果保存到日志文件中。

在以前的版本中(使用`os.walk`)，我们只是在`logname`打开一个文件，并将结果写入文件。在这个版本中，我们必须遍历结果中的多个类别，每个类别对应一个扩展。我们将把`found`中的每个结果列表连接到我们的结果字符串，我们现在将其标识为`logbody`。我们还将在日志文件中添加一个小标题，*日志标题*:

```py

# The header in our logfile

loghead = 'Search log from filefind for files in {}\n\n'.format(os.path.realpath(topdir))
#我们的日志文件的正文
 logbody = ' '
#循环搜索结果
以在 found: 
 #将来自 found dict 
 logbody += " < <结果的结果与扩展名“% s”>>" % search
#使用 str.join 将搜索时的列表转换为 str
log body+= ' \ n \ n % s \ n \ n ' % ' \ n '。加入(找到[搜索]) 

```

结果的格式可以是您喜欢的任何格式，但重要的是我们要遍历所有结果以获得完整的日志。在`logbody`完成之后，我们可以编写我们的日志文件:

```py

# Write results to the logfile

with open(logname, 'w') as logfile:

    logfile.write('%s\n%s' % (loghead, logbody))

```

*注意:*如果解决方案中的任何名称/路径包含非 ASCII 字符，我们必须将`open`模式更改为`wb`，并解码`loghead`和`logbody`(或者在 Python 3.x 中进行编码)，以便成功保存`logfile`。

现在我们终于准备好测试我们的脚本了。在我的系统上运行它会产生这个日志文件(缩短的):

```py

Search log from filefind for files in C:\Python27\Lib\site-packages

<< Results with the extension 'pdf' >>

.\GPL_Full.pdf

.\beautifulsoup4-4.1.3\doc\rfc2425-v2.1.pdf

.\beautifulsoup4-4.1.3\doc\rfc2426-v3.0.pdf
<< Results with the extension 'txt' > > 
。\README.txt 
。\soup.txt 
。\ beautiful soup 4-4 . 1 . 3 \ authors . txt
。\ beautiful soup 4-4 . 1 . 3 \ copy . txt
...
。\ wx-2.8-MSW-unicode \ docs \ changes . txt
。\ wx-2.8-MSW-unicode \ docs \ migration guide . txt
。\ wx-2.8-MSW-unicode \ docs \ readme . win32 . txt
...
。\ wx-2.8-MSW-unicode \ wx \ tools \ XRCed \ todo . txt
<< Results with the extension 'doc' > > 

```

这个日志告诉我们，在`C:\Python27\Lib\site-packages`目录中有几个 PDF 文件，许多文本文件，没有。doc”或 Word 文件。看起来效果不错，扩展搜索列表也很容易更改，但是如果我们不想在`wx-2.8-msw-unicode`树下的“docs”目录中搜索呢？毕竟，我们知道那里可能会有很多文本文件。我们可以通过在主循环中修改`dirnames`列表来忽略这个目录。因为我们可能想要忽略多个目录，所以我们将保留一个它们的列表(当然这将在循环之前进行):

```py

# Directories to ignore

ignore = ['docs', 'doc']

```

现在我们有了列表，我们将在主遍历循环中添加这个小循环(在文件名循环之前):

```py

# Remove directories in ignore
#目录名必须完全匹配！
for idir in ignore:
if idir in dirnames:
dirnames . remove(idir)

```

这将就地编辑`dirnames`,这样遍历循环的下一次迭代将不再包括在 ignore 中命名的文件夹。带有新行走循环的完整脚本现在如下所示:

```py

import os
#文件中名称的第一个参数
 topdir = ' . '
extens = ['txt '，' pdf '，' doc'] #要搜索的扩展名
found = {x: [] for x in extens} #找到的文件列表
#要忽略的目录
 ignore = ['docs '，' doc']
logname = "findfiletypes.log"
print('开始搜索%s' % os.path.realpath(topdir)中的文件)
#遍历 os.walk 中的目录路径、目录名、文件的树
(top dir):
#删除忽略的目录
 #目录名必须完全匹配！
for idir in ignore:
if idir in dirnames:
dirnames . remove(idir)
#遍历当前步骤的文件名
，查找文件名:
 #用“.”分割文件名&获取最后一个元素
 ext = name.lower()。rsplit(' . ', 1)[-1]
#如果 ext 匹配
如果 ext 在 ext:
中找到[ext]则保存全名。append(os.path.join(目录路径，名称))
#我们的日志文件中的标头
 loghead = '在文件中搜索日志在{}中查找文件\n\n '。format(
OS . path . real path(top dir)
)
#我们日志文件的主体
 logbody = ' '
#循环遍历在 found: 
中搜索的结果
#将来自 found dict 
 logbody += " < <结果的结果与扩展名“% s”>>" % search
log body+= ' \ n \ n % s \ n \ n ' % ' \ n '。加入(找到[搜索])
#将结果写入日志文件
，打开(日志名，' w ')作为日志文件:
 logfile.write('%s\n%s' %(日志头，日志体))

```

使用我们新的 ignored files 元素，日志文件看起来像这样(缩短了):

从`filefind`的日志中搜索`C:\Python27\Lib\site-packages`中的文件

```py

<< Results with the extension 'pdf' >>

.\GPL_Full.pdf
<< Results with the extension 'txt' > > 
。\README.txt 
。\soup.txt 
。\ beautiful soup 4-4 . 1 . 3 \ authors . txt
。\ beautiful soup 4-4 . 1 . 3 \ copy . txt
...
。\ beautiful soup 4-4 . 1 . 3 \ scripts \ demonstration _ markup . txt
。\ wx-2.8-MSW-unicode \ wx \ lib \ editor \ readme . txt
...
。\ wx-2.8-MSW-unicode \ wx \ tools \ XRCed \ todo . txt
<< Results with the extension 'doc' > > 

```

我们的忽略列表正如我们所希望的那样工作，删除了`wx-...-unicode`中“docs”目录下的完整树。我们还可以看到，另一个忽略目录(“doc”)从我们的 PDF 结果中删除了另外两个 PDF 文件，对于这两个目录，我们不需要命名完整路径(因为这个名称无论如何都不会是`dirnames`中的完整路径)。这可能很方便，但是请记住，这种方法将删除与`ignore`列表中的名称匹配的名称下的*树的任何*部分(为了避免这种情况，如果您不介意麻烦地命名完整路径，请尝试同时使用`dirpath`和`dirnames`来指定要忽略的完整路径！).

现在我们已经完成了这个版本的文件/目录操作脚本，我们可以快速搜索任意树下的多个文件扩展名，并且只需双击就可以获得所有找到的文件扩展名的记录。如果我们只是想*知道*所有文件存在于哪里，这是很好的，但是由于它们可能不会都在同一个文件夹中，如果我们想将它们全部移动/复制到同一个文件夹中或者同时对它们做其他事情，浏览日志文件的每一行将是**而不是**更好的选择。这就是为什么在下一部分，我们将看看如何升级我们的脚本，以移动，复制/备份，或者擦除我们正在寻找的所有文件。
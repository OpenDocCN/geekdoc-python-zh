# 如何在 Python 中使用进度条移动或复制文件

> 原文：<https://www.pythoncentral.io/how-to-movecopy-a-file-or-directory-folder-with-a-progress-bar-in-python/>

在上一篇名为[如何在 Python 中递归复制文件夹(目录)](https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/ "How to Recursively Copy a Folder (Directory) in Python")的文章中，我介绍了如何将文件夹从一个地方递归复制到另一个地方。

虽然这很有用，但我们还可以添加一些东西，让复制目录变得更加用户友好！

从舞台左侧进入进度条。这是一个有用的工具，所有知名软件都使用它来告诉用户正在发生的事情，以及它正在执行的任务的位置。

那么，我们如何将它合并到复制目录中呢？很高兴你问了，我的问题充满了朋友！

## 进度条

首先，我们将创建一个简单的类，使打印过程简单 100 倍，同时给我们一个很好的可重用类，用于其他可能需要它的项目。

开始了。首先我们将添加`__init__`方法:

```py

# -*- coding: utf-8 -*-

import sys
class progress bar(object):
def _ _ init _ _(self，message，width=20，progressSymbol=u'▣'，empty symbol = u '□'):
self . width = width
如果 self.width < 0:
 self.width = 0
self . message = message
self . progress symbol = progress symbol
self . empty symbol = empty symbol

```

这并不难！我们传入我们想要打印的消息、进度条的宽度以及表示进度完成和空进度的两个符号。我们检查宽度是否总是大于或等于零，因为我们不能有负的长度！🙂

简单的东西。

好了，接下来是最难的部分。我们必须弄清楚如何在输出窗口中更新和显示进度条，而不是每次都打印一行。

这就是回车(`\r`)的用武之地。回车符是一个特殊字符，它允许打印的消息在打印时从一行的开头开始。每次我们打印一些东西，我们只是在我们的行的开始使用它，它应该照顾我们的需要就好了。唯一的限制是，如果进度条碰巧在终端换行，回车将不会像预期的那样工作，只会输出一个新行。

下面是我们的`update`函数，它将打印更新的进度:

```py

def update(self, progress):

totalBlocks = self.width

filledBlocks = int(round(progress / (100 / float(totalBlocks)) ))

emptyBlocks = totalBlocks - filledBlocks
progress bar = self . progress symbol * filled blocks+\
self . empty symbol * empty blocks
如果不是 self . message:
self . message = u ' '
progress message = u“\ r { 0 } { 1 } { 2 } %”。格式(自我信息，
进度条，
进度)
sys . stdout . write(progress message)
sys . stdout . flush()
def calculated and update(self，done，total):
progress = int(round((done/float(total))* 100))
self . update(progress)

```

我们刚刚做的是使用所需的总块数来计算填充块数。看起来有点复杂，我来分解一下。我们希望填充的块数等于`progress`除以 100%除以总块数。`calculateAndUpdate`功能只是为了让我们的生活更轻松一点。它所做的只是计算并打印给定当前项目数和项目总数的完成百分比。

```py

filledBlocks = int(round(progress / (100 / float(totalBlocks)) ))

```

对`float`的调用是为了确保计算以浮点精度完成，然后我们用`round`将浮点数四舍五入，用`int`将它变成整数。

```py

emptyBlocks = totalBlocks - filledBlocks

```

那么空块就是`totalBlocks`减去`filledBlocks`。

```py

progressBar = self.progressSymbol * filledBlocks + \

self.emptySymbol * emptyBlocks

```

我们将填充块的数量乘以`progressSymbol`并将其加到乘以空块数量的`emptySymbol`上。

```py

progressMessage = u'\r{0} {1} {2}%'.format(self.message,

progressBar,

progress)

```

我们在消息的开头使用回车来将打印位置设置为行首，并使消息的格式如下:“message！▣ ▣ ▣ □ □ □ 50%".

```py

sys.stdout.write(progressMessage)

sys.stdout.flush()

```

最后，我们将信息打印到屏幕上。我们为什么不用`print`函数呢？因为`print`函数在我们的消息末尾添加了一个新行，这不是我们想要的。我们希望它按照我们设定的那样打印出来。

继续表演！到复制一个目录功能！

## 使用 Python 复制目录(文件夹)

为了知道我们的进展，我们将不得不牺牲一些速度。我们需要计算我们正在复制的所有文件，以获得剩余的总进度，这需要我们递归地搜索目录并计算所有文件。

让我们写一个函数来做这件事。

```py

import os
def countFiles(目录):
 files = []
if os.path.isdir(目录):
对于 os.walk(目录)中的路径、目录、文件名:
 files.extend(文件名)
返回 len(文件)

```

相当直接。我们刚刚检查了传入的目录是否是一个目录，如果是，那么递归地遍历目录并将目录中的文件添加到文件列表中。然后我们只返回列表的长度。简单。

接下来是复制目录功能。如果你看了本文开头提到的那篇文章，你会注意到我使用了`shutil.copytree`函数。这里我们不能这样做，因为`shutil`不支持进度更新，所以我们必须编写自己的复制函数。

我们开始吧！

首先，我们创建一个`ProgressBar`类的实例。
【python】
p = progress bar('复制文件...')

接下来，我们定义一个函数来创建尚不存在的目录。这将允许我们创建源目录的目录结构。
【python】
def make dirs(dest):
如果不是 OS . path . exists(dest):
OS . make dirs(dest)

如果目录不存在，我们创建它。

现在是复制功能。这是一个有点花哨的东西，所以我会把它们都放在一个地方，然后再把它们分开。

```py

import shutil
def copyfilewithprogress(src，dest):
numFiles = count files(src)
如果 numFiles > 0: 
 makedirs(dest)
numCopied = 0
对于 os.walk(src)中的路径、目录、文件名:
对于目录中的目录:
 destDir = path.replace(src，dest)
make dirs(OS . path . join(destDir，directory))
对于文件名中的 sfile:
src file = OS . path . join(path，sfile)
dest file = OS . path . join(path . replace(src，dest)，sfile)
shutil.copy(srcFile，destFile)
numCopied += 1
p . calculated and update(numCopied，numFiles) 
 print 

```

首先，我们计算文件的数量。

```py

def copyFilesWithProgress(src, dest):

numFiles = countFiles(src)

```

这没什么难的。

接下来，如果确实有要复制的文件，我们将创建目标文件夹(如果它不存在),并初始化一个变量来计算当前复制的文件数。

```py

if numFiles &gt; 0:

makedirs(dest)
num copied = 0

```

然后，我们遍历目录树并创建所有需要的目录。

```py

for path, dirs, filenames in os.walk(src):

for directory in dirs:

destDir = path.replace(src, dest)

makedirs(os.path.join(destDir, directory))

```

这里唯一棘手的部分是，我用目标文件夹路径替换了 path 中的源目录字符串，它是根路径。

接下来，我们复制文件和更新进度！

```py

for path, dirs, filenames in os.walk(src):

(...)

for sfile in filenames:

srcFile = os.path.join(path, sfile)
dest file = OS . path . join(path . replace(src，dest)，sfile)
shutil.copy(srcFile，destFile)
numCopied += 1
p . calculated and update(num copied，numFiles) 

```

在这里，我们遍历所有文件，复制它们，更新到目前为止复制的文件数量，然后绘制我们的进度条。

就是这样！我们完了！现在你有了一个不错的复制功能，它会在复制文件时提醒你进度。

## 在 Python 中移动目录(文件夹)

注意，你也可以将`copyFilesWithProgress`中的`shutil.copy`函数调用改为`shutil.move`，并有一个移动进度条。所有其他代码都是一样的，所以我不打算在这里重写。我会让你决定的！🙂

下次再见，我向你道别。
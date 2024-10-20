# 如何在 Python 中重命名(移动)文件

> 原文：<https://www.pythoncentral.io/how-to-rename-move-a-file-in-python/>

重命名(在 Python 中称为*移动*)Python 中的文件非常简单，由于有一个方便的模块叫做`shutil`，只需几行代码就可以完成。

`shutil`有一个名为`move`的函数，它准确地执行函数名所暗示的功能。它将文件或目录从一个位置移动到另一个位置。

这是一个非常简单但完整的例子:

```py

import shutil
def move(src, dest): 
 shutil.move(src, dest) 

```

简单的东西。该函数接收源文件或目录，并将其移动到目标文件或目录。

## **shutil.copy vs os.rename**

如果文件或目录在当前的本地文件系统上，`shutil.move`使用`os.rename`来移动文件或目录。否则，它使用`shutil.copy2`将文件或目录复制到目的地，然后删除源。

我们之所以用`shutil.move`而不用`os.rename`是因为上面的原因。函数`shutil.move`已经处理了一个文件不在当前文件系统上的情况，它还处理将目录复制到目的地的操作。任何`os.rename`抛出的异常在`shutil.move`中也会得到妥善处理，因此无需担心。

`shutil.move`确实以`shutil.Error`的形式抛出自己的异常。当目标文件或目录已经存在时，或者当您试图将源文件或目录复制到自身时，它会这样做。

不幸的是，`shutil.move`中没有选项提供用于测量进度的回调函数。你必须为此编写自己的复制函数，由于需要计算文件数量和测量它们的大小，这可能会慢一点。

真的就这么简单。另一件需要注意的事情是，如果在当前的文件系统上，我们调用`move`函数的时间将是瞬时的，而当用于移动到一个单独的驱动器时，例如，调用将花费与典型的复制操作相同的时间。

如果你有兴趣了解`shutil.move`函数是如何实现的，请点击[链接](http://hg.python.org/cpython/file/2.7/Lib/shutil.py#l264 "Python shutil Move Source Code")获取源代码。

超级短，超级简洁，超级牛逼。

后来的 Pythonistas！
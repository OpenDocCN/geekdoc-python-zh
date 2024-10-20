# Python 101:如何编写清理脚本

> 原文：<https://www.blog.pythonlibrary.org/2013/11/14/python-101-how-to-write-a-cleanup-script/>

有一天，有人问我是否可以写一个脚本来清理目录中所有大于或等于 X 天的文件。我最终使用 Python 的核心模块来完成这项任务。我们将花一些时间来看看做这个有用练习的一种方法。

**公平警告:本文中的代码旨在删除文件。使用风险自担！**

这是我想出的代码:

```py

import os
import sys
import time

#----------------------------------------------------------------------
def remove(path):
    """
    Remove the file or directory
    """
    if os.path.isdir(path):
        try:
            os.rmdir(path)
        except OSError:
            print "Unable to remove folder: %s" % path
    else:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            print "Unable to remove file: %s" % path

#----------------------------------------------------------------------
def cleanup(number_of_days, path):
    """
    Removes files from the passed in path that are older than or equal 
    to the number_of_days
    """
    time_in_secs = time.time() - (number_of_days * 24 * 60 * 60)
    for root, dirs, files in os.walk(path, topdown=False):
        for file_ in files:
            full_path = os.path.join(root, file_)
            stat = os.stat(full_path)

            if stat.st_mtime <= time_in_secs:
                remove(full_path)

        if not os.listdir(root):
            remove(root)

#----------------------------------------------------------------------
if __name__ == "__main__":
    days, path = int(sys.argv[1]), sys.argv[2]
    cleanup(days, path)

```

让我们花几分钟时间看看这段代码是如何工作的。在**清理**函数中，我们采用**天数**参数并将其转换为秒。然后我们从今天的时间中减去这个数量。接下来，我们使用 os 模块的 walk 方法遍历目录。我们将 topdown 设置为 False，告诉 walk 方法从最里面到最外面遍历目录。然后我们遍历最里面的文件夹中的文件，并检查它的最后访问时间。如果该时间小于或等于**时间间隔**(即 X 天前)，则我们尝试移除该文件。当这个循环结束时，我们在 root 上检查它是否有文件(其中 root 是最里面的文件夹)。如果没有，我们就删除这个文件夹。

**移除**功能非常简单。它所做的只是检查传递的路径是否是一个目录。然后，它会尝试使用适当的方法(即 os.rmdir 或 os.remove)删除该路径。

### 删除文件夹/文件的其他方法

还有一些其他的方法来修改文件夹和文件，应该提到。如果你知道你有一组嵌套目录都是空的，你可以使用 os.removedirs()将它们全部删除。另一种更极端的方法是使用 Python 的 shutil 模块。它有一个名为 **rmtree** 的方法，可以删除文件和文件夹！

我在其他脚本中使用这两种方法都取得了很好的效果。我还发现，有时我无法删除 Windows 上的某个特定文件，除非我通过 Windows 资源管理器删除它。为了解决这个问题，我使用 Python 的子进程模块调用 Window 的 **del** 命令及其 **/F** 标志来强制删除。你可以在 Linux 上用它的 **rm -r** 命令做类似的事情。偶尔你会碰到文件被锁定，保护或你只是没有正确的权限，你不能删除它们。

### 要添加的功能

如果您花了一些时间思考上面的脚本，您可能已经想到了一些改进或添加的特性。这里有一些我认为不错的:

*   添加日志记录，这样你就知道什么被删除了，什么没有被删除(或者两者都有)
*   添加上一节中提到的其他一些删除方法
*   使清理脚本能够接受要删除的日期范围或日期列表

我相信你已经想到了其他有趣的想法或解决方案。欢迎在下面的评论中分享它们。

### 进一步阅读

*   如何在 Python 中遍历目录树
*   Python 的 os 模块官方文档
*   Python 的官方文档 [shutil](http://docs.python.org/2/library/shutil.html)
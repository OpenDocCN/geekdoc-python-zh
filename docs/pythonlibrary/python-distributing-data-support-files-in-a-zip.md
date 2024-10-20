# Python:在 zip 中分发数据支持文件

> 原文：<https://www.blog.pythonlibrary.org/2013/07/11/python-distributing-data-support-files-in-a-zip/>

前几天，我听说当我们将一些脚本转移到我们的批处理场中时，我们不能转移子文件夹。因此，如果我有一个如下所示的目录结构，它需要扁平化:

 `Top
--> data
--> font
--> images` 

我需要把我所有的支持文件放在主目录中，而不是为我的字体、图片等设置子文件夹。我认为这是相当蹩脚的，因为我喜欢保持我的目录有组织。如果你的代码遵循模型-视图-控制器(MVC)模型，那么你也会发现这很烦人。所以我思考了一下这个问题，意识到 Python 可以通过使用 [zipfile](http://docs.python.org/2/library/zipfile) 库或者通过 [zipimport](http://docs.python.org/2/library/zipimport.html) 的一些魔法来访问 zip 存档中的文件。现在我可以使用 zipfile，但是如果我将 Python 文件放在子目录中，或者如果我只想将它们放在 zip 存档的顶部，我希望能够导入它们，所以我决定走这条神奇的路线。

现在你不应该直接使用 zipimport。它实际上是 Python 导入机制的一部分，默认情况下是启用的。所以我们不会在我们的代码中使用它，但是我想你应该知道它在幕后做一些事情。无论如何，只是为了额外的背景，我创建了许多使用各种标志和字体的自定义 pdf。因此，我通常将这些文件保存在单独的目录中，以保持有序。我还使用了大量的配置文件，因为一些客户希望以这种方式完成一些事情。因此，我将向您展示一个非常简单的代码片段，您可以用它来导入 Python 文件和提取配置文件。对于后者，我们将使用 Python 方便的 pkgutil 库。

代码如下:

```py

import os
import pkgutil
import StringIO
import sys
from configobj import ConfigObj

base = os.path.dirname(os.path.abspath( __file__ ))
zpath = os.path.join(base, "test.zip")
sys.path.append(zpath)

import hello

#----------------------------------------------------------------------
def getCfgFromZip():
    """
    Extract the config file from the zip file
    """
    cfg_data = pkgutil.get_data("config", "config.ini")
    print cfg_data
    fileLikeObj = StringIO.StringIO(cfg_data)
    cfg = ConfigObj(fileLikeObj)
    cfg_dict = cfg.dict()
    print cfg_dict

if __name__ == "__main__":
    getCfgFromZip()

```

现在，我们还有一个名为 **test.zip** 的 zip 存档，它包含以下内容:

*   hello.py
*   名为 config 的文件夹，包含两个文件:config.ini 和 __init__。巴拉圭

如您所见，Python 知道它可以导入 hello 模块，因为我们通过 **sys.path.append** 将归档文件添加到导入路径中。hello 模块所做的就是向 stdout 输出一条消息。为了提取配置文件，我们使用 pkgutil.get_data(folder_name，file_name)。这将返回一个字符串。因为我们希望将配置加载到 ConfigObj 中，所以我们需要将该字符串转换成类似文件的对象，因此我们使用 StringIO 来实现这个目的。你可以对图像做同样的事情。pkgutil 将返回一个二进制字节字符串，然后您可以将该字符串传递给 reportlab 或 Python 图像库进行进一步处理。我添加了打印语句，以便您可以看到原始数据的样子以及 ConfigObj 的输出。

这就是全部了。我认为这很方便，我希望你在自己的工作中会发现它很有用。

### 进一步阅读

*   python: [可执行 zip 文件可以包含数据文件吗？](http://stackoverflow.com/q/5355694/393194)
*   [用 __main__ 分发一个可执行的 zip 文件。py，如何访问多余的数据？](http://stackoverflow.com/q/2859413/393194)
*   [python 可以直接从 zip 包中导入类或模块吗](http://www.velocityreviews.com/forums/t675118-can-python-import-class-or-module-directly-from-a-zip-package.html)
*   关于 [zipimport](http://docs.python.org/2/library/zipimport.html) 的 Python 文档
*   关于 [pkgutil](http://docs.python.org/2/library/pkgutil.html) 的 Python 文档

### 下载源代码

*   [pyExtract.zip](https://www.blog.pythonlibrary.org/wp-content/uploads/2013/07/pyExtract.zip)
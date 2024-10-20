# Python 101:如何下载文件

> 原文：<https://www.blog.pythonlibrary.org/2012/06/07/python-101-how-to-download-a-file/>

从互联网上下载文件是几乎每个程序员都必须做的事情。Python 在其标准库中提供了几种方法来实现这一点。下载文件最流行的方式可能是使用 urllib 或 urllib2 模块通过 HTTP。Python 还附带了 FTP 下载的 ftplib。最后，有一个新的第三方模块得到了很多关注，称为[请求](http://docs.python-requests.org/en/latest/)。对于本文，我们将重点关注两个 urllib 模块和请求。

由于这是一个非常简单的任务，我们将只展示一个快速而肮脏的脚本，该脚本下载每个库的相同文件，并将结果命名为稍微不同的名称。我们将从这个博客下载一个压缩文件作为我们的示例脚本。让我们来看看:

```py
# Python 2 code
import urllib
import urllib2
import requests

url = 'https://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'

print "downloading with urllib"
urllib.urlretrieve(url, "code.zip")

print "downloading with urllib2"
f = urllib2.urlopen(url)
data = f.read()
with open("code2.zip", "wb") as code:
    code.write(data)

print "downloading with requests"
r = requests.get(url)
with open("code3.zip", "wb") as code:
    code.write(r.content)

```

如您所见， **urllib** 只是一行代码。它的简单性使得它非常容易使用。另一方面，其他两个库也非常简单。对于 **urllib2** ，你只需要打开网址，然后读取并写出数据。事实上，您可以通过执行以下操作将这部分脚本减少一行:

```py
f = urllib2.urlopen(url)
with open("code2.zip", "wb") as code:
    code.write(f.read())

```

无论哪种方式，它都非常有效。**请求**库方法是**获取**，对应 HTTP 获取。然后，您只需获取请求对象并调用其**内容**属性来获取您想要写入的数据。我们将**与**语句一起使用，因为它会自动关闭文件并简化代码。注意，如果文件很大，只使用“read()”可能会有危险。通过传递 **read** a 大小，会更好的阅读。

**更新(2012 年 6 月 8 日)**

正如我的一位读者所指出的，如果你通过 **2to3.py** 运行 urllib，它的内容会发生很大的变化，因此它是 Python 3 格式的。为了完整起见，下面是代码的样子:

```py
# Python 3 code
import urllib.request

url = 'https://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'

print("downloading with urllib")
urllib.request.urlretrieve(url, "code.zip")

print("downloading with urllib2")
f = urllib.request.urlopen(url)
data = f.read()
with open("code2.zip", "wb") as code:
    code.write(data) 

```

你会注意到 urllib2 不再存在，并且 **urllib.urlretrieve** 和 **urllib2.urlopen** 分别变成了**URL lib . request . URL retrieve**和 **urllib.request.urlopen** 。其余都一样。为了简洁起见，我删除了请求部分。

所以你有它！现在你也可以开始使用 Python 2 或 3 下载文件了！

### 进一步阅读

*   如何使用 Python 通过 HTTP 下载文件？
*   通过网络下载文件[食谱](http://code.activestate.com/recipes/496685-downloading-a-file-from-the-web/)
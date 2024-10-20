# 用 Python 下载加密和压缩文件

> 原文：<https://www.blog.pythonlibrary.org/2010/10/20/downloading-encrypted-and-compressed-files-with-python/>

今年早些时候，我负责创建一个应用程序，使用 Python 从我们组织的网站下载信息。棘手的部分是，它将被加密、压缩，有效载荷将是 JSON。Python 能做到这一切吗？这正是我想知道的。现在是时候让你知道我的发现了。

## Python 和加密

当务之急是找出加密的东西。有效载荷应该是 AES 加密的。虽然 Python 似乎没有为这类事情内置的模块，但有一个用于 Python 2.x 的优秀的 [PyCrypto](http://www.dlitz.net/software/pycrypto/) 包，运行得很好。不幸的是，他们的主网站没有列出如何在 Windows 上安装它。你需要自己做一些编译工作(我想是用 Visual Studio)，或者你可以在这里下载迈克尔·福德的版本。我选择了后者。

以下是我最终使用的基本代码:

```py

from Crypto.Cipher import AES

cipher = AES.new(key, AES.MODE_ECB)
gzipData = cipher.decrypt(encData).strip('\000')

```

**encData** 变量就是使用 urllib2 下载的文件。我们很快就会看到如何做到这一点。耐心点。密钥是由我的一个开发伙伴提供的。无论如何，一旦你解密了它，你就得到 gzipped 数据了。

## 解压缩 Gzipped 文件

关于 gzipped 的文档相当混乱。你用 gzip 还是 zlib？我花了不少时间反复试验才弄明白，主要是因为我的同事给了我错误的文件格式。这一部分实际上也非常容易完成:

```py

import zlib
jsonTxt = zlib.decompress(gzipData)

```

如果你这样做了，你将得到解压缩的数据。是的，就是这么简单。

## JSON 和 Python

从 Python 2.6 开始，Python 中提供了一个 json 模块。你可以在这里阅读[。如果你坚持使用旧版本，那么你可以从](http://docs.python.org/library/json.html) [PyPI](http://pypi.python.org/pypi/python-json) 下载这个模块。或者你可以使用 [simplejson](http://pypi.python.org/pypi/simplejson/) 包，我用的就是这个包。

```py

import simplejson
json = simplejson.loads(jsonTxt )

```

现在，您将拥有一个嵌套字典列表。基本上，你会想做这样的事情来使用它:

```py

data = json['keyName']

```

这将返回另一个包含不同数据的字典。您需要稍微研究一下数据结构，以找出访问所需内容的最佳方式。

## 把所有的放在一起

现在，让我们将它们放在一起，并向您展示完整的脚本:

```py

import simplejson
import urllib2
import zlib
from Crypto.Cipher import AES
from platform import node
from win32api import GetUserName

version = "1.0.4"
uid = GetUserName().upper()
machine = node()

#----------------------------------------------------------------------
def getData(url, key):
    """
    Downloads and decrypts gzipped data and returns a JSON string
    """
    try:
        headers = {"X-ActiveCalls-Version":version,
                   "X-ActiveCalls-User-Windows-user-ID":uid,
                   "X-ActiveCalls-Client-Machine-Name":machine}
        request = urllib2.Request(url, headers=headers)
        f = urllib2.urlopen(request)
        encData = f.read()

        cipher = AES.new(key, AES.MODE_ECB)
        gzipData = cipher.decrypt(encData).strip('\000')

        jsonTxt = zlib.decompress(gzipData)
        return jsonTxt
    except:
        msg = "Error: Program unable to contact update server. Please check configuration URL"
        print msg

if __name__ == "__main__":
    json = getData("some url", "some AES key")

```

在这个特定的例子中，我还需要让服务器知道哪个版本的应用程序正在请求数据，用户是谁，以及请求来自哪台机器。为此，我们使用 urllib2 的 **Request** 方法向服务器传递一个包含该信息的特殊头。代码的其余部分应该是不言自明的

## 包扎

我希望这些都有意义，并且对您的 Python 冒险有所帮助。如果没有，请查看我在各个部分提供的链接，并做一点研究。玩得开心！
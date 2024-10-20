# 使用 Python 解析 MP3 中的 ID3 标签

> 原文：<https://www.blog.pythonlibrary.org/2010/04/22/parsing-id3-tags-from-mp3s-using-python/>

在开发我的 Python [mp3 播放器](https://www.blog.pythonlibrary.org/2010/04/20/wxpython-creating-a-simple-mp3-player/)时，我意识到我需要研究 Python 为解析 ID3 标签提供了什么。有大量的项目，但是大部分看起来不是死了，就是没有文档，或者两者都有。在这篇文章中，你将和我一起探索 Python 中 mp3 标签解析的广阔世界，我们将看看是否能找到一些我可以用来增强我的 MP3 播放器项目的东西。

在本练习中，我们将尝试从解析器中获取以下信息:

*   艺术家
*   相册标题
*   音轨标题
*   轨道长度
*   专辑发行日期

我们可能需要更多的元数据，但这是我在 mp3 播放体验中最关心的东西。我们将查看以下第三方库，看看它们的表现如何:

*   [诱变剂](http://pypi.python.org/pypi/mutagen/1.12)
*   [眼睛 3](http://eyed3.nicfit.net/)
*   [id3reader.py](http://nedbatchelder.com/code/modules/id3reader.html)

我们开始吧！

## 诱变剂能扭转乾坤吗？

在这次围捕中包括诱变剂的原因之一是因为除了 MP3 解析之外，它还支持 ASF，FLAC，M4A，Monkey's Audio，Musepack，Ogg FLAC，Ogg Speex，Ogg Theora，Ogg Vorbis，True Audio，WavPack 和 OptimFROG。因此，我们可以潜在地扩展我们的 MP3 播放器。当我发现这个包裹时，我非常兴奋。然而，虽然该软件包似乎正在积极开发，但文档几乎不存在。如果你是一个新的 Python 程序员，你会发现这个库很难直接使用。

要安装诱变剂，你需要打开它，并使用命令行导航到它的文件夹。然后执行以下命令:

 `python setup.py install` 

你也可以使用 [easy_install](http://pypi.python.org/pypi/setuptools) 或 [pip](http://pypi.python.org/pypi/pip) ，尽管他们的网站上并没有具体说明。现在有趣的部分来了:试图弄清楚如何在没有文档的情况下使用这个模块！幸运的是，我找到了一篇[的博文](http://www.mydigitallife.co.za/index.php?option=com_content&task=view&id=1046123&Itemid=43)，给了我一些线索。从我收集的信息来看，诱变剂非常接近 ID3 规范,所以你实际上是在阅读 ID3 文本框架并使用它们的术语，而不是将其抽象化，这样你就有了类似 GetArtist 的功能。因此，TPE1 =艺术家(或主唱)，TIT2 =标题，等等。让我们看一个例子:

```py

>>> path = r'D:\mp3\12 Stones\2002 - 12 Stones\01 - Crash.mp3'
>>> from mutagen.id3 import ID3
>>> audio = ID3(path)
>>> audio
>>> audio['TPE1']
TPE1(encoding=0, text=[u'12 Stones'])
>>> audio['TPE1'].text
[u'12 Stones']

```

这里有一个更恰当的例子:

```py

from mutagen.id3 import ID3

#----------------------------------------------------------------------
def getMutagenTags(path):
    """"""
    audio = ID3(path)

    print "Artist: %s" % audio['TPE1'].text[0]
    print "Track: %s" % audio["TIT2"].text[0]
    print "Release Year: %s" % audio["TDRC"].text[0]

```

我个人觉得这很难阅读和使用，所以我不会在我的 mp3 播放器上使用这个模块，除非我需要添加额外的数字文件格式。还要注意的是，我不知道如何获取音轨的播放长度或专辑名称。让我们继续下一个 ID3 解析器，看看它的表现如何。

## 眼睛 3

如果你去 eyeD3 的网站，你会发现它似乎不支持 Windows。这是许多用户的一个问题，几乎让我放弃了这个综述。幸运的是，我发现了一个论坛，其中提到了一种使它工作的方法。我们的想法是将主文件夹中的“setup.py.in”文件重命名为“setup.py”，将“__init__.py.in”文件重命名为“__init__”。py”，您可以在“src\eyeD3”中找到它。然后就可以用常用的“python setup.py install”来安装了。一旦你安装了它，它真的很容易使用。检查以下功能:

```py

import eyeD3

#----------------------------------------------------------------------
def getEyeD3Tags(path):
    """"""
    trackInfo = eyeD3.Mp3AudioFile(path)
    tag = trackInfo.getTag()
    tag.link(path)

    print "Artist: %s" % tag.getArtist()
    print "Album: %s" % tag.getAlbum()
    print "Track: %s" % tag.getTitle()
    print "Track Length: %s" % trackInfo.getPlayTimeString()
    print "Release Year: %s" % tag.getYear()

```

这个包确实满足我们任意的要求。该软件包唯一令人遗憾的方面是它缺乏官方的 Windows 支持。我们将保留判断，直到我们尝试了我们的第三种可能性。

## Ned Batchelder 的 id3reader.py

这个模块可能是三个模块中最容易安装的，因为它只是一个文件。你需要做的就是下载它，然后把文件放到站点包或者 Python 路径上的其他地方。这个解析器的主要问题是 Batchelder 不再支持它。让我们看看是否有一种简单的方法来获得我们需要的信息。

```py

import id3reader

#----------------------------------------------------------------------
def getTags(path):
    """"""
    id3r = id3reader.Reader(path)

    print "Artist: %s" % id3r.getValue('performer')
    print "Album: %s" % id3r.getValue('album')
    print "Track: %s" % id3r.getValue('title')
    print "Release Year: %s" % id3r.getValue('year')

```

在不了解 ID3 规格的情况下，我看不出有什么明显的方法可以用这个模块获得走线长度。唉！虽然我喜欢这个模块的简单和强大，但缺乏支持和超级简单的 API 使我拒绝了它，而支持 eyeD3。目前，这将是我的 mp3 播放器的选择库。如果你知道一个很棒的 ID3 解析脚本，请在评论中给我留言。我在谷歌上也看到了其他人的名单，但其中有相当一部分人和巴彻尔德一样已经死了。
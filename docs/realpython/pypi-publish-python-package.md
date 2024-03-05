# 如何将开源 Python 包发布到 PyPI

> 原文：<https://realpython.com/pypi-publish-python-package/>

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**如何将自己的 Python 包发布到 PyPI**](/courses/how-to-publish-your-own-python-package-pypi/)

Python 以自带电池而闻名，[标准库](https://docs.python.org/3/library/)中提供了许多复杂的功能。然而，为了释放这种语言的全部潜力，你也应该利用社区在[PyPI](https://pypi.org/):Python 打包索引**上的贡献。**

PyPI，通常读作 *pie-pee-eye* ，是一个包含几十万个包的仓库。这些范围从琐碎的 [`Hello, World`](https://pypi.org/search/?q=hello+world) 实现到高级的[深度学习库](https://pypi.org/project/keras/)。在本教程中，您将学习如何**将自己的包上传到 PyPI** 。发布项目比以前更容易了。然而，仍然有几个步骤。

**在本教程中，您将学习如何:**

*   **准备**您的 Python 包以供发布
*   处理您的包的版本控制
*   **构建**你的包，**上传**到 PyPI
*   理解并使用不同的**构建系统**

在本教程中，您将使用一个示例项目:一个`reader`包，可以用来在您的控制台中阅读真正的 Python 教程。在深入了解如何发布这个包之前，您将得到一个关于这个项目的快速介绍。点击下面的链接访问包含`reader`完整源代码的 GitHub 库:

**获取源代码:** [单击此处获取您将在本教程中使用的真正的 Python 提要阅读器的源代码](https://realpython.com/bonus/pypi-publish-python-package-source-code/)。

## 了解 Python 打包

对于新手和经验丰富的老手来说，用 Python 打包看起来很复杂。你会在互联网上找到相互矛盾的建议，曾经被认为是好的做法现在可能会被人反对。

造成这种情况的主要原因是 Python 是一种相当古老的编程语言。事实上，Python 的第一个版本是在 1991 年发布的，那时,[万维网](https://en.wikipedia.org/wiki/History_of_the_World_Wide_Web)还没有面向大众。自然，在 Python 的早期版本中，没有包括甚至没有计划一个现代的、基于网络的包分发系统。

相反，Python 的打包生态系统在过去几十年中随着用户需求的明确和技术提供新的可能性而有机地发展。第一个打包支持出现在 2000 年秋天，Python 1.6 和 2.0 中包含了 [`distutils`](https://wiki.python.org/moin/Distutils) 库。Python 打包索引(PyPI) [于 2003 年上线](https://web.archive.org/web/20031217222840/http://www.pypi.org/)，最初是作为现有包的纯索引，没有任何托管能力。

**注:** PyPI 参照巨蟒剧团著名的[奶酪店](https://montypython.fandom.com/wiki/Cheese_Shop_sketch)小品，常被称为**巨蟒奶酪店**。时至今日， [`cheeseshop.python.org`](http://cheeseshop.python.org/) 重定向到 PyPI。

在过去的十年里，许多倡议已经改善了包装景观，把它从蛮荒的西部和一个相当现代和有能力的系统。这主要是通过 [Python 增强提案](https://peps.python.org/) (PEPs)来完成的，这些提案由 [Python 打包管理局](https://www.pypa.io/) (PyPA)工作组审查和实施。

定义 Python 打包如何工作的最重要的文档是以下 pep:

*   PEP 427 描述了**车轮**应该如何包装。
*   PEP 440 描述了如何解析**版本号**。
*   PEP 508 描述了如何指定**依赖关系**。
*   PEP 517 描述了一个**构建后端**应该如何工作。
*   PEP 518 描述了如何指定一个**构建系统**。
*   [PEP 621](https://peps.python.org/pep-0621/) 描述了**项目元数据**应该如何编写。
*   PEP 660 描述了**可编辑安装**应该如何执行。

你不需要研究这些技术文件。在本教程中，您将在发布自己的包的过程中学习所有这些规范在实践中是如何结合在一起的。

为了更好地概述 Python 打包的历史，请查看 [Thomas Kluyver 在 PyCon UK 2019 上的](https://twitter.com/takluyver)演讲: [Python 打包:我们是如何来到这里的，我们要去哪里？您还可以在](https://pyvideo.org/pycon-uk-2019/what-does-pep-517-mean-for-packaging.html) [PyPA](https://www.pypa.io/en/latest/presentations/) 网站上找到更多演示。

[*Remove ads*](/account/join/)

## 创建一个小的 Python 包

在这一节中，您将了解一个小的 Python 包，它可以作为一个发布到 PyPI 的例子。如果你已经有了自己想要发布的包，那么请随意浏览这一部分，并在下一部分再次加入。

你在这里看到的这个包叫做`reader`。它既可以作为一个库来下载您自己代码中的真正 Python 教程，也可以作为一个应用程序来阅读您控制台中的教程。

**注意:**本节中显示和解释的源代码是真正的 Python 提要阅读器的简化版，但功能齐全。与目前在 [PyPI](https://pypi.org/project/realpython-reader/) 上发布的版本相比，这个版本缺少一些错误处理和额外的选项。

首先看一下`reader`的目录结构。这个包完全位于一个可以命名为任何名称的目录中。在本例中，它被命名为`realpython-reader/`。源代码被包装在一个`src/`目录中。这并不是绝对必要的，但通常是个好主意。

**注意:**在构建包时使用额外的`src/`目录已经成为 Python 社区中[讨论的焦点](https://github.com/pypa/packaging.python.org/issues/320)好几年了。一般来说，平面目录结构更容易上手，但是随着项目的增长，`src/`结构提供了几个[优势](https://hynek.me/articles/testing-packaging/)。

内部的`src/reader/`目录包含了你所有的源代码:

```py
realpython-reader/
│
├── src/
│   └── reader/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.toml
│       ├── feed.py
│       └── viewer.py
│
├── tests/
│   ├── test_feed.py
│   └── test_viewer.py
│
├── LICENSE
├── MANIFEST.in
├── README.md
└── pyproject.toml
```

这个包的源代码和一个配置文件在一个`src/`子目录中。在单独的`tests/`子目录中有一些测试。测试本身不会在本教程中讨论，但是稍后你会学到如何处理测试目录。你可以在[的 Python 测试入门](https://realpython.com/python-testing/)和[的 Pytest 有效 Python 测试](https://realpython.com/pytest-python-testing/)中了解更多关于测试的知识。

如果您正在使用自己的包，那么您可以使用不同的结构或者在您的包目录中有其他文件。 [Python 应用布局](https://realpython.com/python-application-layouts/)讨论了几种不同的选项。以下发布到 PyPI 的步骤将独立于您使用的布局。

在本节的剩余部分，您将看到`reader`包是如何工作的。在[的下一节](#prepare-your-package-for-publication)中，您将了解更多关于发布您的包所需的特殊文件，如`LICENSE`、`MANIFEST.in`、`README.md`和`pyproject.toml`。

### 使用真正的 Python 阅读器

`reader`是一个基本的 [web feed](https://en.wikipedia.org/wiki/Web_feed) 阅读器，可以从[真实 Python feed](https://realpython.com/contact/#rss-atom-feed) 下载最新的真实 Python 教程。

在这一节中，您将首先看到几个可以从`reader`得到的输出示例。您还不能自己运行这些示例，但是它们应该会让您对该工具的工作原理有所了解。

**注意:**如果您已经下载了`reader`的源代码，那么您可以先创建一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)，然后在这个虚拟环境中本地安装这个包:

```py
(venv) $ python -m pip install -e .
```

在整个教程中，当您运行这个命令时，您将会了解到更多的事情。

第一个示例使用阅读器获取最新文章的列表:

```py
$ python -m reader
The latest tutorials from Real Python (https://realpython.com/)
 0 How to Publish an Open-Source Python Package to PyPI
 1 The Real Python Podcast – Episode #110
 2 Build a URL Shortener With FastAPI and Python
 3 Using Python Class Constructors
 4 Linear Regression in Python
 5 The Real Python Podcast – Episode #109
 6 pandas GroupBy: Your Guide to Grouping Data in Python
 7 Deploying a Flask Application Using Heroku
 8 Python News: What's New From April 2022
 9 The Real Python Podcast – Episode #108
 10 Top Python Game Engines
 11 Testing Your Code With pytest
 12 Python's min() and max(): Find Smallest and Largest Values
 13 Real Python at PyCon US 2022
 14 Why Is It Important to Close Files in Python?
 15 Combining Data in Pandas With merge(), .join(), and concat()
 16 The Real Python Podcast – Episode #107
 17 Python 3.11 Preview: Task and Exception Groups
 18 Building a Django User Management System
 19 How to Get the Most Out of PyCon US
```

此列表显示了最新的教程，因此您的列表可能与上面看到的有所不同。尽管如此，请注意每篇文章都有编号。要阅读一个特定的教程，可以使用相同的命令，但是也要包括教程的编号。

**注意:**真正的 Python 提要包含有限的文章预览。因此，你将无法用`reader`阅读完整的教程。

在这种情况下，要阅读[如何向 PyPI](https://realpython.com/pypi-publish-python-package/) 发布开源 Python 包，您需要在命令中添加`0`:

```py
$ python -m reader 0
How to Publish an Open-Source Python Package to PyPI

Python is famous for coming with batteries included, and many sophisticated
capabilities are available in the standard library. However, to unlock the
full potential of the language, you should also take advantage of the
community contributions at PyPI: the Python Packaging Index.

PyPI, typically pronounced pie-pee-eye, is a repository containing several
hundred thousand packages. These range from trivial Hello, World
implementations to advanced deep learning libraries. In this tutorial,
you'll learn how to upload your own package to PyPI. Publishing your
project is easier than it used to be. Yet, there are still a few
steps involved.

[...]
```

这将使用[减价](https://www.markdownguide.org/basic-syntax)格式将文章打印到控制台。

**注意:** `python -m`是[用来](https://docs.python.org/3/using/cmdline.html#cmdoption-m)执行一个[模块](https://realpython.com/python-import/#modules)或者一个[包](https://realpython.com/python-import/#packages)。对于模块和常规脚本，它的工作方式类似于`python`。比如`python module.py`和`python -m module`大多是等价的。

当你用`-m`运行一个包时，包内的文件 [`__main__.py`](https://docs.python.org/library/__main__.html) 被执行。更多信息参见[致电读者](#call-the-reader)。

现在，您需要从`src/`目录中运行`python -m reader`命令。[稍后](#install-your-package-locally)，您将学习如何从任何工作目录运行该命令。

通过在命令行上更改数字，您可以阅读任何可用的教程。

[*Remove ads*](/account/join/)

### 理解阅读器代码

对于本教程来说，`reader`如何工作的细节并不重要。但是，如果您有兴趣了解关于实现的更多信息，那么您可以扩展下面的部分。该包包含五个文件:



`config.toml`是一个配置文件，用于指定真实 Python 教程的[提要的 URL。这是一个可以被第三方库](https://realpython.com/atom.xml) [`tomli`](https://pypi.org/project/tomli/) 读取的文本文件:

```py
# config.toml [feed] url  =  "https://realpython.com/atom.xml"
```

一般来说，TOML 文件包含被分成几个部分或表格的键值对。这个特定的文件只包含一个区段`feed`，以及一个关键字`url`。

**注意:**对于这个简单的包，一个配置文件可能是多余的。您可以直接在源代码中将 URL 定义为模块级常量。这里包含的配置文件演示了如何使用非代码文件。

[TOML](https://toml.io/) 是一种最近流行起来的配置文件格式。Python 将它用于`pyproject.toml`文件，稍后您将了解到[和](#configure-your-package)。要深入探究 TOML，请查看 [Python 和 TOML:新的最好的朋友](https://realpython.com/python-toml)。

对读取 TOML 文件的支持将[添加](https://realpython.com/python311-tomllib)到 Python 3.11 的标准库中，新增 [`tomllib`](https://docs.python.org/3.11/library/tomllib.html) 库。在此之前，可以使用第三方 [`tomli`](https://pypi.org/project/tomli/) 包。



您将看到的第一个源代码文件是`__main__.py`。双下划线表示这个文件在 Python 中有一个特殊的含义。事实上，当像前面一样用`python -m`执行一个包时，Python 会运行`__main__.py`的内容。

换句话说，`__main__.py`充当程序的入口点，负责主流程，根据需要调用其他部分:

```py
 1# __main__.py
 2
 3import sys
 4
 5from reader import feed, viewer
 6
 7def main():
 8    """Read the Real Python article feed"""
 9
10    # If an article ID is given, then show the article
11    if len(sys.argv) > 1:
12        article = feed.get_article(sys.argv[1])
13        viewer.show(article)
14
15    # If no ID is given, then show a list of all articles
16    else:
17        site = feed.get_site()
18        titles = feed.get_titles()
19        viewer.show_list(site, titles)
20
21if __name__ == "__main__":
22    main()
```

注意最后一行调用了`main()`。如果不调用`main()`，那么程序什么都不会做。正如您之前看到的，该程序可以列出所有教程，也可以打印一个特定的教程。这由第 11 至 19 行的`if` … `else`模块处理。



下一个文件是`__init__.py`。同样，文件名中的双下划线告诉您这是一个特殊的文件。`__init__.py`代表你的包的根。它通常应该非常简单，但是它是放置包常量、文档等的好地方:

```py
# __init__.py

from importlib import resources
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Version of the realpython-reader package
__version__ = "1.0.0"

# Read URL of the Real Python feed from config file
_cfg = tomllib.loads(resources.read_text("reader", "config.toml"))
URL = _cfg["feed"]["url"]
```

特殊变量`__version__`是 Python 中向包中添加版本号的约定。它是在 [PEP 396](https://peps.python.org/pep-0396/) 中引入的。稍后你会学到更多关于版本控制的知识。

**注意:**您可以使用 [`importlib.metadata`](https://realpython.com/python38-new-features/#importlibmetadata) 来检查您已经安装的任何软件包的版本:

>>>

```py
>>> from importlib import metadata
>>> metadata.version("realpython-reader")
'1.0.0'
```

`importlib.metadata`通过包的 PyPI 名来识别包，这就是为什么你要查找`realpython-reader`而不是`reader`。稍后你会学到更多关于你的项目的不同名字[。](#name-your-package)

严格地说，没有必要指定`__version__`。`importlib.metadata`机制使用项目元数据来查找版本号。然而，这仍然是一个有用的约定，像 Setuptools 和 Flit 这样的工具可以使用它来自动更新您的包元数据。

要从配置文件中读取提要的 URL，可以使用`tomllib`或`tomli`来获得 TOML 支持。如上所述使用`try` … `except`可以确保在`tomllib`可用时使用它，如果不可用时[退回到](https://github.com/hukkin/tomli#building-a-tomlitomllib-compatibility-layer)到`tomli`。

[`importlib.resources`](https://docs.python.org/library/importlib.html#module-importlib.resources) 用于从一个包中导入非代码或资源文件，而不必找出它们的完整文件路径。当您将您的包发布到 PyPI 并且不能完全控制包的安装位置和使用方式时，这尤其有用。资源文件甚至可能在 zip 存档中结束[。](https://realpython.com/python-zip-import/)

`importlib.resources`在 [Python 3.7](https://realpython.com/python37-new-features/#importing-data-files-with-importlibresources) 中成为 Python 标准库的一部分。更多信息，请参见 [Barry Warsaw 在 2018 年](https://www.youtube.com/watch?v=ZsGFU2qh73E)PyCon 上的演讲。

在`__init__.py`中定义的变量成为包名称空间中的变量:

>>>

```py
>>> import reader
>>> reader.__version__
'1.0.0'

>>> reader.URL
'https://realpython.com/atom.xml'
```

您可以直接在`reader`上访问作为属性的包常量。



在`__main__.py`中，您导入两个模块`feed`和`viewer`，并使用它们从提要中读取并显示结果。这些模块完成了大部分实际工作。

首先考虑`feed.py`。这个文件包含了从一个 web 提要中读取并解析结果的函数。幸运的是，已经有很好的库可以做到这一点。`feed.py`依赖于 PyPI 上已经有的两个模块: [`feedparser`](https://pypi.org/project/feedparser/) 和 [`html2text`](https://pypi.org/project/html2text/) 。

`feed.py`由几个功能组成。你会一次看一个。

阅读网络订阅源可能需要一些时间。为了避免从 web 提要中读取不必要的内容，可以在提要第一次被读取时使用`@cache`到[缓存](https://realpython.com/lru-cache-python/):

```py
# feed.py

from functools import cache

import feedparser
import html2text

import reader

@cache
def _get_feed(url=reader.URL):
    """Read the web feed, use caching to only read it once"""
    return feedparser.parse(url)
```

从网上读取一个 feed，并以一种看起来像字典的结构返回。为了避免多次下载提要，函数[用`@cache`修饰了](https://realpython.com/primer-on-python-decorators/)，它会记住`_get_feed()`返回的值，并在以后的调用中重用它。

**注:**`@cache`装饰器是在 [Python 3.9](https://realpython.com/python39-new-features/) 中引入到`functools`的。在老版本的 Python 上，可以用 [`@lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache) 代替。

您在`_get_feed()`前面加了一个下划线，表示它是一个支持函数，并不意味着您的软件包的用户可以直接使用它。

通过查看`.feed`元数据，您可以获得关于提要的一些基本信息。下面的函数挑选出包含提要的网站的标题和链接:

```py
# feed.py

# ...

def get_site(url=reader.URL):
    """Get name and link to website of the feed"""
    feed = _get_feed(url).feed
    return f"{feed.title} ({feed.link})"
```

除了`.title`和`.link`之外，[属性](https://realpython.com/python-property/)像`.subtitle`、`.updated`、`.id`也可以。

提要中可用的文章可以在`.entries`列表中找到。文章标题可以通过[列表理解](https://realpython.com/list-comprehension-python/)找到:

```py
# feed.py

# ...

def get_titles(url=reader.URL):
    """List titles in the feed"""
    articles = _get_feed(url).entries
    return [a.title for a in articles]
```

`.entries`按时间顺序列出提要中的文章，因此最新的文章是`.entries[0]`。

为了获得一篇特定文章的内容，您使用它在`.entries`中的索引作为文章 ID:

```py
# feed.py

# ...

def get_article(article_id, url=reader.URL):
    """Get article from feed with the given ID"""
    articles = _get_feed(url).entries
    article = articles[int(article_id)]
    html = article.content[0].value
    text = html2text.html2text(html)
    return f"# {article.title}\n\n{text}"
```

从`.entries`中选择正确的文章后，您找到 HTML 格式的文章文本，并将其存储为`article`。接下来，`html2text`出色地将 HTML 翻译成可读性更好的降价文本。HTML 不包含文章标题，所以标题是在文章文本返回之前添加的。



最后的模块是`viewer.py`。它由两个小函数组成。实际上，你可以在`__main__.py`中直接使用`print()`，而不是调用`viewer`函数。然而，将功能分离出来使得以后用更高级的东西替换它变得更加简单。

这里，您使用简单的 [`print()`](https://realpython.com/python-print/) 语句向用户显示内容。作为一种改进，也许你想给你的阅读器添加[更丰富的格式](https://realpython.com/podcasts/rpp/80/)或 [GUI 界面](https://realpython.com/python-pyqt-gui-calculator/)。为此，您只需替换以下两个函数:

```py
# viewer.py

def show(article):
    """Show one article"""
    print(article)

def show_list(site, titles):
    """Show list of article titles"""
    print(f"The latest tutorials from {site}")
    for article_id, title in enumerate(titles):
        print(f"{article_id:>3}  {title}")
```

`show()`打印一篇文章到控制台，而`show_list()`打印标题列表。后者还创建文章 id，在选择阅读一篇特定文章时使用。

除了这些源代码文件，您还需要添加一些特殊的文件，然后才能发布您的包。您将在后面的章节中讨论这些文件。

### 召唤读者

当你的项目越来越复杂时，一个挑战就是让你的用户知道他们如何使用你的项目。由于`reader`由四个不同的源代码文件组成，用户如何知道执行哪个文件才能使用应用程序？

**注意:**单个 Python 文件通常被称为**脚本**或[T5】模块](https://realpython.com/python-import/#modules) 。你可以把一个 [**包**](https://realpython.com/python-import/#packages) 看成是模块的集合。

最常见的是，你通过提供文件名来运行 Python 脚本。例如，如果您有一个名为`hello.py`的脚本，那么您可以如下运行它:

```py
$ python hello.py
Hi there!
```

当您运行这个假想的脚本时，它会将`Hi there!`打印到您的控制台上。同样，您可以使用`python`解释程序的`-m`选项，通过指定其模块名而不是文件名来运行脚本:

```py
$ python -m hello
Hi there!
```

对于当前目录中的模块，模块名与文件名相同，只是省略了后缀`.py`。

使用`-m`的一个好处是它允许你调用所有在你的 [Python 路径](https://realpython.com/python-import/#pythons-import-path)中的模块，包括那些内置在 Python 中的模块。一个例子是称 [`antigravity`](http://python-history.blogspot.com/2010/06/import-antigravity.html) :

```py
$ python -m antigravity
Created new window in existing browser session.
```

如果你想运行一个没有`-m`的内置模块，那么你需要首先查找它在你的系统中的存储位置，然后用它的完整路径调用它。

使用`-m`的另一个优点是它既适用于模块也适用于包。正如您前面所学的，只要在您的工作目录中有`reader/`目录，您就可以用`-m`调用`reader`包:

```py
$ cd src/
$ python -m reader
```

因为`reader`是一个包，所以名字只指一个目录。Python 如何决定运行该目录中的哪个代码？它寻找一个名为`__main__.py`的文件。如果这样的文件存在，那么它被执行。如果它不存在，则会打印一条错误消息:

```py
$ python -m urllib
python: No module named urllib.__main__; 'urllib' is a package and
 cannot be directly executed
```

错误信息显示标准库的 [`urllib`包](https://realpython.com/urllib-request/)没有定义`__main__.py`文件。

如果你正在创建一个应该被执行的包，那么你应该包含一个`__main__.py`文件。你也可以效仿里奇的好例子，用`python -m rich`向[展示](https://pypi.org/project/rich/)你的软件包的功能。

[稍后](#configure-your-package)，您将看到如何为您的包创建**入口点**，其行为类似于常规命令行程序。这些对于您的最终用户来说会更容易使用。

[*Remove ads*](/account/join/)

## 准备您的软件包以供发布

您有一个想要发布的包。可能你抄袭了`reader`，也可能你有自己的包。在这一节中，您将看到在将您的包上传到 PyPI 之前需要采取哪些步骤。

### 将您的包命名为

第一步——可能也是最难的一步——是为你的包取一个好名字。PyPI 上的所有包都需要有唯一的名称。现在 PyPI 上有几十万个包，所以你最喜欢的名字可能已经被使用了。

举个例子，PyPI 上已经有一个名为`reader`的包。使包名唯一的一种方法是在名字前添加一个可识别的前缀。在这个例子中，您将使用`realpython-reader`作为`reader`包的 PyPI 名称。

无论您为您的包选择哪个 PyPI 名称，这都是您在使用`pip`安装它时将使用的名称:

```py
$ python -m pip install realpython-reader
```

请注意，PyPI 名称不需要与包名称匹配。这里，包仍然被命名为`reader`，这是您在导入包时需要使用的名称:

>>>

```py
>>> import reader
>>> reader.__version__
'1.0.0'

>>> from reader import feed
>>> feed.get_titles()
['How to Publish an Open-Source Python Package to PyPI', ...]
```

有时你需要为你的包使用不同的名字。但是，如果包名和 PyPI 名相同，那么对用户来说事情就简单多了。

请注意，尽管软件包名称不需要像 PyPI 名称那样是全局唯一的，但是它需要在您运行它的环境中是唯一的。

如果你用相同的包名安装了两个包，例如`reader`和`realpython-reader`，那么像`import reader`这样的声明是有歧义的。Python 通过导入它在导入路径中首先找到的包来解决这个问题。通常，这将是按字母顺序排列名称时的第一个包。但是，你不应该依赖这种行为。

通常，您会希望您的包名尽可能地唯一，同时用一个简短而简洁的名称来平衡这一点。`realpython-reader`是一个专门的提要阅读器，而 PyPI 上的`reader`则更加通用。出于本教程的目的，没有理由两者都需要，所以使用非唯一名称的折衷可能是值得的。

### 配置您的软件包

为了准备在 PyPI 上发布您的包，您需要提供一些关于它的信息。通常，您需要指定两种信息:

1.  构建系统的配置
2.  包的配置

一个**构建系统**负责创建你将上传到 PyPI 的实际文件，通常是以[轮](https://realpython.com/python-wheels/)或[源代码分发(sdist)](https://packaging.python.org/en/latest/specifications/source-distribution-format/) 格式。很长一段时间，这都是由 [`distutils`](https://docs.python.org/3.11/library/distutils.html) 或者 [`setuptools`](https://setuptools.pypa.io/) 来完成的。然而， [PEP 517](https://peps.python.org/pep-0517/) 和 [PEP 518](https://peps.python.org/pep-0518/) 引入了一种指定定制构建系统的方法。

注意:您可以选择在您的项目中使用哪个构建系统。不同构建系统之间的主要区别在于如何配置包，以及运行哪些命令来构建和上传包。

本教程将重点介绍如何使用`setuptools`作为构建系统。不过，[稍后](#explore-other-build-systems)你会学到如何使用 Flit 和诗歌这样的替代品。

每个 Python 项目都应该使用一个名为`pyproject.toml`的文件来指定它的构建系统。您可以通过将后面的[添加到`pyproject.toml`来使用`setuptools`:](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#basic-use)

```py
# pyproject.toml [build-system] requires  =  ["setuptools>=61.0.0",  "wheel"] build-backend  =  "setuptools.build_meta"
```

这指定了您使用`setuptools`作为构建系统，以及 Python 必须安装哪些依赖项来构建您的包。通常，你选择的构建系统的[文档](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)会告诉你如何在`pyproject.toml`中写`build-system`表。

您需要提供的更有趣的信息与您的包本身有关。 [PEP 621](https://peps.python.org/pep-0621/) 定义了关于你的包的元数据如何被包含在`pyproject.toml`中，在不同的构建系统中以一种尽可能统一的方式。

**注意:**历史上，Setuptools 使用 [`setup.py`](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#setup-py) 来配置你的包。因为这是一个在安装时运行的实际 Python 脚本，所以它非常强大，并且在构建复杂的包时可能仍然需要它。

然而，使用声明性配置文件来表达如何构建您的包通常更好，因为它更容易推理，并且需要担心的缺陷更少。使用 [`setup.cfg`](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html) 是配置 Setuptools 最常见的方式。

然而，Setuptools 是按照 PEP 621 的规定，将移向使用 [`pyproject.toml`](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) 。在本教程中，您将对所有的包配置使用`pyproject.toml`。

`reader`包的一个相当简单的配置如下所示:

```py
# pyproject.toml [build-system] requires  =  ["setuptools>=61.0.0",  "wheel"] build-backend  =  "setuptools.build_meta" [project] name  =  "realpython-reader" version  =  "1.0.0" description  =  "Read the latest Real Python tutorials" readme  =  "README.md" authors  =  [{  name  =  "Real Python",  email  =  "info@realpython.com"  }] license  =  {  file  =  "LICENSE"  } classifiers  =  [ "License :: OSI Approved :: MIT License", "Programming Language :: Python", "Programming Language :: Python :: 3", ] keywords  =  ["feed",  "reader",  "tutorial"] dependencies  =  [ "feedparser >= 5.2.0", "html2text", 'tomli; python_version < "3.11"', ] requires-python  =  ">=3.9" [project.optional-dependencies] dev  =  ["black",  "bumpver",  "isort",  "pip-tools",  "pytest"] [project.urls] Homepage  =  "https://github.com/realpython/reader" [project.scripts] realpython  =  "reader.__main__:main"
```

这些信息大部分是可选的，还有其他设置可以使用，但没有包括在本例中。查看[文档](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)了解所有细节。

您必须在`pyproject.toml`中包含的最基本信息如下:

*   **`name`** 指定将出现在 PyPI 上的包的名称。
*   **`version`** 设置你的包的当前版本。

如上例所示，您可以包含更多的信息。`pyproject.toml`中的其他几个键解释如下:

*   **`classifiers`** 使用一列[分类器](https://pypi.org/classifiers/)描述你的项目。你应该使用它们，因为它们使你的项目更容易被搜索到。
*   **`dependencies`** 列出您的包对第三方库的所有[依赖关系](https://realpython.com/what-is-pip/)。`reader`取决于`feedparser`、`html2text`、`tomli`，所以在此列出。
*   **`project.urls`** 添加链接，您可以使用这些链接向用户展示有关您的软件包的附加信息。你可以在这里包括几个链接。
*   **`project.scripts`** 创建命令行脚本来调用你的包中的函数。这里，新的`realpython`命令调用`reader.__main__`模块中的`main()`。

`project.scripts`表是可以处理[入口点](https://packaging.python.org/en/latest/specifications/entry-points/)的三个表之一。还可以[包含](https://peps.python.org/pep-0621/#entry-points) `project.gui-scripts`和`project.entry-points`，分别指定 GUI 应用和[插件](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)。

所有这些信息的目的是让你的包在 PyPI 上更有吸引力，更容易被找到。查看 PyPI 上的`realpython-reader` [项目页面](https://pypi.org/project/realpython-reader/)，并与上面的`pyproject.toml`信息进行比较:

[![Information about the `realpython-reader` package at PyPI](img/69b0716be245b72bb6e3cb81fbd403cc.png)](https://files.realpython.com/media/pypi_realpython-reader-1.0.0.0dfba15c7278.png)

PyPI 上的所有信息都来自`pyproject.toml`和`README.md`。比如版本号是以`project.toml`中的台词`version = "1.0.0"`为准，而*阅读最新的真正 Python 教程*则是抄袭`description`。

此外，项目描述是从您的`README.md`文件中提取的。在侧边栏中，您可以找到来自*项目链接*部分的`project.urls`以及来自*元*部分的`license`和`authors`的信息。您在`classifiers`中指定的值可以在侧边栏的底部看到。

有关所有按键的详细信息，请参见 [PEP 621](https://peps.python.org/pep-0621/#specification) 。在下一小节中，您将了解更多关于`dependencies`和`project.optional-dependencies`的内容。

[*Remove ads*](/account/join/)

### 指定您的软件包依赖关系

您的包可能依赖于不属于标准库的第三方库。您应该在`pyproject.toml`的`dependencies`列表中指定这些。在上例中，您执行了以下操作:

```py
dependencies  =  [ "feedparser >= 5.2.0", "html2text", 'tomli; python_version < "3.11"', ]
```

这指定了`reader`依赖于`feedparser`、`html2text`和`tomli`。此外，它说:

*   **`feedparser`** 必须是 5.2.0 或更高版本。
*   **`html2text`** 可以是任何版本。
*   **`tomli`** 可以是任何版本，但仅在 Python 3.10 或更早版本上是必需的。

这展示了在指定依赖项时可以使用的几种可能性，包括[版本说明符](https://peps.python.org/pep-0508/#specification)和[环境标记](https://peps.python.org/pep-0508/#environment-markers)。您可以使用后者来说明不同的操作系统、Python 版本等等。

但是，请注意，您应该努力只指定库或应用程序工作所需的最低要求。这个列表将被`pip`用来在任何时候安装你的包时解析依赖关系。通过保持这个列表的最小化，您可以确保您的包尽可能地兼容。

你可能听说过你应该**固定你的依赖关系**。那就是[伟大的建议](https://realpython.com/python-virtual-environments-a-primer/#pin-your-dependencies)！然而，它在这种情况下不成立。您固定您的依赖项，以确保您的环境是可复制的。另一方面，您的包应该有望跨许多不同的 Python 环境工作。

向`dependencies`添加包时，您应该遵循这些[经验法则](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/):

*   只列出你的直接依赖。例如，`reader`导入了`feedparser`、`html2text`和`tomli`，所以这些都列出来了。另一方面，`feedparser`依赖于 [`sgmllib3k`](https://pypi.org/project/sgmllib3k/) ，但是`reader`没有直接使用这个库，所以没有指定。
*   永远不要用`==`把你的依赖固定在一个特定的版本上。
*   如果您依赖于在您的依赖关系的特定版本中添加的功能，使用`>=`来添加一个下限。
*   如果您担心在主要版本升级中依赖关系可能会破坏兼容性，请使用`<`添加上限。在这种情况下，您应该努力测试这样的升级，并在可能的情况下移除或增加上限。

请注意，当您配置一个可供他人使用的包时，这些规则也适用。如果您正在部署您的包，那么您应该将您的依赖项固定在一个[虚拟环境](https://realpython.com/python-virtual-environments-a-primer/)中。

[pip-tools](https://pypi.org/project/pip-tools/) 项目是管理固定依赖关系的好方法。它附带了一个`pip-compile`命令，可以创建或更新依赖项的完整列表。

例如，假设您正在将`reader`部署到虚拟环境中。然后，您可以使用 pip-tools 创建一个可复制的环境。事实上，`pip-compile`可以直接与你的`pyproject.toml`文件一起工作:

```py
(venv) $ python -m pip install pip-tools
(venv) $ pip-compile pyproject.toml
feedparser==6.0.8
 via realpython-reader (pyproject.toml)
html2text==2020.1.16
 via realpython-reader (pyproject.toml)
sgmllib3k==1.0.0
 via feedparser
tomli==2.0.1 ; python_version < "3.11"
 via realpython-reader (pyproject.toml)
```

`pip-compile`创建一个详细的`requirements.txt`文件，其内容类似于上面的输出。您可以使用`pip install`或`pip-sync`将这些依赖项安装到您的环境中:

```py
(venv) $ pip-sync
Collecting feedparser==6.0.8
 ...
Installing collected packages: sgmllib3k, tomli, html2text, feedparser
Successfully installed feedparser-6.0.8 html2text-2020.1.16 sgmllib3k-1.0.0
 tomli-2.0.1
```

更多信息参见 [pip-tools 文档](https://pip-tools.readthedocs.io/)。

您还可以在一个名为`project.optional-dependencies`的单独的表中指定软件包的可选依赖项。您经常使用它来指定您在开发或测试过程中使用的依赖项。但是，您也可以指定用于支持软件包中某些功能的额外依赖项。

在上面的示例中，您包括了以下部分:

```py
[project.optional-dependencies] dev  =  ["black",  "bumpver",  "isort",  "pip-tools",  "pytest"]
```

这将添加一组可选依赖项`dev`。您可以有几个这样的组，您可以根据需要命名这些组。

默认情况下，安装软件包时不包括可选的依赖项。但是，通过在运行`pip`时在方括号中添加组名，您可以手动指定应该安装它们。例如，您可以通过执行以下操作来安装`reader`的额外`dev`依赖项:

```py
(venv) $ python -m pip install realpython-reader[dev]
```

通过使用`--extra`命令行选项，您还可以在使用`pip-compile`固定您的依赖关系时包含可选的依赖关系:

```py
(venv) $ pip-compile --extra dev pyproject.toml
attrs==21.4.0
 via pytest
black==22.3.0
 via realpython-reader (pyproject.toml)
...
tomli==2.0.1 ; python_version < "3.11"
 via
 black
 pytest
 realpython-reader (pyproject.toml)
```

这将创建一个包含常规依赖项和开发依赖项的固定的`requirements.txt`文件。

[*Remove ads*](/account/join/)

### 记录您的包

你应该在向全世界发布你的软件包之前添加一些[文档](https://realpython.com/documenting-python-code/)。根据您的项目，您的文档可以小到单个`README`文件，也可以大到包含教程、示例库和 API 参考的完整网页。

至少，您应该在项目中包含一个`README`文件。一个好的 [`README`](https://readme.so/) 应该快速描述你的项目，以及解释如何安装和使用你的软件包。通常，你想在`pyproject.toml`的`readme`键中引用你的`README`。这也将在 PyPI 项目页面上显示信息。

您可以使用 [Markdown](https://www.markdownguide.org/basic-syntax) 或 [reStructuredText](http://docutils.sourceforge.net/rst.html) 作为项目描述的格式。PyPI [根据文件扩展名判断出](https://peps.python.org/pep-0621/#readme)你使用的是哪种格式。如果您不需要 reStructuredText 的任何高级特性，那么您通常最好对您的`README`使用 Markdown。它更简单，在 PyPI 之外有更广泛的支持。

对于较大的项目，您可能希望提供比单个文件所能容纳的更多的文档。在这种情况下，你可以把你的文档放在类似于 [GitHub](https://github.com/) 或[的网站上，阅读文档](https://readthedocs.org/)并从 PyPI 项目页面链接到它。

您可以通过在`pyproject.toml`中的`project.urls`表中指定链接到其他 URL。在示例中，URL 部分用于链接到`reader` GitHub 存储库。

### 测试您的软件包

当你开发你的包时，测试是有用的，你应该包括它们。如前所述，本教程中不会涉及测试，但是您可以看看`tests/`源代码目录中`reader`的测试。

你可以在[用 Pytest](https://realpython.com/pytest-python-testing/) 进行有效的 Python 测试中了解更多关于测试的知识，在[用测试驱动开发(TDD)构建 Python 中的哈希表](https://realpython.com/python-hash-table/)和 [Python 练习题:解析 CSV 文件](https://realpython.com/python-interview-problem-parsing-csv-files/)中获得一些测试驱动开发(TDD)的实践经验。

在准备要出版的包时，您应该意识到测试所扮演的角色。它们通常只对开发人员感兴趣，所以它们应该*而不是*包含在您通过 PyPI 发布的包中。

Setuptools 的更高版本在[代码发现](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#automatic-discovery)方面非常出色，通常会将您的源代码包含在软件包发行版中，但会省略您的测试、文档和类似的开发工件。

您可以通过使用`pyproject.toml`中的`find`指令来精确控制您的包中包含的内容。更多信息参见[设置工具文档](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#custom-discovery)。

### 版本化您的软件包

您的包需要有一个版本。此外，PyPI 只允许你上传一次特定版本的包。换句话说，如果你想在 PyPI 上更新你的包，那么你首先需要增加版本号。这是一件好事，因为它有助于保证可再现性:具有相同版本的给定包的两个环境应该表现相同。

您可以使用许多不同的[版本控制方案](https://en.wikipedia.org/wiki/Software_versioning)。对于 Python 项目， [PEP 440](https://peps.python.org/pep-0440/) 给出了一些建议。但是，为了灵活起见，PEP 中的描述比较复杂。对于一个简单的项目，您应该坚持使用简单的版本控制方案。

[语义版本化](https://semver.org/)是一个很好的默认使用方案，尽管它[并不完美](https://hynek.me/articles/semver-will-not-save-you/)。您将版本指定为三个数字部分，例如`1.2.3`。这些组件分别称为主要组件、次要组件和修补组件。以下是关于何时增加每个组件的建议:

> *   When making incompatible API changes, add the major version.
> *   When adding features in a backward compatible way, add minor versions.
> *   Add patch version when making backward compatible bug fixes. ( [source](https://semver.org/#summary) )

当你增加 MINOR 时，你应该重置 PATCH 到`0`，当你增加 MAJOR 时，重置 PATCH 和 MINOR 到`0`。

[日历版本化](https://calver.org/)是语义版本化的一种替代方案，越来越受欢迎，被像 [Ubuntu](http://www.ubuntu.com/) 、 [Twisted](https://twistedmatrix.com/) 、 [Black](https://black.readthedocs.io/) 和 [`pip`](https://pip.pypa.io/) 这样的项目所使用。日历版本也由几个数字组成，但其中一个或几个与当前的年、月或周相关联。

通常，您希望在项目的不同文件中指定版本号。例如，`reader`包中的`pyproject.toml`和`reader/__init__.py`都提到了版本号。为了帮助你确保版本号保持一致，你可以使用像 [BumpVer](https://pypi.org/project/bumpver/) 这样的工具。

BumpVer 允许你将版本号直接写入你的文件，然后根据需要更新这些版本号。例如，您可以安装 BumpVer 并将其集成到您的项目中，如下所示:

```py
(venv) $ python -m pip install bumpver
(venv) $ bumpver init
WARNING - Couldn't parse pyproject.toml: Missing version_pattern
Updated pyproject.toml
```

`bumpver init`命令在您的`pyproject.toml`中创建一个部分，允许您为您的项目配置工具。根据您的需要，您可能需要更改许多默认设置。对于`reader`，你可能会得到如下结果:

```py
[tool.bumpver] current_version  =  "1.0.0" version_pattern  =  "MAJOR.MINOR.PATCH" commit_message  =  "Bump version {old_version} -> {new_version}" commit  =  true tag  =  true push  =  false [tool.bumpver.file_patterns] "pyproject.toml"  =  ['current_version = "{version}"',  'version = "{version}"'] "src/reader/__init__.py"  =  ["{version}"]
```

为了让 BumpVer 正常工作，您必须在`file_patterns`小节中指定包含您的版本号的所有文件。请注意，BumpVer 与 Git 配合得很好，可以在更新版本号时自动提交、标记和推送。

**注:** BumpVer 与您的[版本控制系统](https://realpython.com/python-git-github-intro/)集成。如果你的存储库中有未提交的变更，它会拒绝更新你的文件。

设置好配置后，您可以用一个命令在所有文件中添加版本。例如，要增加`reader`的次要版本，您需要执行以下操作:

```py
(venv) $ bumpver update --minor
INFO    - Old Version: 1.0.0
INFO    - New Version: 1.1.0
```

这将把`pyproject.toml`和`__init__.py`中的版本号从 1.0.0 更改为 1.1.0。您可以使用`--dry`标志来查看 BumpVer 会做出哪些更改，而无需实际执行它们。

[*Remove ads*](/account/join/)

### 将资源文件添加到您的包中

有时，您的包中会有一些不是源代码文件的文件。示例包括数据文件、二进制文件、文档，以及(如本例中所示的)配置文件。

为了确保在生成项目时包含这些文件，可以使用清单文件。对于许多项目，您不需要担心清单:默认情况下，Setuptools 在构建中包含所有源代码文件和`README`文件。

如果您有其他资源文件并且需要更新清单，那么您需要在项目的基本目录中的`pyproject.toml`旁边创建一个名为`MANIFEST.in`的文件。该文件指定包括和排除哪些文件的规则:

```py
# MANIFEST.in

include src/reader/*.toml
```

这个例子将包括`src/reader/`目录中的所有`.toml`文件。实际上，这是配置文件。

参见[文档](https://packaging.python.org/en/latest/guides/using-manifest-in/)获取更多关于设置你的清单的信息。[检查清单](https://pypi.org/project/check-manifest/)工具对于使用`MANIFEST.in`也很有用。

### 许可您的软件包

如果您正在与他人共享您的软件包，那么您需要在您的软件包中添加一个许可证，解释如何允许他人使用您的软件包。例如，`reader`是根据 [MIT license](https://mit-license.org/) 发布的。

许可证是法律文件，你通常不想写你自己的。相反，你应该[从](https://choosealicense.com/)[众多](https://choosealicense.com/appendix/)可用的执照中选择一个。

您应该将一个名为`LICENSE`的文件添加到您的项目中，该文件包含您选择的许可文本。然后您可以在`pyproject.toml`中引用这个文件，使许可证在 PyPI 上可见。

### 在本地安装您的软件包

您已经为您的包完成了所有必要的设置和配置。在下一节的[中，您将了解如何最终在 PyPI 上获得您的包裹。不过，首先，您将了解**可编辑安装**。这是一种使用`pip`在本地安装你的包的方式，让你在安装后编辑你的代码。](#publish-your-package-to-pypi)

**注意:**正常情况下，`pip`会做一个**常规安装**，将一个包放到你的 [`site-packages/`文件夹](https://realpython.com/python-virtual-environments-a-primer/#why-do-you-need-virtual-environments)中。如果你安装你的本地项目，那么源代码将被复制到`site-packages/`。这样做的结果是，您以后所做的更改将不会生效。您需要首先重新安装您的软件包。

在开发过程中，这可能既无效又令人沮丧。可编辑安装通过直接链接到您的源代码来解决这个问题。

可编辑安装已经在 [PEP 660](https://peps.python.org/pep-0660/) 中正式化。这些在您开发软件包时非常有用，因为您可以测试软件包的所有功能并更新源代码，而无需重新安装。

通过添加`-e`或`--editable`标志，您可以使用`pip`在可编辑模式下安装您的软件包:

```py
(venv) $ python -m pip install -e .
```

注意命令末尾的句点(`.`)。这是该命令的必要部分，它告诉`pip`您想要安装位于当前工作目录中的软件包。一般来说，这应该是包含您的`pyproject.toml`文件的目录的路径。

**注意:**您可能会得到一条错误消息，说*“项目文件有一个‘py Project . toml’并且它的构建后端缺少‘build _ editable’钩子。”*这是由于 PEP 660 的 Setuptools 支持中的[限制](https://github.com/pypa/setuptools/issues/2816)。您可以通过添加包含以下内容的名为`setup.py`的文件来解决这个问题:

```py
# setup.py

from setuptools import setup

setup()
```

该填充程序将可编辑安装的工作委托给 Setuptools 的遗留机制，直到对 PEP 660 的本机支持可用。

一旦成功安装了项目，它就可以在您的环境中使用，独立于您的当前目录。此外，您的脚本已经设置好，因此您可以运行它们。回想一下，`reader`定义了一个名为`realpython`的脚本:

```py
(venv) $ realpython
The latest tutorials from Real Python (https://realpython.com/)
 0 How to Publish an Open-Source Python Package to PyPI
 [...]
```

您也可以从任何目录使用`python -m reader`,或者从 REPL 或另一个脚本导入您的包:

>>>

```py
>>> from reader import feed
>>> feed.get_titles()
['How to Publish an Open-Source Python Package to PyPI', ...]
```

在开发过程中以可编辑模式安装您的包会让您的开发体验更加愉快。这也是定位某些 bug 的好方法，在这些 bug 中，您可能不自觉地依赖于当前工作目录中可用的文件。

这需要一些时间，但是这已经完成了您需要为您的包做的准备工作。在下一节中，您将学习如何实际发布它！

[*Remove ads*](/account/join/)

## 将您的包发布到 PyPI

您的软件包终于准备好迎接计算机外部的世界了！在本节中，您将学习如何构建您的包并将其上传到 PyPI。

如果你还没有 PyPI 账户，那么现在是时候[在 PyPI](https://pypi.org/account/register/) 上注册你的账户了。同时，你也应该[在 TestPyPI](https://test.pypi.org/account/register/) 上注册一个账户。TestPyPI 很有用！如果你搞砸了，你可以尝试发布一个包的所有步骤而不会有任何后果。

要构建您的包并上传到 PyPI，您将使用两个工具，称为[构建](https://pypa-build.readthedocs.io/)和[缠绕](https://twine.readthedocs.io)。您可以像往常一样使用`pip`安装它们:

```py
(venv) $ python -m pip install build twine
```

在接下来的小节中，您将学习如何使用这些工具。

### 构建您的软件包

PyPI 上的包不是以普通源代码的形式发布的。相反，它们被打包成分发包。发行包最常见的格式是源文件和 [Python wheels](https://realpython.com/python-wheels/) 。

**注:**车轮是参照**奶酪车轮**而命名为的[，是](https://www.youtube.com/watch?v=s5lJsFzv_iI&t=11m08s)[奶酪店里最重要的物品](https://wiki.python.org/moin/CheeseShop)。

源档案由你的源代码和任何支持文件组成，它们被打包成一个 [`tar`文件](https://en.wikipedia.org/wiki/Tar_%28computing%29)。类似地，轮子本质上是一个包含代码的 zip 存档。您应该为您的包提供源档案和轮子。对于您的最终用户来说，Wheels 通常更快、更方便，而源归档提供了一种灵活的备份替代方案。

要为您的包创建一个源归档文件和一个轮子，您可以使用 Build:

```py
(venv) $ python -m build
[...]
Successfully built realpython-reader-1.0.0.tar.gz and
 realpython_reader-1.0.0-py3-none-any.whl
```

正如输出的最后一行所说，这创建了一个源档案和一个轮子。您可以在新创建的`dist`目录中找到它们:

```py
realpython-reader/
│
└── dist/
    ├── realpython_reader-1.0.0-py3-none-any.whl
    └── realpython-reader-1.0.0.tar.gz
```

`.tar.gz`文件是您的源档案，而`.whl`文件是您的车轮。这些是您将上传到 PyPI 的文件，并且`pip`将在以后安装您的软件包时下载它们。

### 确认您的软件包版本

在上传新构建的分发包之前，您应该检查它们是否包含您期望的文件。车轮文件实际上是一个扩展名不同的 [ZIP 文件](https://realpython.com/python-zipfile/)。您可以按如下方式解压缩并检查其内容:

*   [*视窗*](#windows-1)
**   [**Linux + macOS**](#linux-macos-1)*

```py
(venv) PS> cd .\dist
(venv) PS> Copy-Item .\realpython_reader-1.0.0-py3-none-any.whl reader-whl.zip
(venv) PS> Expand-Archive reader-whl.zip
(venv) PS> tree .\reader-whl\ /F
C:\REALPYTHON-READER\DIST\READER-WHL
├───reader
│       config.toml
│       feed.py
│       viewer.py
│       __init__.py
│       __main__.py
│
└───realpython_reader-1.0.0.dist-info
 entry_points.txt
 LICENSE
 METADATA
 RECORD
 top_level.txt
 WHEEL
```

您首先重命名 wheel 文件，使其具有一个`.zip`扩展名，以便您可以扩展它。

```py
(venv) $ cd dist/
(venv) $ unzip realpython_reader-1.0.0-py3-none-any.whl -d reader-whl
(venv) $ tree reader-whl/
reader-whl/
├── reader
│   ├── config.toml
│   ├── feed.py
│   ├── __init__.py
│   ├── __main__.py
│   └── viewer.py
└── realpython_reader-1.0.0.dist-info
 ├── entry_points.txt
 ├── LICENSE
 ├── METADATA
 ├── RECORD
 ├── top_level.txt
 └── WHEEL

2 directories, 11 files
```

您应该看到列出了所有的源代码，以及一些已经创建的新文件，这些文件包含您在`pyproject.toml`中提供的信息。特别要确保所有的子包和支持文件像`config.toml`都包含在内。

你也可以看看源文件的内部，因为它被打包成一个[焦油球](https://en.wikipedia.org/wiki/Tar_(computing))。然而，如果您的轮子包含您期望的文件，那么源归档也应该是好的。

Twine 还可以[检查](https://twine.readthedocs.io/en/stable/#twine-check)你的包描述是否会在 PyPI 上正确呈现。您可以对在`dist`中创建的文件运行`twine check`:

```py
(venv) $ twine check dist/*
Checking distribution dist/realpython_reader-1.0.0-py3-none-any.whl: Passed
Checking distribution dist/realpython-reader-1.0.0.tar.gz: Passed
```

这不会捕捉到您可能遇到的所有问题，但这是很好的第一道防线。

[*Remove ads*](/account/join/)

### 上传您的包

现在您已经准备好将您的包上传到 PyPI 了。为此，您将再次使用 Twine 工具，告诉它上传您已经构建的分发包。

首先，您应该上传到 [TestPyPI](https://packaging.python.org/guides/using-testpypi/) 以确保一切按预期运行:

```py
(venv) $ twine upload -r testpypi dist/*
```

Twine 会要求您输入用户名和密码。

**注意:**如果您以`reader`包为例按照教程进行操作，那么前面的命令可能会失败，并显示一条消息，告诉您不允许上传到`realpython-reader`项目。

你可以把`pyproject.toml`中的`name`改成独特的，比如`test-<your-username>`。然后再次构建项目，并将新构建的文件上传到 TestPyPI。

如果上传成功，那么您可以快速前往 [TestPyPI](https://test.pypi.org/) ，向下滚动，查看您的项目在新版本中自豪地展示！点击你的包裹，确保一切正常。

如果您一直在使用`reader`包，那么本教程到此结束！虽然您可以尽情地使用 TestPyPI，但是您不应该仅仅为了测试而将示例包上传到 PyPI。

注意: TestPyPI 对于检查您的包上传是否正确以及您的项目页面看起来是否如您所愿非常有用。您也可以尝试从 TestPyPI 安装您的软件包:

```py
(venv) $ python -m pip install -i https://test.pypi.org/simple realpython-reader
```

但是，请注意，这可能会失败，因为不是所有的依赖项在 TestPyPI 上都可用。这不是问题。当你上传到 PyPI 的时候，你的包应该还能工作。

如果您有自己的软件包要发布，那么这一时刻终于到来了！所有准备工作就绪后，最后一步很简单:

```py
(venv) $ twine upload dist/*
```

要求时提供您的用户名和密码。就是这样！

前往 [PyPI](https://pypi.org/) 查看你的包裹。你既可以通过[搜索](https://pypi.org/search/)，也可以通过查看你的项目页面的[T5，或者直接进入你的项目的网址:](https://pypi.org/manage/projects/)[pypi.org/project/your-package-name/](https://pypi.org/project/realpython-reader/)。

恭喜你！你的包裹发布在 PyPI 上！

### 安装您的软件包

花点时间沐浴在 PyPI 网页的蓝色光芒中，向你的朋友吹嘘。

然后再打开一个终端。还有一个更大的回报！

随着你的包上传到 PyPI，你也可以用`pip`来安装它。首先，创建一个新的虚拟环境并激活它。然后运行以下命令:

```py
(venv) $ python -m pip install your-package-name
```

用您为包选择的名称替换`your-package-name`。例如，要安装`reader`包，您需要执行以下操作:

```py
(venv) $ python -m pip install realpython-reader
```

看到自己的代码由`pip`安装——就像任何其他第三方库一样——是一种美妙的感觉！

[*Remove ads*](/account/join/)

## 探索其他构建系统

在本教程中，您已经使用 Setuptools 构建了您的包。无论好坏，Setuptools 都是创建包的长期标准。虽然它被广泛使用和信任，但它也有很多可能与你无关的特性。

有几个可供选择的构建系统可以用来代替 Setuptools。在过去的几年里，Python 社区已经完成了标准化 Python 打包生态系统的重要工作。这使得在不同的构建系统之间移动变得更加简单，并使用最适合您的工作流和包的构建系统。

在本节中，您将简要了解两种可用于创建和发布 Python 包的备选构建系统。除了 Flit 和诗歌，你接下来会学到，你还可以看看 [pbr](https://docs.openstack.org/pbr/) 、 [enscons](https://pypi.org/project/enscons/) 和[孵卵](https://pypi.org/project/hatchling/)。此外， [`pep517`](https://pypi.org/project/pep517/) 包为创建您自己的构建系统提供了支持。

### 掠过

[Flit](https://flit.pypa.io/) 是一个伟大的小项目，目的是在打包时“让简单的事情变得简单”([来源](https://flit.pypa.io/en/latest/rationale.html))。Flit 不支持像那些创建 [C 扩展](https://realpython.com/build-python-c-extension-module/)的高级包，一般来说，它在设置你的包时不会给你很多选择。相反，Flit 赞同应该有一个明显的工作流来发布一个包的理念。

**注意:**你不能同时用 Setuptools 和 Flit 配置你的包。为了测试这一部分中的工作流，您应该将您的 Setuptools 配置安全地存储在您的版本控制系统中，然后删除`pyproject.toml`中的`build-system`和`project`部分。

首先用`pip`安装 Flit:

```py
(venv) $ python -m pip install flit
```

尽可能多地，Flit 自动完成你需要用你的包做的准备工作。要开始配置新的软件包，请运行`flit init`:

```py
(venv) $ flit init
Module name [reader]:
Author: Real Python
Author email: info@realpython.com
Home page: https://github.com/realpython/reader
Choose a license (see http://choosealicense.com/ for more info)
1\. MIT - simple and permissive
2\. Apache - explicitly grants patent rights
3\. GPL - ensures that code based on this is shared with the same terms
4\. Skip - choose a license later
Enter 1-4: 1

Written pyproject.toml; edit that file to add optional extra info.
```

`flit init`命令将根据您对几个问题的回答创建一个`pyproject.toml`文件。在使用这个文件之前，您可能需要稍微编辑一下。对于`reader`项目，Flit 的`pyproject.toml`文件看起来如下:

```py
# pyproject.toml [build-system] requires  =  ["flit_core >=3.2,<4"] build-backend  =  "flit_core.buildapi" [project] name  =  "realpython-reader" authors  =  [{  name  =  "Real Python",  email  =  "info@realpython.com"  }] readme  =  "README.md" license  =  {  file  =  "LICENSE"  } classifiers  =  [ "License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3", ] dynamic  =  ["version",  "description"] [project.urls] Home  =  "https://github.com/realpython/reader" [project.scripts] realpython  =  "reader.__main__:main"
```

请注意，大多数`project`项目与您原来的`pyproject.toml`完全相同。不过，一个不同之处是`version`和`description`是在`dynamic`字段中指定的。Flit 实际上通过使用`__version__`和`__init__.py`文件中定义的 [docstring](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings) 自己解决了这些问题。 [Flit 的文档](https://flit.pypa.io/)解释了关于`pyproject.toml`文件的一切。

Flit 可以构建您的包并将其发布到 PyPI。您不需要使用构建和缠绕。要构建您的包，只需执行以下操作:

```py
(venv) $ flit build
```

这创建了一个源档案和一个轮子，类似于您之前对`python -m build`所做的。如果您愿意，也可以使用 Build。

要将您的包上传到 PyPI，您可以像前面一样使用 Twine。但是，您也可以直接使用 Flit:

```py
(venv) $ flit publish
```

如果需要的话,`publish`命令将构建您的包，然后将文件上传到 PyPI，提示您输入用户名和密码。

要查看 Flit 的早期但可识别的版本，请查看来自 EuroSciPy 2017 的 Thomas Kluyver 的[闪电谈话](https://www.youtube.com/watch?v=qTgk2DUM6G0&t=11m50s)。[演示](https://asciinema.org/a/135744)展示了如何在两分钟内配置、编译并发布到 PyPI。

[*Remove ads*](/account/join/)

### 诗歌

诗歌是另一个工具，你可以用它来构建和上传你的包。与 Flit 相比，poem 有更多的特性可以帮助你开发包，包括强大的[依赖管理](https://realpython.com/dependency-management-python-poetry/)。

在你使用诗歌之前，你需要安装它。用`pip`装诗是可以的。然而，[的维护者建议](https://python-poetry.org/docs/#installation)使用定制的安装脚本来避免潜在的依赖冲突。参见[文档](https://python-poetry.org/docs/#installation)中的说明。

**注意:**你不能同时用设置工具和诗歌来配置你的软件包。为了测试这一部分中的工作流，您应该将您的 Setuptools 配置安全地存储在您的版本控制系统中，然后删除`pyproject.toml`中的`build-system`和`project`部分。

安装完诗歌后，您可以通过一个`init`命令开始使用它，类似于 Flit:

```py
(venv) $ poetry init

This command will guide you through creating your pyproject.toml config.

Package name [code]: realpython-reader
Version [0.1.0]: 1.0.0
Description []: Read the latest Real Python tutorials
...
```

这将基于您对关于您的包的问题的回答创建一个`pyproject.toml`文件。

**注意:**诗歌[目前不支持](https://github.com/python-poetry/poetry/issues/3332) PEP 621，因此`pyproject.toml`内部的实际规格目前在诗歌和其他工具之间有所不同。

对于诗歌，`pyproject.toml`文件最终看起来如下:

```py
# pyproject.toml [build-system] requires  =  ["poetry-core>=1.0.0"] build-backend  =  "poetry.core.masonry.api" [tool.poetry] name  =  "realpython-reader" version  =  "1.0.0" description  =  "Read the latest Real Python tutorials" authors  =  ["Real Python <info@realpython.com>"] readme  =  "README.md" homepage  =  "https://github.com/realpython/reader" license  =  "MIT" [tool.poetry.dependencies] python  =  ">=3.9" feedparser  =  "^6.0.8" html2text  =  "^2020.1.16" tomli  =  "^2.0.1" [tool.poetry.scripts] realpython  =  "reader.__main__:main"
```

您应该能从前面的`pyproject.toml`讨论中认出所有这些项目，尽管这些部分的名称不同。

需要注意的一点是，poems 会根据您指定的许可和 Python 版本自动添加分类器。诗歌也要求你明确你的依赖版本。事实上，依赖管理是诗歌的优点之一。

就像 Flit 一样，诗歌可以构建包并上传到 PyPI。`build`命令创建一个源档案和一个轮子:

```py
(venv) $ poetry build
```

这将在`dist`子目录中创建两个常用文件，您可以像前面一样使用 Twine 上传它们。您也可以使用诗歌发布到 PyPI:

```py
(venv) $ poetry publish
```

这将把你的包上传到 PyPI。除了有助于建设和出版，诗歌可以帮助你在这个过程的早期。诗歌可以帮助你用`new`命令启动一个新项目。它还支持使用虚拟环境。详见[诗歌的文档](https://python-poetry.org/docs/)。

除了配置文件略有不同，Flit 和 poem 的工作方式非常相似。诗歌的范围更广，因为它也旨在帮助依赖管理，而 Flit 已经存在了一段时间。

## 结论

您现在知道了如何准备您的项目并将其上传到 PyPI，以便其他人可以安装和使用它。虽然您需要完成一些步骤，但是在 PyPI 上看到您自己的包是一个很大的回报。让别人发现你的项目有用就更好了！

在本教程中，您已经学习了如何通过以下方式发布自己的包:

*   为你的包裹找到一个好名字
*   **使用`pyproject.toml`配置**您的软件包
*   **大厦**你的包裹
*   **上传**您的包到 PyPI

此外，您还了解了 Python 打包社区中标准化工具和流程的计划。

如果你还有问题，请在下面的评论区联系我们。另外， [Python 打包用户指南](https://packaging.python.org/)提供了比你在本教程中看到的更多的详细信息。

**获取源代码:** [单击此处获取您将在本教程中使用的真正的 Python 提要阅读器的源代码](https://realpython.com/bonus/pypi-publish-python-package-source-code/)。

*立即观看**本教程有真实 Python 团队创建的相关视频课程。和文字教程一起看，加深理解: [**如何将自己的 Python 包发布到 PyPI**](/courses/how-to-publish-your-own-python-package-pypi/)***************
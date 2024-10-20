# 用单个文件运行 Python

> 原文：<https://www.blog.pythonlibrary.org/2014/06/19/pyrun-running-python-with-a-single-file/>

eGenix 本周宣布，他们将发布一个“开源、一个文件、无安装版本的 Python”。你可以在这里阅读他们的[新闻稿中的全部声明。如果你想看看实际的产品，你可以在以下网址得到它:【http://www.egenix.com/products/python/PyRun/](http://www.egenix.com/company/news/eGenix-PyRun-2.0.0-GA.html)[。](http://www.egenix.com/products/python/PyRun/)

这是一个基于 Unix 的 Python，他们声明他们只提供 Linux，FreeBSD 和 Mac OS X，作为 32 位和 64 位版本，所以如果你是一个 Windows 的家伙，你就不走运了。我认为这个工具最酷的一个方面是 Python 2 的 11MB 和 Python 3 的 13MB，但它仍然支持大多数 Python 应用程序和您编写的代码。如果你写了很多依赖于第三方工具的代码，那么我不认为 PyRun 会对你有很大帮助。但是如果你主要依赖 Python 的标准库，那么我认为这是一个非常方便的工具。

PyRun 的一个替代方案是 [Portable Python](http://portablepython.com/) ，它提供了大量的第三方库，这些库与标准库结合在一起。我已经在我自己的几个应用程序中使用了 Portable Python，在这些应用程序中我不想安装 Python 本身。无论如何，我希望这些信息对你自己的努力有用。
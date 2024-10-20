# Python 恶意软件可能会入侵您附近的计算机

> 原文：<https://www.blog.pythonlibrary.org/2020/08/04/python-malware-may-be-coming-to-a-computer-near-you/>

Cyborg Security [最近](https://www.cyborgsecurity.com/python-malware-on-the-rise/)报道称，使用 Python 编程语言编写的恶意软件开始出现。传统上，大多数恶意软件都是用编译语言编写的，如 C 或 C++。

原因很简单。编译语言让攻击者创建更小、更难检测的可执行文件。然而，Python 的流行和易用性使得它对恶意软件作者更有吸引力。Python 对于恶意软件的最大问题是，它往往比用 C 或 C++编写的恶意软件使用更多的 RAM 和 CPU。

当然，随着个人电脑像现在这样强大，这不再是一个问题。尤其是当你想到有如此多的应用程序是用电子语言编写的。你的网络浏览器现在是一个巨大的资源猪！

正如 Cyborg Security 网站所指出的，您可以使用 PyInstaller 或 py2exe 来创建 Python 代码的可执行文件。这篇文章没有提到的是，有人还需要对该软件进行数字签名，才能让它在 Windows 10 上运行。这篇文章提到的一件让我感兴趣的事情是，你可以使用 Nuitka 将你的 Python 代码转换成 C 语言，最终你会得到一个比使用 PyInstaller 或 py2exe 小得多的可执行文件。

Python 恶意软件的具体例子包括 2015 年和 2016 年针对民主党全国委员会使用的 [SeaDuke](https://unit42.paloaltonetworks.com/unit-42-technical-analysis-seaduke/) 。他们还提到了 PWOBot，这是一种类似的 Python 恶意软件，可以进行按键记录以及下载和执行其他应用程序。

[趋势科技](https://blog.trendmicro.com/trendlabs-security-intelligence/a-closer-look-at-the-locky-poser-pylocky-ransomware/)覆盖了基于 Python 的勒索软件 PyLocky。它可以使用 3DES 加密文件。

提到的最后一款恶意软件是 PoetRAT，这是一款特洛伊木马，今年曾被用于攻击阿塞拜疆政府和能源部门。

查看[全文](https://www.cyborgsecurity.com/python-malware-on-the-rise/)。这真的很有趣，涵盖了更多关于这个话题的内容。
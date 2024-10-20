# 黑莓发布了一款用 Python 编写的反恶意软件工具

> 原文：<https://www.blog.pythonlibrary.org/2020/08/26/blackberry-released-an-anti-malware-tool-written-in-python/>

如果你[错过了这个月早些时候的](https://www.techrepublic.com/article/blackberry-launches-free-tool-for-reverse-engineering-to-fight-cybersecurity-attacks/)，黑莓发布了他们的一个工具，他们用它来逆向工程恶意软件。这个工具叫做 [PE 树](https://github.com/blackberry/pe_tree)，是开源的，用 Python 编写。

Blackberry 使用流行的 PyQt5 GUI 工具包来编写显示可移植可执行文件的树形视图，这使得更容易转储和重建内存中的恶意软件。

公关树工具适用于视窗、苹果和 Linux。它可以作为一个独立的应用程序运行，也可以作为一个插件运行，这本身就是一个反汇编器的插件。

这听起来像是一个非常好的工具。如果没有别的，它将是学习如何用 Python 创建真实世界 GUI 的一个很好的应用程序。
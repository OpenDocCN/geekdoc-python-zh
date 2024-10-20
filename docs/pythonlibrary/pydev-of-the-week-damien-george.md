# 本周 PyDev:达米恩·乔治

> 原文：<https://www.blog.pythonlibrary.org/2016/08/29/pydev-of-the-week-damien-george/>

本周我们欢迎达米恩·乔治成为我们的本周派德夫！Damien 是 MicroPython 项目背后的人，这个项目允许你在微控制器上运行 Python 的一个版本。你可以在达米恩的[网站](http://dpgeorge.net/)或者访问他的 [Github 页面](https://github.com/dpgeorge/)了解更多关于他的信息。让我们花些时间来更好地了解我们的同胞 Pythonista！

![damien](img/bd7b2af786553db088a5b205508aa337.png)

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我在澳大利亚的墨尔本长大，很早就开始玩电脑和电子产品。我年轻的时候有一台 Commodore 64，还记得从当地图书馆借的关于如何用汇编语言编程的书。实际上，它只是给你一堆“数据”语句来输入，但最终你可以做出一些非常酷的东西。高中时，我喜欢物理和数学，后来我上了大学，获得了科学和工程学位。我主修物理和计算机工程，然后攻读了物理学博士学位，之后我去了荷兰，最终去了英国，从事理论高能物理学的研究(额外维度、超对称、宇宙学等等)。在我作为一名物理学家的职业生涯中，我对编程和机器人技术保持着浓厚的兴趣，同时也做了很多兼职项目，包括一台自制的数控机床(见[http://dpgeorge.net/cnc/](http://dpgeorge.net/cnc/))。

**你为什么开始使用 Python？**

因为我需要一种既强大又允许快速开发的语言。我想我用 Python 做的第一个大项目是我和一个同事开发的 Paperscape 网站，它可视化了物理学方面的科学论文(见[http://paperscape.org](http://paperscape.org))。Python 在后端被广泛用于解析 TeX、LaTeX 和 PDF 文件，提取引用信息，以及维护论文的 SQL 数据库。

你还知道哪些编程语言，你最喜欢哪一种？

*这些年我用过很多语言，包括:很多架构的汇编语言，BASIC，Fortran(大部分是 77)，C，C++，Java，Haskell，Go，JavaScript。我非常喜欢 C，可以说它是我的最爱，但 C++也很好，现在是现代标准下更好的语言。我喜欢 Haskell，但不要把它用于任何严肃的事情。* 

你现在在做什么项目？

我几乎把所有的时间都花在了开发 MicroPython 上(见[https://micropython.org](https://micropython.org)和[https://github.com/micropython/micropython](https://github.com/micropython/micropython))。我想添加许多功能，进行一些优化，并开发新的平台来运行它。现在，我正专注于让 MicroPython 在 ESP8266 Wi-Fi 芯片上运行良好，这是我们今年早些时候运行的 [Kickstarter](https://www.kickstarter.com/projects/214379695/micropython-on-the-esp8266-beautifully-easy-iot) 的一部分。这涉及到编程以及文档，教程和论坛管理。我也在帮助开发 MicroPython 的 BBC micro:bit 端口(见[https://github.com/bbcmicrobit/micropython](https://github.com/bbcmicrobit/micropython))。很多人以不同的方式为 MicroPython 项目做贡献，这很棒，有助于项目的发展。我期待看到 MicroPython 变得更大，并以新的有趣的方式使用。

哪些 Python 库是你最喜欢的(核心或第三方)？

因为它们在 MicroPython 中的特殊需求，我喜欢 sys、os 和 struct 这样无聊的库。但是更有趣的是我们的“机器”模块，它以 Pythonic 的方式抽象了底层硬件。例如 GPIO 引脚、I2C 总线、模数转换器等等。在 BBC micro:bit 上有一个“microbit”模块，里面有很多有趣的东西，让孩子们可以轻松地对设备进行编程。

作为一门编程语言，你认为 Python 将何去何从？

Python 不断发展并跟上了程序员的需求以及当前的技术，这是一件非常重要的事情，我认为这将使它在未来很长一段时间内成为五大编程语言之一。对不同语言的需求总是存在的，每种语言都有特定的需求，Python 将仍然是通用快速开发编程的最佳选择。

你还有什么想说的吗？

我认为 Python 社区很棒。它友好、开放，有很多聪明人致力于让 Python 不仅成为一门伟大的语言，而且成为一个伟大的编程生态系统。更广泛地说，我认为开源社区和开源软件是当今软件和计算的基础部分，我很高兴能够为之做出贡献。

感谢您接受采访！
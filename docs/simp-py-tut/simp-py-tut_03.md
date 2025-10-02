# 第一章 介绍

**目录表**

*   简介
*   Python 的特色
*   概括
*   为什么不使用 Perl？
*   程序员的话

Python 语言是少有的一种可以称得上即**简单**又**功能强大**的编程语言。你将惊喜地发现 Python 语言是多么地简单，它注重的是如何解决问题而不是编程语言的语法和结构。

Python 的官方介绍是：

> Python 是一种简单易学，功能强大的编程语言，它有高效率的高层数据结构，简单而有效地实现面向对象编程。Python 简洁的语法和对动态输入的支持，再加上解释性语言的本质，使得它在大多数平台上的许多领域都是一个理想的脚本语言，特别适用于快速的应用程序开发。

我会在下一节里详细地讨论 Python 的这些特点。

注释 Python 语言的创造者 Guido van Rossum 是根据英国广播公司的节目“蟒蛇飞行马戏”命名这个语言的，并非他本人特别喜欢蛇缠起它们的长身躯碾死动物觅食。

# Python 的特色

简单

Python 是一种代表简单主义思想的语言。阅读一个良好的 Python 程序就感觉像是在读英语一样，尽管这个英语的要求非常严格！Python 的这种伪代码本质是它最大的优点之一。它使你能够专注于解决问题而不是去搞明白语言本身。

易学

就如同你即将看到的一样，Python 极其容易上手。前面已经提到了，Python 有极其简单的语法。

免费、开源

Python 是 FLOSS（自由/开放源码软件）之一。简单地说，你可以自由地发布这个软件的拷贝、阅读它的源代码、对它做改动、把它的一部分用于新的自由软件中。FLOSS 是基于一个团体分享知识的概念。这是为什么 Python 如此优秀的原因之一——它是由一群希望看到一个更加优秀的 Python 的人创造并经常改进着的。

高层语言

当你用 Python 语言编写程序的时候，你无需考虑诸如如何管理你的程序使用的内存一类的底层细节。

可移植性

由于它的开源本质，Python 已经被移植在许多平台上（经过改动使它能够工作在不同平台上）。如果你小心地避免使用依赖于系统的特性，那么你的所有 Python 程序无需修改就可以在下述任何平台上面运行。

这些平台包括 Linux、Windows、FreeBSD、Macintosh、Solaris、OS/2、Amiga、AROS、AS/400、BeOS、OS/390、z/OS、Palm OS、QNX、VMS、Psion、Acom RISC OS、VxWorks、PlayStation、Sharp Zaurus、Windows CE 甚至还有 PocketPC！

解释性

这一点需要一些解释。

一个用编译性语言比如 C 或 C++写的程序可以从源文件（即 C 或 C++语言）转换到一个你的计算机使用的语言（二进制代码，即 0 和 1）。这个过程通过编译器和不同的标记、选项完成。当你运行你的程序的时候，连接/转载器软件把你的程序从硬盘复制到内存中并且运行。

而 Python 语言写的程序不需要编译成二进制代码。你可以直接从源代码 运行 程序。在计算机内部，Python 解释器把源代码转换成称为字节码的中间形式，然后再把它翻译成计算机使用的机器语言并运行。事实上，由于你不再需要担心如何编译程序，如何确保连接转载正确的库等等，所有这一切使得使用 Python 更加简单。由于你只需要把你的 Python 程序拷贝到另外一台计算机上，它就可以工作了，这也使得你的 Python 程序更加易于移植。

面向对象

Python 即支持面向过程的编程也支持面向对象的编程。在 面向过程 的语言中，程序是由过程或仅仅是可重用代码的函数构建起来的。在 面向对象 的语言中，程序是由数据和功能组合而成的对象构建起来的。与其他主要的语言如 C++和 Java 相比，Python 以一种非常强大又简单的方式实现面向对象编程。

可扩展性

如果你需要你的一段关键代码运行得更快或者希望某些算法不公开，你可以把你的部分程序用 C 或 C++编写，然后在你的 Python 程序中使用它们。

可嵌入性

你可以把 Python 嵌入你的 C/C++程序，从而向你的程序用户提供脚本功能。

丰富的库

Python 标准库确实很庞大。它可以帮助你处理各种工作，包括正则表达式、文档生成、单元测试、线程、数据库、网页浏览器、CGI、FTP、电子邮件、XML、XML-RPC、HTML、WAV 文件、密码系统、GUI（图形用户界面）、Tk 和其他与系统有关的操作。记住，只要安装了 Python，所有这些功能都是可用的。这被称作 Python 的“功能齐全”理念。

除了标准库以外，还有许多其他高质量的库，如[wxPython](http://www.wxpython.org)、[Twisted](http://www.twistedmatrix.com/products/twisted)和[Python 图像库](http://www.pythonware.com/products/pil/index.htm)等等。

Python 确实是一种十分精彩又强大的语言。它合理地结合了高性能与使得编写程序简单有趣的特色。

# 为什么不使用 Perl？

也许你以前并不知道，Perl 是另外一种极其流行的开源解释性编程语言。

如果你曾经尝试过用 Perl 语言编写一个大程序，你一定会自己回答这个问题。在规模较小的时候，Perl 程序是简单的。它可以胜任于小型的应用程序和脚本，“使工作完成”。然而，当你想开始写一些大一点的程序的时候，Perl 程序就变得不实用了。我是通过为 Yahoo 编写大型 Perl 程序的经验得出这样的总结的！

与 Perl 相比，Python 程序一定会更简单、更清晰、更易于编写，从而也更加易懂、易维护。我确实也很喜欢 Perl，用它来做一些日常的各种事情。不过当我要写一个程序的时候，我总是想到使用 Python，这对我来说已经成了十分自然的事。Perl 已经经历了多次大的修正和改变，遗憾的是，即将发布的 Perl 6 似乎仍然没有在这个方面做什么改进。

我感到 Perl 唯一也是十分重要的优势是它庞大的[CPAN](http://cpan.perl.org)库——综合 Perl 存档网络。就如同这个名字所指的意思一样，这是一个巨大的 Perl 模块集，它大得让人难以置信——你几乎用这些模块在计算机上做任何事情。Perl 的模块比 Python 多的原因之一是 Perl 拥有更加悠久的历史。或许我会在[comp.lang.python](http://groups.google.com/groups?q=comp.lang.python)上建议把 Perl 模块移植到 Python 上的计划。

另外，新的[Parrot 虚拟机](http://www.parrotcode.org)按设计可以运行完全重新设计的 Perl 6 也可以运行 Python 和其他解释性语言如 Ruby、PHP 和 Tcl 等等。这意味着你将来 或许 可以在 Python 上使用所有 Perl 的模块。这将成为两全其美的事——强大的 CPAN 库与强大的 Python 语言结合在一起。我们将拭目以待。

# 程序员的话

读一下像 ESR 这样的超级电脑高手谈 Python 的话，你会感到十分有意思：

*   **Eric S. Raymond**是《The Cathedral and the Bazaar》的作者、“开放源码”一词的提出人。他说[Python 已经成为了他最喜爱的编程语言](http://linuxjournal.com/article.php?sid=3882)。这篇文章也是促使我第一次接触 Python 的真正原动力。

*   **Bruce Eckel**著名的《Thinking in Java》和《Thinking in C++》的作者。他说没有一种语言比得上 Python 使他的工作效率如此之高。同时他说 Python 可能是唯一一种旨在帮助程序员把事情弄得更加简单的语言。请阅读[完整的采访](http://www.artima.com/inv/aboutme.html)以获得更详细的内容。

*   **Peter Norvig**是著名的 Lisp 语言书籍的作者和 Google 公司的搜索质量主任（感谢 Guido van Rossum 告诉我这一点）。他说 Python 始终是 Google 的主要部分。事实上你看一下[Google 招聘](http://www.google.com/jobs/index.html)的网页就可以验证这一点。在那个网页上，Python 知识是对软件工程师的一个必需要求。

*   **Bruce Perens**是 OpenSource.org 和 UserLinux 项目的一位共同创始人。UserLinux 旨在创造一个可以被多家发行商支持标准的 Linux 发行版。Python 击败了其它竞争对手如 Perl 和 Ruby 成为 UserLinux 支持的主要编程语言。
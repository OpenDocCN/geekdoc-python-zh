# 本周 PyDev:马里亚诺·莱因加特

> 原文：<https://www.blog.pythonlibrary.org/2014/12/29/pydev-of-the-week-mariano-reingart/>

本周我们邀请了 Mariano Reingart 作为本周的 PyDev。Mariano 为 Packt Publishing 合著了 [web2py 应用程序开发食谱](http://www.amazon.com/gp/product/B007KHZ1AA/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=B007KHZ1AA&linkCode=as2&tag=thmovsthpy-20&linkId=6PZQNBO3FD42TFZL),最近为 wxPython 完成了 wxQt 的部分移植。让我们花些时间和马里亚诺在一起，看看他有什么要说的！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

我是一名自由职业的开发人员和教师，已婚，有两个小孩。

当我还是个孩子的时候，我就开始编程，我父亲在 80 年代末给我带来了一台 ZX Spectrum TK-90。*1989 年，年仅 11 岁的我用 MSX 计算机上了第一堂“正式”编程课“Basic I”，并于 1991 年获得“D.O.S .”。* *完成小学教育后，我于 1992 年进入一所“信息学学士”定向学院(中级“预备”水平)，在那里教授 Visual Basic 和 Visual Fox Pro 的最后课程。*

一毕业，我就在那所学校当助教，大约在 1997 年安装了我的第一台 Slackware Linux 服务器以连接互联网，并为学院系统编程(最初使用 VB + Access)。

*我的第一次“开源”编程经历是开发一个 [Linux 的内核模块](https://docs.google.com/leaf?id=0B__UYqYT4LNaMzdiODk0YjUtZWEyYi00ZTliLTgwYTctZTQxNWE5OTMwN2Mx&hl=en)来使用 128Kps 的电话专线(我甚至不得不导入通信板)，和一个 [ucLinux 内核驱动程序](https://docs.google.com/uc?id=0B__UYqYT4LNaNTc3YWNkMzQtMzM4MC00Zjg1LWI0MzEtMmRkYzQwOWI1M2Ni&export=download&hl=en)用于 PiCoTux(一个运行 Linux 的小型嵌入式微型计算机)。*

另外，我完成了将数据库迁移到 PostgreSQL 的第一步，并使用了早期的“WiFi”设备。

然后，我学习了几年电子工程，但我意识到我的技能更多地与软件相关，所以我转到了当地的一所大学学习计算机系统分析(相当于理学学士学位)。

*经过几年的“休假”，我终于在 2011 年毕业，在那里我获得了开发小型 ERP 的工作经验，比如为中小企业开发业务系统和 911 应用程序(与我的一个朋友为当地警察局合作)*

*目前我正在完成一个自由软件的硕士学位(加泰罗尼亚开放大学)，和一个“教授职位”(正式的高等教师培训)*

**你为什么开始使用 Python？**

我从 2000 年开始使用 python，寻找 Visual Basic 的替代品。

*我 2006 年的学位论文选择它“[个人软件过程下的快速应用开发](https://docs.google.com/document/d/1tQYzvI4_2Nq5Wv6srdCZPG5Da0ZzCLpabwsYGm0db5I/edit?hl=es)”(不好意思，是西班牙语，不过你可以看更多 PyCon US 2012 上展示的[海报](https://us.pycon.org/2012/schedule/presentation/147/))*

从 2009 年开始，我在一所高等院校教 Python。它开始只是一门课程，但后来我们(和其他教授)同意实施一个面向自由软件的[机构课程，主要是:](http://docs.google.com/View?id=dd9bm82g_428g8zvfvdx)

*   *《计算机编程 I》、《计算机编程 II》(含 wxPython/web2py 入门)*
*   *[数据库](http://reingart.blogspot.com.ar/p/materia-base-de-datos.html)(设计与实现)*
*   *[操作系统](http://reingart.blogspot.com.ar/p/materia-sistemas-operativos.html)(文件系统、进程、IPC 等。)*
*   *[网间互联](http://reingart.blogspot.com.ar/p/materia-interconectividad-redes.html)(“计算机网络”):套接字、互联网应用等。)*
*   *[专业实践](http://reingart.blogspot.com.ar/p/materia-practica-profesional.html)(最终工作:web app 开发)*

在此基础上，我们和其他同事正在准备一个“自由软件文凭”(开放本科学位，一年课程计划，重点是 Python、PostgreSQL 和 GNU/Linux，更多信息见本文[文章](http://43jaiio.sadio.org.ar/proceedings/STS/859-Reingart.pdf))。

*令人高兴的是，现在我靠出售开源商业支持计划谋生，这要感谢 Python，因为在 2008 年，项目 [PyAfipWs，“电子发票”](http://pyafipws.googlecode.com/)，从 PyAr 邮件列表开始，当地的 [Python Argentina](http://www.python.org.ar/) 社区，然后成长起来，被相对较大的用户群(包括公司、中小企业和专业人士)*

你还知道哪些编程语言，你最喜欢哪一种？

*Python 当然是我的最爱，对 PHP、C/C++、Java 等语言也有一定的经验。*

*遗憾的是，我仍然不得不使用 Visual Basic Classic (6.0)来开发我的一些遗留 ERP 系统(它们很大，大约有几十万行，我还没有时间/资源来迁移到 Python)*

*现在正在调查 [vb2py](http://vb2py.sourceforge.net/) ，一个把 vb 转换成 Python 代码的项目。* *我想知道为什么它没有达到临界质量/牵引力(开发似乎从 2014 年就停止了)，因为仍然有很多 VB 代码挂在周围...*

对于电子发票项目，我使用 [pywin32](http://sourceforge.net/projects/pywin32/) (来自马克·哈蒙德等人)来使 python 组件可以从遗留语言(VB，VFP)中使用，并使用一些库，如 [dbf](https://pypi.python.org/pypi/dbf/) (来自伊森·弗曼)来与更古老的语言(Clipper，xBase 等)交互。). *最近我也开始玩 [Jython](http://www.jython.org/) ，也开始使用来自 Java 的 python 项目。*

你现在在做什么项目？

为了我的学位论文，我开始开发 [rad2py](https://code.google.com/p/rad2py/) ，一个实验性的集成开发环境(IDE)和案例支持工具，设想用于商业、教育&学术目的。

*现在我正在完成我的硕士论文研究“[高质量自由软件快速开发](https://docs.google.com/document/d/1Jo-_Nf_vMeKvszEuWA24yrfrqGGU-T73cczMPSBZ9ss/edit?usp=sharing)”，试图让开发人员的生活变得更容易，集成最近的方法，如以任务为中心的界面(Eclipse Mylyn)和敏捷 ALM(应用程序生命周期管理)，用严格的软件工程原则来保证质量和持续的自我改进。*

哪些 Python 库是你最喜欢的(核心或第三方)？

*我尝试只使用 Python 标准库+ [wxPython](http://www.wxpython.org/) 和 [web2py](http://web2py.com/) ，对于一些项目使用 pywin32 和 dbf，如前所述。*

我发现 web2py 很吸引人，因为它有一个几乎平坦的学习曲线，允许快速原型开发，一个友好的在线 IDE，等等。 *它的非传统方式也带来了其他观点和“批判性思维”,这要特别感谢它的创造者马西莫·迪·皮埃罗(Massimo Di Pierro)的开放、热情的领导，当然还有它热情的社区。**免责声明**:我是一个“主要”开发人员，最近因为时间不够不太活跃，但是贡献了一个[在线调试器](http://reingart.blogspot.com.ar/2012/02/new-web2py-online-python-debugger.html)和其他增强功能，与人合著了 Packt 的书“ [web2py 应用程序开发食谱](https://www.packtpub.com/web-development/web2py-application-development-cookbook)”等等。*

*wxPython 也值得特别一提，罗宾·邓恩的作品很棒([凤凰](http://wiki.wxpython.org/ProjectPhoenix) py3k！)，Andrea Gavana pure python[agw widgets](http://xoomer.virgilio.it/infinity77/AGW_Docs/)确实令人印象深刻，这只是提到了那个社区的几个开发者。我还发现 wxWidgets 比其他替代产品更正交、更易于使用。* *今年我试图与 [GSoC 2014](http://www.google-melange.com/gsoc/proposal/public/google/gsoc2014/reingart/5629499534213120) 中的实验性 [wxQt 端口](https://wiki.wxwidgets.org/WxQt)进行更深入的合作(它现在已经可用，甚至来自 wxPython！在 Android 下，至少是 C++部分...).*

当我需要一些不隐蔽的东西时，我倾向于寻找简单的解决方案(很多时候受 PHP 扩展的影响)，有时会开始、继续或分叉其他项目:

*   *[PySimpleSoap](https://github.com/pysimplesoap/pysimplesoap) :构建 Soap 客户端和服务器的临时轻量级 web 服务接口*
*   *[PyFPDF](http://pyfpdf.googlecode.com/) :基于 PHP 的 FPDF 及其衍生物的简单 PDF 生成类*
*   *[gui2py](https://code.google.com/p/gui2py/):python card 的一个分支(“rapid”&简单 gui 构建工具包)对其进行进化和现代化*

除了一些开发人员认为它们是“幼稚”的努力(至少可以这么说)，他们中的大多数已经找到了自己的位置，现在其他合作者也做出了许多发现它们有用的贡献。它们还有助于理解底层技术，提供纯 python 的替代实现，并尝试 Python 3 迁移。

你还有什么想说的吗？

不，很抱歉回答这么长，但是你的面试是不可抗拒的🙂 ...另外，很抱歉我的英语不好，我的母语是西班牙语，所以有时我很难找到合适的词。感谢你的项目和努力，感谢 Python 的整个社区！

**谢谢！**

### 前一周的 PyDevs

*   巴斯卡尔·乔德里
*   蒂姆·罗伯茨
*   汤姆·克里斯蒂
*   史蒂夫·霍尔登
*   卡尔查看
*   迈克尔·赫尔曼
*   布莱恩·柯廷
*   卢西亚诺·拉马拉
*   沃纳·布鲁欣
*   浸信会二次方阵
*   本·劳什
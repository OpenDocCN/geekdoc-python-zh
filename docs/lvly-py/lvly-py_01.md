# 书前

Contents:

*   哲思自由软件图书序
*   preface by Guido van Rossum
    *   可爱的 Python（第一版）推荐序
*   书序
    *   目标读者
    *   内容组织
    *   本书结构
    *   本书行文体例
*   前言
    *   本书阅读技巧
    *   代码阅读技巧
*   感谢
    *   人物
    *   社区
    *   行者
    *   校对
    *   工具
    *   还有,,,

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# 哲思自由软件图书序

> *   作者：徐继哲 <bill@zeuux.org>
> *   2009 年 2 月 16 日 北京

什么是自由软件？自由软件强调的是用户运行，学习，修改和发行软件的自由，而不是价格。具体来讲，自由软件赋予用户下面 4 个自由度:

```py
0\. 出于任何目的，运行软件的自由。
1\. 学习软件如何工作，以及为了满足自己的需要修改软件的自由。（显然，这个自由度的前提是能够访问软件的源代码）
2\. 为了帮助你的邻居，将软件拷贝给他的自由。
3\. 为了能够让整个社区受益，公开发行改进之后的软件的自由。（显然，这个自由度的前提是能够访问软件的源代码）
```

在 1983 年，Richard Stallman 发起了自由软件运动。经过多年的努力，自由软件运动早已开花结果，在计算机工业、科学研究、教育、法律等领域都取得了巨大的成功，自由软件赋予了每个人运行、学习、修改和再发行软件的自由。现在，使用自由软件可以完成生活、工作中的各类任务，从构建服务器集群到个人计算机桌面，几乎无所不能。

> *   自由软件运动所倡导的哲学思想再次提醒我们自由、平等和互助是人类社会不断向前发展的基础；在自由软件运动中诞生的对称版权（copyleft）为我们展示了全新的软件发行模式，在这一模式下，软件像知识一样被积累和传播，为人类造福；无数满怀激情的程序员创造了大量的优秀自由软件，比如：GNU、Linux、BSD、Apache、PostgreSQL、Python、Postfix 等，众多优秀的公司，比如：Yahoo!、Google 和新浪等，都建立在自由软件技术之上。
> *   为了进一步促进自由软件在中国的发展，哲思社区启动了哲思自由软件图书计划。我们将创作一批优秀的自由软件图书，涉及的领域包括：自由软件哲学、技术、法律、经济、管理和人文等领域，以服务中国自由软件社区和 IT 产业。
> *   社区是自由的、开放的，希望更都的朋友能够加入自由软件社区，一起协作创新！

徐继哲 <bill@zeuux.org>

> *   2009 年 2 月 16 日 下午 北京
> *   哲思自由软件社区 创始人
> *   [`www.zeuux.org`](http://www.zeuux.org)

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# preface by Guido van Rossum

`~ of <<Lovely Python(editon 1)>>`

> Dear Reader,

Congratulations on picking up a book on programming!

Programming is the ultimate puzzle, and studying it will keep your mind fresh forever. That is certainly how it has worked for me: I have always had fun programming, and eventually this led me to create my own programming language, so that I could have even more fun. Python is one of the most fun programming languages around. It is incredibly flexible, and is very easy to learn -- this book teaches you how to write Python programs, even if you have never learned another programming language. In fact, Python is one of the easiest programming languages to learn, and definitely the most fun.

Python is not just for beginners: professional programmers all over the world are having fun using Python in their work. For example, at Google (where I work) about 15% of all programs are written in Python. Programmers everywhere are having fun using Python for websites, games, databases, virtual reality, and so on.

Of course I want everyone to have as much fun programming as I do, and that is why I have made Python free software. Python's source code can be downloaded for free by everyone in the world, and that's not all: you can also pass it around to others. This is the true open source spirit of free software: share your creations with everyone. I hope you will use Python to create something wonderful, and share it with the world.

Even if you're not quite that ambitious, I assure you that you will have fun using Python and other free software like it.

`--Guido van Rossum`

creator of Python & mentor of ZEUUX

## 可爱的 Python（第一版）推荐序

作者：Guido van Rossum

亲爱的读者：

恭喜你挑选了一本关于编程的书！

编程要克服重重困难，因此学习编程将使你智慧永驻。我就是这样，编程让我获得了很多乐趣，这最终导致我创造了自己的编程语言，因此我获得了更多的乐趣。 Python 是最有趣的编程语言之一。她非常灵活，而且非常易于学习。这本书将教你如何用 Python 编程，哪怕你还没有学过任何一门编程语言。事实上，Python 是最容易学习的编程语言之一，当然也是最有趣的。

Python 不仅适合初学者，全世界众多的专业程序员正在使用 Python 语言，并享受其中。比如，在（我所工作的）Google 公司，15％的程序是用 Python 写的。各地的程序员正在网站、游戏、数据库和虚拟现实等领域用 Python 编程，他们都非常快乐！

我当然希望每一个人都能像我一样享受编程的乐趣，这就是我将 Python 作为自由软件发布的原因。世界上的任何人都可以免费下载 Python 的源代码，不仅如此，你还可以把她传递给你周围的人。这就是自由软件的精神：和大家分享你的创新！我希望你将用 Python 创造一些美妙的东西，并和世人分享。

即使你没那么大的野心，也没关系，我敢打赌，在你使用 Python 和其他自由软件的过程中，一定会获得巨大的乐趣。

`--Guido van Rossum`

Python 发明人、哲思社区顾问

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# 书序

[Python](http://python.org) [http://python.org] 是蕴含快乐编程思想的奇妙脚本语言,但是在中国程序员世界里并不为人所知,原因有很多; 本书试图使用一种比较草根的叙述形式来推广这一美好的语言, 决不教条或对比贬低其它"热门/主流"语言;-)

```py
def foo():
    print "Love Python, Love FreeDome"
    print "E 文标点,.0123456789,中文标点,. "

```

## 目标读者

> 本书假设读者有如下基本技能::
> 
> *   英文 1.618 级 -- 认 26 字母,会查字典,有基本 1024 个单词量,可以使用简单的 Cnglish 同外国友人沟通
> *   有至少一种计算机语言的编程经验,从 C 到 JavaScript 任何一种类型的计算机语言都可以.
> 
> 本书假定读者有如下希求::
> 
> *   期望有种工具语言可以快速解决日常的一些处理工作
> *   期望有种快速语言可以立即验证自个儿的想法
> *   期望有种胶水语言可以平滑的将旧有的各种系统统合在一起
> *   期望...

`那么,尝试 Python 吧!` 我们尽力将不同行业背景中喜欢上 Python 的感觉包含在文字中传达给具有丰富好奇心和学习全新技术勇气的你.

## 内容组织

本书内容主要来自 [CPyUG](http://wiki.woodpecker.org.cn/moin/CPUG) [http://wiki.woodpecker.org.cn/moin/CPUG] (中文 Python 用户组)的邮件列表,虽已尽可能的让各方面的叙述完整无缺,但是笔者们都不是什么作家,完全是因为对 Python 的热爱而组织起来,期望同中国的程序员们分享一下自个儿的快乐;所以,各种论述都带有很强烈的感情因素,而且因为篇幅所限无法深入讨论到 Python 的各种高级特性上去;对于真正的高人,本书最多是个散文的随想录了;

因为 Python 语言本身是种非常灵活的动态脚本语言,同一个目标可以使用多种方式完成,笔者们为了各种不同技术背景的读者可以快速无碍的理解,可能选择了种实际上比较笨拙的方式来实现功能,聪明的读者一定可以看出来的,那么请会心一笑,因为您已经和我们一样棒了!

## 本书结构

本书主要分成四部分:

> 第一部分 CDays 光盘实例故事::
> 
> *   根据设定的自制光盘管理软件的剧情,分成 10 日讲述使用 Python 从无到有自在的创建自个儿中意的软件的过程
> *   习题解答: [`wiki.woodpecker.org.cn/moin/ObpLovelyPython/LpyAttAnswerCdays`](http://wiki.woodpecker.org.cn/moin/ObpLovelyPython/LpyAttAnswerCdays)
> 
> > *   精巧地址: [`bit.ly/XzYIX`](http://bit.ly/XzYIX) ; SVN 下载: [`bit.ly/EGgXM`](http://bit.ly/EGgXM)
> 
> 第二部分 KDays 实用网站开发故事::
> 
> *   讲述如何 Pythonic 的运用即有框架在网络中解决实际问题;
> *   习题解答: [`wiki.woodpecker.org.cn/moin/ObpLovelyPython/LpyAttAnswerKdays`](http://wiki.woodpecker.org.cn/moin/ObpLovelyPython/LpyAttAnswerKdays)
> 
> > *   精巧地址: [`bit.ly/axi7`](http://bit.ly/axi7) ; SVN 下载: [`bit.ly/naqE7`](http://bit.ly/naqE7)
> 
> 第三部分 Py 初学者作弊条汇集::
> 
> *   同 Py 日实例故事呼应,以精简模式讲述各个关键语言知识点;并提供各种实用代码片段;
> *   分成以下几组:
>     1.  环境篇 ; ^(分享各种 Python 常用环境的使用技巧)
>     2.  语法篇 ; ^(说明 Python 语言最基础也的语法点)
>     3.  模块篇 ; ^(分享故事中涉及的各种常用模块的使用)
>     4.  框架篇 ; ^(介绍流行的几个 Python Web 应用框架)
>     5.  友邻篇 ; ^(分享一些在 Python 开发之外的相关领域基础知识)
> *   代码下载: [`openbookproject.googlecode.com/svn/trunk/LovelyPython/PCS/`](http://openbookproject.googlecode.com/svn/trunk/LovelyPython/PCS/)
> 
> > *   精巧地址: [`bit.ly/1IWqQW`](http://bit.ly/1IWqQW)
> 
> 第四部分 附录::
> 
> *   对以上所有内容的总结,给读者提供另一种理解 Python 的思维方式;
>     1.  行者箴言 ; ^(行者们的言论...包含很多靠谱的经验的,不听白不听;-))
>     2.  术语索引 ; ^(面对全新的动态对象脚本语言,不是各种术语是可以快速理解的,这里行者们尝试快速解说一下)
>     3.  Z 跋 ; ^(笔者记述的行者和编辑发生的各种故事)

注意

SVN(Subversion) 是一个流行的非常强大的版本管理系统,使用手册在:

> *   访问地址: [`www.subversion.org.cn/svnbook/1.4/index.html`](http://www.subversion.org.cn/svnbook/1.4/index.html)
> *   精巧地址: [`bit.ly/rgVp`](http://bit.ly/rgVp)

一般讲使用官方社区提供的图形化工具--TortoiseSVN(优秀的免费开源客户端)可以非常自然的在桌面上使用远程版本仓库,使用手册在:

> *   访问地址: [`svndoc.iusesvn.com/tsvn/1.5/`](http://svndoc.iusesvn.com/tsvn/1.5/)
> *   精巧地址: [`bit.ly/uPrd`](http://bit.ly/uPrd)
> *   软件下载: [`tortoisesvn.net/downloads`](http://tortoisesvn.net/downloads)

## 本书行文体例

本书使用不同的体例来区分不同的情景.

### 字体设定

`除非浏览器支持 HTML5 的服务端字体,否则只能是印刷的设定了`

> *   正文: 文泉驿正黑体 "wqy-zenhei"
> *   代码: Monaco 有灰底色,例如:`print map(foo, range(10))`
> *   旁注: 有边框效果 前导符号 出现在旁白/页脚

### 精巧地址

> 本书包含很多外部网站的 URL 地址,但是图书必竟不是网页,读者无法点击进入相关网站;所以,笔者尝试使用 URL 精简工具来帮助读者可以快速输入自动跳转到原有网站来访问;
> 
> > *   比如说: 本书的维基入口 [`wiki.woodpecker.org.cn/moin/ObpLovelyPython`](http://wiki.woodpecker.org.cn/moin/ObpLovelyPython)
> > *   精巧地址: [`bit.ly/2QA425`](http://bit.ly/2QA425)
> > *   输入的字符量少了三倍! 这是借助 [`bit.ly`](http://bit.ly) 提供的网址精简服务达到的效果;
> > *   提醒:毕竟这是借用外国的免费服务进行的精简,如果读者输入后不能自动跳转的话,可能是网络问题也可能是服务问题,那就只能麻烦读者重新使用原有的 URL 进入了;

### 程序体例

使用有语法颜色的代码引用

```py
def foo():
    print "Love Python, Love FreeDome"
    print "E 文标点,.0123456789,中文标点,. "

```

### 文本体例

**技巧警示:**

Note

(~_~)

*   This icon signifies a tip, suggestion, or general note.

Warning

(#_#)

*   警告得注意的...

See also

(^.^)

*   指向参考的...

**附加说明:**

进一步的

包含题外的信息,笔者心路,等等和正文有关,但是不直接的信息

**名词解释:**

是也乎是也乎

*   可以这么解释吧;
*   也可以这么来吧;

**知识引用:**

边注

表示以下内容出现在页面边注中

表示以下内容出现在边注中 将涉及内容指向后面的 `PCS*`

*   使用边注
*   追随正文
*   活动説明
*   效果如右

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# 前言

如果以厨艺来作比喻的话:

> *   "Learning Python" Mark Lutz / David Ascher ~ 等入门教材图书,应该算 白米饭
> *   "Python in a Nutshell" Alex Martelli ~ 等手册参考大书,应该算 大盘的素菜／汤
> *   "Dive Into Python" Mark Pilgrim ~ 等技术精解图书,应该算 极入味的 荤菜
> *   "Text Processing in Python" David Mertz ~ 等专门领域的详解图书,应该算风味名吃

众所周知：不吃主食得饿死的,不食蔬菜要生病的,光吃大荤一样没救的！到地头不来点风味会水土不服地;

"Lovely Python" 就是下酒的老醋花生,解酒的胡辣汤,下饭的泡菜！

*   PS: 中国特种部队野外装备里是使用 涪凌榨菜 作综合性盐/矿物质/维生素 补充品的 ;-)

本书总结中国一批先学先用了 Python 的行者们的亲身体验,为从来没有听说过 Python 的人,准备的一份实用的导学性图书;

*   试图将优化后的学习体验,通过故事的方式传达给只有学校系统学习体验的读者,同时也分享了蟒样(Pythonic)的知识获取技巧;
*   试图将最常用的代码和思路,通过 作弊条(Cheat Sheet~提示表单) 的形式分享给有初步基础的 Python 用户,来帮助大家多快好省的完成功能;

"Lovely Python" 期望成为学习使用 Python 的同好们的沟通话题,引发进一步的学习/应用/创造/推广!

行者

名词解释

*   (1)∶佛教语. 即“头陀”. 行脚乞食的苦行僧人;又指方丈的侍者及在寺院服杂役尚未剃发的出家者
*   (2)∶泛指修行佛道之人
*   (3)∶《西游记》中孙悟空的别名
*   在 [啄木鸟社区](http://wiki.woodpecker.org.cn/moin/WoodpeckerHackers) [http://wiki.woodpecker.org.cn/moin/WoodpeckerHackers] (精巧地址:[`bit.ly/TUzr3`](http://bit.ly/TUzr3))

被借用成为 Hacker 的中文专用词,意指在自由软件技术世界不断探寻前行的学习者...

## 本书阅读技巧

`Pythonic` ~简单的说就是使用 Python 的思维去解决问题的态度,记住 Python 就是 Python, 如果你拿 JAVA 的思路和方式来使用 Python 不是不可以,而是会得不偿失的...详细的,大家跟着内容蹓一圏,再和以往使用其它语言解决类似问题时的过相比较就知道了 ;-)

*   本书不是教材,不要期望可以根据本书泡制出考试大纲来获得什么认证
*   现实生活中的各种需求,不会根据教材的编制来要求的;所以,一切从需求出发,关注数据的处理,快速使用即有功能来完成愿望才是 Pythonic 的真髓!

`建议阅读态度:` ~ 学习 Python 不是什么大事儿 -- 和学习自行车类似,千万不要用学汽车的劲头来学习自行车:"非要先会拆修自行车了才敢骑行" -- 非要将 Python 的所有语法规则学完之后才敢真正使用 Python 来解决实际问题; 反了!其实这才是本末倒置的

> 1.  记住学习的目的就是要解决实际问题,如果一个语言的技法看不出与你现在的问题有关,就先甭理她! 看的多用的多了自然就会在合适的时机使用了,真的! ~ 这和学习英语时所谓“语感”类似的;)
> 2.  跟着实例故事走,不要想当前问题以外的事儿,依照眼前问题的解决顺序来逐步学习--虚无缥缈的语法规则是最难记忆的,只有和具体问题的解决绑定,才记的牢!
> 3.  看似零散的知识点,其实都是相通的,好比任何计算都可以归结为加减运算一样,不论多高深的技法,都可以使用粗浅直白的代码来完成相同的工作, **任何简陋但是可运行的代码,都比精致美观但是无法运行的代码要好!**
> 
> > *   所以,背好唐诗三百首,不会作诗也会吟! 背好英语 900 句,不会作文也得分!
> > *   甭总想着要跟着一个完美的教程走完才可以成为 Pythoner; 其实常见问题的处置代码看熟了,想不会写 Py 脚本也难了!

## 代码阅读技巧

没有技巧!

*   只要将代码 copy 到你的机器中运行,然后保持好奇心,有针对性的尝试小小修改一点,立即运行一下,看是否吻合自个儿的预想,就是最好的代码阅读技法!
*   Python 被设计成友好的,容易理解和使用的脚本语言,最好的学习方式就是使用之!期望大家在尝试后,平常也注意积累一些自个儿中意的代码片段,如果可以进一步分享回来那就太好了!

本书的所有代码都可以使用 SVN(Subversion)公开的下载:

*   下载地址: [`openbookproject.googlecode.com/svn/trunk/LovelyPython/`](http://openbookproject.googlecode.com/svn/trunk/LovelyPython/)

*   可爱的 Python 图书源码目录约定:

    LovelyPython/ +-- CDays (CDays 实例故事代码) +-- KDays (KDays 实例故事代码) +-- PCS (Python Cheat Sheet ~ Python 作弊条 内容) +-- exercise (各章练习,按照章节对应收集) -- pages (图书正文 维基格式文本目录)

`提醒:`

*   如果读者下载了相关代码时,发觉和图书中引用的代码有不同,不要惊奇,那是勤劳的行者们,在不断的优化实例代码!

### 反馈渠道

*本书是开放的,永远接受各种建议,看不过眼的聪明的读者可以直接在本书的专用邮件列表上进行交流,这样也许下一版的图书就有你的贡献了.*

> *   如果发现本书内容上任何方面的错误,行者们都将倾情接受指教;-) 别有心得的读者,任何时候想改进/改正/改善/改革本书的文字/代码/图片,都可以加入到图书专用讨论列表来,汇同行者们一齐来完善这本有趣的好书!
> *   提供方式与行者们沟通:

在线资源:

> *   邮件列表: [`www.zeuux.org/mailman/listinfo/zeuux-python`](http://www.zeuux.org/mailman/listinfo/zeuux-python)
> 
> > *   精巧地址: [`bit.ly/3rJucf`](http://bit.ly/3rJucf)
> > *   订阅后,就可以和所有本书的读者以及作者,以及所有订阅了此列表的中国 Pythoner 们分享图书以及 Pythonic 体验了!
> 
> *   意见反馈：[`code.google.com/p/openbookproject/issues`](http://code.google.com/p/openbookproject/issues)
> 
> > *   精巧地址: [`bit.ly/U5fAB`](http://bit.ly/U5fAB)
> > *   使用方式: 通过 Google 公司提供的项目管理管理环境,使用提案(Issue)的方式来提交意见,相关使用文档:
> > 
> > > *   快速使用 Issue 教程: [`code.google.com/p/openbookproject/wiki/UsageIssue`](http://code.google.com/p/openbookproject/wiki/UsageIssue)
> > > 
> > > > *   精巧地址: [`bit.ly/xxSHq`](http://bit.ly/xxSHq)
> > > 
> > > *   Issue 标签详解 [`code.google.com/p/openbookproject/wiki/IssueTags`](http://code.google.com/p/openbookproject/wiki/IssueTags)
> > > 
> > > > *   精巧地址: [`bit.ly/DOCdK`](http://bit.ly/DOCdK)
> > > 
> > > *   Issue 流程概述 [`code.google.com/p/openbookproject/wiki/IssueFlow`](http://code.google.com/p/openbookproject/wiki/IssueFlow)
> > > 
> > > > *   精巧地址: [`bit.ly/IXkNt`](http://bit.ly/IXkNt)
> > 
> > *   针对本书的各方面,自由提出看法,我们会认真处理,并及时同步到在线版本的图书中，持续修订到再版就可以将你的建议合并进来, 谢先 ;-)
> > *   反馈 Issue
> > 
> > > *   访问地址: [`code.google.com/p/openbookproject/issues/detail?id=1000`](http://code.google.com/p/openbookproject/issues/detail?id=1000)
> > > *   精巧地址: [`bit.ly/aa0F`](http://bit.ly/aa0F)
> > > *   样例

```py
标题:[LovPy]PCS304 AbtEurasia 增补缺少的,,
正文:
现在内容:
  Eurasia 项目沿革 中,,,
  "不过后来我们的团队很快编写了上百万行的智能,"
修订建议:
  增补上下文,好象没有说完!
  以及,作者简介也没有完成,,,
理由:
  读不通
```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# 感谢

想找本轻松言之有物的技术入门书,是非常困难的事儿;反推之,想写成一本有趣并有用的入门书也一准是非常困难的一件事儿;

这本书之所以可以诞生,不是个人的意志决定的,是由 Python 这门优秀语言自身的巨大吸引力凝聚而成的一大批中国 Pythoner 共同意识促生的!

真的!这本书的完成,要感谢的太多了! 所以,专门组织了这一独立章节,进行认认真真的``感谢``

## 人物

Guido van Rossum:

> *   ![../_images/beginning-1-zeuux-fashion-guido.jpg](img/beginning-1-zeuux-fashion-guido.jpg)
> *   后排穿"人生苦短 我用 Python" 字样 T 裇的帅大叔就是!
>     
>     
> *   饮水不忘挖井人,如果没有 Guido 蟒爹一时兴起挖出的这泓灵泉,就没有这本书了;-)
>     
>     
> 
> 让我们一齐感谢他！支持他！推广他的 "Simple is better" 世界观！

topic

此 T 裇由哲思自由软件社区设计并发售: 访问地址:[`www.zeuux.org/community/fashion/fashion.cn.html`](http://www.zeuux.org/community/fashion/fashion.cn.html) 精巧地址:[`bit.ly/1QjO0K`](http://bit.ly/1QjO0K)

> > 题词 No matter where you go, there you are.
> 
> —Buckaroo Banzai

Note

此 T 裇由哲思自由软件社区设计并发售: 访问地址:[`www.zeuux.org/community/fashion/fashion.cn.html`](http://www.zeuux.org/community/fashion/fashion.cn.html) 精巧地址:[`bit.ly/1QjO0K`](http://bit.ly/1QjO0K)

旁注

Optional Sidebar Subtitle

Subsequent indented lines comprise the body of the sidebar, and are interpreted as body elements.

## 社区

> 啄木鸟 Python 社区:: 由来自五湖四海,为了一个共同的革命目标--推广 Python 在中国的学习和应用--聚集在一起的行者形成. 今天已经领导着超过 5000 人口的根据地,但是相对中国的程序员群体而言还不够,还需要更大些,才能真正促成全国软件界的思想解放.
> 
> > *   访问地址: [`www.woodpecker.org.cn`](http://www.woodpecker.org.cn)
> > *   在此基础上又成立了以 *PyUG 为名的一批区域性 Python 用户组,主要的活动形式是线上列表讨论,项目组织,以及线下的``会课``--在各个城市的 Python 爱好者自发组织的一种线下进行的会面及技术课题交流活动--主要在北京和上海进行,从 08 年起珠三角/南昌/安徽/武汉各地也相继开展;比较活跃的有:
> > 
> > > *   CPyUG ~ 中文 Python 用户组,是最早成立的邮件列表之一,因其开放自由热情的讨论风格逐渐形成最有人气的中文 Python 技术讨论列表!
> > > 
> > > > *   访问地址: [`groups-beta.google.com/group/python-cn`](http://groups-beta.google.com/group/python-cn)
> > > > *   精巧地址: [`bit.ly/13BJb`](http://bit.ly/13BJb)
> > > 
> > > *   BPyUG ~ 北京 Python 用户组,形成了良好的不定期会课制度,组织 Python 行者们当面交流
> > > *   ZPyUG ~ 珠三角地区 Python 用户组,由 Zoom.Quiet 南下发起,同样以会课为主要形式进行线下交流
> 
> CZUG.org:: China Zope User Group ~ 中国 Zope 用户组;
> 
> > *   访问地址: [`czug.org`](http://czug.org)
> > *   Zope 是一个开放源代码的 Web 应用平台.Plone 是 Zope 上的一个用户友好、功能强大的开放源代码内容管理系统. Plone 适合用作门户网站、企业内外网站、文档发布系统、协同群件工具,Plone 也是一个应用开发平台. CZUG.org 里是 Zope 开源 web 应用服务器和 Plone 开源内容管理系统的中文技术社区.
> > *   几乎所有啄木鸟 Python 社区的早期成员都来自 CZUG.org,可想此社区的历史;
> 
> 新浪网:: 新浪在全球范围内注册用户超过 2.3 亿,各种付费服务的常用用户超过 4200 万,日浏览量超过 7 亿多次,是中国大陆及全球华人社群中最受推崇的互联网品牌. 是啄木鸟社区的主要赞助商.
> 
> > *   访问地址: [`sina.com.cn`](http://sina.com.cn)
> 
> 博文视点:: 电子工业出版社博文视点资讯有限公司是信息产业部直属的中央一级科技与教育出版社——电子工业出版社,与国内最大的 IT 技术网站——CSDN.NET 和最具有专业水准的 IT 杂志社——《程序员》杂志社联合成立的,以 IT 图书出版为主,并开展相关信息和知识增值服务的出版公司. 博文视点致力于 IT 出版,为 IT 专业人士提供真正专业、经典的好书.
> 
> > *   博文视点的宗旨是：IT 出版以教育为本. 博文视点愿与向上的心合作,共同成长！
> > *   博文视点专家团是利用 Google 提供的 Group ~ 邮件列表服务建立的一个作者/编辑自由交流的社区:
> > 
> > > *   访问地址: [`groups.google.com/group/BVtougao`](http://groups.google.com/group/BVtougao)
> > > *   精巧地址: [`bit.ly/MP2B`](http://bit.ly/MP2B)

## 行者

行者~中国 Python 社区中的自称,这本书就是由众多华蟒行者们完成的,当然要大力感谢(按照掺合图书工程的先后顺序排列,收集各自的成书感言和贡献):

> Zoom.Quiet::
> 
> *   贡献: 图书创意/工程管理;实例故事;书/谢/Z 序;部分 PCS^简述/PCS301~303/PCS304(引述及章节设计)/PCS400~404^;附录引言;行者箴言;资源索引;后记故事;
> *   工作: 珠海,金山软件股份有限公司,过程改进经理
> *   经验: 2000 年从 Zope 开始接触 Python;主要进行 Web 应用/数据分析;组织以 Trac 为核心的敏捷开发支持平台;关注社会化学习和知识管理;学习 PyPy 并尝试和 Erlang 结合 ing;
> *   环境: * HP 520(GQ349AA) * Ubuntu 8.04 - Hardy Heron * Python 2.5.2 (r252:60911, Jun 21 2008, 09:47:06)
> 
> 清风::
> 
> *   贡献:PCS 模块篇 200~207,209~214;PCS300 部分(回收的章节:Py 常见任务处理);SVN 到维基自动批量更新脚本;
> *   工作:新浪网
> *   经验:学习使用 Python 5 年左右.目前是某个 Python+Django 项目的 leader
> *   环境: * iBook G4 * Mac OS X * Python 2.4.3
> 
> XYB::
> 
> *   贡献:(回收章节:实例 CookBook 索引)
> *   工作: 豆瓣，软件工程师
> *   经验: 2000 年接触 Python，用它来写系统维护脚本；2003 年开始以 Python 开发谋生。
> *   环境:
> 
> > *   Mac Book Pro
> > *   OS X 10.5
> > *   Python 2.5
> 
> 黄毅::
> 
> *   贡献: PCS300;PCS304.Django~最流行框架快速体验教程+深入探索 Python 的 Web 开发;
> *   工作：腾讯，程序员，主要是 web 前后台。
> *   经验：2003 年末，大三的时候开始接触 Python，通过 Python 学习到很多很多好东西，在大学期间也用 Python 完成了几个 web/gui 的项目赚点零花钱 ;-) 很可惜目前工作跟 Python 没有什么关系。
> *   环境：thinkpad x61; ubuntu8.10; Python2.5.2
> 
> 张沈鹏::
> 
> *   贡献: (回收章节:Py2.5 绝对简明手册)
> *   工作: 北京,豆瓣,程序员
> *   经验: 从一个抓网页的小程序开始结识 Python，关注 Python 在互联网方面的应用，并喜欢用 Boost:Python 给 Python 写扩展。常在博客上记录一些 Python 学习心得,访问地址: [`zsp.javaeye.com/`](http://zsp.javaeye.com/) (路过打酱油的,曾尝试写一个章节,因种种原因最终未完成.但 Zoom.Quiet 大人居然把我列在这里,博文视点的编辑还亲自打电话来,让我很汗颜...)
> *   环境: Dell INSPIRON 2200 + WindowsXP + SSH 远程登陆 Gentoo 编程
> 
> 盛艳(Liz)::
> 
> *   贡献: 实例故事练习题设计/附录:故事练习解答;PCS 环境篇/语法篇(除 PCS114 FP 初体验);术语索引;Py 资源索引等等文字校对;
> *   工作: 扬州大学信息学院计算机系研二学生,主研究方向是数据挖掘,概念格.
> *   经验: 从 2007 年 10 月开始学习 Python,非常喜欢她的风格,目前还在不断深入学习中.主要进行 Web 应用开发和编写些小脚本,非常高兴能够掺合"可爱的 Python"的编写,以后会继续努力,为社区贡献一份力量.
> *   环境: 当前日常工作环境(软硬件) * 方正尊越 A360 * Ubuntu 8.04 * Python 2.5.2
> 
> 刘鑫::
> 
> *   贡献: PCS114 FP 初体验;
> *   工作: 珠海,金山软件股份有限公司,软件架构师
> *   经验: 从 2002 年接触 Python,现在使用 Python 搭建中间服务器，曾经尝试在己有的游戏服务器中嵌入 Python 进行功能扩展。也一直使用 Python 编写各种开发过程中所需的辅助工具。
> *   环境: * HP 520(GQ349AA)/组装 PC(AMD2300+/1G) * Ubuntu 8.04 - Hardy Heron * Python 2.5.2 (r252:60911, Jun 21 2008, 09:47:06)
> 
> Limodou::
> 
> *   贡献: PCS208;PCS304.UliWeb ~ 关注平衡感的自制框架;
> *   工作: 北京,程序员;
> *   经验: 2000 看开始学 Python，从此之后 Python 成为我掌握最熟练，最喜欢的语言了。曾担任 Linuxforum.net 的 Python 版版主。CPyUG(Chinese Python User Group，2005 年创建)创始人之一，也是 Python-cn 邮件列表(2004 年创建，目前为 CPyUG 的主力邮件列表)创建人。
> 
> > *   喜欢编程，喜欢分享，喜欢与人交流，喜欢技术博客，到目前为计，已经写了近 1000 篇左右 Blog。在 CPyUG 的多次会课中进行心得的分享。
> > *   参与过多项开源项目，并于 2004 年开始开发 NewEdit，后改名为 UliPad，此作品曾参加第一届中国开源软件竞赛银奖。还自主开发过其它小型开源项目。目前 Python 仍然是业余爱好，但是会一直坚持下去。
> 
> *   环境:
> 
> > *   主要是在 Windows 下，有时在 Ubuntu 下。
> > *   Python 2.4+
> 
> 沈崴::
> 
> *   贡献: PCS304.Eurasia ~ 关注高性能的原创框架
> *   工作: 上海, 高级架构师
> *   经验: 1993 年的程序员, 2001 年初完全转到 Python。
> *   环境:
> 
> > *   硬件: IBM Thinkpad (数个型号)、EeePC、AMD64、MIPSEL 等机型
> > *   系统: Debian 系 (包括 Ubuntu 704-810)、BSD 系、OpenWRT 等操作系统, 使用 Stackless Python 2.5.2
> 
> 洪强宁/QiangningHong/hongqn::
> 
> *   贡献：PCS304.Quxiote ~ 豆瓣动力核心
> *   工作：北京豆瓣互动科技有限公司，技术负责人
> *   经验：C 背景程序员，2002 年开始接触 Python，2004 年开始完全使用 Python 工作。2006 年加入豆瓣以来，用 Python 作为网站开发的利器，得心应手，十分快活。
> *   环境：
> 
> > *   桌面：
> > 
> > > *   MacBook Pro 133
> > > *   Mac OS X 10.5.5
> > > *   Python 2.5.2 (r252:60911, Sep 30 2008, 12:02:56)
> > 
> > *   服务器：
> > 
> > > *   自攒 AMD64 服务器二十余台
> 
> 潘俊勇::
> 
> *   贡献: PCS304.Zope ~ 超浑厚框架
> *   工作: 上海润普广州公司(zopen.cn)技术总监
> *   经验: 2002 年开始折腾 zope 至今，李木头(Limodou)当年就是俺崇拜的对象，俺专一一点就这个优点。在 CZUG.org 中提供了 Zope/Plone 上的全套中文/阴历支持包；
> *   环境: * Ubuntu 8.04 - Hardy Heron * Python 2.5.2

`特别要指出的是`:核心撰写团队成员大多使用非 Windows 操作系统作为日常工作环境的,所以,如果在截屏或是代码运行结果上和你在本地的尝试结果不同时不要惊讶,应该惊喜--Python 是跨平台的! 不论大家工作/生活在什么操作系统中,都可以友好快捷的協助完成你想要的功能!

## 校对

在图书工程的最后时刻哲思社区的西安邮电学院成员,主动担当了技术校对,并高效及时的进行了 3 遍复查,为保证图书质量作出了重大贡献,特此感谢:

> 孔建军(kongove)::
> 
> *   贡献: 负责审校团队工作协调;完成 CDaysKDays 模块篇等章节的审校;PCS215/216;
> *   工作: 就读于西安邮电学院，网络工程专业，热衷于 Web 应用和系统开发
> *   经验: 接触 Python 不到一年，做过一些网络编程和 GUI 程序，目前正在关注哲思系统的开发
> *   环境:
> 
> > *   组装 AMD 2800+/1.5G 内存台式机
> > *   Ubuntu 7.10 - Gutsy Gibbon
> > *   Python 2.5.1 (r251:54863, Jul 31 2008, 23:17:40)
> 
> 高辉(aurthmyth)::
> 
> *   贡献: 实例故事练习题设计和解答，环境篇语法篇等审校;PCS217;
> *   工作: 就读于西安邮电学院计算机系
> *   经验:接触 Python 不到一年,正在深入学习中，很惊叹她的高效和风格.非常幸运地参与"可爱的 Python"的校对,为 Python 在国内的推广贡献一份力量.
> *   环境: 当前日常工作环境
> 
> > *   Ubuntu 7.04 - Feisty Fawn
> > *   Python 2.5.2
> 
> 张斌/SK::
> 
> *   贡献: 完成相关审校工作
> *   工作: 就读于西安邮电学院，主要探索 web 开发
> *   经验: 接触 Python 仅有半年时间，熟悉 Python 的基础知识，平时写写小的脚本，做过一些 Python 的 web 开发
> *   环境:
> 
> > *   HPV3414/双核 T2130
> > *   Ubuntu 8.04 - Hardy Heron
> > *   Python 2.5.2
> 
> 潘猛::
> 
> *   贡献: 部分章节的审校
> *   工作: 就读于西安邮电学院
> *   经验: 接触 Python 不到一年，目前正在用 Python 开发一个电子书管理系统
> *   环境:
> 
> > *   组装 AMD Athlon 64 X2 3600+/1G 内存台式机
> > *   Ubuntu 8.04 - Hardy Heron
> > *   Python 2.5.2
> 
> 冯立强::
> 
> *   贡献: 部分章节的审校
> *   工作: 就读于西安邮电学院
> *   经验: 接触 Python 不到一年，目前正在用 Python 开发一个电子书管理系统
> *   环境:
> 
> > *   组装 AMD Athlon 64 X2 4000+/1G 内存台式机
> > *   opensuse11.0
> > *   Python 2.5.2

## 工具

> UliPad:: UliPad 是一个编辑器,你可以用它来进行你的文档写作,编程开发. 它使用 Python 编程语言开发,用户界面基于 wxPython . 它除了想要完成一般编辑器的常用功能之外,最主要是想实现一种方便、灵活的框架,开发者可以方便地开发新的功能.
> 
> > *   UliPad 支持 代码着色、智能补全、代码调试、Python 类浏览、代码片段、Ftp 功能、目录浏览等等强大功能,其 Doc 目录下的文档更是非常丰富,是你编写 Python 的绝世好帮手！
> > 
> > > *   访问地址: [`wiki.woodpecker.org.cn/moin/UliPad`](http://wiki.woodpecker.org.cn/moin/UliPad)
> > > *   精巧地址：[`bit.ly/3khJdv`](http://bit.ly/3khJdv)
> 
> 中蟒:: 中蟒和 Python 基本上是相容的,对象是电脑编程的初学者、 对编程概念不了解的人以及不打算以程序员为职业的人. 对这些人来说, 能用中文来学习并完成简单的编程工作是一件很不错的事. 对于英文基础不好的入门者来说, 以中文进行学习可以专注于编程的概念, 常用的算法, 程序的逻辑这些东西.
> 
> > *   访问地址: [`www.chinesePython.org`](http://www.chinesePython.org)
> 
> 周蟒:: 周蟒(zhpy)是 Python 的替身环境,完全利用 Python 原生环境,包裹中文关键字替换模块形成的中文编程环境. 和中蟒不同在于:周蟒没有改动 Python 本身任何代码,是个标准的 Python 软件;
> 
> > *   访问地址: [`code.google.com/p/zhpy`](http://code.google.com/p/zhpy)
> > *   精巧地址: [`bit.ly/ZMWU`](http://bit.ly/ZMWU)

## 还有,,,

> 更多的感谢::
> 
> *   感谢博文视点(武汉)出版社的编辑们,他们前赴后继的鼓励我们,不断的鞭策我们坚持不懈的撰写这部小书,要没有他们的奋斗,这本书可能还得等几年!
> *   感谢 CPyUG/BPyUG/ZPyUG 等等相关列表中不知名的朋友们的意见和鼓励,本书作为一个开放图书工程,没有他们的参与是无法成功的!

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest
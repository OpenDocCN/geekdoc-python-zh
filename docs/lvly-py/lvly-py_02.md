# 实例故事

*   实例故事
*   CDays
    *   CDay -5: Python 初体验和原始需求
    *   CDay -4
    *   CDay -3
    *   CDay -2
    *   CDay -1
    *   CDay 0
    *   CDay +1
    *   CDay +2
    *   CDay +3
    *   CDay +N
*   KDays
    *   KDay 0
    *   KDay 1
    *   KDay 2
    *   KDay 3
    *   KDay 4
    *   KDay 5
    *   KDay 6
    *   KDay N

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# 实例故事

~ Realy Taste Storys

> 笔者自 2000 年通过自学 PHP,从而进入自由软件世界接触到 Python,到 2004 年参与啄木鸟 Python 社区,2005 参与创立 CPyUG,对 Python 的兴趣和信心是一直增长着的; 但发觉世人对这一优秀语言的认知非常非常有限,即使各个综合技术社区中已经出现 Python 板块,又甚或 Google 公司的成功,都没有促动起 Python 在中国都的发展;
> 
> 这才开始思考 Python 的推广,反思自个儿的 Python 学习体验,寻找是否有更加轻松的学习方式以及好书来促进 Python 在中国落地生根;再后来通过对社会化学习方面的体验,意识到:

学习的阶级 从低到高应该是:

```py
不知己不知 < 不知己知 < 知己知 < 知己不知
```

为什么这么说?

*   开始涉及一个全新技术领域时,我们不知道我们对这个领域的任何信息,也就是连`不知道什么`都没有概念;
*   后来,通过不自觉的下意识的信息接收,或是主动的学习,获得了部分相关信息,但是从来没有考虑使用这领域的知识来解决问题,所以,不知道自个儿已经知道了哪些领域内的实用信息;
*   再后来,通过实践,切实掌握了领域的基础知识,并对领域内的所有方面都有所了解,明确在领域内自个儿掌握了什么;好比举着油灯进在一个大图书馆中,刚好照亮了周围几排书架,但是不清楚在光圈以外还有多少书架;
*   最后,推而广之,对领域相关的所有知识领域都有所了解,掌握了领域间的关系,明白自个儿究竟还有多少知识不知道;好比那个大图书馆中,打开了所有的电灯,明白自个儿身边这一整书架同整个图书馆其它书架的关系;

感觉,只有快速达到`知己知`的阶段,才可能事半功倍的高效率/自信/自觉的继续学习下去;

*   幸好,Python 真的的确是个易学好用的语言,怎么向其它没有摸到 Python 秉性的人们宣传这种体验呢?

*   developerWorks 中 David Mertz 创作的“可爱的 Python”系列 ：

    *   访问地址: [`www.ibm.com/developerworks/cn/linux/theme/special/index.html#python`](http://www.ibm.com/developerworks/cn/linux/theme/special/index.html#python)
    *   精巧地址: [`bit.ly/mfUT`](http://bit.ly/mfUT)

也就成为本书的原型结构了:

*   将 Python 最可爱的方面,以小篇幅的轻松形式组织发布出来!
*   接下来就是两个以解决实际问题为出发点,有剧情有人物的,模拟真实事件发展而记述的,一个小白的成长故事. 希望读者可以跟随小白轻松体验到 Pythonic ;-)

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDays

*   CDay -5: Python 初体验和原始需求
    *   剧本背景
        *   人物
        *   约定
    *   事件
    *   发动
        *   Python!
        *   Hello World!
        *   文档
    *   原始需求
    *   小结
    *   练习
*   CDay -4
    *   需求
    *   练习
*   CDay -3
    *   需求
    *   练习
*   CDay -2
    *   需求
    *   练习
*   CDay -1
    *   需求
    *   练习
*   CDay 0
    *   需求
    *   练习
*   CDay +1
    *   需求
    *   练习
*   CDay +2
    *   需求
    *   练习
*   CDay +3
    *   需求
    *   练习
*   CDay +N
    *   需求
    *   练习

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay -5: Python 初体验和原始需求

`Just use it! don't learn!` -- 用之,不学!

## 剧本背景

嗯嗯嗯,所谓实例故事,就是设计一个具体情景,让代表读者的初学者,同代表作者的行者沟通,从而完成一件事儿,在过程中引导式的展示给大家 Python 的乐趣;

当然读者不一定什么都不知道,作者可能也高明不过读者, 但是,有个具体的事,也就好具体的讲起来.

好的,这就开始,依照正统说书的,也得来个定场诗什么的活跃一下气氛,先:

那么...

```py
侧有咖啡,后宝石；还是灵蟒最贴心！
最贴心,不费心, 用好还要想清楚.
想清楚,就清楚, 一切清楚才清爽！
要清爽,常重构！ 刚刚够用是王道！
```

### 人物

小白

读者一方,没有或是仅有一点编程体验的好奇宝宝,想快速上手使用 Python 解决实际问题

行者

嗯嗯嗯!啄木鸟/CPyUG 等等中国活跃 Python 社区的那群热心的 Python 用户,说话可能有些颠三倒四,但是绝对是好心人们

### 约定

列表

指 `邮件列表` -- 一种仅仅通过邮件进行群体异步交流的服务形式,是比 BBS 更加古老和有效的沟通方式

详细::

邮件列表有古老的规范和礼节

*   访问地址: [`www.woodpecker.org.cn/share/classes/050730-CPUG/usMaillist/`](http://www.woodpecker.org.cn/share/classes/050730-CPUG/usMaillist/)
*   精简地址: [`bit.ly/43WKcR`](http://bit.ly/43WKcR)

CPyUG 社区有丰富的列表资源 - 访问地址: [`wiki.woodpecker.org.cn/moin/CPUGres`](http://wiki.woodpecker.org.cn/moin/CPUGres) - 精简地址: [`bit.ly/vrqUk`](http://bit.ly/vrqUk)

小结

指每日故事最后的独立章节,将当日故事情节中涉及的知识点/领域技术 进行集中简述,以便读者明确要点;

练习

> 指每日故事最后的额外章节,和故事内容可能没有关联的几个小实用问题,因为必须使用前述涉及的知识点/领域技术才可以解决,所以,特别列出,建议读者独立进行尝试,加强相关知识的体验.

*   习题解答发布在图书维基: [`wiki.woodpecker.org.cn/moin/ObpLovelyPython/LpyAttAnswerCdays`](http://wiki.woodpecker.org.cn/moin/ObpLovelyPython/LpyAttAnswerCdays)
*   精巧地址: [`bit.ly/XzYIX`](http://bit.ly/XzYIX)
*   用 SVN 下载: [`openbookproject.googlecode.com/svn/trunk/LovelyPython/exercise/part1-CDays/`](http://openbookproject.googlecode.com/svn/trunk/LovelyPython/exercise/part1-CDays/)

## 事件

小白忽然间厌烦了不断的下载安装,破解,却总是找不到称心的软件的生活： “怒了! 什么破软件这么不好使,还要 150$!!! 我!要!自个儿写!”

## 发动

怎么回事儿呢? 小白到列表中一说,原来是买了台刻录机,这一下, eMule 的下载更加是没日没夜了,才一个月刻录出来的光盘就有上百张了,结果想找回一个专辑的 MP3,简直不可能了...

想要一种工具:`可以不用插入光盘就可以搜索所有光盘的内容`

就这么简单的一个愿望,乍就是找不到好用的软件?!

### Python!

OK!你们都说 Python 好用,那么来尝试一下吧! 我是菜鸟我怕谁?!

运行环境:

*   推荐 `ActivePython` ,虽然此乃商业产品,却是一个有自由软件版权保证的完善的 Python 开发应用环境,关键是文档及相关模块预设非常齐备;
*   GNU/Linux 环境中,当然推荐使用原生的 Python.org,主流的自由操作系统发行版都内置了 Python 环境,或是对应的软件仓库中都有合适的 Python 版本可以选择,安装和使用也非常方便;

详细::

PCS3 交互环境之 winpy

*   包含在 `ActivePython`_ 支行版中,商业公司,但是对自由软件支持良多
*   `Python`_.org Python 语言本身的大本营

好了,下载,安装,没什么说的,这再不会,先进行电脑基本操作扫盲,再来学习编程吧... ;-)

### Hello World!

灰常灰常著名的,但凡是编程语言,第一课都要玩的例程,如果你也想看一看 Python 的?

图 CDay-5-1 Hello World 示例

attachment:cday-5-hello-world.png

再 Show 一个类似的,但是推荐的体验环境 iPython

详细::

PCS2 交互环境之 iPython

[`ipython.scipy.org/`](http://ipython.scipy.org/) 是个融合了 N 多 Unix Shell 环境特质的 Python 交互式命令行环境 ,推荐使用,你会爱上 TAB 键的;-)

图 CDay-5-2 Hello World 示例(iPython)

attachment:cday-5-hello-world-ipython.png

是也乎,就是这么简单,告诉 Python 打印"Hello World!" 就好.

所以说,对于 Python, 勿学,即用就好!

### 文档

但是丰富的文档还是可以安抚我们面对未知的恐惧的*,推荐以下深入阅读资料,但是不推荐现在就全面阅读

> *   Python Tutorial -- Python 教程中文版本
> 
> > *   在线访问: [`wiki.woodpecker.org.cn/moin/March_Liu/PyTutorial`](http://wiki.woodpecker.org.cn/moin/March_Liu/PyTutorial)
> > *   快速地址: [`tinyurl.com/6h2q7g`](http://tinyurl.com/6h2q7g)
> > *   是 CPyUG ~ Chinese Python User Group 中国 Python 用户组的资深专家,刘鑫长期维护的一部基础文档,也是 Python 创造者 Guido.van.Rossum 唯一亲笔撰写的技术文档!
> 
> *   `A Byte Of Python` -- 简明 Python 教程
> 
> > *   在线访问: [`www.woodpecker.org.cn/share/doc/abyteofpython_cn/chinese/index.html`](http://www.woodpecker.org.cn/share/doc/abyteofpython_cn/chinese/index.html)
> > *   快速地址: [`tinyurl.com/5k8pv5`](http://tinyurl.com/5k8pv5)
> > *   沈洁元 翻译的一篇流传甚广的学习 Python 的小书,从初学者的角度,快速说明了一些关键知识点
> > *   原作者是印度的一位年青的程序员,大家可以到这本书的网站直接和作者沟通:
> > 
> > > *   [`www.swaroopch.com/byteofpython/`](http://www.swaroopch.com/byteofpython/)
> 
> *   Python 标准库 中文版
> 
> > *   在线访问: [`www.woodpecker.org.cn/share/doc/Python/_html/PythonStandardLib/`](http://www.woodpecker.org.cn/share/doc/Python/_html/PythonStandardLib/)
> > *   快速地址: [`tinyurl.com/5pmvkn`](http://tinyurl.com/5pmvkn)
> > *   由"Python 江湖 QQ 群"共同完成的 Python 2.0 内置所有标准模块的说明,是初学者开发过程中必备的参考
> 
> *   ASPN -- `Python Reference` ~ Activestate 公司 Python 参考资料汇总:
> 
> > *   在线访问: [`aspn.activestate.com/ASPN/Python/Reference/`](http://aspn.activestate.com/ASPN/Python/Reference/)
> > *   快速地址: [`tinyurl.com/58t3sl`](http://tinyurl.com/58t3sl)

## 原始需求

嗯嗯嗯!安装好了 Python 环境,在行者的指点下又收集了一批资料的链接,那么小白想真正开始软件的创造了, 但是,行者又告戒:

> *   '''明晰你的问题,当问题真正得到定义时,问题已经解决了一半'''
> 
> > 1\. 因为,程序不过是将人的思想转述为机器可以理解的操作序列而已 1\. 对于寻求快速解决问题,而不是研究问题的小白和 Pythoner 们,精确,恰当的描述问题,就等于写好了程序框架,余下的不过是让程序可以运行罢了

好的,于是小白直觉的将软件需求细化了一下:

> *   '''可以不用插入光盘就可以搜索所有光盘的内容''', 等于说...
> 
> > *   可以将光盘内容索引自动储存到硬盘上
> > *   可以根据储存到硬盘上的光盘信息进行搜索

仅仅就这两点,也仅此两点的需求,可以?如何?以及怎样通过 Python 实现?小白和读者一同期待...

## 小结

作为开始,今天小白决定使用 Python 来解决光盘内容管理,这一实际问题; 安装了 python 环境,运行了 "Hello World!" 实例.

OK!轻松的开始,但是,你知道,你同时也获得了免费的绝对强大的科学计算器嘛?

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay -4

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay -3

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay -2

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay -1

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay 0

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay +1

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay +2

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay +3

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# CDay +N

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDays

*   KDay 0
    *   需求
    *   练习
*   KDay 1
    *   需求
    *   练习
*   KDay 2
    *   需求
    *   练习
*   KDay 3
    *   需求
    *   练习
*   KDay 4
    *   需求
    *   练习
*   KDay 5
    *   需求
    *   练习
*   KDay 6
    *   需求
    *   练习
*   KDay N
    *   需求
    *   练习

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 0

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 1

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 2

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 3

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 4

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 5

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay 6

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest

# KDay N

## 需求

## 练习

```py
@route('%s/'%ini.urlprefix)
def index():
    __urlog("INFO","idx++")
    return template('index.tpl',urlprefix=ini.urlprefix)

```

© Copyright 2010, Zoom.Quiet. Last updated on 130306 09:15. Created using [Sphinx](http://sphinx.pocoo.org/) 1.1.3.

Brought to you by Read the Docs

*   latest
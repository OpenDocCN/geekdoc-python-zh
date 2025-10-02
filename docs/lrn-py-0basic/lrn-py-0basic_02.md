# 二、基本数据类型

## 数和四则运算

一提到计算机，当然现在更多人把她叫做电脑，这两个词都是指 computer。不管什么，只要提到她，普遍都会想到她能够比较快地做加减乘除，甚至乘方开方等。乃至于，有的人在口语中区分不开计算机和计算器。

有一篇名为[《计算机前世》](http://www.flickering.cn/%E5%85%AB%E5%8D%A6%E5%A4%A9%E5%9C%B0/2015/02/%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%89%8D%E4%B8%96%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%8C%E5%A7%91%E5%A8%98%E8%AE%A1%E7%AE%97%E6%9C%BA%EF%BC%89/)的文章，这样讲到：

> 还是先来看看计算机（computer）这个词是怎么来的。 英文学得好的小伙伴看到这货，computer 第一反应好像是：“compute-er”是吧，应该是个什么样的人就对了，就是啊，“做计算的人”。 叮咚！恭喜你答对了。 最先被命名为 computer 的确实是人。也就是说，电子计算机（与早期的机械计算机）被给予这个名字是因为他们执行的是此前被分配到人的工作。 “计算机”原来是工作岗位，它被用来定义一个工种，其任务是执行计算诸如导航表，潮汐图表，天文历书和行星的位置要求的重复计算。从事这个工作的人就是 computer，而且大多是女神！

原文还附有如下图片：

![](img/10201.jpg)

所以，以后要用第三人称来称呼 computer，请用 she（她）。现在你明白为什么程序员中那么多“他”了吧，因为 computer 是“她”。

### 数

在 Python 中，对数的规定比较简单，基本在小学数学水平即可理解。

那么，做为零基础学习这，也就从计算小学数学题目开始吧。因为从这里开始，数学的基础知识列位肯定过关了。

```py
>>> 3
3
>>> 3333333333333333333333333333333333333333
3333333333333333333333333333333333333333L
>>> 3.222222
3.222222 
```

上面显示的是在交互模式下，如果输入 3，就显示了 3，这样的数称为整数，这个称呼和小学数学一样。

如果输入一个比较大的数，第二个，那么多个 3 组成的一个整数，在 Python 中称之为长整数。为了表示某个数是长整数，Python 会在其末尾显示一个 L。其实，现在的 Python 已经能够自动将输入的很大的整数视为长整数了。你不必在这方面进行区别。

第三个，在数学里面称为小数，这里你依然可以这么称呼，不过就像很多编程语言一样，习惯称之为“浮点数”。至于这个名称的由来，也是有点说道的，有兴趣可以 google.

上述举例中，可以说都是无符号（或者说是非负数），如果要表示负数，跟数学中的表示方法一样，前面填上负号即可。

值得注意的是，我们这里说的都是十进制的数。

除了十进制，还有二进制、八进制、十六进制都是在编程中可能用到的，当然用六十进制的时候就比较少了（其实时间记录方式就是典型的六十进制）。

具体每个数字，在 Python 中都是一个对象，比如前面输入的 3，就是一个对象。每个对象，在内存中都有自己的一个地址，这个就是它的身份。

```py
>>> id(3)
140574872
>>> id(3.222222)
140612356
>>> id(3.0)
140612356
>>> 
```

用内建函数 id()可以查看每个对象的内存地址，即身份。

> 内建函数，英文为 built-in Function，读者根据名字也能猜个八九不离十了。不错，就是 Python 中已经定义好的内部函数。

以上三个不同的数字，是三个不同的对象，具有三个不同的内存地址。特别要注意，在数学上，3 和 3.0 是相等的，但是在这里，它们是不同的对象。

用 id()得到的内存地址，是只读的，不能修改。

了解了“身份”，再来看“类型”，也有一个内建函数供使用 type()。

```py
>>> type(3)
<type 'int'>
>>> type(3.0)
<type 'float'>
>>> type(3.222222)
<type 'float'> 
```

用内建函数能够查看对象的类型。<type>，说明 3 是整数类型（Interger）；<type>则告诉我们那个对象是浮点型（Floating point real number）。与 id()的结果类似，type()得到的结果也是只读的。</type></type>

至于对象的值，在这里就是对象本身了。

看来对象也不难理解。请保持自信，继续。

### 变量

仅仅写出 3、4、5 是远远不够的，在编程语言中，经常要用到“变量”和“数”（在 Python 中严格来讲是对象）建立一个对应关系。例如：

```py
>>> x = 5
>>> x
5
>>> x = 6
>>> x
6 
```

在这个例子中，`x = 5`就是在变量(x)和数(5)之间建立了对应关系，接着又建立了 x 与 6 之间的对应关系。我们可以看到，x 先“是”5，后来“是”6。

在 Python 中，有这样一句话是非常重要的：**对象有类型，变量无类型**。怎么理解呢？

首先，5、6 都是整数，Python 中为它们取了一个名字，叫做“整数”类型的数据，或者说数据类型是整数，用 int 表示。

当我们在 Python 中写入了 5、6，computer 姑娘就自动在她的内存中某个地方给我们建立这两个对象（对象的定义后面会讲，这里你先用着，逐渐就明晰含义了），就好比建造了两个雕塑，一个是形状似 5，一个形状似 6，这就两个对象，这两个对象的类型就是 int.

那个 x 呢？就好比是一个标签，当`x = 5`时，就是将 x 这个标签拴在了 5 上了，通过这个 x，就顺延看到了 5，于是在交互模式中，`>>> x`输出的结果就是 5，给人的感觉似乎是 x 就是 5，事实是 x 这个标签贴在 5 上面。同样的道理，当`x = 6`时，标签就换位置了，贴到 6 上面。

所以，这个标签 x 没有类型之说，它不仅可以贴在整数类型的对象上，还能贴在其它类型的对象上，比如后面会介绍到的 str（字符串）类型的对象等等。

这是 Python 区别于一些语言非常重要的地方。

### 四则运算

按照下面要求，在交互模式中运行，看看得到的结果和用小学数学知识运算之后得到的结果是否一致

```py
>>> 2+5
7
>>> 5-2
3
>>> 10/2
5
>>> 5*2
10
>>> 10/5+1
3
>>> 2*3-4
2 
```

上面的运算中，分别涉及到了四个运算符号：加(+)、减(-)、乘(*)、除(/)

另外，我相信看官已经发现了一个重要的公理：

**在计算机中，四则运算和小学数学中学习过的四则运算规则是一样的**

要不说人是高等动物呢，自己发明的东西，一定要继承自己已经掌握的知识，别跟自己的历史过不去。伟大的科学家们，在当初设计计算机的时候就想到列位现在学习的需要了，一定不能让后世子孙再学新的运算规则，就用小学数学里面的好了。感谢那些科学家先驱者，泽被后世。

下面计算三个算术题，看看结果是什么

*   4 + 2
*   4.0 + 2
*   4.0 + 2.0

看官可能愤怒了，这么简单的题目，就不要劳驾计算机了，太浪费了。

别着急，还是要运算一下，然后看看结果，有没有不一样？要仔细观察哦。

```py
>>> 4+2
6
>>> 4.0+2
6.0
>>> 4.0+2.0
6.0 
```

不一样的地方是：第一个式子结果是 6，这是一个整数；后面两个是 6.0，这是浮点数。

> 定义 1：类似 4、-2、129486655、-988654、0 这样形式的数，称之为整数
> 定义 2：类似 4.0、-2.0、2344.123、3.1415926 这样形式的数，称之为浮点数

对这两个的定义，不用死记硬背，google 一下。记住爱因斯坦说的那句话：书上有的我都不记忆（是这么的说？好像是，大概意思，反正我也不记忆）。后半句他没说，我补充一下：忘了就 google。

似乎计算机做一些四则运算是不在话下的，但是，有一个问题请你务必注意：在数学中，整数是可以无限大的，但是在计算机中，整数不能无限大。为什么呢？（我推荐你去 google，其实计算机的基本知识中肯定学习过了。）因此，就会有某种情况出现，就是参与运算的数或者运算结果超过了计算机中最大的数了，这种问题称之为“整数溢出问题”。

### 整数溢出问题

这里有一篇专门讨论这个问题的文章，推荐阅读：[整数溢出](http://zhaoweizhuanshuo.blog.163.com/blog/static/148055262201093151439742/)

对于其它语言，整数溢出是必须正视的，但是，在 Python 里面，看官就无忧愁了，原因就是 Python 为我们解决了这个问题，请阅读拙文：大整数相乘 href="https://github.com/qiwsir/algorithm/blob/master/big_int.md")

ok!看官可以在 IDE 中实验一下大整数相乘。

```py
>>> 123456789870987654321122343445567678890098876*1233455667789990099876543332387665443345566
152278477193527562870044352587576277277562328362032444339019158937017801601677976183816L 
```

看官是幸运的，Python 解忧愁，所以，选择学习 Python 就是珍惜光阴了。

上面计算结果的数字最后有一个 Ｌ，就表示这个数是一个长整数，不过，看官不用管这点，反正是 Python 为我们搞定了。

在结束本节之前，有两个符号需要看官牢记（不记住也没关系，可以随时 google，只不过记住后使用更方便）

*   整数，用 int 表示，来自单词：integer
*   浮点数，用 float 表示，就是单词：float

可以用一个命令：type(object)来检测一个数是什么类型。

```py
>>> type(4)
<type 'int'>    #4 是 int，整数
>>> type(5.0)
<type 'float'>　#5.0 是 float，浮点数
type(988776544222112233445566778899887766554433221133344455566677788998776543222344556678)
<type 'long'>   # 是长整数，也是一个整数 
```

* * *

[总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 除法

除法啰嗦，不仅是 Python。

### 整数除以整数

进入 Python 交互模式之后（以后在本教程中，可能不再重复这类的叙述，只要看到>>>，就说明是在交互模式下），练习下面的运算：

```py
>>> 2 / 5
0
>>> 2.0 / 5
0.4
>>> 2 / 5.0
0.4
>>> 2.0 / 5.0
0.4 
```

看到没有？麻烦出来了（这是在 Python2.x 中），按照数学运算，以上四个运算结果都应该是 0.4。但我们看到的后三个符合，第一个居然结果是 0。why?

因为，在 Python（严格说是 Python2.x 中，Python3 会有所变化）里面有一个规定，像 2/5 中的除法这样，是要取整（就是去掉小数，但不是四舍五入）。2 除以 5，商是 0（整数），余数是 2（整数）。那么如果用这种形式：2/5，计算结果就是商那个整数。或者可以理解为：**整数除以整数，结果是整数（商）**。

比如：

```py
>>> 5 / 2
2
>>> 7 / 2
3
>>> 8 / 2
4 
```

**注意：**得到是商（整数）,而不是得到含有小数位的结果再通过“四舍五入”取整。例如：5/2，得到的是商 2，余数 1，最终`5 / 2 = 2`。并不是对 2.5 进行四舍五入。

### 浮点数与整数相除

这个标题和上面的标题格式不一样，上面的标题是“整数除以整数”，如果按照风格一贯制的要求，本节标题应该是“浮点数除以整数”，但没有，现在是“浮点数与整数相除”，其含义是：

> 假设：x 除以 y。其中 x 可能是整数，也可能是浮点数；y 可能是整数，也可能是浮点数。

出结论之前，还是先做实验：

```py
>>> 9.0 / 2
4.5
>>> 9 / 2.0
4.5
>>> 9.0 / 2.0
4.5

>>> 8.0 / 2
4.0
>>> 8 / 2.0
4.0
>>> 8.0 / 2.0
4.0 
```

归纳，得到规律：**不管是被除数还是除数，只要有一个数是浮点数，结果就是浮点数。**所以，如果相除的结果有余数，也不会像前面一样了，而是要返回一个浮点数，这就跟在数学上学习的结果一样了。

```py
>>> 10.0 / 3
3.3333333333333335 
```

这个是不是就有点搞怪了，按照数学知识，应该是 3.33333...，后面是 3 的循环了。那么你的计算机就停不下来了，满屏都是 3。为了避免这个，Python 武断终结了循环，但是，可悲的是没有按照“四舍五入”的原则终止。当然，还会有更奇葩的出现：

```py
>>> 0.1 + 0.2
0.30000000000000004
>>> 0.1 + 0.1 - 0.2
0.0
>>> 0.1 + 0.1 + 0.1 - 0.3
5.551115123125783e-17
>>> 0.1 + 0.1 + 0.1 - 0.2
0.10000000000000003 
```

越来越糊涂了，为什么 computer 姑娘在计算这么简单的问题上，如此糊涂了呢？不是 computer 姑娘糊涂，她依然冰雪聪明。原因在于十进制和二进制的转换上，computer 姑娘用的是二进制进行计算，上面的例子中，我们输入的是十进制，她就要把十进制的数转化为二进制，然后再计算。但是，在转化中，浮点数转化为二进制，就出问题了。

例如十进制的 0.1，转化为二进制是：0.0001100110011001100110011001100110011001100110011...

也就是说，转化为二进制后，不会精确等于十进制的 0.1。同时，计算机存储的位数是有限制的，所以，就出现上述现象了。

这种问题不仅仅是 Python 中有，所有支持浮点数运算的编程语言都会遇到，它不是 Python 的 bug。

明白了问题原因，怎么解决呢？就 Python 的浮点数运算而言，大多数机器上每次计算误差不超过 2**53 分之一。对于大多数任务这已经足够了，但是要在心中记住这不是十进制算法，每个浮点数计算可能会带来一个新的舍入错误。

一般情况下，只要简单地将最终显示的结果用“四舍五入”到所期望的十进制位数，就会得到期望的最终结果。

对于需要非常精确的情况，可以使用 decimal 模块，它实现的十进制运算适合会计方面的应用和高精度要求的应用。另外 fractions 模块支持另外一种形式的运算，它实现的运算基于有理数（因此像 1/3 这样的数字可以精确地表示）。最高要求则可是使用由 SciPy 提供的 Numerical Python 包和其它用于数学和统计学的包。列出这些东西，仅仅是让看官能明白，解决问题的方式很多，后面会用这些中的某些方式解决上述问题。

关于无限循环小数问题，我有一个链接推荐给诸位，它不是想象的那么简单呀。请阅读：[维基百科的词条：0.999...](http://zh.wikipedia.org/wiki/0.999%E2%80%A6)，会不会有深入体会呢？

> 补充一个资料，供有兴趣的朋友阅读：[浮点数算法：争议和限制](https://docs.python.org/2/tutorial/floatingpoint.html#tut-fp-issues)

Python 总会要提供多种解决问题的方案的，这是她的风格。

### 引用模块解决除法--启用轮子

Python 之所以受人欢迎，一个很重重要的原因，就是轮子多。这是比喻啦。就好比你要跑的快，怎么办？光天天练习跑步是不行滴，要用轮子。找辆自行车，就快了很多。还嫌不够快，再换电瓶车，再换汽车，再换高铁...反正你可以选择的很多。但是，这些让你跑的快的东西，多数不是你自己造的，是别人造好了，你来用。甚至两条腿也是感谢父母恩赐。正是因为轮子多，可以选择的多，就可以以各种不同速度享受了。

轮子是人类伟大的发明。

Python 就是这样，有各种轮子，我们只需要用。只不过那些轮子在 Python 里面的名字不叫自行车、汽车，叫做“模块”，有人承接别的语言的名称，叫做“类库”、“类”。不管叫什么名字吧。就是别人造好的东西我们拿过来使用。

怎么用？可以通过两种形式用：

*   形式 1：import module-name。import 后面跟空格，然后是模块名称，例如：import os
*   形式 2：from module1 import module11。module1 是一个大模块，里面还有子模块 module11，只想用 module11，就这么写了。

不啰嗦了，实验一个：

```py
>>> from __future__ import division
>>> 5 / 2
2.5
>>> 9 / 2
4.5
>>> 9.0 / 2
4.5
>>> 9 / 2.0
4.5 
```

注意了，引用了一个模块之后，再做除法，就不管什么情况，都是得到浮点数的结果了。

这就是轮子的力量。

### 余数

前面计算 5/2 的时候，商是 2，余数是 1

余数怎么得到？在 Python 中（其实大多数语言也都是），用`%`符号来取得两个数相除的余数.

实验下面的操作：

```py
>>> 5 % 2
1
>>> 6%4
2
>>> 5.0%2
1.0 
```

符号：%，就是要得到两个数（可以是整数，也可以是浮点数）相除的余数。

前面说 Python 有很多人见人爱的轮子（模块），她还有丰富的内建函数，也会帮我们做不少事情。例如函数 `divmod()`

```py
>>> divmod(5,2)  # 表示 5 除以 2，返回了商和余数
(2, 1)
>>> divmod(9,2)
(4, 1)
>>> divmod(5.0,2)
(2.0, 1.0) 
```

### 四舍五入

最后一个了，一定要坚持，今天的确有点啰嗦了。要实现四舍五入，很简单，就是内建函数：`round()`

动手试试：

```py
>>> round(1.234567,2)
1.23
>>> round(1.234567,3)
1.235
>>> round(10.0/3,4)
3.3333 
```

简单吧。越简单的时候，越要小心，当你遇到下面的情况，就有点怀疑了：

```py
>>> round(1.2345,3)
1.234               # 应该是：1.235
>>> round(2.235,2)
2.23                # 应该是：2.24 
```

哈哈，我发现了 Python 的一个 bug，太激动了。

别那么激动，如果真的是 bug，这么明显，是轮不到我的。为什么？具体解释看这里，下面摘录官方文档中的一段话：

> **Note:** The behavior of round() for floats can be surprising: for example, round(2.675, 2) gives 2.67 instead of the expected 2.68\. This is not a bug: it’s a result of the fact that most decimal fractions can’t be represented exactly as a float. See [Floating Point Arithmetic: Issues and Limitations](https://docs.Python.org/2/tutorial/floatingpoint.html#tut-fp-issues) for more information.

原来真的轮不到我。归根到底还是浮点数中的十进制转化为二进制惹的祸。

似乎除法的问题到此要结束了，其实远远没有，不过，做为初学者，至此即可。还留下了很多话题，比如如何处理循环小数问题，我肯定不会让有探索精神的朋友失望的，在我的 github 中有这样一个轮子，如果要深入研究，[可以来这里尝试](https://github.com/qiwsir/algorithm/blob/master/divide.py)。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 常用数学函数和运算优先级

在数学之中，除了加减乘除四则运算之外——这是小学数学——还有其它更多的运算，比如乘方、开方、对数运算等等，要实现这些运算，需要用到 Python 中的一个模块：Math

> 模块(module)是 Python 中非常重要的东西，你可以把它理解为 Python 的扩展工具。换言之，Python 默认情况下提供了一些可用的东西，但是这些默认情况下提供的还远远不能满足编程实践的需要，于是就有人专门制作了另外一些工具。这些工具被称之为“模块”
> 任何一个 Pythoner 都可以编写模块，并且把这些模块放到网上供他人来使用。
> 当安装好 Python 之后，就有一些模块默认安装了，这个称之为“标准库”，“标准库”中的模块不需要安装，就可以直接使用。
> 
> 如果没有纳入标准库的模块，需要安装之后才能使用。模块的安装方法，我特别推荐使用 pip 来安装。这里仅仅提一下，后面会专门进行讲述，性急的看官可以自己 google。

### 使用 math 模块

math 模块是标准库中的，所以不用安装，可以直接使用。使用方法是：

```py
>>> import math 
```

用 import 就将 math 模块引用过来了，下面就可以使用这个模块提供的工具了。比如，要得到圆周率：

```py
>>> math.pi
3.141592653589793 
```

这个模块都能做哪些事情呢？可以用下面的方法看到：

```py
>>> dir(math)
['__doc__', '__name__', '__package__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'hypot', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc'] 
```

`dir(module)`是一个非常有用的指令，可以通过它查看任何模块中所包含的工具。从上面的列表中就可以看出，在 math 模块中，可以计算正 sin(a),cos(a),sqrt(a)......

这些我们称之为函数，也就是在模块 math 中提供了各类计算的函数，比如计算乘方，可以使用 pow 函数。但是，怎么用呢？

Python 是一个非常周到的姑娘，她早就提供了一个命令，让我们来查看每个函数的使用方法。

```py
>>> help(math.pow) 
```

在交互模式下输入上面的指令，然后回车，看到下面的信息：

```py
Help on built-in function pow in module math:

pow(...)
    pow(x, y)

    Return x**y (x to the power of y). 
```

这里展示了 math 模块中的 pow 函数的使用方法和相关说明。

1.  第一行意思是说这里是 math 模块的内建函数 pow 帮助信息（所谓 built-in，称之为内建函数，是说这个函数是 Python 默认就有的)
2.  第三行，表示这个函数的参数，有两个，也是函数的调用方式
3.  第四行，是对函数的说明，返回 `x**y` 的结果，并且在后面解释了 `x**y` 的含义。
4.  最后，按 q 键返回到 Python 交互模式

从上面看到了一个额外的信息，就是 pow 函数和 `x**y` 是等效的，都是计算 x 的 y 次方。

```py
>>> 4**2
16
>>> math.pow(4,2)
16.0
>>> 4*2
8 
```

特别注意，`4**2` 和 `4*2` 是有很大区别的。

用类似的方法，可以查看 math 模块中的任何一个函数的使用方法。

> 关于“函数”的问题，在这里不做深入阐述，看管姑且按照自己在数学中所学到去理解。后面会有专门研究函数的章节。

下面是几个常用的 math 模块中函数举例，看官可以结合自己调试的进行比照。

```py
>>> math.sqrt(9)
3.0
>>> math.floor(3.14)
3.0
>>> math.floor(3.92)
3.0
>>> math.fabs(-2)    # 等价于 abs(-2)
2.0
>>> abs(-2)
2
>>> math.fmod(5,3)    # 等价于 5%3
2.0
>>> 5%3
2 
```

### 几个常见函数

有几个常用的函数，列一下，如果记不住也不要紧，知道有这些就好了，用的时候就 google。

**求绝对值**

```py
>>> abs(10)
10
>>> abs(-10)
10
>>> abs(-1.2)
1.2 
```

**四舍五入**

```py
>>> round(1.234)
1.0
>>> round(1.234,2)
1.23

>>> # 如果不清楚这个函数的用法，可以使用下面方法看帮助信息
>>> help(round)

Help on built-in function round in module __builtin__:

round(...)
    round(number[, ndigits]) -> floating point number

    Round a number to a given precision in decimal digits (default 0 digits).
    This always returns a floating point number.  Precision may be negative. 
```

### 运算优先级

从小学数学开始，就研究运算优先级的问题，比如四则运算中“先乘除，后加减”，说明乘法、除法的优先级要高于加减。

对于同一级别的，就按照“从左到右”的顺序进行计算。

下面的表格中列出了 Python 中的各种运算的优先级顺序。不过，就一般情况而言，不需要记忆，完全可以按照数学中的去理解，因为人类既然已经发明了数学，在计算机中进行的运算就不需要从新编写一套新规范了，只需要符合数学中的即可。

| 运算符 | 描述 |
| --- | --- |
| lambda | Lambda 表达式 |
| or | 布尔“或” |
| and | 布尔“与” |
| not x | 布尔“非” |
| in，not in | 成员测试 |
| is，is not | 同一性测试 |
| <，<=，>，>=，!=，== | 比较 |
| &#124; | 按位或 |
| ^ | 按位异或 |
| & | 按位与 |
| <<，>> | 移位 |
| +，- | 加法与减法 |
| *，/，% | 乘法、除法与取余 |
| +x，-x | 正负号 |
| ~x | 按位翻转 |
| ** | 指数 |
| x.attribute | 属性参考 |
| x[index] | 下标 |
| x[index:index] | 寻址段 |
| f(arguments...) | 函数调用 |
| (experession,...) | 绑定或元组显示 |
| [expression,...] | 列表显示 |
| {key:datum,...} | 字典显示 |
| 'expression,...' | 字符串转换 |

上面的表格将 Python 中用到的与运算符有关的都列出来了，是按照**从低到高**的顺序列出的。虽然有很多还不知道是怎么回事，不过先列出来，等以后用到了，还可以回来查看。

最后，要提及的是运算中的绝杀：括号。只要有括号，就先计算括号里面的。这是数学中的共识，无需解释。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 写一个简单的程序

通过对四则运算的学习，已经初步接触了 Python 中内容，如果看官是零基础的学习者，可能有点迷惑了。难道敲几个命令，然后看到结果，就算编程了？这也不是那些能够自动运行的程序呀？

的确。到目前为止，还不能算编程，只能算会用一些指令（或者叫做命令）来做点简单的工作。

稍安勿躁，下面就开始编写一个真正的但是简单程序。

### 程序

下面一段，关于程序的概念，内容来自维基百科：

*   先阅读一段英文的：[computer program and source code](http://en.wikipedia.org/wiki/Computer_program)，看不懂不要紧，可以跳过去，直接看下一条。

> A computer program, or just a program, is a sequence of instructions, written to perform a specified task with a computer.[1] A computer requires programs to function, typically executing the program's instructions in a central processor.[2] The program has an executable form that the computer can use directly to execute the instructions. The same program in its human-readable source code form, from which executable programs are derived (e.g., compiled), enables a programmer to study and develop its algorithms. A collection of computer programs and related data is referred to as the software.
> 
> Computer source code is typically written by computer programmers.[3] Source code is written in a programming language that usually follows one of two main paradigms: imperative or declarative programming. Source code may be converted into an executable file (sometimes called an executable program or a binary) by a compiler and later executed by a central processing unit. Alternatively, computer programs may be executed with the aid of an interpreter, or may be embedded directly into hardware.
> 
> Computer programs may be ranked along functional lines: system software and application software. Two or more computer programs may run simultaneously on one computer from the perspective of the user, this process being known as multitasking.

*   [计算机程序](http://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A8%8B%E5%BA%8F)

> 计算机程序（Computer Program）是指一组指示计算机或其他具有信息处理能力装置每一步动作的指令，通常用某种程序设计语言编写，运行于某种目标体系结构上。打个比方，一个程序就像一个用汉语（程序设计语言）写下的红烧肉菜谱（程序），用于指导懂汉语和烹饪手法的人（体系结构）来做这个菜。
> 
> 通常，计算机程序要经过编译和链接而成为一种人们不易看清而计算机可解读的格式，然后运行。未经编译就可运行的程序，通常称之为脚本程序（script）。

程序，简而言之，就是指令的集合。但是，有的程序需要编译，有的不需要。Python 编写的程序就不需要，因此她也被称之为解释性语言，编程出来的层序被叫做脚本程序。在有的程序员头脑中，有一种认为“编译型语言比解释性语言高价”的认识。这是错误的。不要认为编译的就好，不编译的就不好；也不要认为编译的就“高端”，不编译的就属于“低端”。有一些做了很多年程序的程序员或者其它什么人，可能会有这样的想法，这是毫无根据的。

不争论。用得妙就是好。

### 用 IDLE 的编程环境

能够写 Python 程序的工具很多，比如记事本就可以。当然，很多人总希望能用一个专门的编程工具，Python 里面自带了一个，作为简单应用是足够了。另外，可以根据自己的喜好用其它的工具，比如我用的是 vim，有不少人也用 eclipse，还有 notepad++，等等。软件领域为编程提供了丰富多彩的工具。

以 Python 默认的 IDE 为例，如下所示：

操作：File->New window

![](img/10501.png)

这样，就出现了一个新的操作界面，在这个界面里面，看不到用于输入指令的提示符：>>>，这个界面有点像记事本。说对了，本质上就是一个记事本，只能输入文本，不能直接在里面贴图片。

![](img/10502.png)

### 写两个大字：Hello,World

Hello,World.是面向世界的标志，所以，写任何程序，第一句一定要写这个，因为程序员是面向世界的，绝对不畏缩在某个局域网内，所以，所以看官要会科学上网，才能真正与世界 Hello。

直接上代码，就这么一行即可。

```py
print "Hello,World" 
```

如下图的样式

![](img/10503.png)

前面说过了，程序就是指令的集合，现在，这个程序里面，就一条指令。一条指令也可以成为集合。

注意观察，菜单上有一个 RUN，点击这个菜单，在下拉列表里面选择 Run Module。

![](img/10504.png)

会弹出对话框，要求把这个文件保存，这就比较简单了，保存到一个位置，看官一定要记住这个位置，并且取个文件名，文件名是以.py 为扩展名的。

都做好之后，点击确定按钮，就会发现在另外一个带有 >>> 的界面中，就自动出来了 Hello,World 两个大字。

成功了吗？成功了也别兴奋，因为还没有到庆祝的时候。

在这种情况系，我们依然是在 IDLE 的环境中实现了刚才那段程序的自动执行，如果脱离这个环境呢？

下面就关闭 IDLE，打开 shell(如果看官在使用苹果的 Mac OS 操作系统或者某种 linux 发行版的操作系统，比如我使用的是 ubuntu)，或者打开 cmd(windows 操作系统的用户，特别提醒用 windows 的用户，使用 windows 不是你的错，错就错在你只会使用鼠标点来点去，而不想也不会使用命令，更不想也不会使用 linux 的命令，还梦想成为优秀程序员。)，通过命令的方式，进入到你保存刚才的文件目录。

下图是我保存那个文件的地址，我把那个文件命名为 105.py，并保存在一个文件夹中。

![](img/10505.png)

然后在这个 shell 里面，输入：Python 105.py

上面这句话的含义就是告诉计算机，给我运行一个 Python 语言编写的程序，那个程序文件的名称是 105.py

我的计算机我做主。于是它给我乖乖地执行了这条命令。如下图：

![](img/10506.png)

还在沉默？可以欢呼了，德国队 7:1 胜巴西队，列看官中，不管是德国队还是巴西队的粉丝，都可以欢呼，因为你在程序员道路上迈出了伟大的第二步（什么迈出的第一步？）。顺便预测一下，本届世界杯最终冠军应该是：中国队。（还有这么扯的吗？）

### 解一道题目

请计算：19+2*4-8/2

代码如下：

```py
#!/usr/bin/env python
#coding:utf-8

"""
请计算：
19+2*4-8/2
"""

a = 19+2*4-8/2
print a 
```

提醒初学者，别复制这段代码，而是要一个字一个字的敲进去。然后保存(我保存的文件名是:105-1.py)。

在 shell 或者 cmd 中，执行：Python (文件名.py)

执行结果如下图：

![](img/10507.png)

好像还是比较简单。

下面对这个简单程序进行一一解释。

```py
#!/usr/bin/env python 
```

这一行是必须写的，它能够引导程序找到 Python 的解析器，也就是说，不管你这个文件保存在什么地方，这个程序都能执行，而不用制定 Python 的安装路径。

```py
#coding:utf-8 
```

这一行是告诉 Python，本程序采用的编码格式是 utf-8，什么是编码？什么是 utf-8？这是一个比较复杂且有历史的问题，此处暂不讨论。只有有了上面这句话，后面的程序中才能写汉字，否则就会报错了。看官可以把你的程序中的这行删掉，看看什么结果？

```py
"""
请计算：
19+2*4-8/2
""" 
```

这一行是让人看的，计算机看不懂。在 Python 程序中（别的编程语言也是如此），要写所谓的注释，就是对程序或者某段语句的说明文字，这些文字在计算机执行程序的时候，被计算机姑娘忽略，但是，注释又是必不可少的，正如前面说的那样，程序在大多数情况下是给人看的。注释就是帮助人理解程序的。

写注释的方式有两种，一种是单行注释，用 `#` 开头，另外一种是多行注释，用一对`'''`包裹起来。比如：

```py
"""
请计算：
19+2*4-8/2
""" 
```

用 `#` 开头的注释，可以像下面这样来写：

```py
# 请计算：19+2*4-8/2 
```

这种注释通常写在程序中的某个位置，比如某个语句的前面或者后面。计算机也会忽略这种注释的内容，只是给人看的。以 `#` 开头的注释，会在后面的编程中大量使用。

一般在程序的开头部分，都要写点东西，主要是告诉别人这个程序是用来做什么的。

```py
a = 19+2*4-8/2 
```

所谓语句，就是告诉程序要做什么事情。程序就是有各种各样的语句组成的。这条语句，又有一个名字，叫做复制语句。`19+2*4-8/2` 是一个表达式，最后要计算出一个结果，这个结果就是一个对象（又遇到了对象这个术语。在某些地方的方言中，把配偶、男女朋友也称之为对象，“对象”是一个应用很广泛的术语）。`=` 不要理解为数学中的等号，它的作用不是等于，而是完成赋值语句中“赋值”的功能。`a` 就是变量。这样就完成了一个赋值过程。

> 语句和表达式的区别：“表达式就是某件事”，“语句是做某件事”。

```py
print a 
```

这还是一个语句，称之为 print 语句，就是要打印出 a 的值（这种说法不是非常非常严格，但是通常总这么说。按照严格的说法，是打印变量 a 做对应的对象的值。嫌这种说法啰嗦，就直接说打印 a 的值）。

是不是在为看到自己写的第一个程序而欣慰呢？

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字符串(1)

如果对自然语言分类，有很多中分法，比如英语、法语、汉语等，这种分法是最常见的。在语言学里面，也有对语言的分类方法，比如什么什么语系之类的。我这里提出一种分法，这种分法尚未得到广大人民群众和研究者的广泛认同，但是，我相信那句“真理是掌握在少数人的手里”，至少在这里可以用来给自己壮壮胆。

我的分法：一种是语言中的两个元素（比如两个字）拼接在一起，出来一个新的元素（比如新的字）；另外一种是两个元素拼接在一起，只是得到这两个元素的并列显示。比如“好”和“人”，两个元素拼接在一起是“好人”，而 3 和 5 拼接（就是整数求和）在一起是 8，如果你认为是 35，那就属于第二类了。

把我的这种分法抽象一下：

*   一种是：△ + □ = ○
*   另外一种是：△ + □ = △ □

我们的语言中，离不开以上两类，不是第一类就是第二类。

太天才了。请鼓掌。

### 字符串

在我洋洋自得的时候，我 google 了一下，才发现，自己没那么高明，看[维基百科的字符串词条](http://zh.wikipedia.org/wiki/%E5%AD%97%E7%AC%A6%E4%B8%B2)是这么说的：

> 字符串（String），是由零个或多个字符组成的有限串行。一般记为 s=a[1]a[2]...a[n]。

看到维基百科的伟大了吧，它已经把我所设想的一种情况取了一个形象的名称，叫做字符串，本质上就是一串字符。

根据这个定义，在前面两次让一个程序员感到伟大的"Hello,World"，就是一个字符串。或者说不管用英文还是中文还是别的某种文，写出来的文字都可以做为字符串对待，当然，里面的特殊符号，也是可以做为字符串的，比如空格等。

严格地说，在 Python 中的字符串是一种对象类型，这种类型用 str 表示，通常单引号`''`或者双引号`""`包裹起来。

> 字符串和前面讲过的数字一样，都是对象的类型，或者说都是值。当然，表示方式还是有区别的。

```py
>>> "I love Python."
'I love Python.'
>>> 'I LOVE PYTHON.'
'I LOVE PYTHON.' 
```

从这两个例子中可以看出来，不论使用单引号还是双引号，结果都是一样的。

```py
>>> 250
250
>>> type(250)
<type 'int'>

>>> "250"
'250'
>>> type("250")
<type 'str'> 
```

仔细观察上面的区别，同样是 250，一个没有放在引号里面，一个放在了引号里面，用 `type()`函数来检验一下，发现它们居然是两种不同的对象类型，前者是 int 类型，后者则是 str 类型，即字符串类型。所以，请大家务必注意，不是所有数字都是 int（or float）,必须要看看，它在什么地方，如果在引号里面，就是字符串了。如果搞不清楚是什么类型，就让 `type()`来帮忙搞定。

操练一下字符串吧。

```py
>>> print "good good study, day day up"
good good study, day day up
>>> print "----good---study---day----up"
----good---study---day----up 
```

在 print 后面，打印的都是字符串。注意，是双引号里面的，引号不是字符串的组成部分。它是在告诉计算机，它里面包裹着的是一个字符串。

爱思考的看官肯定发现上面这句话有问题了。如果我要把下面这句话看做一个字符串，应该怎么做？

```py
What's your name? 
```

这个问题非常好，因为在这句话中有一个单引号，如果直接在交互模式中像上面那样输入，就会这样：

```py
>>> 'What's your name?'
File "<stdin>", line 1
 'What's your name?'
      ^
SyntaxError: invalid syntax 
```

出现了 `SyntaxError`（语法错误）引导的提示，这是在告诉我们这里存在错误，错误的类型就是 `SyntaxError`，后面是对这种错误的解释“invalid syntax”（无效的语法）。特别注意，错误提示的上面，有一个 ^ 符号，直接只着一个单引号，不用多说，你也能猜测出，大概在告诉我们，可能是这里出现错误了。

> 在 python 中，这一点是非常友好的，如果语句存在错误，就会将错误输出来，供程序员改正参考。当然，错误来源有时候比较复杂，需要根据经验和知识进行修改。还有一种修改错误的好办法，就是讲错误提示放到 google 中搜索。

上面那个值的错误原因是什么呢？仔细观察，发现那句话中事实上有三个单引号，本来一对单引号之间包裹的是一个字符串，现在出现了三个（一对半）单引号，computer 姑娘迷茫了，她不知道单引号包裹的到底是谁。于是报错。

**解决方法一：**双引号包裹单引号

```py
>>> "What's your name?"
"What's your name?" 
```

用双引号来包裹，双引号里面允许出现单引号。其实，反过来，单引号里面也可以包裹双引号。这个可以笼统地成为二者的嵌套。

**解决方法二：**使用转义符

所谓转义，就是让某个符号不在表示某个含义，而是表示另外一个含义。转义符的作用就是它能够转变符号的含义。在 Python 中，用 `\` 作为转义符（其实很多语言，只要有转义符的，都是用这个符号）。

```py
>>> 'What\'s your name?'
"What's your name?" 
```

是不是看到转义符 `\` 的作用了。

本来单引号表示包括字符串，它不是字符串一部分，但是如果前面有转义符，那么它就失去了原来的含义，转化为字符串的一部分，相当于一个特殊字符了。

### 变量和字符串

前面讲过**变量无类型，对象有类型**了，比如在数字中:

```py
>>> a = 5
>>> a
5 
```

其本质含义是变量 a 相当于一个标签，贴在了对象 5 上面。并且我们把这个语句叫做赋值语句。

同样，在对字符串类型的对象，也是这样，能够通过赋值语句，将对象与某个标签（变量）关联起来。

```py
>>> b = "hello,world"
>>> b
'hello,world'
>>> print b
hello,world 
```

还记得我们曾经用过一个 type 命令吗？现在它还有用，就是检验一个变量，到底跟什么类型联系着，是字符串还是数字？

```py
>>> type(a)
<type 'int'>
>>> type(b)
<type 'str'> 
```

有时候，你会听到一种说法：把 a 称之为数字型变量，把 b 叫做字符（串）型变量。这种说法，在某些语言中是成立的。某些语言，需要提前声明变量，然后变量就成为了一个筐，将值装到这个筐里面。但是，Python 不是这样的。要注意区别。

### 拼接字符串

还记得我在本节开篇提出的那个伟大发现吗？就是将两个东西拼接起来。

对数字，如果拼接，就是对两个数字求和。如：3+5，就计算出为 8。那么对字符串都能进行什么样的操作呢？试试吧：

```py
>>> "Py" + "thon"
'Python' 
```

跟我那个不为大多数人认可的发现是一样的，你还不认可吗？两个字符串相加，就相当于把两个字符串连接起来。(别的运算就别尝试了，没什么意义，肯定报错，不信就试试）

```py
>>> "Py" - "thon"     # 这么做的人，是脑袋进水泥了吧？
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for -: 'str' and 'str' 
```

用 `+` 号实现连接，的确比较简单，不过，有时候你会遇到这样的问题：

```py
>>> a = 1989
>>> b = "free"
>>> print b+a
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: cannot concatenate 'str' and 'int' objects 
```

> 这里引入了一个指令：`print`，意思就是打印后面的字符串（或者指向字符串的变量），上面是 Python2 中的使用方式，在 Python3 中，它变成了一个函数。应该用 `print(b+a)`的样式了。

报错了，其错误原因已经打印出来了（一定要注意看打印出来的信息）：`cannot concatenate 'str' and 'int' objects`。原来 `a` 对应的对象是一个 `int` 类型的，不能将它和 `str` 对象连接起来。怎么办？

原来，用 `+` 拼接起来的两个对象，必须是同一种类型的。如果两个都是数字，毫无疑问是正确的，就是求和；如果都是字符串，那么就得到一个新的字符串。

修改上面的错误，可以通过以下方法：

```py
>>> print b + `a`       
free1989 
```

注意，`\` 是反引号，不是单引号，就是键盘中通常在数字 1 左边的那个，在英文半角状态下输入的符号。这种方法，在编程实践中比较少应用，特别是在 Python3 中，已经把这种方式弃绝了。我想原因就是这个符号太容易和单引号混淆了。在编程中，也不容易看出来，可读性太差。

常言道：“困难只有一个，解决困难的方法不止一种”，既然反引号可读性不好，在编程实践中就尽量不要使用。于是乎就有了下面的方法，这是被广泛采用的。不但简单，更主要是直白，一看就懂什么意思了。

```py
>>> print b + str(a)    
free1989 
```

用 `str(a)`实现将整数对象转换为字符串对象。虽然 str 是一种对象类型，但是它也能够实现对象类型的转换，这就起到了一个函数的作用。其实前面已经讲过的 int 也有类似的作用。比如：

```py
>>> a = "250"
>>> type(a)
<type 'str'>
>>> b = int(a)
>>> b
250
>>> type(b)
<type 'int'> 
```

> 提醒列位，如果你对 int 和 str 比较好奇，可以在交互模式中，使用 help(int)，help(str)查阅相关的更多资料。

还有第三种：

```py
>>> print b + repr(a)   #repr(a)与上面的类似
free1989 
```

这里 repr()是一个函数，其实就是反引号的替代品，它能够把结果字符串转化为合法的 python 表达式。

可能看官看到这个，就要问它们三者之间的区别了。首先明确，repr()和 `\` 是一致的，就不用区别了。接下来需要区别的就是 repr()和 str，一个最简单的区别，repr 是函数，str 是跟 int 一样，一种对象类型。不过这么说是不能完全解惑的。幸亏有那么好的 google 让我辈使用，你会找到不少人对这两者进行区分的内容，我推荐这个：

> 1.  When should i use str() and when should i use repr() ?
> 
> Almost always use str when creating output for end users.
> 
> repr is mainly useful for debugging and exploring. For example, if you suspect a string has non printing characters in it, or a float has a small rounding error, repr will show you; str may not.
> 
> repr can also be useful for for generating literals to paste into your source code. It can also be used for persistence (with ast.literal_eval or eval), but this is rarely a good idea--if you want editable persisted values, something like JSON or YAML is much better, and if you don't plan to edit them, use pickle.
> 
> 2.In which cases i can use either of them ?
> Well, you can use them almost anywhere. You shouldn't generally use them except as described above.
> 
> 3.What can str() do which repr() can't ?
> Give you output fit for end-user consumption--not always (e.g., str(['spam', 'eggs']) isn't likely to be anything you want to put in a GUI), but more often than repr.
> 
> 4.What can repr() do which str() can't
> Give you output that's useful for debugging--again, not always (the default for instances of user-created classes is rarely helpful), but whenever possible.
> 
> And sometimes give you output that's a valid Python literal or other expression--but you rarely want to rely on that except for interactive exploration.

以上英文内容来源：[`stackoverflow.com/questions/19331404/str-vs-repr-functions-in-python-2-7-5`](http://stackoverflow.com/questions/19331404/str-vs-repr-functions-in-python-2-7-5)

### Python 转义字符

在字符串中，有时需要输入一些特殊的符号，但是，某些符号不能直接输出，就需要用转义符。所谓转义，就是不采用符号本来的含义，而采用另外一含义了。下面表格中列出常用的转义符：

| 转义字符 | 描述 |
| --- | --- |
| \ | (在行尾时) 续行符 |
| \ | 反斜杠符号 |
| \' | 单引号 |
| \" | 双引号 |
| \a | 响铃 |
| \b | 退格(Backspace) |
| \e | 转义 |
| \000 | 空 |
| \n | 换行 |
| \v | 纵向制表符 |
| \t | 横向制表符 |
| \r | 回车 |
| \f | 换页 |
| \oyy | 八进制数，yy 代表的字符，例如：\o12 代表换行 |
| \xyy | 十六进制数，yy 代表的字符，例如：\x0a 代表换行 |
| \other | 其它的字符以普通格式输出 |

以上所有转义符，都可以通过交互模式下 print 来测试一下，感受实际上是什么样子的。例如：

```py
>>> print "hello.I am qiwsir.\                  # 这里换行，下一行接续
... My website is 'http://qiwsir.github.io'."
hello.I am qiwsir.My website is 'http://qiwsir.github.io'.

>>> print "you can connect me by qq\\weibo\\gmail"  #\\ 是为了要后面那个 \
you can connect me by qq\weibo\gmail 
```

看官自己试试吧。如果有问题，可以联系我解答。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字符串(2)

### raw_input 和 print

自从本课程开始以来，我们还没有感受到 computer 姑娘的智能。最简单的智能应该体现在哪里呢？想想小孩子刚刚回说话的时候情景吧。

> 小孩学说话，是一个模仿的过程，孩子周围的人怎么说，她（他）往往就是重复。看官可以忘记自己当初是怎么学说话了吧？就找个小孩子观察一下吧。最好是自己的孩子。如果没有，就要抓紧了。

通过 Python 能不能实现这个简单的功能呢？当然能，要不然 Python 如何横行天下呀。

不过在写这个功能前，要了解两个函数：raw_input 和 print

> 这两个都是 Python 的内建函数（built-in function）。关于 Python 的内建函数，下面这个表格都列出来了。所谓内建函数，就是能够在 Python 中直接调用，不需要做其它的操作。

Built-in Functions

| abs() | divmod() | input() | open() | staticmethod() |
| :-- | :-- | :-- | :-- | :-- |
| all() | enumerate() | int() | ord() | str() |
| any() | eval() | isinstance() | pow() | sum() |
| basestring() | execfile() | issubclass() | print() | super() |
| bin() | file() | iter() | property() | tuple() |
| bool() | filter() | len() | range() | type() |
| bytearray() | float() | list() | raw_input() | unichr() |
| callable() | format() | locals() | reduce() | unicode() |
| chr() | frozenset() | long() | reload() | vars() |
| classmethod() | getattr() | map() | repr() | xrange() |
| cmp() | globals() | max() | reversed() | zip() |
| compile() | hasattr() | memoryview() | round() | **import**() |
| complex() | hash() | min() | set() | apply() |
| delattr() | help() | next() | setattr() | buffer() |
| dict() | hex() | object() | slice() | coerce() |
| dir() | id() | oct() | sorted() | intern() |

这些内建函数，怎么才能知道哪个函数怎么用，是干什么用的呢？

不知道你是否还记得我在前面使用过的方法，这里再进行演示，这种方法是学习 Python 的法宝。

```py
>>> help(raw_input) 
```

然后就出现：

```py
Help on built-in function raw_input in module __builtin__:

raw_input(...)
    raw_input([prompt]) -> string

    Read a string from standard input.  The trailing newline is stripped.
    If the user hits EOF (Unix: Ctl-D, Windows: Ctl-Z+Return), raise EOFError.
    On Unix, GNU readline is used if enabled.  The prompt string, if given,
    is printed without a trailing newline before reading. 
```

从中是不是已经清晰地看到了 `raw_input()`的使用方法了。

还有第二种方法，那就是到 Python 的官方网站，查看内建函数的说明。[`docs.Python.org/2/library/functions.html`](https://docs.Python.org/2/library/functions.html)

其实，我上面那个表格，就是在这个网页中抄过来的。

例如，对 `print()`说明如下：

```py
 print(*objects, sep=' ', end='\n', file=sys.stdout)

    Print objects to the stream file, separated by sep and followed by end. sep, end and file, if present, must be given as keyword arguments.

    All non-keyword arguments are converted to strings like str() does and written to the stream, separated by sep and followed by end. Both sep and end must be strings; they can also be None, which means to use the default values. If no objects are given, print() will just write end.

    The file argument must be an object with a write(string) method; if it is not present or None, sys.stdout will be used. Output buffering is determined by file. Use file.flush() to ensure, for instance, immediate appearance on a screen. 
```

分别在交互模式下，将这个两个函数操练一下。

```py
>>> raw_input("input your name:")
input your name:python
'python' 
```

输入名字之后，就返回了输入的内容。用一个变量可以获得这个返回值。

```py
>>> name = raw_input("input your name:")
input your name:python
>>> name
'python'
>>> type(name)
<type 'str'> 
```

而且，返回的结果是 str 类型。如果输入的是数字呢？

```py
>>> age = raw_input("How old are you?")
How old are you?10
>>> age
'10'
>>> type(age)
<type 'str'> 
```

返回的结果，仍然是 str 类型。

再试试 `print()`，看前面对它的说明，是比较复杂的。没关系，我们从简单的开始。在交互模式下操作：

```py
>>> print("hello, world")
hello, world
>>> a = "python"
>>> b = "good"
>>> print a
python
>>> print a,b
python good 
```

比较简单吧。当然，这是没有搞太复杂了。

特别要提醒的是，`print()`默认是以 `\n` 结尾的，所以，会看到每个输出语句之后，输出内容后面自动带上了 `\n`，于是就换行了。

有了以上两个准备，接下来就可以写一个能够“对话”的小程序了。

```py
#!/usr/bin/env python
# coding=utf-8

name = raw_input("What is your name?")
age = raw_input("How old are you?")

print "Your name is:", name
print "You are " + age + " years old."

after_ten = int(age) + 10
print "You will be " + str(after_ten) + " years old after ten years." 
```

对这段小程序中，有几点说明

前面演示了 `print()`的使用，除了打印一个字符串之外，还可以打印字符串拼接结果。

```py
print "You are " + age + " years old." 
```

注意，那个变量 `age` 必须是字符串，如最后的那个语句中：

```py
print "You will be " + str(after_ten) + " years old after ten years." 
```

这句话里面，有一个类型转化，将原本是整数型 `after_ten` 转化为了 str 类型。否则，就包括，不信，你可以试试。

同样注意，在 `after_ten = int(age) + 10` 中，因为通过 `raw_input` 得到的是 str 类型，当 age 和 10 求和的时候，需要先用 `int()`函数进行类型转化，才能和后面的整数 10 相加。

这个小程序，是有点综合的，基本上把已经学到的东西综合运用了一次。请看官调试一下，如果没有通过，仔细看报错信息，你能够从中获得修改方向的信息。

### 原始字符串

所谓原始字符串，就是指字符串里面的每个字符都是原始含义，比如反斜杠，不会被看做转义符。如果在一般字符串中，比如

```py
>>> print "I like \npython"
I like 
python 
```

这里的反斜杠就不是“反斜杠”的原始符号含义，而是和后面的 n 一起表示换行（转义了）。当然，这似乎没有什么太大影响，但有的时候，可能会出现问题，比如打印 DOS 路径（DOS，有没有搞错，现在还有人用吗？）

```py
>>> dos = "c:\news"
>>> dos
'c:\news'        # 这里貌似没有什么问题
>>> print dos    # 当用 print 来打印这个字符串的时候，就出问题了。
c:
ews 
```

如何避免？用前面讲过的转义符可以解决：

```py
>>> dos = "c:\\news"
>>> print dos
c:\news 
```

此外，还有一种方法，如：

```py
>>> dos = r"c:\news"
>>> print dos
c:\news
>>> print r"c:\news\python"
c:\news\python 
```

状如 `r"c:\news"`，由 r 开头引起的字符串，就是原始字符串，在里面放任何字符都表示该字符的原始含义。

这种方法在做网站设置网站目录结构的时候非常有用。使用了原始字符串，就不需要转义了。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字符串(3)

关于字符串的内容，已经有两节进行介绍了。不过，它是一个话题中心，还要再继续。

例如这样一个字符串 `Python`，还记得前面对字符串的定义吗？它就是几个字符：P,y,t,h,o,n，排列起来。这种排列是非常严格的，不仅仅是字符本身，而且还有顺序，换言之，如果某个字符换了，就编程一个新字符串了；如果这些字符顺序发生变化了，也成为了一个新字符串。

在 Python 中，把像字符串这样的对象类型（后面还会冒出来类似的其它有这种特点的对象类型，比如列表），统称为序列。顾名思义，序列就是“有序排列”。

比如水泊梁山的 108 个好汉（里面分明也有女的，难道女汉子是从这里来的吗？），就是一个“有序排列”的序列。从老大宋江一直排到第 108 位金毛犬段景住。在这个序列中，每个人有编号，编号和每个人一一对应。1 号是宋江，2 号是卢俊义。反过来，通过每个人的姓名，也能找出他对应的编号。武松是多少号？14 号。李逵呢？22 号。

在 Python 中，给这些编号取了一个文雅的名字，叫做**索引**(别的编程语言也这么称呼，不是 Python 独有的。)。

### 索引和切片

前面用梁山好汉的为例说明了索引。再看 Python 中的例子：

```py
>>> lang = "study Python"
>>> lang[0]
's'
>>> lang[1]
't' 
```

有一个字符串，通过赋值语句赋给了变量 lang。如果要得到这个字符串的第一个单词 `s`，可以用 `lang[0]`。当然，如果你不愿意通过赋值语句，让变量 lang 来指向那个字符串，也可以这样做：

```py
>>> "study Python"[0]
's' 
```

效果是一样的。因为 lang 是标签，就指向了 `"study Python"` 字符串。当让 Python 执行 `lang[0]` 的时候，就是要转到那个字符串对象，如同上面的操作一样。只不过，如果不用 lang 这么一个变量，后面如果再写，就费笔墨了，要每次都把那个字符串写全了。为了省事，还是复制给一个变量吧。变量就是字符串的代表了。

字符串这个序列的排序方法跟梁山好汉有点不同，第一个不是用数字 1 表示，而是用数字 0 表示。不仅仅 Python，其它很多语言都是从 0 开始排序的。为什么这样做呢？这就是规定。当然，这个规定是有一定优势的。此处不展开，有兴趣的网上去 google 一下，有专门对此进行解释的文章。

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| s | t | u | d | y | l | p | y | t | h | o | n |

上面的表格中，将这个字符串从第一个到最后一个进行了排序，特别注意，两个单词中间的那个空格，也占用了一个位置。

通过索引能够找到该索引所对应的字符，那么反过来，能不能通过字符，找到其在字符串中的索引值呢？怎么找？

```py
>>> lang.index("p")
6 
```

就这样，是不是已经能够和梁山好汉的例子对上号了？只不过区别在于第一个的索引值是 0。

如果某一天，宋大哥站在大石头上，向着各位弟兄大喊：“兄弟们，都排好队。”等兄弟们排好之后，宋江说：“现在给各位没有老婆的兄弟分配女朋友，我这里已经有了名单，我念叨的兄弟站出来。不过我是按照序号来念的。第 29 号到第 34 号先出列，到旁边房子等候分配女朋友。”

在前面的例子中 lang[1] 能够得到原来字符串的第二个字符 t，就相当于从原来字符串中把这个“切”出来了。不过，我们这么“切”却不影响原来字符串的完整性，当然可以理解为将那个字符 t 赋值一份拿出来了。

那么宋江大哥没有一个一个“切”，而是一下将几个兄弟叫出来。在 Python 中也能做类似事情。

```py
>>> lang
'study Python'    #在前面“切”了若干的字符之后，再看一下该字符串，还是完整的。
>>> lang[2:9]
'udy pyt' 
```

通过 `lang[2:9]`要得到部分（不是一个）字符，从返回的结果中可以看出，我们得到的是序号分别对应着 `2,3,4,5,6,7,8`(跟上面的表格对应一下)字符（包括那个空格）。也就是，这种获得部分字符的方法中，能够得到开始需要的以及最后一个序号之前的所对应的字符。有点拗口，自己对照上面的表格数一数就知道了。简单说就是包括开头，不包括结尾。

上述，不管是得到一个还是多个，通过索引得到字符的过程，称之为**切片**。

切片是一个很有意思的东西。可以“切”出不少花样呢？

```py
>>> lang
'study Python'
>>> b = lang[1:]    # 得到从 1 号到最末尾的字符，这时最后那个需要不用写
>>> b
'tudy Python'
>>> c = lang[:]    # 得到所有字符
>>> c
'study Python'
>>> d = lang[:10]    # 得到从第一个到 10 号之前的字符
>>> d
'study pyth' 
```

在获取切片的时候，如果分号的前面或者后面的序号不写，就表示是到最末（后面的不写）或第一个（前面的不写）

`lang[:10]`的效果和 `lang[0:10]`是一样的。

```py
>>> e = lang[0:10]
>>> e
'study pyth' 
```

那么，`lang[1:]`和 `lang[1:11]`效果一样吗？请思考后作答。

```py
>>> lang[1:11]
'tudy pytho'
>>> lang[1:]
'tudy python' 
```

果然不一样，你思考对了吗？原因就是前述所说的，如果分号后面有数字，所得到的切片，不包含该数字所对应的序号（前包括，后不包括）。那么，是不是可以这样呢？`lang[1:12]`，不包括 12 号（事实没有 12 号），是不是可以得到 1 到 11 号对应的字符呢？

```py
>>> lang[1:12]
'tudy python'
>>> lang[1:13]
'tudy python' 
```

果然是。并且不仅仅后面写 12，写 13，也能得到同样的结果。但是，我这个特别要提醒，这种获得切片的做法在编程实践中是不提倡的。特别是如果后面要用到循环的时候，这样做或许在什么时候遇到麻烦。

如果在切片的时候，冒号左右都不写数字，就是前面所操作的 `c = lang[:]`，其结果是变量 c 的值与原字符串一样，也就是“复制”了一份。注意，这里的“复制”我打上了引号，意思是如同复制，是不是真的复制呢？可以用下面的方式检验一下

```py
>>> id(c)
3071934536L
>>> id(lang)
3071934536L 
```

`id()`的作用就是查看该对象在内存地址（就是在内存中的位置编号）。从上面可以看出，两个的内存地址一样，说明 c 和 lang 两个变量指向的是同一个对象。用 `c=lang[:]`的方式，并没有生成一个新的字符串，而是将变量 c 这个标签也贴在了原来那个字符串上了。

```py
>>> lang = "study python"
>>> c = lang 
```

如果这样操作，变量 c 和 lang 是不是指向同一个对象呢？或者两者所指向的对象内存地址如何呢？看官可以自行查看。

### 字符串基本操作

字符串是一种序列，所有序列都有如下基本操作：

1.  len()：求序列长度
2.  *   ：连接 2 个序列
3.  *   : 重复序列元素
4.  in :判断元素是否存在于序列中
5.  max() :返回最大值
6.  min() :返回最小值
7.  cmp(str1,str2) :比较 2 个序列值是否相同

通过下面的例子，将这几个基本操作在字符串上的使用演示一下：

#### “+”连接字符串

```py
>>> str1 + str2
'abcdabcde'
>>> str1 + "-->" + str2
'abcd-->abcde' 
```

这其实就是拼接，不过在这里，看官应该有一个更大的观念，我们现在只是学了字符串这一种序列，后面还会遇到列表、元组两种序列，都能够如此实现拼接。

#### in

```py
>>> "a" in str1
True
>>> "de" in str1
False
>>> "de" in str2
True 
```

`in` 用来判断某个字符串是不是在另外一个字符串内，或者说判断某个字符串内是否包含某个字符串，如果包含，就返回 `True`，否则返回 `False`。

#### 最值

```py
>>> max(str1)
'd'
>>> max(str2)
'e'
>>> min(str1)
'a' 
```

一个字符串中，每个字符在计算机内都是有编码的，也就是对应着一个数字，`min()`和 `max()`就是根据这个数字里获得最小值和最大值，然后对应出相应的字符。关于这种编号是多少，看官可以 google 有关字符编码，或者 ASCII 编码什么的，很容易查到。

#### 比较

```py
>>> cmp(str1, str2)
-1 
```

将两个字符串进行比较，也是首先将字符串中的符号转化为对一个的数字，然后比较。如果返回的数值小于零，说明第一个小于第二个，等于 0，则两个相等，大于 0，第一个大于第二个。为了能够明白其所以然，进入下面的分析。

```py
>>> ord('a')
97
>>> ord('b')
98
>>> ord(' ')
32 
```

`ord()`是一个内建函数，能够返回某个字符（注意，是一个字符，不是多个字符组成的串）所对一个的 ASCII 值（是十进制的），字符 a 在 ASCII 中的值是 97，空格在 ASCII 中也有值，是 32。顺便说明，反过来，根据整数值得到相应字符，可以使用 `chr()`：

```py
>>> chr(97)
'a'
>>> chr(98)
'b' 
```

于是，就得到如下比较结果了：

```py
>>> cmp("a","b")    #a-->97, b-->98, 97 小于 98，所以 a 小于 b
-1
>>> cmp("abc","aaa") 
1
>>> cmp("a","a")
0 
```

看看下面的比较，是怎么进行的呢？

```py
>>> cmp("ad","c")
-1 
```

在字符串的比较中，是两个字符串的第一个字符先比较，如果相等，就比较下一个，如果不相等，就返回结果。直到最后，如果还相等，就返回 0。位数不够时，按照没有处理（注意，没有不是 0，0 在 ASCII 中对应的是 NUL），位数多的那个天然大了。ad 中的 a 先和后面的 c 进行比较，显然 a 小于 c，于是就返回结果 -1。如果进行下面的比较，是最容易让人迷茫的。看官能不能根据刚才阐述的比较远离理解呢？

```py
>>> cmp("123","23")
-1
>>> cmp(123,23)    # 也可以比较整数，这时候就是整数的直接比较了。
1 
```

#### “*”

字符串中的“乘法”，这个乘法，就是重复那个字符串的含义。在某些时候很好用的。比如我要打印一个华丽的分割线：

```py
>>> str1*3
'abcdabcdabcd'
>>> print "-"*20    # 不用输入很多个`-`
-------------------- 
```

#### len()

要知道一个字符串有多少个字符，一种方法是从头开始，盯着屏幕数一数。哦，这不是计算机在干活，是键客在干活。

> 键客，不是剑客。剑客是以剑为武器的侠客；而键客是以键盘为武器的侠客。当然，还有贱客，那是贱人的最高境界，贱到大侠的程度，比如岳不群之流。

键客这样来数字符串长度：

```py
>>> a="hello"
>>> len(a)
5 
```

使用的是一个函数 len(object)。得到的结果就是该字符串长度。

```py
>>> m = len(a)  # 把结果返回后赋值给一个变量
>>> m
5
>>> type(m)     # 这个返回值（变量）是一个整数型
<type 'int'> 
```

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字符串(4)

字符串的内容的确不少，甚至都有点啰嗦了。但是，本节依然还要继续，就是因为在编程实践中，经常会遇到有关字符串的问题，而且也是很多初学者容易迷茫的。

### 字符串格式化输出

什么是格式化？在维基百科中有专门的词条，这么说的：

> 格式化是指对磁盘或磁盘中的分区（partition）进行初始化的一种操作，这种操作通常会导致现有的磁盘或分区中所有的文件被清除。

不知道你是否知道这种“格式化”。显然，此格式化非我们这里所说的，我们说的是字符串的格式化，或者说成“格式化字符串”，都可以，表示的意思就是：

> 格式化字符串，是 C、C++ 等程序设计语言 printf 类函数中用于指定输出参数的格式与相对位置的字符串参数。其中的转换说明（conversion specification）用于把随后对应的 0 个或多个函数参数转换为相应的格式输出；格式化字符串中转换说明以外的其它字符原样输出。

这也是来自维基百科的定义。在这个定义中，是用 C 语言作为例子，并且用了其输出函数来说明。在 Python 中，也有同样的操作和类似的函数 `print`，此前我们已经了解一二了。

如果将那个定义说的通俗一些，字符串格式化化，就是要先制定一个模板，在这个模板中某个或者某几个地方留出空位来，然后在那些空位填上字符串。那么，那些空位，需要用一个符号来表示，这个符号通常被叫做占位符（仅仅是占据着那个位置，并不是输出的内容）。

```py
>>> "I like %s"
'I like %s' 
```

在这个字符串中，有一个符号：`%s`，就是一个占位符，这个占位符可以被其它的字符串代替。比如：

```py
>>> "I like %s" % "python"
'I like python'
>>> "I like %s" % "Pascal"
'I like Pascal' 
```

这是较为常用的一种字符串输出方式。

另外，不同的占位符，会表示那个位置应该被不同类型的对象填充。下面列出许多，供参考。不过，不用记忆，常用的只有 `%s` 和 `%d`，或者再加上 `%f`，其它的如果需要了，到这里来查即可。

| 占位符 | 说明 |
| :-- | :-- |
| %s | 字符串(采用 str()的显示) |
| %r | 字符串(采用 repr()的显示) |
| %c | 单个字符 |
| %b | 二进制整数 |
| %d | 十进制整数 |
| %i | 十进制整数 |
| %o | 八进制整数 |
| %x | 十六进制整数 |
| %e | 指数 (基底写为 e) |
| %E | 指数 (基底写为 E) |
| %f | 浮点数 |
| %F | 浮点数，与上相同 |
| %g | 指数(e) 或浮点数 (根据显示长度) |
| %G | 指数(E)或浮点数 (根据显示长度) |

看例子：

```py
>>> a = "%d years" % 15
>>> print a
15 years 
```

当然，还可以在一个字符串中设置多个占位符，就像下面一样

```py
>>> print "Suzhou is more than %d years. %s lives in here." % (2500, "qiwsir")
Suzhou is more than 2500 years. qiwsir lives in here. 
```

对于浮点数字的打印输出，还可以限定输出的小数位数和其它样式。

```py
>>> print "Today's temperature is %.2f" % 12.235
Today's temperature is 12.23
>>> print "Today's temperature is %+.2f" % 12.235
Today's temperature is +12.23 
```

注意，上面的例子中，没有实现四舍五入的操作。只是截取。

关于类似的操作，还有很多变化，比如输出格式要宽度是多少等等。如果看官在编程中遇到了，可以到网上查找。我这里给一个参考图示，也是从网上抄来的。

![](img/10901.png)

其实，上面这种格式化方法，常常被认为是太“古老”了。因为在 Python 中还有新的格式化方法。

```py
>>> s1 = "I like {}".format("python")
>>> s1
'I like python'
>>> s2 = "Suzhou is more than {} years. {} lives in here.".format(2500, "qiwsir") 
>>> s2
'Suzhou is more than 2500 years. qiwsir lives in here.' 
```

这就是 Python 非常提倡的 `string.format()`的格式化方法，其中 `{}` 作为占位符。

这种方法真的是非常好，而且非常简单，只需要将对应的东西，按照顺序在 format 后面的括号中排列好，分别对应占位符 `{}` 即可。我喜欢的方法。

如果你觉得还不明确，还可以这样来做。

```py
>>> print "Suzhou is more than {year} years. {name} lives in here.".format(year=2500, name="qiwsir") 
Suzhou is more than 2500 years. qiwsir lives in here. 
```

真的很简洁，看成优雅。

其实，还有一种格式化的方法，被称为“字典格式化”，这里仅仅列一个例子，如果看官要了解字典的含义，本教程后续会有的。

```py
>>> lang = "Python"
>>> print "I love %(program)s"%{"program":lang}
I love Python 
```

列举了三种基本格式化的方法，你喜欢那种？我推荐：`string.format()`

### 常用的字符串方法

字符串的方法很多。可以通过 dir 来查看：

```py
>>> dir(str)
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__init__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_formatter_field_name_split', '_formatter_parser', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill'] 
```

这么多，不会一一介绍，要了解某个具体的含义和使用方法，最好是使用 help 查看。举例：

```py
>>> help(str.isalpha)

Help on method_descriptor:

isalpha(...)
    S.isalpha() -> bool

    Return True if all characters in S are alphabetic
    and there is at least one character in S, False otherwise. 
```

按照这里的说明，就可以在交互模式下进行实验。

```py
>>> "python".isalpha()    # 字符串全是字母，应该返回 True
True
>>> "2python".isalpha()    # 字符串含非字母，返回 False
False 
```

#### split

这个函数的作用是将字符串根据某个分割符进行分割。

```py
>>> a = "I LOVE PYTHON"
>>> a.split(" ")
['I', 'LOVE', 'PYTHON'] 
```

这是用空格作为分割，得到了一个名字叫做列表（list）的返回值，关于列表的内容，后续会介绍。还能用别的分隔吗？

```py
>>> b = "www.itdiffer.com"
>>> b.split(".")
['www', 'itdiffer', 'com'] 
```

#### 去掉字符串两头的空格

这个功能，在让用户输入一些信息的时候非常有用。有的朋友喜欢输入结束的时候敲击空格，比如让他输入自己的名字，输完了，他来个空格。有的则喜欢先加一个空格，总做的输入的第一个字前面应该空两个格。

这些空格是没用的。Python 考虑到有不少人可能有这个习惯，因此就帮助程序员把这些空格去掉。

方法是：

*   S.strip() 去掉字符串的左右空格
*   S.lstrip() 去掉字符串的左边空格
*   S.rstrip() 去掉字符串的右边空格

例如：

```py
>>> b=" hello "    # 两边有空格
>>> b.strip()
'hello'
>>> b
' hello ' 
```

特别注意，原来的值没有变化，而是新返回了一个结果。

```py
>>> b.lstrip()    # 去掉左边的空格
'hello '
>>> b.rstrip()    # 去掉右边的空格
' hello' 
```

#### 字符大小写的转换

对于英文，有时候要用到大小写转换。最有名驼峰命名，里面就有一些大写和小写的参合。如果有兴趣，可以来这里看[自动将字符串转化为驼峰命名形式的方法 href="https://github.com/qiwsir/algorithm/blob/master/string_to_hump.md")。

在 Python 中有下面一堆内建函数，用来实现各种类型的大小写转化

*   S.upper() #S 中的字母大写
*   S.lower() #S 中的字母小写
*   S.capitalize() # 首字母大写
*   S.isupper() #S 中的字母是否全是大写
*   S.islower() #S 中的字母是否全是小写
*   S.istitle()

看例子：

```py
>>> a = "qiwsir,Python" 
>>> a.upper()       # 将小写字母完全变成大写字母
'QIWSIR,PYTHON'
>>> a               # 原数据对象并没有改变
'qiwsir,Python'
>>> b = a.upper()
>>> b
'QIWSIR,PYTHON'
>>> c = b.lower()   # 将所有的小写字母变成大写字母
>>> c
'qiwsir,Python'

>>> a
'qiwsir,Python'
>>> a.capitalize()  # 把字符串的第一个字母变成大写
'Qiwsir,Python'
>>> a               # 原数据对象没有改变
'qiwsir,Python'
>>> b = a.capitalize() # 新建立了一个
>>> b
'Qiwsir,Python'

>>> a = "qiwsir,github"    # 这里的问题就是网友白羽毛指出的，非常感谢他。
>>> a.istitle()
False
>>> a = "QIWSIR"        # 当全是大写的时候，返回 False
>>> a.istitle()
False
>>> a = "qIWSIR"
>>> a.istitle()
False
>>> a = "Qiwsir,github"  # 如果这样，也返回 False
>>> a.istitle()
False
>>> a = "Qiwsir"        # 这样是 True
>>> a.istitle()
True
>>> a = 'Qiwsir,Github' # 这样也是 True
>>> a.istitle()
True

>>> a = "Qiwsir"
>>> a.isupper()
False
>>> a.upper().isupper()
True
>>> a.islower()
False
>>> a.lower().islower()
True 
```

顺着白羽毛网友指出的，再探究一下，可以这么做：

```py
>>> a = "This is a Book"
>>> a.istitle()
False
>>> b = a.title()     # 这样就把所有单词的第一个字母转化为大写
>>> b
'This Is A Book'
>>> b.istitle()       # 判断每个单词的第一个字母是否为大写
True 
```

#### join 拼接字符串

用“+”能够拼接字符串，但不是什么情况下都能够如愿的。比如，将列表（关于列表，后续详细说，它是另外一种类型）中的每个字符（串）元素拼接成一个字符串，并且用某个符号连接，如果用“+”，就比较麻烦了（是能够实现的，麻烦）。

用字符串的 join 就比较容易实现。

```py
>>> b
'www.itdiffer.com'
>>> c = b.split(".")
>>> c
['www', 'itdiffer', 'com']
>>> ".".join(c)
'www.itdiffer.com'
>>> "*".join(c)
'www*itdiffer*com' 
```

这种拼接，是不是简单呢？

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字符编码

我在第一版的《零基础学 Python》中，这个标题前面加了“坑爹”两个字。在后来的实践中，很多朋友都在网上问我关于编码的事情。说明这的确是一个“坑”。

首先说明，在 Python2 中，编码问题的确有点麻烦。但是，Python3 就不用纠结于此了。但是，正如前面所说的原因，至少本教程还是用 Python2，所以，必须要搞清楚编码。当然了，搞清楚，也不是坏事。

字符编码，在编程中，是一个让学习者比较郁闷的东西，比如一个 str，如果都是英文，好说多了。但恰恰不是如此，中文是我们不得不用的。所以，哪怕是初学者，都要了解并能够解决字符编码问题。

```py
>>> name = '老齐'
>>> name
'\xe8\x80\x81\xe9\xbd\x90' 
```

在你的编程中，你遇到过上面的情形吗？认识最下面一行打印出来的东西吗？看人家英文，就好多了

```py
>>> name = "qiwsir"
>>> name
'qiwsir' 
```

难道这是中文的错吗？看来投胎真的是一个技术活。是的，投胎是技术活，但上面的问题不是中文的错。

### 编码

什么是编码？这是一个比较玄乎的问题。也不好下一个普通定义。我看到有的教材中有定义，不敢说他的定义不对，至少可以说不容易理解。

古代打仗，击鼓进攻、鸣金收兵，这就是编码。把要传达给士兵的命令对应为一定的其它形式，比如命令“进攻”，经过如此的信息传递：

![](img/11001.png)

1.  长官下达进攻命令，传令员将这个命令编码为鼓声（如果复杂点，是不是有几声鼓响，如何进攻呢？）。
2.  鼓声在空气中传播，比传令员的嗓子吼出来的声音传播的更远，士兵听到后也不会引起歧义，一般不会有士兵把鼓声当做打呼噜的声音。这就是“进攻”命令被编码成鼓声之后的优势所在。
3.  士兵听到鼓声，就是接收到信息之后，如果接受过训练或者有人告诉过他们，他们就知道这是让我进攻。这个过程就是解码。所以，编码方案要有两套。一套在信息发出者那里，另外一套在信息接受者这里。经过解码之后，士兵明白了，才行动。

以上过程比较简单。其实，真实的编码和解码过程，要复杂了。不过，原理都差不多的。

举一个似乎遥远，其实不久前人们都在使用的东西做例子：[电报](http://zh.wikipedia.org/wiki/%E7%94%B5%E6%8A%A5)

> 电报是通信业务的一种，在 19 世纪初发明，是最早使用电进行通信的方法。电报大为加快了消息的流通，是工业社会的其中一项重要发明。早期的电报只能在陆地上通讯，后来使用了海底电缆，开展了越洋服务。到了 20 世纪初，开始使用无线电拨发电报，电报业务基本上已能抵达地球上大部份地区。电报主要是用作传递文字讯息，使用电报技术用作传送图片称为传真。
> 
> 中国首条出现电报线路是 1871 年，由英国、俄国及丹麦敷设，从香港经上海至日本长崎的海底电缆。由于清政府的反对，电缆被禁止在上海登陆。后来丹麦公司不理清政府的禁令，将线路引至上海公共租界，并在 6 月 3 日起开始收发电报。至于首条自主敷设的线路，是由福建巡抚丁日昌在台湾所建，1877 年 10 月完工，连接台南及高雄。1879 年，北洋大臣李鸿章在天津、大沽及北塘之间架设电报线路，用作军事通讯。1880 年，李鸿章奏准开办电报总局，由盛宣怀任总办。并在 1881 年 12 月开通天津至上海的电报服务。李鸿章説：“五年来，我国创设沿江沿海各省电线，总计一万多里，国家所费无多，巨款来自民间。当时正值法人挑衅，将帅报告军情，朝廷传达指示，均相机而动，无丝毫阻碍。中国自古用兵，从未如此神速。出使大臣往来问答，朝发夕至，相隔万里好似同居庭院。举设电报一举三得，既防止外敌侵略，又加强国防，亦有利于商务。”天津官电局于庚子遭乱全毁。1887 年，台湾巡抚刘铭传敷设了福州至台湾的海底电缆，是中国首条海底电缆。1884 年，北京电报开始建设，采用"安设双线，由通州展至京城，以一端引入署中，专递官信，以一端择地安置用便商民"，同年 8 月 5 日，电报线路开始建设，所有电线杆一律漆成红色。8 月 22 日，位于北京崇文门外大街西的喜鹊胡同的外城商用电报局开业。同年 8 月 30 日，位于崇文门内泡子和以西的吕公堂开局，专门收发官方电报。
> 
> 为了传达汉字，电报部门准备由 4 位数字或 3 位罗马字构成的代码，即中文电码，采用发送前将汉字改写成电码发出，收电报后再将电码改写成汉字的方法。

列位看官注意了，这里出现了电报中用的“[中文电码](http://zh.wikipedia.org/wiki/%E4%B8%AD%E6%96%87%E9%9B%BB%E7%A2%BC)”，这就是一种编码，将汉字对应成阿拉伯数字，从而能够用电报发送汉字。

> 1873 年,法国驻华人员威基杰参照《康熙字典》的部首排列方法,挑选了常用汉字 6800 多个,编成了第一部汉字电码本《电报新书》。

电报中的编码被称为[摩尔斯电码，英文是 Morse Code](http://zh.wikipedia.org/wiki/%E6%91%A9%E6%96%AF%E7%94%B5%E7%A0%81)

> 摩尔斯电码（英语：Morse Code）是一种时通时断的信号代码，通过不同的排列顺序来表达不同的英文字母、数字和标点符号。是由美国人萨缪尔·摩尔斯在 1836 年发明。
> 
> 摩尔斯电码是一种早期的数字化通信形式，但是它不同于现代只使用 0 和 1 两种状态的二进制代码，它的代码包括五种：点（.）、划（-）、每个字符间短的停顿（在点和划之间的停顿）、每个词之间中等的停顿、以及句子之间长的停顿

看来电报员是一个技术活，不同长短的停顿都代表了不同意思。哦，对了，有一个老片子《永不消逝的电波》，看完之后保证你才知道，里面根本就没有讲电报是怎么编码的。

> 摩尔斯电码在海事通讯中被作为国际标准一直使用到 1999 年。1997 年，当法国海军停止使用摩尔斯电码时，发送的最后一条消息是：“所有人注意，这是我们在永远沉寂之前最后的一声呐喊！”

![](img/11002.png)

我瞪着眼看了老长时间，这两行不是一样的吗？

不管这个了，总之，这就是编码。

### 计算机中的字符编码

先抄一段[维基百科对字符编码](http://zh.wikipedia.org/wiki/%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81)的解释：

> 字符编码（英语：Character encoding）、字集码是把字符集中的字符编码为指定集合中某一对象（例如：比特模式、自然数串行、8 位组或者电脉冲），以便文本在计算机中存储和通过通信网络的传递。常见的例子包括将拉丁字母表编码成摩斯电码和 ASCII。其中，ASCII 将字母、数字和其它符号编号，并用 7 比特的二进制来表示这个整数。通常会额外使用一个扩充的比特，以便于以 1 个字节的方式存储。
> 
> 在计算机技术发展的早期，如 ASCII（1963 年）和 EBCDIC（1964 年）这样的字符集逐渐成为标准。但这些字符集的局限很快就变得明显，于是人们开发了许多方法来扩展它们。对于支持包括东亚 CJK 字符家族在内的写作系统的要求能支持更大量的字符，并且需要一种系统而不是临时的方法实现这些字符的编码。

在这个世界上，有好多不同的字符编码。但是，它们不是自己随便搞搞的。而是要有一定的基础，往往是以名叫 [ASCII](http://zh.wikipedia.org/wiki/ASCII) 的编码为基础，这里边也应该包括北朝鲜吧（不知道他们用什么字符编码，瞎想的，别当真，不代表本教材立场，只代表瞎想）。

> ASCII（pronunciation: 英语发音：/ˈæski/ ASS-kee[1]，American Standard Code for Information Interchange，美国信息交换标准代码）是基于拉丁字母的一套电脑编码系统。它主要用于显示现代英语，而其扩展版本 EASCII 则可以部分支持其他西欧语言，并等同于国际标准 ISO/IEC 646。由于万维网使得 ASCII 广为通用，直到 2007 年 12 月，逐渐被 Unicode 取代。

上面的引文中已经说了，现在我们用的编码标准已经变成 Unicode 了，那么什么是 Unicode 呢？还是抄一段来自[维基百科的说明](http://zh.wikipedia.org/wiki/Unicode)

> Unicode（中文：万国码、国际码、统一码、单一码）是计算机科学领域里的一项业界标准。它对世界上大部分的文字系统进行了整理、编码，使得电脑可以用更为简单的方式来呈现和处理文字。
> 
> Unicode 伴随着通用字符集的标准而发展，同时也以书本的形式对外发表。Unicode 至今仍在不断增修，每个新版本都加入更多新的字符。目前最新的版本为 7.0.0，已收入超过十万个字符（第十万个字符在 2005 年获采纳）。Unicode 涵盖的数据除了视觉上的字形、编码方法、标准的字符编码外，还包含了字符特性，如大小写字母。

听这名字：万国码，那就一定包含了中文喽。的确是。但是，光有一个 Unicode 还不行，因为....（此处省略若干字，看官可以到上面给出的维基百科链接中看），还要有其它的一些编码实现方式，Unicode 的实现方式称为 Unicode 转换格式（Unicode Transformation Format，简称为 UTF），于是乎有了一个我们在很多时候都会看到的 utf-8。

什么是 utf-8，还是看[维基百科](http://zh.wikipedia.org/wiki/UTF-8)上怎么说的吧

> UTF-8（8-bit Unicode Transformation Format）是一种针对 Unicode 的可变长度字符编码，也是一种前缀码。它可以用来表示 Unicode 标准中的任何字符，且其编码中的第一个字节仍与 ASCII 兼容，这使得原来处理 ASCII 字符的软件无须或只须做少部份修改，即可继续使用。因此，它逐渐成为电子邮件、网页及其他存储或发送文字的应用中，优先采用的编码。

不再多引用了，如果要看更多，请到原文。

看官现在是不是就理解了，前面写程序的时候，曾经出现过：coding:utf-8 的字样。就是在告诉 python 我们要用什么字符编码呢。

### encode 和 decode

历史部分说完了，接下怎么讲？比较麻烦了。因为不管怎么讲，都不是三言两语说清楚的。姑且从 encode()和 decode()两个内置函数起吧。

> codecs.encode(obj[, encoding[, errors]]):Encodes obj using the codec registered for encoding. codecs.decode(obj[, encoding[, errors]]):Decodes obj using the codec registered for encoding.

Python2 默认的编码是 ascii，通过 encode 可以将对象的编码转换为指定编码格式（称作“编码”），而 decode 是这个过程的逆过程（称作“解码”）。

做一个实验，才能理解：

```py
>>> a = "中"
>>> type(a)
<type 'str'>
>>> a
'\xe4\xb8\xad'
>>> len(a)
3

>>> b = a.decode()
>>> b
u'\u4e2d'
>>> type(b)
<type 'unicode'>
>>> len(b)
1 
```

这个实验不做之前，或许看官还不是很迷茫（因为不知道，知道的越多越迷茫），实验做完了，自己也迷茫了。别急躁，对编码问题的理解，要慢慢来，如果一时理解不了，也肯定理解不了，就先注意按照要求做，做着做着就豁然开朗了。

上面试验中，变量 a 引用了一个字符串，所谓字符串(str)，严格地将是字节串，它是经过编码后的字节组成的序列。也就是你在上面的实验中，看到的是“中”这个字在计算机中编码之后的字节表示。（关于字节，看官可以 google 一下）。用 len(a)来度量它的长度，它是由三个字节组成的。

然后通过 decode 函数，将**字节串**转变为**字符串**，并且这个字符串是按照 unicode 编码的。在 unicode 编码中，一个汉字对应一个字符，这时候度量它的长度就是 1.

反过来，一个 unicode 编码的字符串，也可以转换为字节串。

```py
>>> c = b.encode('utf-8')
>>> c
'\xe4\xb8\xad'
>>> type(c)
<type 'str'>
>>> c == a
True 
```

关于编码问题，先到这里，点到为止吧。因为再扯，还会扯出问题来。看官肯定感到不满意，因为还没有知其所以然。没关系，请尽情 google，即可解决。

### Python 中如何避免中文是乱码

这个问题是一个具有很强操作性的问题。我这里有一个经验总结，分享一下，供参考：

首先，提倡使用 utf-8 编码方案，因为它跨平台不错。

经验一：在开头声明：

```py
# -*- coding: utf-8 -*- 
```

有朋友问我-*-有什么作用，那个就是为了好看，爱美之心人皆有，更何况程序员？当然，也可以写成：

```py
# coding:utf-8 
```

经验二：遇到字符（节）串，立刻转化为 unicode，不要用 str()，直接使用 unicode()

```py
unicode_str = unicode('中文', encoding='utf-8')
print unicode_str.encode('utf-8') 
```

经验三：如果对文件操作，打开文件的时候，最好用 codecs.open，替代 open(这个后面会讲到，先放在这里)

```py
import codecs
codecs.open('filename', encoding='utf8') 
```

我还收集了网上的一片文章，也挺好的，推荐给看官：Python2.x 的中文显示方法 href="https://github.com/qiwsir/ITArticles/blob/master/Python/Python%E7%9A%84%E4%B8%AD%E6%96%87%E6%98%BE%E7%A4%BA%E6%96%B9%E6%B3%95.md")

最后告诉给我，如果用 Python3，坑爹的编码问题就不烦恼了。

* * *

[总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 列表(1)

前面的学习中，我们已经知道了两种 Python 的数据类型：int 和 str。再强调一下对数据类型的理解，这个世界是由数据组成的，数据可能是数字（注意，别搞混了，数字和数据是有区别的），也可能是文字、或者是声音、视频等。在 Python 中（其它高级语言也类似）把状如 2,3 这样的数字划分为一个类型，把状如“你好”这样的文字划分一个类型，前者是 int 类型，后者是 str 类型（这里就不说翻译的名字了，请看官熟悉用英文的名称，对日后编程大有好处，什么好处呢？谁用谁知道！）。

前面还学习了变量，如果某个变量指向一个对象（某种类型的数据）行话是：赋值），通常这个变量我们就把它叫做 int 类型的变量（注意，这种说法是不严格的，或者是受到别的语言影响的，在 Python 中，特别要注意：**变量没有类型，对象有类型**。在 Python 里，变量不用提前声明（在某些语言，如 JAVA 中需要声明变量之后才能使用。这个如果看官没有了解，不用担心，因为我们是学习 Python，以后学习的语言多了，自然就能体会到这点区别了），随用随命名。

这一讲中的 list 类型，也是 Python 的一种数据类型。翻译为：列表。下面的黑字，请看官注意了：

**LIST 在 Python 中具有非常强大的功能。**

### 定义

在 Python 中，用方括号表示一个 list，[ ]

在方括号里面，可以是 int，也可以是 str 类型的数据，甚至也能够是 True/False 这种布尔值。看下面的例子，特别注意阅读注释。

```py
>>> a=[]        #定义了一个变量 a，它是 list 类型，并且是空的。
>>> type(a)
<type 'list'>   #用内置函数 type()查看变量 a 的类型，为 list
>>> bool(a)     #用内置函数 bool()看看 list 类型的变量 a 的布尔值，因为是空的，所以为 False
False
>>> print a     #打印 list 类型的变量 a
[] 
```

`bool()`是一个布尔函数，这个东西后面会详述。它的作用就是来判断一个对象是“真”还是“空”（假）。如果想上面例子那样，list 中什么也没有，就是空的，用 bool()函数来判断，得到 False，从而显示它是空的。

不能总玩空的，来点实的吧。

```py
>>> a=['2',3,'qiwsir.github.io']
>>> a
['2', 3, 'qiwsir.github.io']
>>> type(a)
<type 'list'>
>>> bool(a)
True
>>> print a
['2', 3, 'qiwsir.github.io'] 
```

用上述方法，定义一个 list 类型的变量和数据。

本讲的标题是“有容乃大的 list”，就指明了 list 的一大特点：可以无限大，就是说 list 里面所能容纳的元素数量无限，当然这是在硬件设备理想的情况下。

> 如果看官以后或者已经了解了别的语言，比如比较常见的 Java，里面有一个跟 list 相似的数据类型——数组——但是两者还是有区别的。在 Java 中，数组中的元素必须是基本数据类型中某一个，也就是要么都是 int 类型，要么都是 char 类型等，不能一个数组中既有 int 类型又有 char 类型。这是因为 java 中的数组，需要提前声明，声明的时候就确定了里面元素的类型。但是 python 中的 list，尽管跟 java 中的数组有类似的地方——都是`[]`包裹的—— list 中的元素是任意类型的，可以是 int,str，甚至还可以是 list，乃至于是以后要学的 dict 等。所以，有一句话说：List 是 python 中的苦力，什么都可以干。

### 索引和切片

尚记得在《字符串(3)》中，曾经给“索引”(index)和“切片”。

```py
>>> url = "qiwsir.github.io"
>>> url[2]
'w'
>>> url[:4]
'qiws'
>>> url[3:9]
'sir.gi' 
```

在 list 中，也有类似的操作。只不过是以元素为单位，不是以字符为单位进行索引了。看例子就明白了。

```py
>>> a
['2', 3, 'qiwsir.github.io']
>>> a[0]    #索引序号也是从 0 开始
'2'
>>> a[1]
3
>>> [2]
[2]
>>> a[:2]   #跟 str 中的类似，切片的范围是：包含开始位置，到结束位置之前
['2', 3]    #不包含结束位置
>>> a[1:]
[3, 'qiwsir.github.io'] 
```

list 和 str 两种类型的数据，有共同的地方，它们都属于序列（都是一些对象按照某个次序排列起来，这就是序列的最大特征），因此，就有很多类似的地方。如刚才演示的索引和切片，是非常一致的。

```py
>>> lang = "python"
>>> lang.index("y")
1
>>> lst = ['python','java','c++']
>>> lst.index('java')
1 
```

在前面讲述字符串索引和切片的时候，以及前面的演示，所有的索引都是从左边开始编号，第一个是 0，然后依次增加 1。此外，还有一种编号方式，就是从右边开始，右边第一个可以编号为 `-1`，然后向左依次是：-2,-3,...，依次类推下来。这对字符串、列表等各种序列类型都是用。

```py
>>> lang
'python'
>>> lang[-1]
'n'
>>> lst
['python', 'java', 'c++']
>>> lst[-1]
'c++' 
```

从右边开始编号，第 -1 号是右边第一个。但是，如果要切片的话，应该注意了。

```py
>>> lang[-1:-3]
''
>>> lang[-3:-1]
'ho'
>>> lst[-3:-1]
['python', 'java'] 
```

序列的切片，一定要左边的数字小有右边的数字，`lang[-1:-3]`就没有遵守这个规则，返回的是一个空。

### 反转

这个功能作为一个独立的项目提出来，是因为在编程中常常会用到。通过举例来说明反转的方法：

```py
>>> alst = [1,2,3,4,5,6]
>>> alst[::-1]    #反转
[6, 5, 4, 3, 2, 1]
>>> alst
[1, 2, 3, 4, 5, 6] 
```

当然，对于字符串也可以

```py
>>> lang
'python'
>>> lang[::-1]
'nohtyp'
>>> lang
'python' 
```

看官是否注意到，上述不管是 str 还是 lst 反转之后，再看原来的值，没有改变。这就说明，这里的反转，不是在“原地”把原来的值倒过来，而是新生成了一个值，那个值跟原来的值相比，是倒过来了。

这是一种非常简单的方法，虽然我在写程序的时候常常使用，但是，我不是十分推荐，因为有时候让人感觉迷茫。Python 还有另外一种方法让 list 反转，是比较容易理解和阅读的，特别推荐之：

```py
>>> list(reversed(alst))
[6, 5, 4, 3, 2, 1] 
```

比较简单，而且很容易看懂。不是吗？

顺便给出 reversed 函数的详细说明：

```py
>>> help(reversed)
Help on class reversed in module __builtin__:

class reversed(object)
 |  reversed(sequence) -> reverse iterator over values of the sequence
 |  
 |  Return a reverse iterator 
```

它返回一个可以迭代的对象（关于迭代的问题，后续会详述之），不过是已经将原来的序列对象反转了。比如：

```py
>>> list(reversed("abcd"))
['d', 'c', 'b', 'a'] 
```

很好，很强大，特别推荐使用。

### 对 list 的操作

任何一个行业都有自己的行话，如同古代的强盗，把撤退称之为“扯乎”一样，纵然是一个含义，但是强盗们愿意用他们自己的行业用语，俗称“黑话”。各行各业都如此。这样做的目的我理解有两个，一个是某种保密；另外一个是行外人士显示本行业的门槛，让别人感觉这个行业很高深，从业者有一定水平。

不管怎么，在 Python 和很多高级语言中，都给本来数学角度就是函数的东西，又在不同情况下有不同的称呼，如方法、类等。当然，这种称呼，其实也是为了区分函数的不同功能。

前面在对 str 进行操作的时候，有一些内置函数，比如 s.strip()，这是去掉左右空格的内置函数，也是 str 的方法。按照一贯制的对称法则，对 list 也会有一些操作方法。

在讲述字符串的时候，提到过，所有的序列，都有几种基本操作。list 当然如此。

#### 基本操作

*   len()

在交互模式中操作：

```py
>>> lst
['python', 'java', 'c++']
>>> len(lst)
3 
```

*   +，连接两个序列

交互模式中：

```py
>>> lst
['python', 'java', 'c++']
>>> alst
[1, 2, 3, 4, 5, 6]
>>> lst + alst
['python', 'java', 'c++', 1, 2, 3, 4, 5, 6] 
```

*   *，重复元素

交互模式中操作

```py
>>> lst
['python', 'java', 'c++']
>>> lst * 3
['python', 'java', 'c++', 'python', 'java', 'c++', 'python', 'java', 'c++'] 
```

*   in

列表 lst 还是前面的值

```py
>>> "python" in lst
True
>>> "c#" in lst
False 
```

*   max()和 min()

以 int 类型元素为例。如果不是，都是按照字符在 ascii 编码中所对应的数字进行比较的。

```py
>>> alst
[1, 2, 3, 4, 5, 6]
>>> max(alst)
6
>>> min(alst)
1
>>> max(lst)
'python'
>>> min(lst)
'c++' 
```

*   cmp()

采用上面的方法，进行比较

```py
>>> lsta = [2,3]
>>> lstb = [2,4]
>>> cmp(lsta,lstb)
-1
>>> lstc = [2]
>>> cmp(lsta,lstc)
1
>>> lstd = ['2','3']
>>> cmp(lsta,lstd)
-1 
```

#### 追加元素

```py
>>> a = ["good","python","I"]      
>>> a
['good', 'python', 'I']
>>> a.append("like")        #向 list 中添加 str 类型 "like"
>>> a
['good', 'python', 'I', 'like']
>>> a.append(100)           #向 list 中添加 int 类型 100
>>> a
['good', 'python', 'I', 'like', 100] 
```

[官方文档](https://docs.python.org/2/tutorial/datastructures.html)这样描述 list.append()方法

> list.append(x)
> 
> Add an item to the end of the list; equivalent to a[len(a):] = [x].

从以上描述中，以及本部分的标题“追加元素”，是不是能够理解 list.append(x)的含义呢？即将新的元素 x 追加到 list 的尾部。

列位看官，如果您注意看上面官方文档中的那句话，应该注意到，还有后面半句： equivalent to a[len(a):] = [x]，意思是说 list.append(x)等效于：a[len(a):]=[x]。这也相当于告诉我们了另外一种追加元素的方法，并且两种方法等效。

```py
>>> a
['good', 'python', 'I', 'like', 100]
>>> a[len(a):]=[3]      #len(a),即得到 list 的长度，这个长度是指 list 中的元素个数。
>>> a
['good', 'python', 'I', 'like', 100, 3]
>>> len(a)
6
>>> a[6:]=['xxoo']
>>> a
['good', 'python', 'I', 'like', 100, 3, 'xxoo'] 
```

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 列表(2)

上一节中已经谈到，list 是 Python 的苦力，那么它都有哪些函数呢？或者它或者对它能做什么呢？在交互模式下这么操作，就看到有关它的函数了。

```py
>>> dir(list)
['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__delslice__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getslice__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort'] 
```

上面的结果中，以双下划线开始和结尾的暂时不管，如`__add__`（以后会管的）。就剩下以下几个了：

> 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort'

下面注意对这些函数进行说明和演示。这都是在编程实践中常常要用到的。

### list 函数

#### append 和 extend

《列表(1)》中，对 list 的基本操作提到了 list.append(x)，也就是将某个元素 x 追加到已知的一个 list 后边。

除了将元素追加到 list 中，还能够将两个 list 合并，或者说将一个 list 追加到另外一个 list 中。按照前文的惯例，还是首先看[官方文档](https://docs.Python.org/2/tutorial/datastructures.html)中的描述：

> list.extend(L)
> 
> Extend the list by appending all the items in the given list; equivalent to a[len(a):] = L.

**向所有正在学习本内容的朋友提供一个成为优秀程序员的必备：看官方文档，是必须的。**

官方文档的这句话翻译过来：

> 通过将所有元素追加到已知 list 来扩充它，相当于 a[len(a):]= L

英语太烂，翻译太差。直接看例子，更明白

```py
>>> la
[1, 2, 3]
>>> lb
['qiwsir', 'python']
>>> la.extend(lb)
>>> la
[1, 2, 3, 'qiwsir', 'python']
>>> lb
['qiwsir', 'python'] 
```

上面的例子，显示了如何将两个 list，一个是 la，另外一个 lb，将 lb 追加到 la 的后面，也就是把 lb 中的所有元素加入到 la 中，即让 la 扩容。

学程序一定要有好奇心，我在交互环境中，经常实验一下自己的想法，有时候是比较愚蠢的想法。

```py
>>> la = [1,2,3]
>>> b = "abc"
>>> la.extend(b)
>>> la
[1, 2, 3, 'a', 'b', 'c']
>>> c = 5
>>> la.extend(c)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  TypeError: 'int' object is not iterable 
```

从上面的实验中，看官能够有什么心得？原来，如果 extend(str)的时候，str 被以字符为单位拆开，然后追加到 la 里面。

如果 extend 的对象是数值型，则报错。

所以，extend 的对象是一个 list，如果是 str，则 Python 会先把它按照字符为单位转化为 list 再追加到已知 list。

不过，别忘记了前面官方文档的后半句话，它的意思是：

```py
>>> la
[1, 2, 3, 'a', 'b', 'c']
>>> lb
['qiwsir', 'python']
>>> la[len(la):]=lb
>>> la
[1, 2, 3, 'a', 'b', 'c', 'qiwsir', 'python'] 
```

list.extend(L) 等效于 list[len(list):] = L，L 是待并入的 list

联想到到上一讲中的一个 list 函数 list.append(),有类似之处。

> extend(...) L.extend(iterable) -- extend list by appending elements from the iterable

上面是在交互模式中输入 `help(list.extend)`后得到的说明。这是非常重要而且简单的获得文档帮助的方法。

从上面内容可知，extend 函数也是将另外的元素增加到一个已知列表中，其元素必须是 iterable，什么是 iterable？这个从现在开始，后面会经常遇到，所以是要搞搞清楚的。

> iterable,中文含义是“可迭代的”。在 Python 中，还有一个词，就是 iterator，这个叫做“迭代器”。这两者有着区别和联系。不过，这里暂且不说那么多，说多了就容易糊涂，我也糊涂了。

为了解释 iterable(可迭代的)，又引入了一个词“迭代”，什么是迭代呢？

> 尽管我们很多文档是用英文写的，但是，如果你能充分利用汉语来理解某些名词，是非常有帮助的。因为在汉语中，不仅仅表音，而且能从词语组合中体会到该术语的含义。比如“激光”，这是汉语。英语是从"light amplification by stimulated emission of radiation"化出来的"laser"，它是一个造出来的词。因为此前人们不知道那种条件下发出来的是什么。但是汉语不然，反正用一个“光”就可以概括了，只不过这个“光”不是传统概念中的“光”，而是由于“受激”辐射得到的光，故名“激光”。是不是汉语很牛叉？
> 
> “迭”在汉语中的意思是“屡次,反复”。如:高潮迭起。那么跟“代”组合，就可以理解为“反复‘代’”，是不是有点“子子孙孙”的意思了？“结婚-生子-子成长-结婚-生子-子成长-...”，你是不是也在这个“迭代”的过程中呢？
> 
> 给个稍微严格的定义，来自维基百科。“迭代是重复反馈过程的活动，其目的通常是为了接近并到达所需的目标或结果。”

某些类型的对象是“可迭代”(iterable)的，这类数据类型有共同的特点。如何判断一个对象是不是可迭代的？下面演示一种方法。事实上还有别的方式。

```py
>>> astr = "Python"
>>> hasattr(astr,'__iter__')
False 
```

这里用内建函数 `hasattr()`判断一个字符串是否是可迭代的，返回了 False。用同样的方式可以判断：

```py
>>> alst = [1,2]
>>> hasattr(alst,'__iter__')
True
>>> hasattr(3, '__iter__')
False 
```

`hasattr()`的判断本质就是看那个类型中是否有`__iter__`函数。看官可以用 `dir()`找一找，在数字、字符串、列表中，谁有`__iter__`。同样还可找一找 dict,tuple 两种类型对象是否含有这个方法。

以上穿插了一个新的概念“iterable”（可迭代的），现在回到 extend 上。这个函数需要的参数就是 iterable 类型的对象。

```py
>>> new = [1,2,3]
>>> lst = ['Python','qiwsir']
>>> lst.extend(new)
>>> lst
['Python', 'qiwsir', 1, 2, 3]
>>> new
[1, 2, 3] 
```

通过 extend 函数，将[1,2,3]中的每个元素都拿出来，然后塞到 lst 里面，从而得到了一个跟原来的对象元素不一样的列表，后面的比原来的多了三个元素。上面说的有点啰嗦，只不过是为了把过程完整表达出来。

还要关注一下，从上面的演示中可以看出，lst 经过 extend 函数操作之后，变成了一个貌似“新”的列表。这句话好像有点别扭，“貌似新”的，之所以这么说，是因为对“新的”可能有不同的理解。不妨深挖一下。

```py
>>> new = [1,2,3]
>>> id(new)
3072383244L

>>> lst = ['python', 'qiwsir']
>>> id(lst)
3069501420L 
```

用 `id()`能够看到两个列表分别在内存中的“窝”的编号。

```py
>>> lst.extend(new)
>>> lst
['python', 'qiwsir', 1, 2, 3]
>>> id(lst)
3069501420L 
```

看官注意到没有，虽然 lst 经过 `extend()`方法之后，比原来扩容了，但是，并没有离开原来的“窝”，也就是在内存中，还是“旧”的，只不过里面的内容增多了。相当于两口之家，经过一番云雨之后，又增加了一个小宝宝，那么这个家是“新”的还是“旧”的呢？角度不同或许说法不一了。

这就是列表的一个**重要特征：列表是可以修改的。这种修改，不是复制一个新的，而是在原地进行修改。**

其实，`append()`对列表的操作也是如此，不妨用同样的方式看看。

**说明：**虽然这里的 lst 内容和上面的一样，但是，我从新在 shell 中输入，所以 id 会变化。也就是内存分配的“窝”的编号变了。

```py
>>> lst = ['Python','qiwsir']
>>> id(lst)     
3069501388L
>>> lst.append(new)
>>> lst
['Python', 'qiwsir', [1, 2, 3]]
>>> id(lst)
3069501388L 
```

显然，`append()`也是原地修改列表。

如果，对于 `extend()`，提供的不是 iterable 类型对象，会如何呢？

```py
>>> lst.extend("itdiffer")
>>> lst
['python', 'qiwsir', 'i', 't', 'd', 'i', 'f', 'f', 'e', 'r'] 
```

它把一个字符串"itdiffer"转化为['i', 't', 'd', 'i', 'f', 'f', 'e', 'r']，然后将这个列表作为参数，提供给 extend，并将列表中的元素塞入原来的列表中。

```py
>>> num_lst = [1,2,3]
>>> num_lst.extend(8)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'int' object is not iterable 
```

这就报错了。错误提示中告诉我们，那个数字 8，是 int 类型的对象，不是 iterable 的。

这里讲述的两个让列表扩容的函数 `append()`和 `extend()`。从上面的演示中，可以看到他们有相同的地方：

*   都是原地修改列表
*   既然是原地修改，就不返回值

原地修改没有返回值，就不能赋值给某个变量。

```py
>>> one = ["good","good","study"]
>>> another = one.extend(["day","day","up"])    #对于没有提供返回值的函数，如果要这样，结果是：
>>> another                                     #这样的，什么也没有得到。
>>> one
['good', 'good', 'study', 'day', 'day', 'up'] 
```

那么两者有什么不一样呢？看下面例子：

```py
>>> lst = [1,2,3]
>>> lst.append(["qiwsir","github"])
>>> lst
[1, 2, 3, ['qiwsir', 'github']]  #append 的结果
>>> len(lst)
4

>>> lst2 = [1,2,3]
>>> lst2.extend(["qiwsir","github"])
>>> lst2
[1, 2, 3, 'qiwsir', 'github']   #extend 的结果
>>> len(lst2)
5 
```

append 是整建制地追加，extend 是个体化扩编。

#### count

上面的 len(L)，可得到 list 的长度，也就是 list 中有多少个元素。python 的 list 还有一个函数，就是数一数某个元素在该 list 中出现多少次，也就是某个元素有多少个。官方文档是这么说的：

> list.count(x)
> 
> Return the number of times x appears in the list.

一定要不断实验，才能理解文档中精炼的表达。

```py
>>> la = [1,2,1,1,3]
>>> la.count(1)
3
>>> la.append('a')
>>> la.append('a')
>>> la
[1, 2, 1, 1, 3, 'a', 'a']
>>> la.count('a')
2
>>> la.count(2)
1
>>> la.count(5)     #NOTE:la 中没有 5,但是如果用这种方法找，不报错，返回的是数字 0
0 
```

#### index

《列表(1)》中已经提到，这里不赘述，但是为了完整，也占个位置吧。

```py
>>> la
[1, 2, 3, 'a', 'b', 'c', 'qiwsir', 'python']
>>> la.index(3)
2
>>> la.index('qi')      #如果不存在，就报错
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  ValueError: 'qi' is not in list
>>> la.index('qiwsir')
6 
```

list.index(x)，x 是 list 中的一个元素，这样就能够检索到该元素在 list 中的位置了。这才是真正的索引，注意那个英文单词 index。

依然是上一条官方解释：

> list.index(x)
> 
> Return the index in the list of the first item whose value is x. It is an error if there is no such item.

是不是说的非常清楚明白了？

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 列表(3)

接着上节内容。下面是上节中说好要介绍的列表方法：

> 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort'

已经在上节讲解了前四个。

继续。

### list 函数

#### insert

前面有向 list 中追加元素的方法，那个追加是且只能是将新元素添加在 list 的最后一个。如：

```py
>>> all_users = ["qiwsir","github"]
>>> all_users.append("io")
>>> all_users
['qiwsir', 'github', 'io'] 
```

与 `list.append(x)`类似，`list.insert(i,x)`也是对 list 元素的增加。只不过是可以在任何位置增加一个元素。

还是先看[官方文档来理解](https://docs.python.org/2/tutorial/datastructures.html)：

> list.insert(i, x)
> 
> Insert an item at a given position. The first argument is the index of the element before which to insert, so a.insert(0, x) inserts at the front of the list, and a.insert(len(a), x) is equivalent to a.append(x).

这次就不翻译了。如果看不懂英语，怎么了解贵国呢？一定要硬着头皮看英语，不仅能够学好程序，更能...（此处省略两千字）

根据官方文档的说明，我们做下面的实验，请看官从实验中理解：

```py
>>> all_users
['qiwsir', 'github', 'io']
>>> all_users.insert("python")      
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: insert() takes exactly 2 arguments (1 given) 
```

请注意看报错的提示信息，`insert()`应该供给两个参数，但是这里只给了一个。所以报错没商量啦。

```py
>>> all_users.insert(0,"python")
>>> all_users
['python', 'qiwsir', 'github', 'io']

>>> all_users.insert(1,"http://")
>>> all_users
['python', 'http://', 'qiwsir', 'github', 'io'] 
```

`list.insert(i, x)`中的 i 是将元素 x 插入到 list 中的位置，即将 x 插入到索引值是 i 的元素前面。注意，索引是从 0 开始的。

有一种操作，挺有意思的，如下：

```py
>>> length = len(all_users)
>>> length
5       
>>> all_users.insert(length,"algorithm")
>>> all_users
['python', 'http://', 'qiwsir', 'github', 'io', 'algorithm'] 
```

在 all_users 中，没有索引最大到 4，如果要 `all_users.insert(5,"algorithm")`，则表示将`"algorithm"`插入到索引值是 5 的前面，但是没有。换个说法，5 前面就是 4 的后面。所以，就是追加了。

其实，还可以这样：

```py
>>> a = [1,2,3]
>>> a.insert(9,777)
>>> a
[1, 2, 3, 777] 
```

也就是说，如果遇到那个 i 已经超过了最大索引值，会自动将所要插入的元素放到列表的尾部，即追加。

#### pop 和 remove

list 中的元素，不仅能增加，还能被删除。删除 list 元素的方法有两个，它们分别是：

> list.remove(x)
> 
> Remove the first item from the list whose value is x. It is an error if there is no such item.
> 
> list.pop([i])
> 
> Remove the item at the given position in the list, and return it. If no index is specified, a.pop() removes and returns the last item in the list. (The square brackets around the i in the method signature denote that the parameter is optional, not that you should type square brackets at that position. You will see this notation frequently in the Python Library Reference.)

我这里讲授 Python，有一个习惯，就是用学习物理的方法。如果看官当初物理没有学好，那么一定是没有用这种方法，或者你的老师没有用这种教学法。这种方法就是：自己先实验，然后总结规律。

先实验 list.remove(x)，注意看上面的描述。这是一个能够删除 list 元素的方法，同时上面说明告诉我们，如果 x 没有在 list 中，会报错。

```py
>>> all_users
['Python', 'http://', 'qiwsir', 'github', 'io', 'algorithm']
>>> all_users.remove("http://")
>>> all_users       #的确是把"http://"删除了
['Python', 'qiwsir', 'github', 'io', 'algorithm']

>>> all_users.remove("tianchao")        #原 list 中没有“tianchao”，要删除，就报错。
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: list.remove(x): x not in list

>>> lst = ["python","java","python","c"]
>>> lst.remove("python")
>>> lst
['java', 'python', 'c'] 
```

重点解释一下第三个操作。哦，忘记一个提醒，我在前面的很多操作中，也都给列表的变量命名为 lst，但是不是 list,为什么呢？因为 list 是 Python 的保留字。

还是继续第三段操作，列表中有两个'Python'字符串，当删除后，发现结果只删除了第一个'Python'字符串，第二个还在。请仔细看前面的文档说明：**remove the first item ...**

注意两点：

*   如果正确删除，不会有任何反馈。没有消息就是好消息。并且是对列表进行原地修改。
*   如果所删除的内容不在 list 中，就报错。注意阅读报错信息：x not in list

> 什么是保留字？在 Python 中，当然别的语言中也是如此啦。某些词语或者拼写是不能被用户拿来做变量／函数／类等命名，因为它们已经被语言本身先占用了。这些就是所谓保留字。在 Python 中，以下是保留字，不能用于你自己变成中的任何命名。
> 
> and, assert, break, class, continue, def, del, elif, else, except, exec, finally, for, from, global, if, import, in, is, lambda, not, or, pass, print, raise, return, try, while, with,yield
> 
> 这些保留字，都是我们在编程中要用到的。有的你已经在前面遇到了。

看官是不是想到一个问题？如果能够在删除之前，先判断一下这个元素是不是在 list 中，如果在就删，不在就不删，不是更智能吗？

如果看官想到这里，就是在编程的旅程上一进步。Python 的确让我们这么做。

```py
>>> all_users
['python', 'qiwsir', 'github', 'io', 'algorithm']
>>> "Python" in all_users       #这里用 in 来判断一个元素是否在 list 中，在则返回 True，否则返回 False
True

>>> if "Python" in all_users:
...     all_users.remove("python")
...     print all_users
... else:
...     print "'Python' is not in all_users"
... 
['qiwsir', 'github', 'io', 'algorithm']     #删除了"Python"元素

>>> if "Python" in all_users:
...     all_users.remove("Python")
...     print all_users
... else:
...     print "'Python' is not in all_users"
... 
'Python' is not in all_users        #因为已经删除了，所以就没有了。 
```

上述代码，就是两段小程序，我是在交互模式中运行的，相当于小实验。这里其实用了一个后面才会讲到的东西：if-else 语句。不过，我觉得即使没有学习，你也能看懂，因为它非常接近自然语言了。

另外一个删除 list.pop([i])会怎么样呢？看看文档，做做实验。

```py
>>> all_users
['qiwsir', 'github', 'io', 'algorithm']
>>> all_users.pop()     #list.pop([i]),圆括号里面是[i]，表示这个序号是可选的
'algorithm'             #如果不写，就如同这个操作，默认删除最后一个，并且将该结果返回

>>> all_users
['qiwsir', 'github', 'io']

>>> all_users.pop(1)        #指定删除编号为 1 的元素"github"
'github'

>>> all_users
['qiwsir', 'io']
>>> all_users.pop()
'io'

>>> all_users           #只有一个元素了，该元素编号是 0
['qiwsir']
>>> all_users.pop(1)    #但是非要删除编号为 1 的元素，结果报错。注意看报错信息
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: pop index out of range      #删除索引超出范围，就是 1 不在 list 的编号范围之内 
```

简单总结一下，`list.remove(x)`中的参数是列表中元素，即删除某个元素；`list.pop([i])`中的 i 是列表中元素的索引值，这个 i 用方括号包裹起来，意味着还可以不写任何索引值，如上面操作结果，就是删除列表的最后一个。

给看官留下一个思考题，如果要像前面那样，能不能事先判断一下要删除的编号是不是在 list 的长度范围（用 len(list)获取长度)以内？然后进行删除或者不删除操作。

#### reverse

reverse 比较简单，就是把列表的元素顺序反过来。

```py
>>> a = [3,5,1,6]
>>> a.reverse()
>>> a
[6, 1, 5, 3] 
```

注意，是原地反过来，不是另外生成一个新的列表。所以，它没有返回值。跟这个类似的有一个内建函数 reversed，建议读者了解一下这个函数的使用方法。

> 因为 `list.reverse()`不返回值，所以不能实现对列表的反向迭代，如果要这么做，可以使用 reversed 函数。

#### sort

sort 就是对列表进行排序。帮助文档中这么写的：

> sort(...)
> 
> L.sort(cmp=None, key=None, reverse=False) -- stable sort *IN PLACE*; cmp(x, y) -> -1, 0, 1

```py
>>> a = [6, 1, 5, 3]
>>> a.sort()
>>> a
[1, 3, 5, 6] 
```

`list.sort()`也是让列表进行原地修改，没有返回值。默认情况，如上面操作，实现的是从小到大的排序。

```py
>>> a.sort(reverse=True)
>>> a
[6, 5, 3, 1] 
```

这样做，就实现了从大到小的排序。

在前面的函数说明中，还有一个参数 key，这个怎么用呢？不知道看官是否用过电子表格，里面就是能够设置按照哪个关键字进行排序。这里也是如此。

```py
>>> lst = ["Python","java","c","pascal","basic"]
>>> lst.sort(key=len)
>>> lst
['c', 'java', 'basic', 'Python', 'pascal'] 
```

这是以字符串的长度为关键词进行排序。

对于排序，也有一个更为常用的内建函数 sorted。

顺便指出，排序是一个非常有研究价值的话题。不仅仅是现在这么一个函数。有兴趣的读者可以去网上搜一下排序相关知识。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 回顾 list 和 str

list 和 str 两种类型数据，有不少相似的地方，也有很大的区别。本讲对她们做个简要比较，同时也是对前面有关两者的知识复习一下，所谓“温故而知新”。

### 相同点

#### 都属于序列类型的数据

所谓序列类型的数据，就是说它的每一个元素都可以通过指定一个编号，行话叫做“偏移量”的方式得到，而要想一次得到多个元素，可以使用切片。偏移量从 0 开始，总元素数减 1 结束。

例如：

```py
>>> welcome_str = "Welcome you"
>>> welcome_str[0]
'W'
>>> welcome_str[1]
'e'
>>> welcome_str[len(welcome_str)-1]
'u'
>>> welcome_str[:4]
'Welc'
>>> a = "python"
>>> a*3
'pythonpythonpython'

>>> git_list = ["qiwsir","github","io"]
>>> git_list[0]
'qiwsir'
>>> git_list[len(git_list)-1]
'io'
>>> git_list[0:2]
['qiwsir', 'github']
>>> b = ['qiwsir']
>>> b*7
['qiwsir', 'qiwsir', 'qiwsir', 'qiwsir', 'qiwsir', 'qiwsir', 'qiwsir'] 
```

对于此类数据，下面一些操作是类似的：

```py
>>> first = "hello,world"
>>> welcome_str
'Welcome you'
>>> first+","+welcome_str   #用 + 号连接 str
'hello,world,Welcome you'
>>> welcome_str             #原来的 str 没有受到影响，即上面的+号连接后重新生成了一个字符串
'Welcome you'
>>> first
'hello,world'

>>> language = ['python']
>>> git_list
['qiwsir', 'github', 'io']
>>> language + git_list     #用 + 号连接 list，得到一个新的 list
['python', 'qiwsir', 'github', 'io']
>>> git_list
['qiwsir', 'github', 'io']
>>> language
['Python']

>>> len(welcome_str)    # 得到字符数
11
>>> len(git_list)       # 得到元素数
3 
```

另外，前面的讲述中已经说明了关于序列的基本操作，此处不再重复。

### 区别

list 和 str 的最大区别是：list 是可以改变的，str 不可变。这个怎么理解呢？

首先看对 list 的这些操作，其特点是在原处将 list 进行了修改：

```py
>>> git_list
['qiwsir', 'github', 'io']

>>> git_list.append("python")
>>> git_list
['qiwsir', 'github', 'io', 'python']

>>> git_list[1]               
'github'
>>> git_list[1] = 'github.com'
>>> git_list
['qiwsir', 'github.com', 'io', 'python']

>>> git_list.insert(1,"algorithm")
>>> git_list
['qiwsir', 'algorithm', 'github.com', 'io', 'python']

>>> git_list.pop()
'python'

>>> del git_list[1]
>>> git_list
['qiwsir', 'github.com', 'io'] 
```

以上这些操作，如果用在 str 上，都会报错，比如：

```py
>>> welcome_str
'Welcome you'

>>> welcome_str[1]='E'
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment

>>> del welcome_str[1]
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: 'str' object doesn't support item deletion

>>> welcome_str.append("E")
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
AttributeError: 'str' object has no attribute 'append' 
```

如果要修改一个 str，不得不这样。

```py
>>> welcome_str
'Welcome you'
>>> welcome_str[0]+"E"+welcome_str[2:]  #从新生成一个 str
'WElcome you'
>>> welcome_str                         #对原来的没有任何影响
'Welcome you' 
```

其实，在这种做法中，相当于重新生成了一个 str。

### 多维 list

这个也应该算是两者的区别了，虽然有点牵强。在 str 中，里面的每个元素只能是字符，在 list 中，元素可以是任何类型的数据。前面见的多是数字或者字符，其实还可以这样：

```py
>>> matrix = [[1,2,3],[4,5,6],[7,8,9]]
>>> matrix = [[1,2,3],[4,5,6],[7,8,9]]
>>> matrix[0][1]
2
>>> mult = [[1,2,3],['a','b','c'],'d','e']
>>> mult
[[1, 2, 3], ['a', 'b', 'c'], 'd', 'e']
>>> mult[1][1]
'b'
>>> mult[2]
'd' 
```

以上显示了多维 list 以及访问方式。在多维的情况下，里面的 list 被当成一个元素对待。

### list 和 str 转化

以下涉及到的 `split()`和 `join()`在前面字符串部分已经见过。一回生，二回熟，这次再见面，特别是在已经学习了列表的基础上，应该有更深刻的理解。

#### str.split()

这个内置函数实现的是将 str 转化为 list。其中 str=""是分隔符。

在看例子之前，请看官在交互模式下做如下操作：

```py
>>>help(str.split) 
```

得到了对这个内置函数的完整说明。**特别强调：**这是一种非常好的学习方法

> split(...) S.split([sep [,maxsplit]]) -> list of strings
> 
> Return a list of the words in the string S, using sep as the delimiter string. If maxsplit is given, at most maxsplit splits are done. If sep is not specified or is None, any whitespace string is a separator and empty strings are removed from the result.

不管是否看懂上面这段话，都可以看例子。还是希望看官能够理解上面的内容。

```py
>>> line = "Hello.I am qiwsir.Welcome you." 

>>> line.split(".")     #以英文的句点为分隔符，得到 list
['Hello', 'I am qiwsir', 'Welcome you', '']

>>> line.split(".",1)   #这个 1,就是表达了上文中的：If maxsplit is given, at most maxsplit splits are done.
['Hello', 'I am qiwsir.Welcome you.']       

>>> name = "Albert Ainstain"    #也有可能用空格来做为分隔符
>>> name.split(" ")
['Albert', 'Ainstain'] 
```

下面的例子，让你更有点惊奇了。

```py
>>> s = "I am, writing\npython\tbook on line"   #这个字符串中有空格，逗号，换行\n，tab 缩进\t 符号
>>> print s         #输出之后的样式
I am, writing
python  book on line
>>> s.split()       #用 split(),但是括号中不输入任何参数
['I', 'am,', 'writing', 'Python', 'book', 'on', 'line'] 
```

如果 split()不输入任何参数，显示就是见到任何分割符号，就用其分割了。

#### "[sep]".join(list)

join 可以说是 split 的逆运算，举例：

```py
>>> name
['Albert', 'Ainstain']
>>> "".join(name)       #将 list 中的元素连接起来，但是没有连接符，表示一个一个紧邻着
'AlbertAinstain'
>>> ".".join(name)      #以英文的句点做为连接分隔符
'Albert.Ainstain'
>>> " ".join(name)      #以空格做为连接的分隔符
'Albert Ainstain' 
```

回到上面那个神奇的例子中，可以这么使用 join.

```py
>>> s = "I am, writing\npython\tbook on line" 
>>> print s
I am, writing
python  book on line
>>> s.split()
['I', 'am,', 'writing', 'Python', 'book', 'on', 'line']
>>> " ".join(s.split())         #重新连接，不过有一点遗憾，am 后面逗号还是有的。怎么去掉？
'I am, writing python book on line' 
```

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 元组

### 定义

先看一个例子：

```py
>>># 变量引用 str
>>> s = "abc"
>>> s
'abc'

>>>#如果这样写，就会是...
>>> t = 123,'abc',["come","here"]
>>> t
(123, 'abc', ['come', 'here']) 
```

上面例子中看到的变量 t，并没有报错，也没有“最后一个有效”，而是将对象做为一个新的数据类型：tuple（元组），赋值给了变量 t。

**元组是用圆括号括起来的，其中的元素之间用逗号隔开。（都是英文半角）**

元组中的元素类型是任意的 Python 数据。

> 这种类型，可以歪着想，所谓“元”组，就是用“圆”括号啦。
> 
> 其实，你不应该对元组陌生，还记得前面讲述字符串的格式化输出时，有这样一种方式：

```py
>>> print "I love %s, and I am a %s" % ('python', 'programmer')
I love Python, and I am a programmer 
```

> 这里的圆括号，就是一个元组。

显然，tuple 是一种序列类型的数据，这点上跟 list/str 类似。它的特点就是其中的元素不能更改，这点上跟 list 不同，倒是跟 str 类似；它的元素又可以是任何类型的数据，这点上跟 list 相同，但不同于 str。

```py
>>> t = 1,"23",[123,"abc"],("python","learn")   #元素多样性，近 list
>>> t
(1, '23', [123, 'abc'], ('python', 'learn'))

>>> t[0] = 8　                                  #不能原地修改，近 str
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment

>>> t.append("no")  
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'tuple' object has no attribute 'append'
    >>> 
```

从上面的简单比较似乎可以认为，tuple 就是一个融合了部分 list 和部分 str 属性的杂交产物。此言有理。

### 索引和切片

因为前面有了关于列表和字符串的知识，它们都是序列类型，元组也是。因此，元组的基本操作就和它们是一样的。

例如：

```py
>>> t
(1, '23', [123, 'abc'], ('python', 'learn'))
>>> t[2]
[123, 'abc']
>>> t[1:]
('23', [123, 'abc'], ('python', 'learn'))

>>> t[2][0]     #还能这样呀，哦对了，list 中也能这样
123
>>> t[3][1]
'learn' 
```

关于序列的基本操作在 tuple 上的表现，就不一一展示了。看官可以去试试。

但是这里要特别提醒，如果一个元组中只有一个元素的时候，应该在该元素后面加一个半角的英文逗号。

```py
>>> a = (3)
>>> type(a)
<type 'int'>

>>> b = (3,)
>>> type(b)
<type 'tuple'> 
```

以上面的例子说明，如果不加那个逗号，就不是元组，加了才是。这也是为了避免让 Python 误解你要表达的内容。

顺便补充：如果要想看一个对象是什么类型，可以使用 `type()`函数，然后就返回该对象的类型。

**所有在 list 中可以修改 list 的方法，在 tuple 中，都失效。**

分别用 list()和 tuple()能够实现两者的转化:

```py
>>> t         
(1, '23', [123, 'abc'], ('python', 'learn'))
>>> tls = list(t)                           #tuple-->list
>>> tls
[1, '23', [123, 'abc'], ('python', 'learn')]

>>> t_tuple = tuple(tls)                    #list-->tuple
>>> t_tuple
(1, '23', [123, 'abc'], ('python', 'learn')) 
```

### tuple 用在哪里？

既然它是 list 和 str 的杂合，它有什么用途呢？不是用 list 和 str 都可以了吗？

在很多时候，的确是用 list 和 str 都可以了。但是，看官不要忘记，我们用计算机语言解决的问题不都是简单问题，就如同我们的自然语言一样，虽然有的词汇看似可有可无,用别的也能替换之,但是我们依然需要在某些情况下使用它们.

一般认为,tuple 有这类特点,并且也是它使用的情景:

*   Tuple 比 list 操作速度快。如果您定义了一个值的常量集，并且唯一要用它做的是不断地遍历它，请使用 tuple 代替 list。
*   如果对不需要修改的数据进行 “写保护”，可以使代码更安全。使用 tuple 而不是 list 如同拥有一个隐含的 assert 语句，说明这一数据是常量。如果必须要改变这些值，则需要执行 tuple 到 list 的转换 (需要使用一个特殊的函数)。
*   Tuples 可以在 dictionary（字典，后面要讲述） 中被用做 key，但是 list 不行。Dictionary key 必须是不可变的。Tuple 本身是不可改变的，但是如果您有一个 list 的 tuple，那就认为是可变的了，用做 dictionary key 就是不安全的。只有字符串、整数或其它对 dictionary 安全的 tuple 才可以用作 dictionary key。
*   Tuples 可以用在字符串格式化中。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字典(1)

字典，这个东西你现在还用吗？随着网络的发展，用的人越来越少了。不少人习惯于在网上搜索，不仅有 web 版，乃至于已经有手机版的各种字典了。我在上小学的时候曾经用过一本小小的《新华字典》，记得是拾了不少废品，然后换钱，最终花费了 1.01 元人民币买的。

> 《新华字典》是中国第一部现代汉语字典。最早的名字叫《伍记小字典》，但未能编纂完成。自 1953 年，开始重编，其凡例完全采用《伍记小字典》。从 1953 年开始出版，经过反复修订，但是以 1957 年商务印书馆出版的《新华字典》作为第一版。原由新华辞书社编写，1956 年并入中科院语言研究所（现中国社科院语言研究所）词典编辑室。新华字典由商务印书馆出版。历经几代上百名专家学者 10 余次大规模的修订，重印 200 多次。成为迄今为止世界出版史上最高发行量的字典。

这里讲到字典，不是为了回忆青葱岁月。而是提醒看官想想我们如何使用字典：先查索引（不管是拼音还是偏旁查字），然后通过索引找到相应内容。不用从头开始一页一页地找。

这种方法能够快捷的找到目标。

正是基于这种需要，Python 中有了一种叫做 dictionary 的数据类型，翻译过来就是“字典”，用 dict 表示。

假设一种需要，要存储城市和电话区号，苏州的区号是 0512，唐山的是 0315，北京的是 011，上海的是 012。用前面已经学习过的知识，可以这么来做：

```py
>>> citys = ["suzhou", "tangshan", "beijing", "shanghai"]
>>> city_codes = ["0512", "0315", "011", "012"] 
```

用一个列表来存储城市名称，然后用另外一个列表，一一对应地保存区号。假如要输出苏州的区号，可以这么做：

```py
>>> print "{} : {}".format(citys[0], city_codes[0])
suzhou : 0512 
```

> 请特别注意，我在 city_codes 中，表示区号的元素没有用整数型，而是使用了字符串类型，你知道为什么吗？ 如果用整数，就是这样的。

```py
>>> suzhou_code = 0512
>>> print suzhou_code
330 
```

> 怎么会这样？原来在 Python 中，如果按照上面那样做，0512 并没有被认为是一个八进制的数，用 print 打印的时候，将它转换为了十进制输出。关于进制转换问题，看官可以网上搜索一下有关资料。此处不详述。一般是用几个内建函数实现：`int()`, `bin()`, `oct()`, `hex()`

这样来看，用两个列表分别来存储城市和区号，似乎能够解决问题。但是，这不是最好的选择，至少在 Python 里面。因为 Python 还提供了另外一种方案，那就是字典(dict)。

### 创建 dict

**方法 1:**

创建一个空的 dict，这个空 dict，可以在以后向里面加东西用。

```py
>>> mydict = {}
>>> mydict
{} 
```

不要小看“空”，“色即是空，空即是色”，在编程中，“空”是很重要。一般带“空”字的人都很有名，比如孙悟空，哦。好像他应该是猴、或者是神。举一个人的名字，带“空”字，你懂得。

创建有内容的 dict。

```py
>>> person = {"name":"qiwsir","site":"qiwsir.github.io","language":"python"}
>>> person
{'name': 'qiwsir', 'language': 'python', 'site': 'qiwsir.github.io'} 
```

`"name":"qiwsir"`，有一个优雅的名字：键值对。前面的 name 叫做键（key），后面的 qiwsir 是前面的键所对应的值(value)。在一个 dict 中，键是唯一的，不能重复。值则是对应于键，值可以重复。键值之间用(:)英文的分号，每一对键值之间用英文的逗号(,)隔开。

```py
>>> person['name2']="qiwsir"    #这是一种向 dict 中增加键值对的方法
>>> person
{'name2': 'qiwsir', 'name': 'qiwsir', 'language': 'Python', 'site': 'qiwsir.github.io'} 
```

用这样的方法可以向一个 dict 类型的数据中增加“键值对”，也可以说是增加数值。那么，增加了值之后，那个字典还是原来的吗？也就是也要同样探讨一下，字典是否能原地修改？（列表可以，所以列表是可变的；字符串和元组都不行，所以它们是不可变的。）

```py
>>> ad = {}
>>> id(ad)
3072770636L
>>> ad["name"] = "qiwsir"
>>> ad
{'name': 'qiwsir'}
>>> id(ad)
3072770636L 
```

实验表明，字典可以原地修改，即它是可变的。

**方法 2:**

利用元组在建构字典，方法如下：

```py
>>> name = (["first","Google"],["second","Yahoo"])      
>>> website = dict(name)
>>> website
{'second': 'Yahoo', 'first': 'Google'} 
```

或者用这样的方法：

```py
>>> ad = dict(name="qiwsir", age=42)
>>> ad
{'age': 42, 'name': 'qiwsir'} 
```

**方法 3:**

这个方法，跟上面的不同在于使用 fromkeys

```py
>>> website = {}.fromkeys(("third","forth"),"facebook")
>>> website
{'forth': 'facebook', 'third': 'facebook'} 
```

需要提醒的是，这种方法是重新建立一个 dict。

需要提醒注意的是，在字典中的“键”，必须是不可变的数据类型；“值”可以是任意数据类型。

```py
>>> dd = {(1,2):1}
>>> dd
{(1, 2): 1}
>>> dd = {[1,2]:1}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list' 
```

### 访问 dict 的值

dict 数据类型是以键值对的形式存储数据的，所以，只要知道键，就能得到值。这本质上就是一种映射关系。

> 映射，就好比“物体”和“影子”的关系，“形影相吊”，两者之间是映射关系。此外，映射也是一个严格数学概念：A 是非空集合，A 到 B 的映射是指：A 中每个元素都对应到 B 中的某个元素。

既然是映射，就可以通过字典的“键”找到相应的“值”。

```py
>>> person
{'name2': 'qiwsir', 'name': 'qiwsir', 'language': 'python', 'site': 'qiwsir.github.io'}
>>> person['name']
'qiwsir'
>>> person['language']
'python' 
```

如同前面所讲，通过“键”能够增加 dict 中的“值”，通过“键”能够改变 dict 中的“值”，通过“键”也能够访问 dict 中的“值”。

本节开头那个城市和区号的关系，也可以用字典来存储和读取。

```py
>>> city_code = {"suzhou":"0512", "tangshan":"0315", "beijing":"011", "shanghai":"012"}
>>> print city_code["suzhou"]
0512 
```

既然 dict 是键值对的映射，就不用考虑所谓“排序”问题了，只要通过键就能找到值，至于这个键值对位置在哪里就不用考虑了。比如，刚才建立的 city_code

```py
>>> city_code
{'suzhou': '0512', 'beijing': '011', 'shanghai': '012', 'tangshan': '0315'} 
```

虽然这里显示的和刚刚赋值的时候顺序有别，但是不影响读取其中的值。

在 list 中，得到值是用索引的方法。那么在字典中有索引吗？当然没有，因为它没有顺序，哪里来的索引呢？所以，在字典中就不要什么索引和切片了。

> dict 中的这类以键值对的映射方式存储数据，是一种非常高效的方法，比如要读取值得时候，如果用列表，Python 需要从头开始读，直到找到指定的那个索引值。但是，在 dict 中是通过“键”来得到值。要高效得多。 正是这个特点，键值对这样的形式可以用来存储大规模的数据，因为检索快捷。规模越大越明显。所以，mongdb 这种非关系型数据库在大数据方面比较流行了。

### 基本操作

字典虽然跟列表有很大的区别，但是依然有不少类似的地方。它的基本操作：

*   len(d)，返回字典(d)中的键值对的数量
*   d[key]，返回字典(d)中的键(key)的值
*   d[key]=value，将值(value)赋给字典(d)中的键(key)
*   del d[key]，删除字典(d)的键(key)项（将该键值对删除）
*   key in d，检查字典(d)中是否含有键为 key 的项

下面依次进行演示。

```py
>>> city_code
{'suzhou': '0512', 'beijing': '011', 'shanghai': '012', 'tangshan': '0315'}
>>> len(city_code)
4 
```

以 city_code 为操作对象，len(city_code)的值是 4，表明有四组键值对，也可以说是四项。

```py
>>> city_code["nanjing"] = "025"
>>> city_code
{'suzhou': '0512', 'beijing': '011', 'shanghai': '012', 'tangshan': '0315', 'nanjing': '025'} 
```

向其中增加一项

```py
>>> city_code["beijing"] = "010"
>>> city_code
{'suzhou': '0512', 'beijing': '010', 'shanghai': '012', 'tangshan': '0315', 'nanjing': '025'} 
```

突然发现北京的区号写错了。可以这样修改。这进一步说明字典是可变的。

```py
>>> city_code["shanghai"]
'012'
>>> del city_code["shanghai"] 
```

通过 `city_code["shanghai"]`能够查看到该键(key)所对应的值(value)，结果发现也错了。干脆删除，用 del，将那一项都删掉。

```py
>>> city_code["shanghai"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'shanghai'
>>> "shanghai" in city_code
False 
```

因为键是"shanghai"的那个键值对项已经删除了，随意不能找到，用 `in` 来看看，返回的是 `False`。

```py
>>> city_code
{'suzhou': '0512', 'beijing': '010', 'tangshan': '0315', 'nanjing': '025'} 
```

真的删除了哦。没有了。

### 字符串格式化输出

这是一个前面已经探讨过的话题，请参看《字符串(4)》，这里再次提到，就是因为用字典也可以实现格式化字符串的目的。虽然在《字符串(4)》那节中已经有了简单演示，但是我还是愿意重复一下。

```py
>>> city_code = {"suzhou":"0512", "tangshan":"0315", "hangzhou":"0571"}
>>> " Suzhou is a beautiful city, its area code is %(suzhou)s" % city_code
' Suzhou is a beautiful city, its area code is 0512' 
```

这种写法是非常简洁，而且很有意思的。有人说它简直是酷。

其实，更酷还是下面的——模板

在做网页开发的时候，通常要用到模板，也就是你只需要写好 HTML 代码，然后将某些部位空出来，等着 Python 后台提供相应的数据即可。当然，下面所演示的是玩具代码，基本没有什么使用价值，因为在真实的网站开发中，这样的姿势很少用上。但是，它绝非花拳绣腿，而是你能够明了其本质，至少了解到一种格式化方法的应用。

```py
>>> temp = "<html><head><title>%(lang)s<title><body><p>My name is %(name)s.</p></body></head></html>"
>>> my = {"name":"qiwsir", "lang":"python"}
>>> temp % my
'<html><head><title>python<title><body><p>My name is qiwsir.</p></body></head></html>' 
```

temp 就是所谓的模板，在双引号所包裹的实质上是一段 HTML 代码。然后在 dict 中写好一些数据，按照模板的要求在相应位置显示对应的数据。

是不是一个很有意思的屠龙之技？

> 什么是 HTML? 下面是在《维基百科》上抄录的：
> 
> 超文本标记语言（英文：HyperText Markup Language，HTML）是为「网页创建和其它可在网页浏览器中看到的信息」设计的一种标记语言。HTML 被用来结构化信息——例如标题、段落和列表等等，也可用来在一定程度上描述文档的外观和语义。1982 年由蒂姆·伯纳斯-李创建，由 IETF 用简化的 SGML（标准通用标记语言）语法进行进一步发展的 HTML，后来成为国际标准，由万维网联盟（W3C）维护。
> 
> HTML 经过发展，现在已经到 HTML5 了。现在的 HTML 设计，更强调“响应式”设计，就是能够兼顾 PC、手机和各种 PAD 的不同尺寸的显示器浏览。如果要开发一个网站，一定要做到“响应式”设计，否则就只能在 PC 上看，在手机上看就不得不左右移动。

### 知识

什么是关联数组？以下解释来自[维基百科](http://zh.wikipedia.org/wiki/%E5%85%B3%E8%81%94%E6%95%B0%E7%BB%84)

> 在计算机科学中，关联数组（英语：Associative Array），又称映射（Map）、字典（Dictionary）是一个抽象的数据结构，它包含着类似于（键，值）的有序对。一个关联数组中的有序对可以重复（如 C++ 中的 multimap）也可以不重复（如 C++ 中的 map）。
> 
> 这种数据结构包含以下几种常见的操作：
> 
> > 1.  向关联数组添加配对
> > 2.  从关联数组内删除配对
> > 3.  修改关联数组内的配对
> > 4.  根据已知的键寻找配对
> 
> 字典问题是设计一种能够具备关联数组特性的数据结构。解决字典问题的常用方法，是利用散列表，但有些情况下，也可以直接使用有地址的数组，或二叉树，和其他结构。
> 
> 许多程序设计语言内置基本的数据类型，提供对关联数组的支持。而 Content-addressable memory 则是硬件层面上实现对关联数组的支持。

什么是哈希表？关于哈希表的叙述比较多，这里仅仅截取了概念描述，更多的可以到[维基百科上阅读](http://zh.wikipedia.org/wiki/%E5%93%88%E5%B8%8C%E8%A1%A8)。

> 散列表（Hash table，也叫哈希表），是根据关键字（Key value）而直接访问在内存存储位置的数据结构。也就是说，它通过把键值通过一个函数的计算，映射到表中一个位置来访问记录，这加快了查找速度。这个映射函数称做散列函数，存放记录的数组称做散列表。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 字典(2)

### 字典方法

跟前面所讲述的其它数据类型类似，字典也有一些方法。通过这些方法，能够实现对字典类型数据的操作。这回可不是屠龙之技的。这些方法在编程实践中经常会用到。

#### copy

拷贝，这个汉语是 copy 的音译，标准的汉语翻译是“复制”。我还记得当初在学 DOS 的时候，那个老师说“拷贝”，搞得我晕头转向，他没有说英文的“copy”发音，而是用标准汉语说“kao(三声)bei(四声)”，对于一直学习过英语、标准汉语和我家乡方言的人来说，理解“拷贝”是有点困难的。谁知道在编程界用的是音译呢。

在一般的理解中，copy 就是将原来的东西再搞一份。但是，在 Python 里面（乃至于很多编程语言中），copy 可不是那么简单的。

```py
>>> a = 5
>>> b = a
>>> b
5 
```

这样做，是不是就得到了两个 5 了呢？表面上看似乎是，但是，不要忘记我在前面反复提到的：**对象有类型，变量无类型**，正是因着这句话，变量其实是一个标签。不妨请出法宝：`id()`，专门查看内存中对象编号

```py
>>> id(a)
139774080
>>> id(b)
139774080 
```

果然，并没有两个 5，就一个，只不过是贴了两张标签而已。这种现象普遍存在于 Python 的多种数据类型中。其它的就不演示了，就仅看看 dict 类型。

```py
>>> ad = {"name":"qiwsir", "lang":"Python"}
>>> bd = ad
>>> bd
{'lang': 'Python', 'name': 'qiwsir'}
>>> id(ad)
3072239652L
>>> id(bd)
3072239652L 
```

是的，验证了。的确是一个对象贴了两个标签。这是用赋值的方式，实现的所谓“假装拷贝”。那么如果用 copy 方法呢？

```py
>>> cd = ad.copy()
>>> cd
{'lang': 'Python', 'name': 'qiwsir'}
>>> id(cd)
3072239788L 
```

果然不同，这次得到的 cd 是跟原来的 ad 不同的，它在内存中另辟了一个空间。如果我尝试修改 cd，就应该对原来的 ad 不会造成任何影响。

```py
>>> cd["name"] = "itdiffer.com"
>>> cd 
{'lang': 'Python', 'name': 'itdiffer.com'}
>>> ad
{'lang': 'Python', 'name': 'qiwsir'} 
```

真的是那样，跟推理一模一样。所以，要理解了“变量”是对象的标签，对象有类型而变量无类型，就能正确推断出 Python 能够提供的结果。

```py
>>> bd
{'lang': 'Python', 'name': 'qiwsir'}
>>> bd["name"] = "laoqi"
>>> ad
{'lang': 'Python', 'name': 'laoqi'}
>>> bd
{'lang': 'Python', 'name': 'laoqi'} 
```

这是又修改了 bd 所对应的“对象”，结果发现 ad 的“对象”也变了。

然而，事情没有那么简单，看下面的，要仔细点，否则就迷茫了。

```py
>>> x = {"name":"qiwsir", "lang":["Python", "java", "c"]}
>>> y = x.copy()
>>> y
{'lang': ['Python', 'java', 'c'], 'name': 'qiwsir'}
>>> id(x)
3072241012L
>>> id(y)
3072241284L 
```

y 是从 x 拷贝过来的，两个在内存中是不同的对象。

```py
>>> y["lang"].remove("c") 
```

为了便于理解，尽量使用短句子，避免用很长很长的复合句。在 y 所对应的 dict 对象中，键"lang"的值是一个列表，为['Python', 'java', 'c']，这里用 `remove()`这个列表方法删除其中的一个元素"c"。删除之后，这个列表变为：['Python', 'java']

```py
>>> y
{'lang': ['Python', 'java'], 'name': 'qiwsir'} 
```

果然不出所料。那么，那个 x 所对应的字典中，这个列表变化了吗？应该没有变化。因为按照前面所讲的，它是另外一个对象，两个互不干扰。

```py
>>> x
{'lang': ['Python', 'java'], 'name': 'qiwsir'} 
```

是不是有点出乎意料呢？我没有作弊哦。你如果不信，就按照操作自己在交互模式中试试，是不是能够得到这个结果呢？这是为什么？

但是，如果要操作另外一个键值对：

```py
>>> y["name"] = "laoqi"
>>> y
{'lang': ['python', 'java'], 'name': 'laoqi'}
>>> x
{'lang': ['python', 'java'], 'name': 'qiwsir'} 
```

前面所说的原理是有效的，为什么到值是列表的时候就不奏效了呢？

要破解这个迷局还得用 `id()`

```py
>>> id(x)
3072241012L
>>> id(y)
3072241284L 
```

x,y 对应着两个不同对象，的确如此。但这个对象（字典）是由两个键值对组成的。其中一个键的值是列表。

```py
>>> id(x["lang"])
3072243276L
>>> id(y["lang"])
3072243276L 
```

发现了这样一个事实，列表是同一个对象。

但是，作为字符串为值得那个键值对，是分属不同对象。

```py
>>> id(x["name"])
3072245184L
>>> id(y["name"])
3072245408L 
```

这个事实，就说明了为什么修改一个列表，另外一个也跟着修改；而修改一个的字符串，另外一个不跟随的原因了。

但是，似乎还没有解开深层的原因。深层的原因，这跟 Python 存储的数据类型特点有关，Python 只存储基本类型的数据，比如 int,str，对于不是基础类型的，比如刚才字典的值是列表，Python 不会在被复制的那个对象中重新存储，而是用引用的方式，指向原来的值。如果读者没有明白这句话的意思，我就只能说点通俗的了（我本来不想说通俗的，装着自己有学问），Python 在所执行的复制动作中，如果是基本类型的数据，就在内存中重新建个窝，如果不是基本类型的，就不新建窝了，而是用标签引用原来的窝。这也好理解，如果比较简单，随便建立新窝简单；但是，如果对象太复杂了，就别费劲了，还是引用一下原来的省事。（这么讲有点忽悠了）。

所以，在编程语言中，把实现上面那种拷贝的方式称之为“浅拷贝”。顾名思义，没有解决深层次问题。言外之意，还有能够解决深层次问题的方法喽。

的确是，在 Python 中，有一个“深拷贝”(deep copy)。不过，要用下一 `import` 来导入一个模块。这个东西后面会讲述，前面也遇到过了。

```py
>>> import copy
>>> z = copy.deepcopy(x)
>>> z
{'lang': ['python', 'java'], 'name': 'qiwsir'} 
```

用 `copy.deepcopy()`深拷贝了一个新的副本，看这个函数的名字就知道是深拷贝(deepcopy)。用上面用过的武器 id()来勘察一番：

```py
>>> id(x["lang"])
3072243276L
>>> id(z["lang"])
3072245068L 
```

果然是另外一个“窝”，不是引用了。如果按照这个结果，修改其中一个列表中的元素，应该不影响另外一个了。

```py
>>> x
{'lang': ['Python', 'java'], 'name': 'qiwsir'}
>>> x["lang"].remove("java")
>>> x
{'lang': ['Python'], 'name': 'qiwsir'}
>>> z
{'lang': ['Python', 'java'], 'name': 'qiwsir'} 
```

果然如此。再试试，才过瘾呀。

```py
>>> x["lang"].append("c++")
>>> x
{'lang': ['Python', 'c++'], 'name': 'qiwsir'} 
```

这就是所谓浅拷贝和深拷贝。

#### clear

在交互模式中，用 help 是一个很好的习惯

```py
>>> help(dict.clear)

clear(...)
    D.clear() -> None.  Remove all items from D. 
```

这是一个清空字典中所有元素的操作。

```py
>>> a = {"name":"qiwsir"}
>>> a.clear()
>>> a
{} 
```

这就是 `clear` 的含义，将字典清空，得到的是“空”字典。这个上节说的 `del` 有着很大的区别。`del` 是将字典删除，内存中就没有它了，不是为“空”。

```py
>>> del a
>>> a
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'a' is not defined 
```

果然删除了。

另外，如果要清空一个字典，还能够使用 `a = {}`这种方法，但这种方法本质是将变量 a 转向了`{}`这个对象，那么原来的呢？原来的成为了断线的风筝。这样的东西在 Python 中称之为垃圾，而且 Python 能够自动的将这样的垃圾回收。编程者就不用关心它了，反正 Python 会处理了。

#### get,setdefault

get 的含义是：

```py
get(...)
    D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None. 
```

注意这个说明中，“if k in D”，就返回其值，否则...(等会再说)。

```py
>>> d
{'lang': 'python'}
>>> d.get("lang")
'python' 
```

`dict.get()`就是要得到字典中某个键的值，不过，它不是那么“严厉”罢了。因为类似获得字典中键的值得方法，上节已经有过，如 `d['lang']`就能得到对应的值`"Python"`，可是，如果要获取的键不存在，如：

```py
>>> print d.get("name")
None

>>> d["name"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'name' 
```

这就是 `dict.get()`和 `dict['key']`的区别。

前面有一个半句话，如果键不在字典中，会返回 None，这是一种情况。还可以这样：

```py
>>> d = {"lang":"Python"}
>>> newd = d.get("name",'qiwsir')
>>> newd
'qiwsir'
>>> d
{'lang': 'Python'} 
```

以 `d.get("name",'qiwsir')`的方式，如果不能得到键"name"的值，就返回后面指定的值"qiwsir"。这就是文档中那句话：`D[k] if k in D, else d.`的含义。这样做，并没有影响原来的字典。

另外一个跟 get 在功能上有相似地方的 `D.setdefault(k)`，其含义是：

```py
setdefault(...)
    D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D 
```

首先，它要执行 `D.get(k,d)`,就跟前面一样了，然后，进一步执行另外一个操作，如果键 k 不在字典中，就在字典中增加这个键值对。当然，如果有就没有必要执行这一步了。

```py
>>> d
{'lang': 'Python'}
>>> d.setdefault("lang")
'Python' 
```

在字典中，有"lang"这个键，那么就返回它的值。

```py
>>> d.setdefault("name","qiwsir")
'qiwsir'
>>> d
{'lang': 'Python', 'name': 'qiwsir'} 
```

没有"name"这个键，于是返回 `d.setdefault("name","qiwsir")`指定的值"qiwsir"，并且将键值对`'name':"qiwsir"`添加到原来的字典中。

如果这样操作：

```py
>>> d.setdefault("web") 
```

什么也没有返回吗？不是，返回了，只不过没有显示出来，如果你用 print 就能看到了。因为这里返回的是一个 None.不妨查看一下那个字典。

```py
>>> d
{'lang': 'Python', 'web': None, 'name': 'qiwsir'} 
```

是不是键"web"的值成为了 None

#### items/iteritems, keys/iterkeys, values/itervalues

这个标题中列出的是三组 dict 的函数，并且这三组有相似的地方。在这里详细讲述第一组，其余两组，我想凭借读者的聪明智慧是不在话下的。

```py
>>> help(dict.items)

items(...)
    D.items() -> list of D's (key, value) pairs, as 2-tuples 
```

这种方法是惯用的伎俩了，只要在交互模式中鼓捣一下，就能得到帮助信息。从中就知道 `D.items()`能够得到一个关于字典的列表，列表中的元素是由字典中的键和值组成的元组。例如：

```py
>>> dd = {"name":"qiwsir", "lang":"python", "web":"www.itdiffer.com"}
>>> dd_kv = dd.items()
>>> dd_kv
[('lang', 'Python'), ('web', 'www.itdiffer.com'), ('name', 'qiwsir')] 
```

显然，是有返回值的。这个操作，在后面要讲到的循环中，将有很大的作用。

跟 `items` 类似的就是 `iteritems`，看这个词的特点，是由 iter 和 items 拼接而成的，后部分 items 就不用说了，肯定是在告诉我们，得到的结果跟 `D.items()`的结果类似。是的，但是，还有一个 iter 是什么意思？在《列表(2)中，我提到了一个词“iterable”，它的含义是“可迭代的”，这里的 iter 是指的名词 iterator 的前部分，意思是“迭代器”。合起来，"iteritems"的含义就是：

```py
iteritems(...)
    D.iteritems() -> an iterator over the (key, value) items of D 
```

你看，学习 Python 不是什么难事，只要充分使用帮助文档就好了。这里告诉我们，得到的是一个“迭代器”（关于什么是迭代器，以及相关的内容，后续会详细讲述），这个迭代器是关于“D.items()”的。看个例子就明白了。

```py
>>> dd
{'lang': 'Python', 'web': 'www.itdiffer.com', 'name': 'qiwsir'}
>>> dd_iter = dd.iteritems()
>>> type(dd_iter)
<type 'dictionary-itemiterator'>
>>> dd_iter
<dictionary-itemiterator object at 0xb72b9a2c>
>>> list(dd_iter)
[('lang', 'Python'), ('web', 'www.itdiffer.com'), ('name', 'qiwsir')] 
```

得到的 dd_iter 的类型，是一个'dictionary-itemiterator'类型，不过这种迭代器类型的数据不能直接输出，必须用 `list()`转换一下，才能看到里面的真面目。

另外两组，含义跟这个相似，只不过是得到 key 或者 value。下面仅列举一下例子，具体内容，读者可以自行在交互模式中看文档。

```py
>>> dd
{'lang': 'Python', 'web': 'www.itdiffer.com', 'name': 'qiwsir'}
>>> dd.keys()
['lang', 'web', 'name']
>>> dd.values()
['Python', 'www.itdiffer.com', 'qiwsir'] 
```

这里先交代一句，如果要实现对键值对或者键或者值的循环，用迭代器的效率会高一些。对这句话的理解，在后面会给大家进行详细分析。

#### pop, popitem

在《列表(3)》中，有关于删除列表中元素的函数 `pop` 和 `remove`，这两个的区别在于 `list.remove(x)`用来删除指定的元素，而 `list.pop([i])`用于删除指定索引的元素，如果不提供索引值，就默认删除最后一个。

在字典中，也有删除键值对的函数。

```py
pop(...)
    D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
    If key is not found, d is returned if given, otherwise KeyError is raised 
```

`D.pop(k[,d])`是以字典的键为参数，删除指定键的键值对，当然，如果输入对应的值也可以，那个是可选的。

```py
>>> dd
{'lang': 'Python', 'web': 'www.itdiffer.com', 'name': 'qiwsir'}
>>> dd.pop("name")
'qiwsir' 
```

要删除指定键"name"，返回了其值"qiwsir"。这样，在原字典中，“'name':'qiwsir'”这个键值对就被删除了。

```py
>>> dd
{'lang': 'Python', 'web': 'www.itdiffer.com'} 
```

值得注意的是，pop 函数中的参数是不能省略的，这跟列表中的那个 pop 有所不同。

```py
>>> dd.pop()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pop expected at least 1 arguments, got 0 
```

如果要删除字典中没有的键值对，也会报错。

```py
>>> dd.pop("name")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'name' 
```

有意思的是 `D.popitem()`倒是跟 `list.pop()`有相似之处，不用写参数（list.pop 是可以不写参数），但是，`D.popitem()`不是删除最后一个，前面已经交代过了，dict 没有顺序，也就没有最后和最先了，它是随机删除一个，并将所删除的返回。

```py
popitem(...)
    D.popitem() -> (k, v), remove and return some (key, value) pair as a 
    2-tuple; but raise KeyError if D is empty. 
```

如果字典是空的，就要报错了

```py
>>> dd
{'lang': 'Python', 'web': 'www.itdiffer.com'}
>>> dd.popitem()
('lang', 'Python')
>>> dd
{'web': 'www.itdiffer.com'} 
```

成功地删除了一对，注意是随机的，不是删除前面显示的最后一个。并且返回了删除的内容，返回的数据格式是 tuple

```py
>>> dd.popitems()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'dict' object has no attribute 'popitems' 
```

错了？！注意看提示信息，没有那个...，哦，果然错了。注意是 popitem，不要多了 s，前面的 `D.items()`中包含 s，是复数形式，说明它能够返回多个结果（多个元组组成的列表），而在 `D.popitem()`中，一次只能随机删除一对键值对，并以一个元组的形式返回，所以，要单数形式，不能用复数形式了。

```py
>>> dd.popitem()
('web', 'www.itdiffer.com')
>>> dd 
{} 
```

都删了，现在那个字典成空的了。如果再删，会怎么样？

```py
>>> dd.popitem()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'popitem(): dictionary is empty' 
```

报错信息中明确告知，字典已经是空的了，没有再供删的东西了。

#### update

`update()`,看名字就猜测到一二了，是不是更新字典内容呢？的确是。

```py
update(...)
    D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
    If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
    If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
    In either case, this is followed by: for k in F: D[k] = F[k] 
```

不过，看样子这个函数有点复杂。不要着急，通过实验可以一点一点鼓捣明白的。

首先，这个函数没有返回值，或者说返回值是 None,它的作用就是更新字典。其参数可以是字典或者某种可迭代的数据类型。

```py
>>> d1 = {"lang":"python"}
>>> d2 = {"song":"I dreamed a dream"}
>>> d1.update(d2)
>>> d1
{'lang': 'Python', 'song': 'I dreamed a dream'}
>>> d2
{'song': 'I dreamed a dream'} 
```

这样就把字典 d2 更新入了 d1 那个字典，于是 d1 中就多了一些内容，把 d2 的内容包含进来了。d2 当然还存在，并没有受到影响。

还可以用下面的方法更新：

```py
>>> d2
{'song': 'I dreamed a dream'}
>>> d2.update([("name","qiwsir"), ("web","itdiffer.com")])
>>> d2
{'web': 'itdiffer.com', 'name': 'qiwsir', 'song': 'I dreamed a dream'} 
```

列表的元组是键值对。

#### has_key

这个函数的功能是判断字典中是否存在某个键

```py
has_key(...)
    D.has_key(k) -> True if D has a key k, else False 
```

跟前一节中遇到的 `k in D` 类似。

```py
>>> d2
{'web': 'itdiffer.com', 'name': 'qiwsir', 'song': 'I dreamed a dream'}
>>> d2.has_key("web")
True
>>> "web" in d2
True 
```

关于 dict 的函数，似乎不少。但是，不用着急，也不用担心记不住，因为根本不需要记忆。只要会用搜索即可。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 集合(1)

回顾一下已经学过的数据类型:int/str/bool/list/dict/tuple

还真的不少了.

不过,Python 是一个发展的语言,没准以后还出别的呢.看官可能有疑问了,出了这么多的数据类型,我也记不住呀,特别是里面还有不少方法.

不要担心记不住,你只要记住爱因斯坦说的就好了.

> 爱因斯坦在美国演讲，有人问：“你可记得声音的速度是多少？你如何记下许多东西？”
> 
> 爱因斯坦轻松答道：“声音的速度是多少，我必须查辞典才能回答。因为我从来不记在辞典上已经印着的东西，我的记忆力是用来记忆书本上没有的东西。”

多么霸气的回答,这回答不仅仅霸气,更告诉我们一种方法:只要能够通过某种方法查找到的,就不需要记忆.

那么,上面那么多数据类型及其各种方法,都不需要记忆了,因为它们都可以通过下述方法但不限于这些方法查到(这句话的逻辑还是比较严密的,包括但不限于...)

*   交互模式下用 dir()或者 help()
*   google(不推荐 Xdu,原因自己体会啦)

在已经学过的数据类型中：

*   能够索引的，如 list/str，其中的元素可以重复
*   可变的，如 list/dict，即其中的元素/键值对可以原地修改
*   不可变的，如 str/int，即不能进行原地修改
*   无索引序列的，如 dict，即其中的元素（键值对）没有排列顺序

现在要介绍另外一种类型的数据，英文是 set，翻译过来叫做“集合”。它的特点是：有的可变，有的不可变；元素无次序，不可重复。

### 创建 set

tuple 算是 list 和 str 的杂合(杂交的都有自己的优势,上一节的末后已经显示了),那么 set 则可以堪称是 list 和 dict 的杂合.

set 拥有类似 dict 的特点:可以用{}花括号来定义；其中的元素没有序列,也就是是非序列类型的数据;而且,set 中的元素不可重复,这就类似 dict 的键.

set 也有一点 list 的特点:有一种集合可以原处修改.

下面通过实验,进一步理解创建 set 的方法:

```py
>>> s1 = set("qiwsir")  
>>> s1                  
set(['q', 'i', 's', 'r', 'w']) 
```

把 str 中的字符拆解开,形成 set.特别注意观察:qiwsir 中有两个 i，但是在 s1 中,只有一个 i,也就是集合中元素不能重复。

```py
>>> s2 = set([123,"google","face","book","facebook","book"])
>>> s2
set(['facebook', 123, 'google', 'book', 'face']) 
```

在创建集合的时候，如果发现了重复的元素，就会过滤一下，剩下不重复的。而且，从 s2 的创建可以看出，查看结果是显示的元素顺序排列与开始建立是不同，完全是随意显示的，这说明集合中的元素没有序列。

```py
>>> s3 = {"facebook",123}       #通过{}直接创建
>>> s3
set([123, 'facebook']) 
```

除了用 `set()`来创建集合。还可以使用`{}`的方式，但是这种方式不提倡使用，因为在某些情况下，Python 搞不清楚是字典还是集合。看看下面的探讨就发现问题了。

```py
>>> s3 = {"facebook",[1,2,'a'],{"name":"Python","lang":"english"},123}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'dict'

>>> s3 = {"facebook",[1,2],123}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list' 
```

从上述实验中,可以看出,通过{}无法创建含有 list/dict 元素的 set.

认真阅读报错信息，有这样的词汇：“unhashable”，在理解这个词之前，先看它的反义词“hashable”，很多时候翻译为“可哈希”，其实它有一个不是音译的名词“散列”，这个在《字典(1)》中有说明。网上搜一下，有不少文章对这个进行诠释。如果我们简单点理解，某数据“不可哈希”(unhashable)就是其可变，如 list/dict，都能原地修改，就是 unhashable。否则，不可变的，类似 str 那样不能原地修改，就是 hashable（可哈希）的。

对于前面已经提到的字典，其键必须是 hashable 数据，即不可变的。

现在遇到的集合，其元素也要是“可哈希”的。上面例子中，试图将字典、列表作为元素的元素，就报错了。而且报错信息中明确告知 list/dict 是不可哈希类型，言外之意，里面的元素都应该是可哈希类型。

继续探索一个情况:

```py
>>> s1
set(['q', 'i', 's', 'r', 'w'])
>>> s1[1] = "I"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'set' object does not support item assignment 
```

这里报错，进一步说明集合没有序列，不能用索引方式对其进行修改。

```py
>>> s1     
set(['q', 'i', 's', 'r', 'w'])
>>> lst = list(s1)
>>> lst
['q', 'i', 's', 'r', 'w']
>>> lst[1] = "I"
>>> lst
['q', 'I', 's', 'r', 'w'] 
```

分别用 `list()`和 `set()`能够实现两种数据类型之间的转化。

特别说明，利用 `set()`建立起来的集合是可变集合，可变集合都是 unhashable 类型的。

### set 的方法

还是用前面已经介绍过多次的自学方法,把 set 的有关内置函数找出来,看看都可以对 set 做什么操作.

```py
>>> dir(set)
['__and__', '__class__', '__cmp__', '__contains__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__iand__', '__init__', '__ior__', '__isub__', '__iter__', '__ixor__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__rand__', '__reduce__', '__reduce_ex__', '__repr__', '__ror__', '__rsub__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__xor__', 'add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection', 'intersection_update', 'isdisjoint', 'issubset', 'issuperset', 'pop', 'remove', 'symmetric_difference', 'symmetric_difference_update', 'union', 'update'] 
```

为了看的清楚,我把双划线 __ 开始的先删除掉(后面我们会有专题讲述这些):

> 'add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection', 'intersection_update', 'isdisjoint', 'issubset', 'issuperset', 'pop', 'remove', 'symmetric_difference', 'symmetric_difference_update', 'union', 'update'

然后用 help()可以找到每个函数的具体使用方法,下面列几个例子:

#### add, update

```py
>>> help(set.add)

Help on method_descriptor:

add(...)
Add an element to a set.  
This has no effect if the element is already present. 
```

下面在交互模式这个最好的实验室里面做实验:

```py
>>> a_set = {}              #我想当然地认为这样也可以建立一个 set
>>> a_set.add("qiwsir")     #报错.看看错误信息,居然告诉我 dict 没有 add.我分明建立的是 set 呀.
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'dict' object has no attribute 'add'
>>> type(a_set)             #type 之后发现,计算机认为我建立的是一个 dict     
<type 'dict'> 
```

特别说明一下,{}这个东西,在 dict 和 set 中都用.但是,如上面的方法建立的是 dict,不是 set.这是 Python 规定的.要建立 set,只能用前面介绍的方法了.

```py
>>> a_set = {'a','i'}       #这回就是 set 了吧
>>> type(a_set)
  <type 'set'>              #果然

>>> a_set.add("qiwsir")     #增加一个元素
>>> a_set                   #原处修改,即原来的 a_set 引用对象已经改变
set(['i', 'a', 'qiwsir'])

>>> b_set = set("python")
>>> type(b_set)
<type 'set'>
>>> b_set
set(['h', 'o', 'n', 'p', 't', 'y'])
>>> b_set.add("qiwsir")
>>> b_set
set(['h', 'o', 'n', 'p', 't', 'qiwsir', 'y'])

>>> b_set.add([1,2,3])      #报错.list 是不可哈希的，集合中的元素应该是 hashable 类型。
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'

>>> b_set.add('[1,2,3]')    #可以这样!
>>> b_set
set(['[1,2,3]', 'h', 'o', 'n', 'p', 't', 'qiwsir', 'y']) 
```

除了上面的增加元素方法之外,还能够从另外一个 set 中合并过来元素,方法是 set.update(s2)

```py
>>> help(set.update)
update(...)
    Update a set with the union of itself and others.

>>> s1
set(['a', 'b'])
>>> s2
set(['github', 'qiwsir'])
>>> s1.update(s2)       #把 s2 的元素并入到 s1 中.
>>> s1                  #s1 的引用对象修改
set(['a', 'qiwsir', 'b', 'github'])
>>> s2                  #s2 的未变
set(['github', 'qiwsir']) 
```

#### pop, remove, discard, clear

```py
>>> help(set.pop)
pop(...)
    Remove and return an arbitrary set element.
    Raises KeyError if the set is empty.

>>> b_set
set(['[1,2,3]', 'h', 'o', 'n', 'p', 't', 'qiwsir', 'y'])
>>> b_set.pop()     #从 set 中任意选一个删除,并返回该值
'[1,2,3]'
>>> b_set.pop()
'h'
>>> b_set.pop()
'o'
>>> b_set
set(['n', 'p', 't', 'qiwsir', 'y'])

>>> b_set.pop("n")  #如果要指定删除某个元素,报错了.
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pop() takes no arguments (1 given) 
```

set.pop()是从 set 中任意选一个元素,删除并将这个值返回.但是,不能指定删除某个元素.报错信息中就告诉我们了,pop()不能有参数.此外,如果 set 是空的了,也报错.这条是帮助信息告诉我们的,看官可以试试.

要删除指定的元素,怎么办?

```py
>>> help(set.remove)

remove(...)
    Remove an element from a set; it must be a member.    

    If the element is not a member, raise a KeyError. 
```

`set.remove(obj)`中的 obj,必须是 set 中的元素,否则就报错.试一试:

```py
>>> a_set
set(['i', 'a', 'qiwsir'])
>>> a_set.remove("i")
>>> a_set
set(['a', 'qiwsir'])
>>> a_set.remove("w")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'w' 
```

跟 remove(obj)类似的还有一个 discard(obj):

```py
>>> help(set.discard)

discard(...)
    Remove an element from a set if it is a member.

    If the element is not a member, do nothing. 
```

与 `help(set.remove)`的信息对比,看看有什么不同.discard(obj)中的 obj 如果是 set 中的元素,就删除,如果不是,就什么也不做,do nothing.新闻就要对比着看才有意思呢.这里也一样.

```py
>>> a_set.discard('a')
>>> a_set       
set(['qiwsir'])
>>> a_set.discard('b')
>>> 
```

在删除上还有一个绝杀,就是 set.clear(),它的功能是:Remove all elements from this set.(看官自己在交互模式下 help(set.clear))

```py
>>> a_set
set(['qiwsir'])
>>> a_set.clear()
>>> a_set
set([])
>>> bool(a_set)     #空了,bool 一下返回 False.
False 
```

### 知识

集合,也是一个数学概念(以下定义来自[维基百科](http://zh.wikipedia.org/wiki/%E9%9B%86%E5%90%88_%28%E6%95%B0%E5%AD%A6%29))

> 集合（或简称集）是基本的数学概念，它是集合论的研究对象。最简单的说法，即是在最原始的集合论─朴素集合论─中的定义，集合就是“一堆东西”。集合里的“东西”，叫作元素。若然 x 是集合 A 的元素，记作 x ∈ A。
> 
> 集合是现代数学中一个重要的基本概念。集合论的基本理论直到十九世纪末才被创立，现在已经是数学教育中一个普遍存在的部分，在小学时就开始学习了。这里对被数学家们称为“直观的”或“朴素的”集合论进行一个简短而基本的介绍；更详细的分析可见朴素集合论。对集合进行严格的公理推导可见公理化集合论。

在计算机中,集合是什么呢?同样来自[维基百科](http://zh.wikipedia.org/wiki/%E9%9B%86%E5%90%88_%28%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6%29),这么说的:

> 在计算机科学中，集合是一组可变数量的数据项（也可能是 0 个）的组合，这些数据项可能共享某些特征，需要以某种操作方式一起进行操作。一般来讲，这些数据项的类型是相同的，或基类相同（若使用的语言支持继承）。列表（或数组）通常不被认为是集合，因为其大小固定，但事实上它常常在实现中作为某些形式的集合使用。
> 
> 集合的种类包括列表，集，多重集，树和图。枚举类型可以是列表或集。

不管是否明白,貌似很厉害呀.

是的,所以本讲仅仅是对集合有一个入门.关于集合的更多操作如运算/比较等,还没有涉及呢.

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。

## 集合(2)

### 不变的集合

《集合(1)》中以 `set()`来建立集合，这种方式所创立的集合都是可原处修改的集合，或者说是可变的，也可以说是 unhashable

还有一种集合，不能在原处修改。这种集合的创建方法是用 `frozenset()`，顾名思义，这是一个被冻结的集合，当然是不能修改了，那么这种集合就是 hashable 类型——可哈希。

```py
>>> f_set = frozenset("qiwsir")
>>> f_set
frozenset(['q', 'i', 's', 'r', 'w'])
>>> f_set.add("python")             #报错，不能修改，则无此方法
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'frozenset' object has no attribute 'add'

>>> a_set = set("github")           #对比看一看，这是一个可以原处修改的 set
>>> a_set
set(['b', 'g', 'i', 'h', 'u', 't'])
>>> a_set.add("python")
>>> a_set
set(['b', 'g', 'i', 'h', 'python', 'u', 't']) 
```

### 集合运算

唤醒一下中学数学（准确说是高中数学中的一点知识）中关于集合的一点知识，当然，你如果是某个理工科的专业大学毕业，更应该熟悉集合之间的关系。

#### 元素与集合的关系

就一种关系，要么术语某个集合，要么不属于。

```py
>>> aset
set(['h', 'o', 'n', 'p', 't', 'y'])
>>> "a" in aset
False
>>> "h" in aset
True 
```

#### 集合与集合的关系

假设两个集合 A、B

*   A 是否等于 B，即两个集合的元素完全一样

在交互模式下实验

```py
>>> a           
set(['q', 'i', 's', 'r', 'w'])
>>> b
set(['a', 'q', 'i', 'l', 'o'])
>>> a == b
False
>>> a != b
True 
```

*   A 是否是 B 的子集，或者反过来，B 是否是 A 的超集。即 A 的元素也都是 B 的元素，但是 B 的元素比 A 的元素数量多。

判断集合 A 是否是集合 B 的子集，可以使用 `A<B`，返回 true 则是子集，否则不是。另外，还可以使用函数 `A.issubset(B)`判断。

```py
>>> a
set(['q', 'i', 's', 'r', 'w'])
>>> c
set(['q', 'i'])
>>> c<a     #c 是 a 的子集
True
>>> c.issubset(a)   #或者用这种方法，判断 c 是否是 a 的子集
True
>>> a.issuperset(c) #判断 a 是否是 c 的超集
True

>>> b
set(['a', 'q', 'i', 'l', 'o'])
>>> a<b     #a 不是 b 的子集
False
>>> a.issubset(b)   #或者这样做
False 
```

*   A、B 的并集，即 A、B 所有元素，如下图所示

![](img/11901.png)

可以使用的符号是“|”，是一个半角状态写的竖线，输入方法是在英文状态下，按下"shift"加上右方括号右边的那个键。找找吧。表达式是 `A | B`.也可使用函数 `A.union(B)`，得到的结果就是两个集合并集，注意，这个结果是新生成的一个对象，不是将结合 A 扩充。

```py
>>> a
set(['q', 'i', 's', 'r', 'w'])
>>> b
set(['a', 'q', 'i', 'l', 'o'])
>>> a | b                       #可以有两种方式，结果一样
set(['a', 'i', 'l', 'o', 'q', 's', 'r', 'w'])
>>> a.union(b)
set(['a', 'i', 'l', 'o', 'q', 's', 'r', 'w']) 
```

*   A、B 的交集，即 A、B 所公有的元素，如下图所示

![](img/11902.png)

```py
>>> a
set(['q', 'i', 's', 'r', 'w'])
>>> b
set(['a', 'q', 'i', 'l', 'o'])
>>> a & b       #两种方式，等价
set(['q', 'i'])
>>> a.intersection(b)
set(['q', 'i']) 
```

我在实验的时候，顺手敲了下面的代码，出现的结果如下，看官能解释一下吗？（思考题）

```py
>>> a and b
set(['a', 'q', 'i', 'l', 'o']) 
```

*   A 相对 B 的差（补），即 A 相对 B 不同的部分元素，如下图所示

![](img/11903.png)

```py
>>> a
set(['q', 'i', 's', 'r', 'w'])
>>> b
set(['a', 'q', 'i', 'l', 'o'])
>>> a - b
set(['s', 'r', 'w'])
>>> a.difference(b)
set(['s', 'r', 'w']) 
```

-A、B 的对称差集，如下图所示

![](img/11904.png)

```py
>>> a
set(['q', 'i', 's', 'r', 'w'])
>>> b
set(['a', 'q', 'i', 'l', 'o'])
>>> a.symmetric_difference(b)
set(['a', 'l', 'o', 's', 'r', 'w']) 
```

以上是集合的基本运算。在编程中，如果用到，可以用前面说的方法查找。不用死记硬背。

* * *

总目录

如果你认为有必要打赏我，请通过支付宝：**qiwsir@126.com**,不胜感激。